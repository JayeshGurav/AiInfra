from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient, models
import structlog

from backend.core.config import Settings
from backend.monitoring.metrics import RETRIEVAL_LATENCY

logger = structlog.get_logger(__name__)


class QdrantVectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncQdrantClient(url=settings.qdrant_url)

    async def close(self) -> None:
        await self.client.close()

    async def ensure_collection(self, vector_size: int | None = None) -> None:
        if await self.client.collection_exists(self.settings.collection_name):
            return
        if vector_size is None:
            raise RuntimeError("Vector dimension is required to create the collection")
        await self.client.create_collection(
            collection_name=self.settings.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        logger.info("qdrant_collection_created", collection=self.settings.collection_name)

    async def initialize(self) -> None:
        if not await self.client.collection_exists(self.settings.collection_name):
            raise RuntimeError(
                "Qdrant collection is not initialized. Run POST /ingest to create it."
            )

    async def ensure_for_vector(self, vector: list[float]) -> None:
        await self.ensure_collection(len(vector))

    async def upsert(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            return
        await self.ensure_for_vector(documents[0]["vector"])
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, str(doc["id"]))),
                vector=doc["vector"],
                payload={"text": doc["text"], **doc["metadata"]},
            )
            for doc in documents
        ]
        await self.client.upsert(
            collection_name=self.settings.collection_name,
            points=points,
            wait=True,
        )

    async def delete_stale_document_chunks(
        self, document_name: str, ingestion_timestamp: str
    ) -> None:
        await self.client.delete(
            collection_name=self.settings.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_name",
                            match=models.MatchValue(value=document_name),
                        )
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="ingestion_timestamp",
                            match=models.MatchValue(value=ingestion_timestamp),
                        )
                    ],
                )
            ),
            wait=True,
        )

    async def search(self, vector: list[float], limit: int) -> list[dict[str, Any]]:
        with RETRIEVAL_LATENCY.time():
            result = await self.client.query_points(
                collection_name=self.settings.collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
            )
        return [
            {
                "text": point.payload.get("text", ""),
                "metadata": {
                    key: value for key, value in point.payload.items() if key != "text"
                },
                "retrieval_score": float(point.score or 0.0),
            }
            for point in result.points
        ]

    async def health(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:
            logger.exception("qdrant_health_check_failed")
            return False

    async def collection_health(self) -> dict[str, Any]:
        if not await self.client.collection_exists(self.settings.collection_name):
            return {"name": self.settings.collection_name, "status": "not_created"}
        info = await self.client.get_collection(self.settings.collection_name)
        return {
            "name": self.settings.collection_name,
            "status": str(info.status),
            "points_count": info.points_count,
        }

    async def collections(self) -> list[dict[str, Any]]:
        response = await self.client.get_collections()
        collection_names = [collection.name for collection in response.collections]
        result = []
        for name in collection_names:
            info = await self.client.get_collection(name)
            result.append(
                {
                    "name": name,
                    "points_count": info.points_count,
                    "status": str(info.status),
                }
            )
        return result
