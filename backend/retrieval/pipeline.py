from typing import Any

import structlog

from backend.core.config import Settings
from backend.core.nvidia import NvidiaClient
from backend.monitoring.metrics import RERANK_LATENCY
from backend.retrieval.vector_store import QdrantVectorStore

logger = structlog.get_logger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        settings: Settings,
        nvidia: NvidiaClient,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.settings = settings
        self.nvidia = nvidia
        self.vector_store = vector_store

    async def retrieve(self, query: str) -> list[dict[str, Any]]:
        query_vector = (await self.nvidia.embed([query]))[0]
        candidates = await self.vector_store.search(query_vector, self.settings.retrieval_top_k)
        if not candidates:
            return []
        with RERANK_LATENCY.time():
            ranking = await self.nvidia.rerank(query, [item["text"] for item in candidates])
        ranked: list[dict[str, Any]] = []
        for index, score in ranking[: self.settings.rerank_top_k]:
            item = candidates[index]
            ranked.append({**item, "rerank_score": score})
        logger.info(
            "retrieval_completed",
            query=query,
            retrieved_count=len(candidates),
            reranked=[
                {
                    "chunk_id": item["metadata"].get("chunk_id"),
                    "retrieval_score": item["retrieval_score"],
                    "rerank_score": item["rerank_score"],
                }
                for item in ranked
            ],
        )
        return ranked
