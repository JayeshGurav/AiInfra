from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from backend.core.config import Settings
from backend.core.nvidia import NvidiaClient
from backend.ingestion.chunker import FixedChunker
from backend.ingestion.extractor import PDFExtractor
from backend.retrieval.vector_store import QdrantVectorStore

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        nvidia: NvidiaClient,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.settings = settings
        self.nvidia = nvidia
        self.vector_store = vector_store
        self.chunker = FixedChunker(settings.chunk_size, settings.chunk_overlap)
        self.extractor = PDFExtractor(
            settings.tesseract_cmd, settings.data_dir / "extracted_images"
        )

    async def ingest_directory(self) -> dict[str, Any]:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(path for path in self.settings.data_dir.glob("*.pdf") if path.is_file())
        if not paths:
            return {"documents": 0, "chunks": 0, "errors": []}
        total_chunks = 0
        successful_documents = 0
        errors: list[dict[str, str]] = []
        for path in paths:
            try:
                chunks = await self.ingest_pdf(path)
                total_chunks += chunks
                successful_documents += 1
            except Exception as exc:
                logger.exception("document_ingestion_failed", document=path.name)
                errors.append({"document": path.name, "error": str(exc)})
        if errors and not successful_documents:
            raise RuntimeError(f"All {len(errors)} PDF documents failed ingestion")
        return {
            "documents": successful_documents,
            "chunks": total_chunks,
            "errors": errors,
        }

    async def ingest_pdf(self, path: Path) -> int:
        timestamp = datetime.now(timezone.utc).isoformat()
        pages = self.extractor.extract(path)
        chunk_rows = []
        for page in pages:
            metadata = {
                "document_name": path.name,
                "ingestion_timestamp": timestamp,
            }
            chunk_rows.extend(self.chunker.split(page["text"], page["page_number"], metadata))
        if not chunk_rows:
            logger.warning("pdf_has_no_extractable_text", document=path.name)
            await self.vector_store.delete_stale_document_chunks(path.name, timestamp)
            return 0
        for start in range(0, len(chunk_rows), self.settings.embedding_batch_size):
            batch = chunk_rows[start : start + self.settings.embedding_batch_size]
            vectors = await self.nvidia.embed([chunk.text for chunk in batch])
            documents = [
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "vector": vector,
                    "metadata": {
                        **chunk.metadata,
                        "chunk_id": chunk.chunk_id,
                        "token_count": len(self.chunker.encoder.encode(chunk.text)),
                    },
                }
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            await self.vector_store.upsert(documents)
        await self.vector_store.delete_stale_document_chunks(path.name, timestamp)
        logger.info("document_ingested", document=path.name, chunks=len(chunk_rows))
        return len(chunk_rows)
