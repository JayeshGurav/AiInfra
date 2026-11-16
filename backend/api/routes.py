from datetime import datetime, timezone
import re
from time import perf_counter
from typing import Any
from uuid import uuid4
from pathlib import Path, PureWindowsPath

import structlog
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, SecretStr
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from backend.monitoring.metrics import (
    END_TO_END_LATENCY,
    INGESTED_DOCUMENTS,
    QDRANT_HEALTHY,
    TOTAL_INGESTIONS,
    TOTAL_QUERIES,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


class APIKeyRequest(BaseModel):
    api_key: SecretStr = Field(min_length=8, max_length=512)


@router.get("/settings/status")
async def settings_status(request: Request) -> dict[str, bool]:
    return {"nvidia_api_key_configured": request.app.state.nvidia.api_key_configured}


@router.post("/settings/nvidia-key")
async def configure_nvidia_key(body: APIKeyRequest, request: Request) -> dict[str, bool]:
    request.app.state.nvidia.configure_api_key(body.api_key.get_secret_value())
    return {"nvidia_api_key_configured": True}


@router.get("/documents")
async def list_documents(request: Request) -> dict[str, Any]:
    data_dir: Path = request.app.state.settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    documents = sorted(
        (
            {"document_name": path.name, "size_bytes": path.stat().st_size}
            for path in data_dir.glob("*.pdf")
            if path.is_file()
        ),
        key=lambda document: document["document_name"].lower(),
    )
    return {"documents": documents}


@router.post("/documents")
async def upload_document(
    request: Request, file: UploadFile = File(...)
) -> dict[str, Any]:
    filename = PureWindowsPath(file.filename or "").name
    filename = re.sub(r"[^A-Za-z0-9._ -]", "_", filename).strip(" .")
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Upload a file with a .pdf extension")
    filename = f"{Path(filename).stem}.pdf"

    data_dir: Path = request.app.state.settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / filename
    temporary = data_dir / f".{uuid4().hex}.upload"
    maximum_size = 50 * 1024 * 1024
    total_size = 0
    try:
        with temporary.open("xb") as destination:
            header = await file.read(5)
            if not header.startswith(b"%PDF-"):
                raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF")
            destination.write(header)
            total_size = len(header)
            while content := await file.read(1024 * 1024):
                total_size += len(content)
                if total_size > maximum_size:
                    raise HTTPException(status_code=413, detail="PDF files must be 50 MB or smaller")
                destination.write(content)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    TOTAL_INGESTIONS.inc()
    try:
        chunks = await request.app.state.ingestion.ingest_pdf(target)
    except Exception as exc:
        logger.exception("uploaded_document_ingestion_failed", document=filename)
        raise HTTPException(
            status_code=502,
            detail=f"{filename} was saved, but ingestion failed. Check the API key and service logs.",
        ) from exc
    INGESTED_DOCUMENTS.inc()
    return {
        "document_name": filename,
        "size_bytes": total_size,
        "chunks": chunks,
        "message": "PDF uploaded and indexed",
    }


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    expected_source_names: list[str] | None = None


class AskResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[dict[str, Any]]
    confidence: str


@router.post("/ingest")
async def ingest(request: Request) -> dict[str, Any]:
    TOTAL_INGESTIONS.inc()
    result = await request.app.state.ingestion.ingest_directory()
    INGESTED_DOCUMENTS.inc(result["documents"])
    return result


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    TOTAL_QUERIES.inc()
    request_id = str(uuid4())
    with END_TO_END_LATENCY.time():
        try:
            chunks = await request.app.state.retrieval.retrieve(body.query)
            generation_started = perf_counter()
            answer = await request.app.state.generation.answer(body.query, chunks)
            generation_seconds = perf_counter() - generation_started
        except Exception as exc:
            logger.exception("ask_request_failed", request_id=request_id, query=body.query)
            raise HTTPException(status_code=502, detail="RAG request failed; check service logs") from exc
    top_score = chunks[0]["retrieval_score"] if chunks else 0.0
    confidence = "high" if top_score >= 0.75 else "medium" if top_score >= 0.5 else "low"
    sources = [
        {
            "document_name": item["metadata"].get("document_name"),
            "page_number": item["metadata"].get("page_number"),
            "chunk_id": item["metadata"].get("chunk_id"),
            "retrieval_score": item["retrieval_score"],
            "rerank_score": item["rerank_score"],
            "text": item["text"],
        }
        for item in chunks
    ]
    request.app.state.source_store.save(
        request_id, sources, datetime.now(timezone.utc).isoformat()
    )
    evaluation_metrics = request.app.state.evaluator.evaluate(
        body.query, answer, chunks, body.expected_source_names
    )
    logger.info(
        "query_answered",
        request_id=request_id,
        query=body.query,
        retrieved_chunks=[
            {
                "chunk_id": item["metadata"].get("chunk_id"),
                "text": item["text"],
                "retrieval_score": item["retrieval_score"],
                "rerank_score": item["rerank_score"],
            }
            for item in chunks
        ],
        generation_time_seconds=generation_seconds,
        evaluation_metrics=evaluation_metrics,
    )
    return AskResponse(
        request_id=request_id,
        answer=answer,
        sources=sources,
        confidence=confidence,
    )


@router.get("/sources/{request_id}")
async def sources(request_id: str, request: Request) -> dict[str, Any]:
    stored = request.app.state.source_store.get(request_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="No sources found for request_id")
    return {"request_id": request_id, "sources": stored}


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    qdrant_healthy = await request.app.state.vector_store.health()
    QDRANT_HEALTHY.set(1 if qdrant_healthy else 0)
    if not qdrant_healthy:
        raise HTTPException(status_code=503, detail={"status": "unhealthy", "qdrant": "down"})
    collection = await request.app.state.vector_store.collection_health()
    return {"status": "healthy", "qdrant": "up", "collection": collection}


@router.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/collections")
async def collections(request: Request) -> dict[str, Any]:
    try:
        return {"collections": await request.app.state.vector_store.collections()}
    except Exception as exc:
        logger.exception("collection_listing_failed")
        raise HTTPException(status_code=503, detail="Unable to query Qdrant collections") from exc
