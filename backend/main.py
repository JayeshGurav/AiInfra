from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.core.nvidia import NvidiaClient
from backend.core.source_store import SourceStore
from backend.evaluation.evaluator import Evaluator
from backend.generation.pipeline import GenerationPipeline
from backend.ingestion.pipeline import IngestionPipeline
from backend.monitoring.metrics import QDRANT_HEALTHY
from backend.retrieval.pipeline import RetrievalPipeline
from backend.retrieval.vector_store import QdrantVectorStore

settings = get_settings()
configure_logging(settings.log_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    nvidia = NvidiaClient(settings)
    vector_store = QdrantVectorStore(settings)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    app.state.nvidia = nvidia
    app.state.vector_store = vector_store
    app.state.settings = settings
    app.state.ingestion = IngestionPipeline(settings, nvidia, vector_store)
    app.state.retrieval = RetrievalPipeline(settings, nvidia, vector_store)
    app.state.generation = GenerationPipeline(nvidia)
    app.state.source_store = SourceStore(settings.log_dir / "sources.sqlite3")
    app.state.evaluator = Evaluator(settings.log_dir / "evaluation.jsonl")
    try:
        QDRANT_HEALTHY.set(1 if await vector_store.health() else 0)
        yield
    finally:
        await nvidia.close()
        await vector_store.close()


app = FastAPI(
    title="Enterprise RAG API",
    version="1.0.0",
    description="Local-first RAG API with NVIDIA NIM and Qdrant.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(router)
