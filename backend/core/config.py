from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    nvidia_api_key: SecretStr
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_embedding_model: str = "nvidia/nemotron-3.5-embed-1b"
    nvidia_rerank_model: str = "nvidia/nemotron-3.5-rerank"
    nvidia_chat_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1"
    nvidia_rerank_url: str = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "enterprise_rag"
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    retrieval_top_k: int = Field(default=10, gt=0)
    rerank_top_k: int = Field(default=5, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0)
    api_timeout_seconds: float = Field(default=60, gt=0)
    tesseract_cmd: str = "tesseract"

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_chunk_size(cls, value: int, info: ValidationInfo) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @property
    def nvidia_key(self) -> str:
        return self.nvidia_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()
