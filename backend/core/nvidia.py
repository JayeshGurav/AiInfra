import asyncio
from collections.abc import Sequence
from typing import Any

import httpx
import structlog

from backend.core.config import Settings
from backend.monitoring.metrics import EMBEDDING_LATENCY

logger = structlog.get_logger(__name__)


class NvidiaAPIError(RuntimeError):
    pass


class NvidiaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._api_key_configured = not settings.nvidia_key.strip().lower().startswith(
            "replace_with"
        )
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.api_timeout_seconds),
            headers={
                "Authorization": f"Bearer {settings.nvidia_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def api_key_configured(self) -> bool:
        return self._api_key_configured

    def configure_api_key(self, api_key: str) -> None:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ValueError("NVIDIA API key cannot be empty")
        self._client.headers["Authorization"] = f"Bearer {normalized_key}"
        self._api_key_configured = True

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(3):
            try:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise NvidiaAPIError("NVIDIA API returned a non-object response")
                return body
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                retryable = status is None or status == 429 or status >= 500
                if attempt == 2 or not retryable:
                    logger.exception("nvidia_api_request_failed", status=status, attempt=attempt + 1)
                    raise NvidiaAPIError(f"NVIDIA API request failed (status={status})") from exc
                await asyncio.sleep(0.5 * (2**attempt))
        raise NvidiaAPIError("NVIDIA API retry loop exhausted")

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        all_vectors: list[list[float]] = []
        batch_size = self.settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            with EMBEDDING_LATENCY.time():
                body = await self._post(
                    f"{self.settings.nvidia_base_url.rstrip('/')}/embeddings",
                    {"model": self.settings.nvidia_embedding_model, "input": batch},
                )
            try:
                rows = sorted(body["data"], key=lambda item: item["index"])
                vectors = [row["embedding"] for row in rows]
                if len(vectors) != len(batch):
                    raise ValueError("Embedding response length does not match request")
                if not all(isinstance(value, (float, int)) for vector in vectors for value in vector):
                    raise ValueError("Embedding response contains non-numeric values")
                all_vectors.extend([[float(value) for value in vector] for vector in vectors])
            except (KeyError, TypeError, ValueError) as exc:
                raise NvidiaAPIError("Invalid NVIDIA embedding response") from exc
        return all_vectors

    async def rerank(self, query: str, passages: Sequence[str]) -> list[tuple[int, float]]:
        if not passages:
            return []
        payload = {
            "model": self.settings.nvidia_rerank_model,
            "query": {"text": query},
            "passages": [{"text": text} for text in passages],
            "truncate": "END",
        }
        body = await self._post(self.settings.nvidia_rerank_url, payload)
        raw = body.get("rankings", body.get("data"))
        if not isinstance(raw, list):
            raise NvidiaAPIError("Invalid NVIDIA reranker response: expected rankings list")
        ranked: list[tuple[int, float]] = []
        try:
            for entry in raw:
                index = int(entry.get("index", entry.get("document_index")))
                score = float(entry.get("logit", entry.get("relevance_score", entry.get("score"))))
                if index < 0 or index >= len(passages):
                    raise ValueError("Reranker returned an out-of-range passage index")
                ranked.append((index, score))
        except (AttributeError, TypeError, ValueError) as exc:
            raise NvidiaAPIError("Invalid ranking entry from NVIDIA reranker") from exc
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    async def generate(self, system_prompt: str, query: str, context: str) -> str:
        body = await self._post(
            f"{self.settings.nvidia_base_url.rstrip('/')}/chat/completions",
            {
                "model": self.settings.nvidia_chat_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Question:\n{query}\n\nContext:\n{context}",
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 2048,
            },
        )
        try:
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty completion")
            return content.strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise NvidiaAPIError("Invalid NVIDIA chat completion response") from exc
