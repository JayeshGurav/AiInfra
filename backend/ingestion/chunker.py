from dataclasses import dataclass
from typing import Any

import tiktoken


@dataclass(frozen=True)
class Chunk:
    text: str
    page_number: int
    chunk_id: str
    metadata: dict[str, Any]


class FixedChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            raise RuntimeError("Unable to initialize the token encoder") from exc
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, page_number: int, base_metadata: dict[str, Any]) -> list[Chunk]:
        token_ids = self.encoder.encode(text)
        if not token_ids:
            return []
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        for start in range(0, len(token_ids), step):
            segment = token_ids[start : start + self.chunk_size]
            if not segment:
                break
            index = len(chunks)
            chunks.append(
                Chunk(
                    text=self.encoder.decode(segment).strip(),
                    page_number=page_number,
                    chunk_id=f"{base_metadata['document_name']}:{page_number}:{index}",
                    metadata={**base_metadata, "page_number": page_number},
                )
            )
            if start + self.chunk_size >= len(token_ids):
                break
        return [chunk for chunk in chunks if chunk.text]
