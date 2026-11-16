from typing import Any

from backend.core.nvidia import NvidiaClient
from backend.monitoring.metrics import GENERATION_LATENCY

SYSTEM_PROMPT = """You are an enterprise knowledge assistant.

Answer only from the provided context.

If information is unavailable in the context, respond with:

'I cannot find sufficient information in the available documents.'

Never hallucinate. Never invent facts. Preserve numerical values exactly. Prioritize table data when available. Cite document source names in every answer.

Output:

Answer: Sources: Confidence:"""


class GenerationPipeline:
    def __init__(self, nvidia: NvidiaClient) -> None:
        self.nvidia = nvidia

    async def answer(self, query: str, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return "I cannot find sufficient information in the available documents."
        context = "\n\n".join(
            (
                f"[Source: {item['metadata'].get('document_name', 'unknown')}; "
                f"page: {item['metadata'].get('page_number', 'unknown')}]\n{item['text']}"
            )
            for item in chunks
        )
        with GENERATION_LATENCY.time():
            return await self.nvidia.generate(SYSTEM_PROMPT, query, context)
