"""
Ollama LLM client — handles communication with the local LLM.

Ollama serves models via a REST API at /api/generate.
We use non-streaming by default for simpler response parsing,
with an optional streaming method for real-time UX.
"""

from __future__ import annotations

import json

import httpx
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import settings
from generation.prompts import SYSTEM_PROMPT, build_rag_prompt, build_context_block

logger = structlog.get_logger()


class LLMClient:
    """
    Async client for Ollama.

    Key responsibilities:
      - Build the RAG prompt from query + retrieved chunks
      - Call Ollama /api/generate
      - Parse response and extract token counts
      - Retry on transient failures
    """

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    # ── Health ───────────────────────────────────────────

    async def is_healthy(self) -> bool:
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    # ── Generate ─────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
    )
    async def generate(
        self,
        query: str,
        retrieved_chunks: list,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Build prompt → call Ollama → return structured result.

        Returns:
            {
                "answer": str,
                "tokens_used": int,
                "model": str,
            }
        """
        context = build_context_block(retrieved_chunks)
        prompt = build_rag_prompt(query, context)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }

        logger.debug("llm_request", model=self.model, prompt_chars=len(prompt))

        resp = await self._client.post("/api/generate", json=payload)
        resp.raise_for_status()

        data = resp.json()
        answer = data.get("response", "").strip()
        prompt_tokens = data.get("prompt_eval_count", 0)
        gen_tokens = data.get("eval_count", 0)
        total_tokens = prompt_tokens + gen_tokens

        logger.info(
            "llm_response",
            model=self.model,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            answer_chars=len(answer),
        )

        return {
            "answer": answer,
            "tokens_used": total_tokens,
            "model": self.model,
        }

    async def generate_stream(self, query: str, retrieved_chunks: list):
        """
        Streaming variant — yields answer tokens in real time.
        Useful for chat interfaces.
        """
        context = build_context_block(retrieved_chunks)
        prompt = build_rag_prompt(query, context)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": True,
            "options": {"temperature": 0.1},
        }

        async with self._client.stream("POST", "/api/generate", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break

    async def close(self):
        await self._client.aclose()
