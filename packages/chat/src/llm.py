import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from openai import AsyncOpenAI, OpenAI

from shared.config import config
import prompts

MODEL_NAME = config.openai.chat_model


def _build_system_message(context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks) if context_chunks else ""
    return prompts.system(context=context)


class LLM:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client = OpenAI(api_key=config.openai.api_key)
        self._async_client = AsyncOpenAI(api_key=config.openai.api_key)

    def extract_search_query(self, question: str) -> str | None:
        """Rewrite a user question into a search-optimized query. Returns None if no search needed."""
        response = self._client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompts.extract_search_query()},
                {"role": "user", "content": question},
            ],
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten == "SKIP":
            self.logger.debug("Query skipped (no search needed): '%s'", question)
            return None
        self.logger.debug("Rewrote query: '%s' -> '%s'", question, rewritten)
        return rewritten

    def chat(self, question: str, context_chunks: list[str]) -> str:
        """Single-turn chat used by the /chat endpoint."""
        return self.chat_messages([{"role": "user", "content": question}], context_chunks)

    def chat_messages(self, messages: list[dict], context_chunks: list[str]) -> str:
        """Multi-turn chat used by /v1/chat/completions (non-streaming)."""
        full_messages = [{"role": "system", "content": _build_system_message(context_chunks)}] + messages
        self.logger.debug("Calling %s with %d messages", MODEL_NAME, len(full_messages))
        response = self._client.chat.completions.create(
            model=MODEL_NAME,
            messages=full_messages,
        )
        return response.choices[0].message.content

    async def stream(self, messages: list[dict], context_chunks: list[str]) -> AsyncIterator[str]:
        """Multi-turn streaming chat used by /v1/chat/completions (stream=True)."""
        full_messages = [{"role": "system", "content": _build_system_message(context_chunks)}] + messages
        self.logger.debug("Streaming %s with %d messages", MODEL_NAME, len(full_messages))
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        response = await self._async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=full_messages,
            stream=True,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [
                        {"index": 0, "delta": {"content": chunk.choices[0].delta.content}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"

        yield "data: [DONE]\n\n"


llm = LLM()
