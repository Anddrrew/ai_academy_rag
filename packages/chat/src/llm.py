import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from openai import AsyncOpenAI, OpenAI

from shared.config import config

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using only the provided context.
If the context does not contain enough information to answer, say so honestly."""

SOURCES_INSTRUCTION = """At the end of your answer add a "Sources:" section listing each source as a markdown link, exactly as provided in the context (e.g. [filename.pdf](url))."""


def _inject_context(messages: list[dict], context_chunks: list[str]) -> list[dict]:
    if context_chunks:
        context = "\n\n".join(context_chunks)
        system_content = f"{SYSTEM_PROMPT}\n\n{SOURCES_INSTRUCTION}\n\nRelevant context:\n{context}"
    else:
        system_content = SYSTEM_PROMPT
    return [{"role": "system", "content": system_content}] + messages


class LLM:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client = OpenAI(api_key=config.llm.api_key)
        self._async_client = AsyncOpenAI(api_key=config.llm.api_key)
        self._model = config.llm.model

    def chat(self, question: str, context_chunks: list[str]) -> str:
        """Single-turn chat used by the legacy /chat endpoint."""
        self.logger.debug("Calling %s with %d context chunks", self._model, len(context_chunks))
        messages = _inject_context([{"role": "user", "content": question}], context_chunks)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content

    def chat_messages(self, messages: list[dict], context_chunks: list[str]) -> str:
        """Multi-turn chat used by /v1/chat/completions (non-streaming)."""
        full_messages = _inject_context(messages, context_chunks)
        self.logger.debug("Calling %s with %d messages", self._model, len(full_messages))
        response = self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
        )
        return response.choices[0].message.content

    async def stream(self, messages: list[dict], context_chunks: list[str]) -> AsyncIterator[str]:
        """Multi-turn streaming chat used by /v1/chat/completions (stream=True)."""
        full_messages = _inject_context(messages, context_chunks)
        self.logger.debug("Streaming %s with %d messages", self._model, len(full_messages))
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            stream=True,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self._model,
                    "choices": [{"index": 0, "delta": {"content": chunk.choices[0].delta.content}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"

        yield "data: [DONE]\n\n"


llm = LLM()
