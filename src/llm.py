import logging

from openai import OpenAI

from config import config

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using only the provided context.
If the context does not contain enough information to answer, say so honestly."""


class LLM:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client = OpenAI(api_key=config.llm.api_key)
        self._model = config.llm.model

    def chat(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(context_chunks)
        self.logger.debug("Calling %s with %d context chunks", self._model, len(context_chunks))
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content


llm = LLM()
