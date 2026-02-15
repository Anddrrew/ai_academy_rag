import logging

import httpx

from shared.types.Chunk import Chunk
from shared.config import config


class Embedder:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._url = config.embedding.url

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        if not chunks:
            return []
        texts = [c.text for c in chunks]
        response = httpx.post(
            self._url,
            json={"inputs": texts},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        response = httpx.post(
            self._url,
            json={"inputs": [text]},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


embedder = Embedder()
