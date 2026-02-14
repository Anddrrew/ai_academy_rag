import logging

from langchain_huggingface import HuggingFaceEmbeddings

from chunker import Chunk
from config import config


class Embedder:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model = HuggingFaceEmbeddings(model_name=config.embedding.model)

    def embed_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self._model.embed_documents([c.text for c in chunks])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)


embedder = Embedder()
