import hashlib
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import config
from chunker import Chunk

class Store:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client = QdrantClient(host=config.qdrant.host, port=config.qdrant.port)

        if config.qdrant.force_clear:
            self._reset_storage()
        elif not self._client.collection_exists(config.qdrant.collection):
            self._create_storage()

    def _reset_storage(self) -> None:
        if self._client.collection_exists(config.qdrant.collection):
            self._client.delete_collection(config.qdrant.collection)
            self.logger.warning("Dropped collection '%s' (force_clear=true)", config.qdrant.collection)
        self._create_storage()

    def _create_storage(self) -> None:
        self._client.create_collection(
            collection_name=config.qdrant.collection,
            vectors_config=VectorParams(size=config.embedding.vector_size, distance=Distance.COSINE),
        )
        self.logger.info("Created collection '%s'", config.qdrant.collection)

    def _make_point_id(self, source: str, index: int) -> int:
        digest = hashlib.sha256(f"{source}:{index}".encode()).digest()
        return int.from_bytes(digest[:8], "big")


    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        points = [
            PointStruct(
                id=self._make_point_id(c.source, c.index),
                vector=vectors[i],
                payload={"text": c.text, "source": c.source, "index": c.index},
            )
            for i, c in enumerate(chunks)
        ]
        self.upsert(points)

    def upsert(self, points: list[PointStruct]) -> None:
        if not points:
            return
        self._client.upsert(collection_name=config.qdrant.collection, points=points)

    def search(self, vector: list[float], k: int = 5) -> list[PointStruct]:
        return self._client.search(
            collection_name=config.qdrant.collection,
            query_vector=vector,
            limit=k,
        )


store = Store()
