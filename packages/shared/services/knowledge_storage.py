import hashlib
import logging
from xmlrpc import client

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams, FilterSelector,  Filter

from shared.config import config
from shared.types.Chunk import Chunk


class KnowledgeStorage:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._collection = config.qdrant.collection
        self._client = QdrantClient(
            host=config.qdrant.host, port=config.qdrant.port)

        self._check_collection_on_init()

    def _check_collection_on_init(self) -> None:
        if (self._client.collection_exists(collection_name=self._collection)):
            return

        self._client.create_collection(
            collection_name=config.qdrant.collection,
            vectors_config=VectorParams(
                size=config.embedding.vector_size, distance=Distance.COSINE),
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
        self._client.upsert(
            collection_name=config.qdrant.collection, points=points)

    def search(self, vector: list[float], k: int = 5) -> list[ScoredPoint]:
        result = self._client.query_points(
            collection_name=config.qdrant.collection,
            query=vector,
            limit=k,
        )
        return result.points

    def reset_storage(self) -> None:
        self._client.delete(collection_name=self._collection,
                            points_selector=FilterSelector(filter=Filter()))

        self.logger.info(
            "Knowledge storage reset: all points deleted from collection '%s'", self._collection)
