import hashlib
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams, FilterSelector,  Filter

from shared.config import config
from shared.types.Chunk import Chunk

QRANT_HOST = config.qdrant.host
QRANT_PORT = config.qdrant.port
QRANT_COLLECTION_NAME = config.qdrant.collection

DEFAULT_SEARCH_LIMIT = config.qdrant.search_k
VECTOR_SIZE = config.embedding.vector_size


class KnowledgeStorage:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client = QdrantClient(host=QRANT_HOST, port=QRANT_PORT)
        self._check_collection_on_init()

    def _check_collection_on_init(self) -> None:
        if (self._client.collection_exists(collection_name=QRANT_COLLECTION_NAME)):
            return

        self._client.create_collection(
            collection_name=QRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        self.logger.info("Created collection '%s'", QRANT_COLLECTION_NAME)

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
            collection_name=QRANT_COLLECTION_NAME, points=points)

    def search(self, vector: list[float], k: int = DEFAULT_SEARCH_LIMIT) -> list[ScoredPoint]:
        result = self._client.query_points(
            collection_name=QRANT_COLLECTION_NAME,
            query=vector,
            limit=k,
        )
        return result.points

    def reset_storage(self) -> None:
        self._client.delete(collection_name=QRANT_COLLECTION_NAME,
                            points_selector=FilterSelector(filter=Filter()))

        self.logger.info(
            "Knowledge storage reset: all points deleted from collection '%s'", QRANT_COLLECTION_NAME)
