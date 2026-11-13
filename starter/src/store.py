"""Persist dense vectors and scope metadata in Qdrant local mode."""
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from .models import Chunk

TOPICS = {"filesystem", "archive", "formats"}


def scope_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    if not filters:
        return None
    if set(filters) != {"topic"} or filters["topic"] not in TOPICS:
        raise ValueError(f"Invalid topic filter: {filters}")
    return models.Filter(must=[
        models.FieldCondition(key="metadata.topic", match=models.MatchValue(value=filters["topic"]))
    ])


class QdrantStore:
    def __init__(self, path: Path, dimensions: int) -> None:
        self.client = QdrantClient(path=str(path))
        self.dimensions = dimensions

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.client.close()

    def count(self, collection: str) -> int:
        return self.client.count(collection, exact=True).count

    def upsert(self, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors) or any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError("Mismatch between chunk count and vector shape")
        if not self.client.collection_exists(collection):
            self.client.create_collection(collection, vectors_config=models.VectorParams(
                size=self.dimensions, distance=models.Distance.COSINE,
            ))
        points = [
            models.PointStruct(id=chunk.chunk_id, vector=vector, payload={
                "document_id": chunk.document_id, "source": chunk.source, "page": chunk.page,
                "text": chunk.text, "metadata": chunk.metadata,
            })
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection, points, wait=True)
        return self.count(collection)

    def search(self, collection: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        points = self.client.query_points(collection, query=query_vector, limit=top_k,
                                          query_filter=scope_filter(filters)).points
        return [{"chunk_id": str(point.id), "score": point.score, **point.payload}
                for point in points]
