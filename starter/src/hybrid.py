"""Optional BM25 + cosine retrieval with reciprocal rank fusion."""
import math
import re
from collections import Counter

from .models import Chunk

CANDIDATES = 15
RRF_K = 60


class BM25:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.words = [Counter(re.findall(r"\w+", chunk.text.lower())) for chunk in chunks]
        self.lengths = [words.total() for words in self.words]
        self.mean = sum(self.lengths) / len(self.lengths)
        self.df = Counter(word for words in self.words for word in words)

    def rank(self, query: str, filters: dict | None) -> list[dict]:
        terms = set(re.findall(r"\w+", query.lower()))
        results = []
        for chunk, words, length in zip(self.chunks, self.words, self.lengths):
            if filters and chunk.metadata["topic"] != filters["topic"]:
                continue
            normalization = 1.5 * (0.25 + 0.75 * length / self.mean)
            score = 0.0
            for term in sorted(terms & words.keys()):
                frequency = words[term]
                df = self.df[term]
                idf = math.log1p((len(self.chunks) - df + 0.5) / (df + 0.5))
                score += idf * frequency * 2.5 / (frequency + normalization)
            if score:
                results.append({"chunk_id": chunk.chunk_id, "score": score,
                                "document_id": chunk.document_id, "source": chunk.source,
                                "page": chunk.page, "text": chunk.text, "metadata": chunk.metadata})
        return sorted(results, key=lambda hit: (-hit["score"], hit["chunk_id"]))[:CANDIDATES]


def fuse(dense: list[dict], lexical: list[dict], limit: int = 3) -> list[dict]:
    combined = {}
    for method, hits in (("cosine", dense), ("bm25", lexical)):
        for rank, hit in enumerate(hits, 1):
            result = combined.setdefault(hit["chunk_id"], {**hit, "score": 0.0})
            result[method + "_rank"] = rank
            result[method + "_score"] = hit["score"]
            result["score"] += 1 / (RRF_K + rank)
    return sorted(combined.values(), key=lambda hit: (-hit["score"], hit["chunk_id"]))[:limit]
