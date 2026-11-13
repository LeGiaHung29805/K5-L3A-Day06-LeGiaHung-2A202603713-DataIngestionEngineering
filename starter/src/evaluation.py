"""Evaluate per-chunk evidence and keep the full retrieval trace."""
from collections import Counter
import json
from pathlib import Path
from typing import Any

from .ingestion import normalize_text
from .models import DocumentPage
from .store import scope_filter


def comparable(text: str) -> str:
    return " ".join(normalize_text(text).split())


def gold_evidence(question: dict) -> list[dict]:
    return [question, *question.get("alternatives", [])]


def load_questions(path: Path, pages: list[DocumentPage]) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    sources = {(page.document_id, page.page): page for page in pages}
    splits = Counter()
    scoped = Counter()
    ids = set()
    for q in questions:
        if q["id"] in ids or q["split"] not in {"dev", "test"}:
            raise ValueError(f"Duplicate ID or invalid split: {q['id']}")
        if not q["question"].strip() or not q["expected_answer"].strip():
            raise ValueError(f"Empty question or answer: {q['id']}")
        ids.add(q["id"])
        splits[q["split"]] += 1
        filters = q.get("filter", {})
        if filters:
            scope_filter(filters)
            if not q.get("filter_context", "").strip():
                raise ValueError(f"No filter context: {q['id']}")
            scoped[q["split"]] += 1
        for gold in gold_evidence(q):
            page = sources.get((gold["gold_source_id"], gold["gold_page"]))
            quote = comparable(gold["evidence_quote"])
            if not page or not quote or quote not in comparable(page.text):
                raise ValueError(f"Gold quote missing from declared source: {q['id']}")
            if filters and page.metadata["topic"] != filters["topic"]:
                raise ValueError(f"Gold falls outside user scope: {q['id']}")
    if splits["dev"] < 15 or splits["test"] < 5 or scoped.total() < 3 or scoped["test"] < 1:
        raise ValueError("Need 15 dev, 5 test, and 3 scoped questions including one test")
    return questions


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = [
        (gold["gold_source_id"], gold["gold_page"], comparable(gold["evidence_quote"]))
        for gold in gold_evidence(question)
    ]
    ranked = []
    for rank, hit in enumerate(hits[:3], 1):
        found = any(source == hit["document_id"] and page == hit["page"]
                    and quote in comparable(hit["text"]) for source, page, quote in evidence)
        ranked.append({**hit, "rank": rank, "evidence_hit": found})
    filters = question.get("filter")
    compliance = None
    if filters:
        compliance = bool(ranked) and all(hit["metadata"]["topic"] == filters["topic"] for hit in ranked)
    return {**question, "hits": ranked, "hit1": bool(ranked and ranked[0]["evidence_hit"]),
            "hit3": any(hit["evidence_hit"] for hit in ranked), "filter_compliant": compliance}


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for strategy in sorted({row["strategy"] for row in traces}):
        group = [row for row in traces if row["strategy"] == strategy]
        scoped = [row for row in group if row["filter_compliant"] is not None]
        summary[strategy] = {
            "questions": len(group), "hit1": sum(row["hit1"] for row in group),
            "hit3": sum(row["hit3"] for row in group), "filter_questions": len(scoped),
            "filter_compliant": sum(row["filter_compliant"] for row in scoped) if scoped else None,
            "mean_retrieval_ms": round(sum(row["retrieval_ms"] for row in group) / len(group), 3),
        }
    return summary
