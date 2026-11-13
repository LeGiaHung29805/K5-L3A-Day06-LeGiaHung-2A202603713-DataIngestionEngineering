"""Index Python documentation, benchmark strategies, or retrieve passages."""
import argparse
import hashlib
from importlib.metadata import version
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from .chunking import by_paragraph, fixed_size
from .embeddings import GeminiEmbedder
from .evaluation import evaluate_query, load_questions, summarize
from .hybrid import BM25, CANDIDATES, RRF_K, fuse
from .ingestion import load_pages
from .store import QdrantStore, TOPICS

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "data/pages.jsonl"
MANIFEST = ROOT / "data/manifest.json"
QUESTIONS = ROOT / "benchmark/questions.json"
DEFAULT_DB = ROOT / "starter/data/.qdrant"
DEFAULT_RESULTS = ROOT / "starter/results"
SUBMISSION_RESULTS = ROOT / "results"
SETTINGS = {"size": 650, "overlap_a": 65, "top_k": 3,
            "hybrid_candidates": CANDIDATES, "rrf_k": RRF_K}


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def chunk_sets(pages):
    return {
        "A": [chunk for page in pages for chunk in fixed_size(
            page, SETTINGS["size"], SETTINGS["overlap_a"])],
        "B": [chunk for page in pages for chunk in by_paragraph(page, SETTINGS["size"])],
    }


def collection_names(chunks, model):
    # Identical snapshots reuse IDs. Changed chunk text or metadata gets a new collection.
    return {
        strategy: strategy + "_" + hashlib.sha256(json.dumps(
            [model.config, [vars(chunk) for chunk in items]],
            sort_keys=True, ensure_ascii=False,
        ).encode()).hexdigest()[:24]
        for strategy, items in chunks.items()
    }


def index(store, names, chunks, model, results, recheck):
    report = {}
    for strategy, items in chunks.items():
        logging.info("Embedding %s: %d chunks", strategy, len(items))
        started = perf_counter()
        usage_before = model.usage_snapshot()
        vectors = model.embed_documents([chunk.text for chunk in items])
        embedding_seconds = perf_counter() - started
        writing = perf_counter()
        count = store.upsert(names[strategy], items, vectors)
        upsert_seconds = perf_counter() - writing
        if count != len(items):
            raise ValueError(f"Index {strategy} count {count}, expected {len(items)}")
        lengths = [len(chunk.text) for chunk in items]
        report[strategy] = {
            "collection": names[strategy], "chunks": count, "characters": sum(lengths),
            "length_min": min(lengths), "length_median": median(lengths),
            "length_mean": round(mean(lengths), 2), "length_max": max(lengths),
            "embedding_usage": model.usage_since(usage_before),
            "embedding_seconds": round(embedding_seconds, 3), "upsert_seconds": round(upsert_seconds, 3),
            "ingest_seconds": round(embedding_seconds + upsert_seconds, 3),
        }
        if recheck:
            reloaded = store.upsert(names[strategy], items, vectors)
            if reloaded != count:
                raise ValueError(f"Second upsert changed the number of {strategy} points")
            report[strategy]["chunks_after_second_upsert"] = reloaded
    report["C"] = {"shares_index": "B", "collection": names["B"],
                   "chunks": report["B"]["chunks"], "incremental_ingest_seconds": 0,
                   "additional_embedding_requests": 0}
    save(results / "index.json", {"settings": SETTINGS, "embedding": model.config,
                                 "qdrant_client": version("qdrant-client"),
                                 "strategy": report})


def benchmark(store, names, chunks, model, questions, split, results):
    traces = []
    lexical = BM25(chunks["B"])
    usage_before = model.usage_snapshot()
    query_times = []
    for question in questions:
        if question["split"] != split:
            continue
        start = perf_counter()
        vector = model.embed_query(question["question"])
        query_times.append(1000 * (perf_counter() - start))
        filters = question.get("filter")
        for strategy in ("A", "B", "C", "Hybrid"):
            collection = names["A" if strategy == "A" else "B"]
            applied = filters if strategy in {"C", "Hybrid"} else None
            start = perf_counter()
            if strategy == "Hybrid":
                dense = store.search(collection, vector, CANDIDATES, applied)
                hits = fuse(dense, lexical.rank(question["question"], applied))
            else:
                hits = store.search(collection, vector, SETTINGS["top_k"], applied)
            elapsed = 1000 * (perf_counter() - start)
            traces.append({**evaluate_query(question, hits), "strategy": strategy,
                           "applied_filter": applied, "retrieval_ms": round(elapsed, 3)})
    summary = summarize(traces)
    save(results / f"{split}_traces.json", traces)
    save(results / f"{split}_metrics.json", {"split": split, "strategies": summary,
                                            "embedding": model.config,
                                            "query_embedding_usage": model.usage_since(usage_before),
                                            "mean_query_embedding_ms": round(mean(query_times), 3)})
    for strategy, row in summary.items():
        print(f"{split:4} {strategy:6} Hit@1 {row['hit1']}/{row['questions']} "
              f"Evidence Hit@3 {row['hit3']}/{row['questions']}")
    return summary


def audit_samples(pages, chunks, results):
    samples = []
    for page in sorted(pages, key=lambda page: (page.document_id, page.page))[:10]:
        position = len(page.text) // 2
        while page.text[position].isspace():
            position += 1
        for strategy, items in [*chunks.items(), ("C", chunks["B"])]:
            match = next(chunk for chunk in items if chunk.source == page.source
                         and chunk.metadata["start"] <= position < chunk.metadata["end"])
            samples.append({"strategy": strategy, "position": position, **vars(match)})
    save(results / "coherence_samples.json", samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                        help="Output directory (default: starter/results/).")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    commands.add_parser("check-api", help="Verify Gemini access and one vector per input.")
    commands.add_parser("index").add_argument("--recheck", action="store_true")
    commands.add_parser("dev")
    commands.add_parser("test")
    query = commands.add_parser("query")
    query.add_argument("question")
    query.add_argument("--strategy", choices=["A", "B", "C", "Hybrid"], default="C")
    query.add_argument("--topic", choices=sorted(TOPICS))
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.StreamHandler(),
        logging.FileHandler(args.results / "pipeline.log", encoding="utf-8"),
    ], format="%(asctime)s %(levelname)s %(message)s")
    try:
        execute(args)
    except Exception as exc:
        logging.error("%s", exc)
        parser.exit(1, "Run failed. See pipeline.log.\n")


def execute(args):
    if args.command == "check-api":
        model = GeminiEmbedder()
        documents = model.embed_documents(["Pathlib handles filesystem paths.", "CSV files store tabular data."])
        query = model.embed_query("What does pathlib handle?")
        print(f"{model.model_name}: {len(documents)} document vectors and 1 query vector "
              f"with {len(query)} dimensions each.")
        return
    pages = load_pages(PAGES, MANIFEST)
    chunks = chunk_sets(pages)
    if args.command == "validate":
        questions = load_questions(QUESTIONS, pages)
        print(f"{len(pages)} source units, {len(questions)} verified quotes, "
              f"{len(chunks['A'])} A chunks, {len(chunks['B'])} B/C chunks")
        print(f"Benchmark SHA-256 {checksum(QUESTIONS)}")
        return
    model = GeminiEmbedder()
    names = collection_names(chunks, model)
    with QdrantStore(args.db, model.dimensions) as store:
        if args.command == "index":
            index(store, names, chunks, model, args.results, args.recheck)
            return
        if any(not store.client.collection_exists(name) or store.count(name) != len(chunks[strategy])
               for strategy, name in names.items()):
            raise ValueError("Index missing or incomplete for this corpus/model. Run index first")
        if args.command == "query":
            strategy = args.strategy
            filters = {"topic": args.topic} if args.topic and strategy in {"C", "Hybrid"} else None
            vector = model.embed_query(args.question)
            collection = names["A" if strategy == "A" else "B"]
            if strategy == "Hybrid":
                hits = fuse(store.search(collection, vector, CANDIDATES, filters),
                            BM25(chunks["B"]).rank(args.question, filters))
            else:
                hits = store.search(collection, vector, SETTINGS["top_k"], filters)
            print(json.dumps({"strategy": strategy, "applied_filter": filters, "hits": hits},
                             ensure_ascii=True, indent=2))
            return

        questions = load_questions(QUESTIONS, pages)
        protocol = {"questions_sha256": checksum(QUESTIONS), "corpus_sha256": checksum(PAGES),
                    "embedding": model.config, "collections": names, "settings": SETTINGS}
        lock = args.results / "selection.json"
        if args.command == "test":
            if not lock.is_file() or json.loads(lock.read_text(encoding="utf-8"))["protocol"] != protocol:
                raise ValueError("Run dev and lock the same model, data and benchmark before test")
            benchmark(store, names, chunks, model, questions, "test", args.results)
            audit_samples(pages, chunks, args.results)
            return
        summary = benchmark(store, names, chunks, model, questions, "dev", args.results)
        # The priority order is specified in the project README.
        winner = max(("B", "C", "A"), key=lambda strategy: (
            summary[strategy]["hit3"], summary[strategy]["hit1"]))
        save(lock, {"protocol": protocol, "selected": winner,
                    "locked_at": datetime.now(timezone.utc).isoformat()})
        print(f"Locked on dev: {winner}")


if __name__ == "__main__":
    main()
