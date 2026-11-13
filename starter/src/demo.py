"""Present saved results and demonstrate live metadata filtering."""
import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .embeddings import GeminiEmbedder
from .evaluation import load_questions
from .ingestion import load_pages
from .run import DEFAULT_DB, MANIFEST, PAGES, QUESTIONS, SUBMISSION_RESULTS, chunk_sets, collection_names
from .store import QdrantStore, TOPICS

QUESTION = "How do I open a file in text mode with a specific encoding and newline handling?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--results", type=Path, default=SUBMISSION_RESULTS,
                        help="Saved results to replay (default: results/ at the repository root).")
    parser.add_argument("--replay", action="store_true", help="Show saved traces without calling Gemini.")
    parser.add_argument("--ui", action="store_true", help="Serve the demo in a local browser.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.ui:
        serve_ui(args.db, args.results, args.port)
        return
    if not args.replay:
        try:
            show_live(args.db)
        except (RuntimeError, ValueError) as exc:
            parser.exit(1, f"{exc}\nUse --replay to show saved traces offline.\n")
    show_traces(args.results)


def show_live(db: Path) -> None:
    pages = load_pages(PAGES, MANIFEST)
    questions = load_questions(QUESTIONS, pages)
    print(f"{len(pages)} source units, {len(questions)} verified quotes")

    comparison = compare_scope(db, QUESTION, "archive")
    print(f"\n{comparison['model']} | requested scope 'archive': {QUESTION}")
    for strategy, row in comparison["strategies"].items():
        print(f"\n{strategy} | filter applied: {row['applied_filter'] or 'none'}")
        for rank, hit in enumerate(row["hits"], 1):
            print(f"  {rank}. {hit['metadata']['topic']:10} page {hit['page']:<2} "
                  f"{hit['score']:.3f} {hit['source']}")
        if row["hits"]:
            excerpt = " ".join(row["hits"][0]["text"].split())[:300]
            print(f"  Top-1 excerpt: {excerpt}")


def compare_scope(db: Path, question: str, topic: str) -> dict:
    chunks = chunk_sets(load_pages(PAGES, MANIFEST))
    model = GeminiEmbedder()
    collection = collection_names(chunks, model)["B"]
    with QdrantStore(db, model.dimensions) as store:
        if not store.client.collection_exists(collection) or store.count(collection) != len(chunks["B"]):
            raise ValueError("Index missing or incomplete. Run python -m src.run index first.")
        vector = model.embed_query(question)
        strategies = {
            strategy: {"applied_filter": filters,
                       "hits": store.search(collection, vector, filters=filters)}
            for strategy, filters in (("B", None), ("C", {"topic": topic}))
        }
    return {"model": model.model_name, "strategies": strategies}


def read_ui_data(results: Path) -> dict:
    data = {
        "manifest": json.loads(MANIFEST.read_text(encoding="utf-8")),
        "questions": json.loads(QUESTIONS.read_text(encoding="utf-8")),
        "live_example": QUESTION,
    }
    for name in ("index", "selection", "dev_metrics", "test_metrics", "dev_traces",
                 "test_traces", "coherence_samples"):
        data[name] = json.loads((results / f"{name}.json").read_text(encoding="utf-8"))
    for name in ("coherence_audit", "grounding_audit", "metadata_audit"):
        with (results / f"{name}.csv").open(encoding="utf-8-sig", newline="") as file:
            data[name] = list(csv.DictReader(file))
    return data


def serve_ui(db: Path, results: Path, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, content: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def json_response(self, status: int, data: dict) -> None:
            self.respond(status, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                         "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if self.path == "/":
                self.respond(200, Path(__file__).with_suffix(".html").read_bytes(),
                             "text/html; charset=utf-8")
            elif self.path == "/api/results":
                try:
                    self.json_response(200, read_ui_data(results))
                except (OSError, ValueError):
                    self.json_response(503, {"error": "Không đọc được bộ kết quả. Kiểm tra thư mục đã chọn qua --results."})
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/api/scope":
                self.send_error(404)
                return
            if self.headers.get_content_type() != "application/json" or self.headers.get("Sec-Fetch-Site") == "cross-site":
                self.send_error(415)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError
                body = json.loads(self.rfile.read(length))
                question, topic = body["question"].strip(), body["topic"]
                if not question or len(question) > 400 or topic not in TOPICS:
                    raise ValueError
            except (ValueError, KeyError, AttributeError, TypeError):
                self.json_response(400, {"error": "Nhập câu hỏi tối đa 400 ký tự và chọn chủ đề hợp lệ."})
                return
            try:
                self.json_response(200, compare_scope(db, question, topic))
            except (RuntimeError, ValueError) as exc:
                self.json_response(503, {"error": str(exc)})

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Demo UI: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def show_traces(results: Path) -> None:
    print(f"\nSaved traces in {results}")
    print(f"{'Split':5} {'ID':4} {'Strategy':8} {'Hit@1':>5} {'Hit@3':>5}  Evidence ranks")
    for split in ("dev", "test"):
        path = results / f"{split}_traces.json"
        if not path.is_file():
            print(f"{split:5} No saved traces. Run python -m src.run {split} to generate them.")
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row["id"] in {"q08", "q12", "q17"}:
                ranks = ", ".join(str(hit["rank"]) for hit in row["hits"] if hit["evidence_hit"])
                print(f"{split:5} {row['id']:4} {row['strategy']:8} "
                      f"{int(row['hit1']):5} {int(row['hit3']):5}  {ranks or '-'}")


if __name__ == "__main__":
    main()
