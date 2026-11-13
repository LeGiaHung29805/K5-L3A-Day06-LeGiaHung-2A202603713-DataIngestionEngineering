"""Load and validate normalized source units with their provenance."""
import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path
from .models import DocumentPage


def normalize_text(text: str) -> str:
    """Keep line breaks, paragraph boundaries and leading code indentation."""
    text = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", lines).strip("\n")


def load_pages(path: Path, manifest_path: Path) -> list[DocumentPage]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {(doc["source_id"], page["page"]):
                (f'{doc["source_path_or_url"]}#{page["anchor"]}', page["characters"], doc["topic"])
                for doc in manifest["sources"] for page in doc["pages"]}
    pages = {}
    raw = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            page = DocumentPage(**json.loads(line))
            key = (page.document_id, page.page)
            page.text = normalize_text(page.text)
            if not page.text.strip():
                raise ValueError("empty content")
            if key in pages:
                raise ValueError(f"duplicate unit {key}")
            if expected.get(key) != (page.source, len(page.text), page.metadata["topic"]):
                raise ValueError(f"unit does not match manifest: {key}")
            pages[key] = page
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            logging.error("%s line %d: %s", path, line_number, exc)

    if len(pages) < 30 or pages.keys() != expected.keys() or len(pages) != manifest["count"]:
        raise ValueError("Corpus is incomplete or has fewer than 30 valid units")
    if hashlib.sha256(raw.encode("utf-8")).hexdigest() != manifest["pages_sha256"]:
        raise ValueError("Corpus checksum does not match the manifest")
    return list(pages.values())
