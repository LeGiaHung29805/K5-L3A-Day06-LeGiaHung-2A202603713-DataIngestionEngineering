"""Build traceable source units from the official Python HTML documentation."""

import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from .ingestion import normalize_text

ROOT = Path(__file__).resolve().parents[2]
BLOCKS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "dt", "li", "td"}


class SectionReader(HTMLParser):
    def __init__(self, anchors: list[str]) -> None:
        super().__init__()
        self.sections: list[str] = []
        self.blocks: dict[str, list[str]] = {anchor: [] for anchor in anchors}
        self.current: tuple[str, str, list[str]] | None = None
        self.heading_link = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "section":
            self.sections.append(attributes.get("id") or "")
        if tag == "a" and attributes.get("class") == "headerlink":
            self.heading_link = True
        if tag in BLOCKS and self.current is None:
            for section in reversed(self.sections):
                if section in self.blocks:
                    self.current = (section, tag, [])
                    break

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self.heading_link = False
        if self.current and tag == self.current[1]:
            section, block_type, parts = self.current
            text = "".join(parts)
            if block_type != "pre":
                text = re.sub(r"\s+", " ", text)
            text = normalize_text(text)
            if text:
                self.blocks[section].append(text)
            self.current = None
        if tag == "section" and self.sections:
            self.sections.pop()

    def handle_data(self, data: str) -> None:
        if self.current and not self.heading_link:
            self.current[2].append(data)


def prepare(html_dir: Path, accessed_at: str) -> None:
    selection = json.loads((ROOT / "data/corpus_sections.json").read_text(encoding="utf-8"))
    snapshot = selection["snapshot"]
    base = f"https://docs.python.org/release/{snapshot}/library"
    pages = []
    sources = []
    for topic, chapter in selection["chapters"].items():
        index_html = (html_dir / "library" / f'{chapter["index"]}.html').read_text(encoding="utf-8")
        for module, anchors in chapter["modules"].items():
            html = (html_dir / "library" / f"{module}.html").read_text(encoding="utf-8")
            if f"Python {snapshot} documentation" not in html:
                raise ValueError(f"{module}: expected Python documentation {snapshot}")
            if f'href="{module}.html' not in index_html:
                raise ValueError(f"{module}: not listed in {chapter['title']}")
            reader = SectionReader(anchors)
            reader.feed(html)
            source = f"{base}/{module}.html"
            manifest_pages = []
            for page, anchor in enumerate(anchors, 1):
                blocks = reader.blocks[anchor]
                text = "\n\n".join(blocks)
                if len(blocks) < 2 or len(text) < 300:
                    raise ValueError(f"{source}#{anchor}: missing substantive content")
                pages.append({"document_id": module, "source": f"{source}#{anchor}",
                              "page": page, "text": text, "metadata": {
                                  "topic": topic, "chapter": chapter["title"],
                                  "section": anchor, "version": snapshot,
                              }})
                manifest_pages.append({"page": page, "anchor": anchor, "heading": blocks[0],
                                       "characters": len(text)})
            sources.append({"source_id": module, "title": f"Python Standard Library — {module}",
                            "source_path_or_url": source, "topic": topic,
                            "metadata_source": f'{base}/{chapter["index"]}.html',
                            "license_or_permission": "PSF License Version 2 (data/NOTICE.md)",
                            "accessed_at": accessed_at, "pages": manifest_pages})
    output = ROOT / "data/pages.jsonl"
    output.write_text("".join(json.dumps(page, ensure_ascii=False) + "\n" for page in pages),
                      encoding="utf-8", newline="\n")
    manifest = {
        "snapshot": snapshot,
        "unit_rule": "One distinct, substantive HTML section per source anchor. The numeric page ID is its 1-based position in the fixed module selection, not a physical page number.",
        "count": len(pages),
        "pages_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "sources": sources,
    }
    (ROOT / "data/manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"{len(pages)} sections across {len(sources)} modules")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_dir", type=Path, help="Documentation directory containing library/")
    parser.add_argument("--accessed-at", required=True, help="Snapshot access date (YYYY-MM-DD)")
    args = parser.parse_args()
    prepare(args.html_dir, args.accessed_at)
