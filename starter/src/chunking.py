"""Split one source unit at a time and preserve its character offsets."""
import re
from uuid import NAMESPACE_URL, uuid5

from .models import Chunk, DocumentPage


def make_chunk(page: DocumentPage, start: int, end: int) -> Chunk:
    chunk_id = str(uuid5(NAMESPACE_URL, f"{page.source}|{start}|{end}"))
    return Chunk(chunk_id, page.document_id, page.source, page.page, page.text[start:end],
                 {**page.metadata, "start": start, "end": end})


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size")
    chunks = []
    for start in range(0, len(page.text), size - overlap):
        end = min(start + size, len(page.text))
        chunks.append(make_chunk(page, start, end))
        if end == len(page.text):
            break
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    if max_size < 1:
        raise ValueError("max_size must be positive")
    chunks = []
    start = end = 0
    for paragraph in re.finditer(r"\S(?:.*?\S)?(?=\n\n|\Z)", page.text, re.DOTALL):
        left, right = paragraph.span()
        if end > start and right - start > max_size:
            chunks.append(make_chunk(page, start, end))
            start = end = left
        if end == start:
            start = left
        while right - start > max_size:
            ceiling = start + max_size
            # Prefer a line or word boundary. A single long token still makes progress.
            boundary = max(page.text.rfind("\n", start, ceiling + 1),
                           page.text.rfind(" ", start, ceiling + 1))
            cut = boundary if boundary > start else ceiling
            chunks.append(make_chunk(page, start, cut))
            start = cut
            while start < right and page.text[start].isspace():
                start += 1
        end = right
    if end > start:
        chunks.append(make_chunk(page, start, end))
    return chunks
