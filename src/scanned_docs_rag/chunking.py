"""Split page text into overlapping chunks that never cross page boundaries.

Keeping each chunk inside one page means every retrieved passage maps to an exact
page number, which is what makes the citations in answers trustworthy.
"""

from __future__ import annotations

import hashlib
import re

from .models import Chunk, Page

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _normalize(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"-\n(?=\w)", "", text)  # re-join words hyphenated across lines (common in OCR)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split(text: str, size: int, separators: list[str]) -> list[str]:
    """Recursively split on the coarsest separator that yields pieces <= size."""
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, rest = separators[0], separators[1:]
    pieces: list[str] = []
    for part in text.split(sep):
        if len(part) > size:
            pieces.extend(_split(part, size, rest))
        elif part:
            pieces.append(part)

    # Greedily merge small pieces back together up to the size limit.
    merged: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = piece
    if current:
        merged.append(current)
    return merged


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = _normalize(text)
    if not text:
        return []
    base = _split(text, chunk_size - overlap, _SEPARATORS)
    if overlap == 0:
        return base
    # Prepend the tail of the previous chunk so context is not lost at the boundaries.
    chunks = [base[0]]
    for prev, cur in zip(base, base[1:], strict=False):
        tail = prev[-overlap:]
        space = tail.find(" ")
        if 0 <= space < len(tail) - 1:
            tail = tail[space + 1 :]  # start the overlap on a word boundary
        chunks.append(f"{tail} {cur}")
    return chunks


def chunk_id(document: str, page: int, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{document}:p{page}:c{index}:{digest}"


def chunk_pages(pages: list[Page], chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for i, text in enumerate(split_text(page.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    id=chunk_id(page.document, page.page_number, i, text),
                    document=page.document,
                    page_number=page.page_number,
                    text=text,
                    ocr=page.ocr,
                )
            )
    return chunks
