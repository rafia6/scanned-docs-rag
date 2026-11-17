"""Shared fixtures: fake models (no API calls) and generated test PDFs."""

from __future__ import annotations

import hashlib
import io
import math
import re
import shutil

import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from scanned_docs_rag.config import Settings
from scanned_docs_rag.pipeline import RAGPipeline
from scanned_docs_rag.store import VectorStore

HAS_TESSERACT = shutil.which("tesseract") is not None
requires_tesseract = pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract not installed")


class FakeEmbedder:
    """Hashed bag-of-words vectors: deterministic, and similar texts get similar vectors."""

    dim = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            v = [0.0] * self.dim
            for word in re.findall(r"\w+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


class FakeChat:
    def __init__(self, reply: str = "The invoice total is 1,250 EUR [1]."):
        self.reply = reply
        self.last_messages: list[dict] | None = None

    def complete(self, messages):
        self.last_messages = messages
        return self.reply

    def stream(self, messages):
        self.last_messages = messages
        yield from re.findall(r"\S+\s*", self.reply)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        chroma_path=str(tmp_path / "chroma"),
        collection="test",
        ocr_languages="eng",
        chunk_size=300,
        chunk_overlap=50,
        top_k=3,
    )


@pytest.fixture
def pipeline(settings) -> RAGPipeline:
    store = VectorStore(FakeEmbedder(), settings.chroma_path, settings.collection)
    return RAGPipeline(store, FakeChat(), settings)


def _text_image(lines: list[str]) -> bytes:
    image = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 48)
    except OSError:
        font = ImageFont.load_default(size=48)
    for i, line in enumerate(lines):
        draw.text((120, 150 + i * 90), line, fill="black", font=font)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def text_pdf(tmp_path):
    """A 'digital' PDF with a real text layer on two pages."""
    path = tmp_path / "contract.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text(
        (72, 72),
        "Rental contract between Alice and Bob.\n"
        "The monthly rent is 900 EUR, due on the first of each month.",
    )
    doc.new_page().insert_text(
        (72, 72),
        "Termination requires three months written notice.\nThe deposit is two months of rent.",
    )
    doc.save(path)
    return path


@pytest.fixture
def scanned_pdf(tmp_path):
    """A 'scanned' PDF: one page that is only an image, with no text layer."""
    path = tmp_path / "invoice_scan.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    png = _text_image(["INVOICE 2026-117", "Customer: Example GmbH", "Total amount: 1250 EUR"])
    page.insert_image(page.rect, stream=png)
    doc.save(path)
    return path
