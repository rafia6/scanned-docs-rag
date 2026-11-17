"""Turn PDFs into page text, using OCR for pages that have no usable text layer."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pymupdf
from PIL import Image

from .models import Page

logger = logging.getLogger(__name__)


def ocr_image(image: Image.Image, languages: str = "eng") -> str:
    """Run Tesseract on a page image."""
    import pytesseract  # imported lazily so the rest of the package works without Tesseract

    return pytesseract.image_to_string(image, lang=languages)


def render_page(page: pymupdf.Page, dpi: int = 300) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def load_pdf(
    path: str | Path,
    *,
    ocr_languages: str = "eng",
    ocr_dpi: int = 300,
    min_text_chars: int = 50,
    force_ocr: bool = False,
) -> list[Page]:
    """Extract text from every page of a PDF.

    Pages with a real text layer are read directly (fast and exact). Scanned pages,
    where the text layer is missing or nearly empty, are rendered to an image and
    passed through OCR.
    """
    path = Path(path)
    pages: list[Page] = []
    with pymupdf.open(path) as pdf:
        for index, page in enumerate(pdf):
            text = "" if force_ocr else page.get_text("text").strip()
            used_ocr = False
            if len(text) < min_text_chars:
                logger.info("OCR on %s page %d", path.name, index + 1)
                text = ocr_image(render_page(page, ocr_dpi), ocr_languages).strip()
                used_ocr = True
            pages.append(Page(document=path.name, page_number=index + 1, text=text, ocr=used_ocr))
    return pages
