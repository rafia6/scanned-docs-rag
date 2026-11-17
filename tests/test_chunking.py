import pytest

from scanned_docs_rag.chunking import chunk_pages, split_text
from scanned_docs_rag.models import Page


def test_short_text_is_one_chunk():
    assert split_text("Hello world.", chunk_size=100, overlap=10) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert split_text("   \n\n ", chunk_size=100, overlap=10) == []


def test_chunks_respect_size_limit():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_overlap_repeats_context():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = split_text(text, chunk_size=120, overlap=30)
    first_words = set(chunks[0].split())
    assert first_words & set(chunks[1].split()), "consecutive chunks should share words"


def test_ocr_hyphenation_is_rejoined():
    assert split_text("the docu-\nment was signed", 100, 10) == ["the document was signed"]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, overlap=10)


def test_chunks_keep_page_numbers_and_unique_ids():
    pages = [Page("a.pdf", 1, "x " * 400), Page("a.pdf", 2, "y " * 400, ocr=True)]
    chunks = chunk_pages(pages, chunk_size=200, overlap=20)
    assert {c.page_number for c in chunks} == {1, 2}
    assert all(c.ocr for c in chunks if c.page_number == 2)
    assert len({c.id for c in chunks}) == len(chunks)
