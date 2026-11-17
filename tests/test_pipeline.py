from conftest import requires_tesseract

from scanned_docs_rag.pipeline import NO_CONTEXT_ANSWER, extract_citations


def test_ask_without_documents(pipeline):
    answer = pipeline.ask("What is the rent?")
    assert answer.answer == NO_CONTEXT_ANSWER
    assert answer.sources == []


def test_ingest_and_retrieve_right_page(pipeline, text_pdf):
    result = pipeline.ingest(text_pdf)
    assert result.pages == 2 and result.ocr_pages == 0 and result.chunks >= 2

    top = pipeline.retrieve("How much notice is needed for termination?")[0]
    assert top.document == "contract.pdf"
    assert top.page == 2


def test_reingest_replaces_instead_of_duplicating(pipeline, text_pdf):
    first = pipeline.ingest(text_pdf)
    pipeline.ingest(text_pdf)
    assert pipeline.store.count() == first.chunks
    assert pipeline.store.documents() == ["contract.pdf"]


def test_answer_has_citations_and_prompt_has_numbered_context(pipeline, text_pdf):
    pipeline.ingest(text_pdf)
    answer = pipeline.ask("What is the monthly rent?")
    assert answer.cited == [1]
    prompt = pipeline.chat.last_messages[-1]["content"]
    assert "[1] (document: contract.pdf, page" in prompt
    assert "What is the monthly rent?" in prompt


def test_stream_returns_sources_and_tokens(pipeline, text_pdf):
    pipeline.ingest(text_pdf)
    sources, tokens = pipeline.ask_stream("rent")
    assert sources
    assert "".join(tokens) == pipeline.chat.reply


def test_extract_citations_ignores_out_of_range():
    assert extract_citations("A [1], B [3][2], C [9]", max_number=3) == [1, 2, 3]


@requires_tesseract
def test_scanned_document_end_to_end(pipeline, scanned_pdf):
    result = pipeline.ingest(scanned_pdf)
    assert result.ocr_pages == 1
    top = pipeline.retrieve("invoice total amount")[0]
    assert top.document == "invoice_scan.pdf"
    assert "1250" in top.text
