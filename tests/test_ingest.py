from conftest import requires_tesseract

from scanned_docs_rag.ingest import load_pdf


def test_text_layer_is_used_without_ocr(text_pdf):
    pages = load_pdf(text_pdf)
    assert [p.page_number for p in pages] == [1, 2]
    assert not any(p.ocr for p in pages)
    assert "900 EUR" in pages[0].text
    assert pages[0].document == "contract.pdf"


@requires_tesseract
def test_scanned_page_falls_back_to_ocr(scanned_pdf):
    [page] = load_pdf(scanned_pdf, ocr_dpi=150)
    assert page.ocr
    assert "INVOICE" in page.text.upper()
    assert "1250" in page.text


@requires_tesseract
def test_force_ocr(text_pdf):
    pages = load_pdf(text_pdf, force_ocr=True, ocr_dpi=150)
    assert all(p.ocr for p in pages)
    assert "rent" in pages[0].text.lower()
