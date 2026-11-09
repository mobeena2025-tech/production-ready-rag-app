import pytest

from rag.loaders import UnsupportedFileError, load_bytes


def test_text_file():
    pages = load_bytes(b"hello there", "notes.txt")
    assert len(pages) == 1
    assert pages[0].text == "hello there"
    assert pages[0].page == 1


def test_pdf_keeps_page_numbers(make_pdf):
    pages = load_bytes(make_pdf(["First page text", "Second page text"]), "doc.pdf")
    assert [p.page for p in pages] == [1, 2]
    assert "Second page" in pages[1].text


def test_unsupported_type():
    with pytest.raises(UnsupportedFileError):
        load_bytes(b"x", "image.png")


def test_blank_file_has_no_pages():
    assert load_bytes(b"   ", "empty.md") == []
