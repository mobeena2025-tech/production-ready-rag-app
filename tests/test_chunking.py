import pytest

from rag.chunking import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n  ", size=100, overlap=10) == []


def test_chunks_respect_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunks_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(50))
    chunks = chunk_text(text, size=150, overlap=60)
    # The start of each chunk should repeat words from the end of the previous one.
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.split()[0] in prev


def test_prefers_paragraph_boundaries():
    text = ("a" * 80) + "\n\n" + ("b" * 80)
    assert chunk_text(text, size=100, overlap=0) == ["a" * 80, "b" * 80]


def test_very_long_word_is_split():
    chunks = chunk_text("x" * 250, size=100, overlap=0)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == "x" * 250


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_bad_arguments(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", size=size, overlap=overlap)
