import pytest

from rag.config import Settings
from rag.pipeline import RAGPipeline
from rag.providers import ExtractiveChat, HashEmbedder
from rag.store import VectorStore


class RecordingChat:
    """Fake LLM that records prompts and returns a canned reply."""

    def __init__(self, reply="Jupiter is the largest planet [1]."):
        self.reply = reply
        self.calls = []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        return self.reply


@pytest.fixture
def settings(tmp_path):
    return Settings(
        provider="local",
        chroma_dir=str(tmp_path / "chroma"),
        chunk_size=300,
        chunk_overlap=50,
        google_api_key=None,
        _env_file=None,
    )


@pytest.fixture
def store(settings):
    return VectorStore(settings.chroma_dir, settings.collection)


@pytest.fixture
def pipeline(settings, store):
    return RAGPipeline(settings, HashEmbedder(), ExtractiveChat(), store)


@pytest.fixture
def make_pdf():
    """Build a small real PDF with one text page per string."""

    def _make(pages: list[str]) -> bytes:
        objs = ["<< /Type /Catalog /Pages 2 0 R >>"]
        kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
        objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
        font_id = 3 + 2 * len(pages)
        for i, text in enumerate(pages):
            stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
            objs.append(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {4 + 2 * i} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
            )
            objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

        out, offsets = "%PDF-1.4\n", []
        for n, body in enumerate(objs, start=1):
            offsets.append(len(out))
            out += f"{n} 0 obj\n{body}\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n"
        out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
        out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
        return out.encode("latin-1")

    return _make
