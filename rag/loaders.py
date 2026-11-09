"""Turn uploaded files into plain-text pages."""

import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_TYPES = {".pdf", ".txt", ".md"}


class UnsupportedFileError(ValueError):
    pass


@dataclass
class Page:
    text: str
    source: str
    page: int  # 1-based; text files count as a single page


def load_bytes(data: bytes, filename: str) -> list[Page]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise UnsupportedFileError(
            f"{filename}: unsupported file type. Use one of {sorted(SUPPORTED_TYPES)}."
        )

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        pages = [
            Page(text=(p.extract_text() or "").strip(), source=filename, page=i)
            for i, p in enumerate(reader.pages, start=1)
        ]
    else:
        pages = [Page(text=data.decode("utf-8", errors="replace").strip(), source=filename, page=1)]

    return [p for p in pages if p.text]


def load_file(path: str | Path) -> list[Page]:
    path = Path(path)
    return load_bytes(path.read_bytes(), path.name)
