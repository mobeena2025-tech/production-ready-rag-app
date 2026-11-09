"""Split text into overlapping chunks that try to end on natural boundaries."""

import re

# Try to split on the biggest boundary first, falling back to smaller ones.
_SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]

    pieces = re.split(seps[0], text)
    out: list[str] = []
    for piece in pieces:
        if len(piece) > size:
            out.extend(_split(piece, size, seps[1:]))
        elif piece:
            out.append(piece)
    return out


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be between 0 and size")

    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    pieces = _split(text, size, _SEPARATORS)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        # Start the next chunk with the tail of the previous one, so a fact
        # that straddles the boundary is still retrievable.
        tail = current[-overlap:] if overlap else ""
        if " " in tail:
            tail = tail[tail.index(" ") + 1 :]
        current = f"{tail} {piece}".strip() if tail else piece
        if len(current) > size:
            current = piece
    if current:
        chunks.append(current)
    return chunks
