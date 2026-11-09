"""Ingest documents and answer questions about them, with citations."""

import hashlib
import re
from dataclasses import dataclass, field

from rag.chunking import chunk_text
from rag.config import Settings
from rag.loaders import load_bytes
from rag.providers import ChatModel, Embedder
from rag.store import Hit, VectorStore

NO_ANSWER = "I cannot find the answer in the uploaded documents."

SYSTEM_PROMPT = f"""You answer questions using only the numbered context passages you are given.
Rules:
- Use only facts stated in the passages. Do not use outside knowledge.
- After each claim, cite the passage it came from, like [1] or [2][3].
- If the passages do not contain the answer, reply exactly: "{NO_ANSWER}"
- Be concise."""


@dataclass
class Citation:
    n: int
    source: str
    page: int
    score: float
    snippet: str


@dataclass
class Answer:
    answer: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class IngestResult:
    doc_id: str
    source: str
    pages: int
    chunks: int
    skipped: bool = False  # True when the same file was already indexed


class RAGPipeline:
    def __init__(self, settings: Settings, embedder: Embedder, llm: ChatModel, store: VectorStore):
        self.settings = settings
        self.embedder = embedder
        self.llm = llm
        self.store = store

    # ----- ingestion -----

    def ingest(self, data: bytes, filename: str) -> IngestResult:
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        if self.store.has_document(doc_id):
            return IngestResult(doc_id, filename, pages=0, chunks=0, skipped=True)

        pages = load_bytes(data, filename)
        ids, texts, metas = [], [], []
        for page in pages:
            for i, chunk in enumerate(
                chunk_text(page.text, self.settings.chunk_size, self.settings.chunk_overlap)
            ):
                ids.append(f"{doc_id}:{page.page}:{i}")
                texts.append(chunk)
                metas.append({"doc_id": doc_id, "source": filename, "page": page.page})

        if texts:
            self.store.add(ids, texts, self.embedder.embed_documents(texts), metas)
        return IngestResult(doc_id, filename, pages=len(pages), chunks=len(texts))

    # ----- question answering -----

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        hits = self.store.query(self.embedder.embed_query(question), k or self.settings.top_k)
        threshold = self.settings.min_score
        if threshold is None:
            threshold = self.embedder.default_min_score
        return [h for h in hits if h.score >= threshold]

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("question is empty")

        hits = self.retrieve(question)
        # Nothing relevant: answer without calling the model, so it can't make things up.
        if not hits:
            return Answer(NO_ANSWER)

        context = "\n\n".join(
            f"[{i}] ({h.source}, page {h.page})\n{h.text}" for i, h in enumerate(hits, start=1)
        )
        prompt = f"Context:\n\n{context}\n\nQuestion: {question}"
        text = self.llm.generate(SYSTEM_PROMPT, prompt)

        if NO_ANSWER.lower() in text.lower():
            return Answer(NO_ANSWER)

        # Only return the sources the model actually cited.
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        citations = [
            Citation(i, h.source, h.page, round(h.score, 3), h.text[:240])
            for i, h in enumerate(hits, start=1)
            if i in cited
        ]
        return Answer(text, citations)


def build_pipeline(settings: Settings) -> RAGPipeline:
    from rag.providers import build_providers

    embedder, llm = build_providers(settings)
    return RAGPipeline(settings, embedder, llm, VectorStore(settings.chroma_dir, settings.collection))
