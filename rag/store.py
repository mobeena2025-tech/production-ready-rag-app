"""Thin wrapper around a persistent Chroma collection."""

from dataclasses import dataclass

import chromadb


@dataclass
class Hit:
    text: str
    source: str
    page: int
    score: float  # cosine similarity, higher is better


class VectorStore:
    def __init__(self, path: str, collection: str):
        self._client = chromadb.PersistentClient(path=path)
        self._name = collection
        self._col = self._open()

    def _open(self):
        # We always pass our own embeddings, so no embedding function is attached.
        return self._client.get_or_create_collection(
            name=self._name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, texts, embeddings, metadatas) -> None:
        if ids:
            self._col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def has_document(self, doc_id: str) -> bool:
        return bool(self._col.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def query(self, embedding: list[float], k: int) -> list[Hit]:
        if self._col.count() == 0:
            return []
        res = self._col.query(query_embeddings=[embedding], n_results=min(k, self._col.count()))
        hits = []
        rows = zip(res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True)
        for text, meta, dist in rows:
            hits.append(Hit(text=text, source=meta["source"], page=int(meta["page"]), score=1 - dist))
        return hits

    def documents(self) -> list[dict]:
        metas = self._col.get(include=["metadatas"])["metadatas"]
        docs: dict[str, dict] = {}
        for m in metas:
            d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "source": m["source"], "chunks": 0})
            d["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["source"])

    def delete_document(self, doc_id: str) -> bool:
        if not self.has_document(doc_id):
            return False
        self._col.delete(where={"doc_id": doc_id})
        return True

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._col = self._open()
