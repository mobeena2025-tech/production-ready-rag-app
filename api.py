"""REST API: upload documents and ask questions about them.

Run:  uvicorn api:app --reload
Docs: http://localhost:8000/docs
"""

from dataclasses import asdict
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag.config import get_settings
from rag.loaders import UnsupportedFileError
from rag.pipeline import RAGPipeline, build_pipeline
from rag.providers import MissingAPIKeyError

app = FastAPI(title="RAG Document Q&A", version="2.0.0")


@lru_cache
def get_pipeline() -> RAGPipeline:
    return build_pipeline(get_settings())


def pipeline_dep() -> RAGPipeline:
    try:
        return get_pipeline()
    except MissingAPIKeyError as e:
        raise HTTPException(503, str(e)) from e


Pipeline = Annotated[RAGPipeline, Depends(pipeline_dep)]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "provider": s.provider, "chat_model": s.chat_model}


@app.post("/documents", status_code=201)
async def upload(file: Annotated[UploadFile, File()], rag: Pipeline):
    data = await file.read()
    limit = rag.settings.max_upload_mb * 1024 * 1024
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {rag.settings.max_upload_mb} MB")
    if not data:
        raise HTTPException(400, "File is empty")
    try:
        result = rag.ingest(data, file.filename or "upload")
    except UnsupportedFileError as e:
        raise HTTPException(415, str(e)) from e
    if not result.skipped and result.chunks == 0:
        raise HTTPException(422, "No text found in this file (scanned PDFs need OCR first)")
    return asdict(result)


@app.get("/documents")
def list_documents(rag: Pipeline):
    return rag.store.documents()


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, rag: Pipeline):
    if not rag.store.delete_document(doc_id):
        raise HTTPException(404, "Document not found")


@app.post("/ask")
def ask(req: AskRequest, rag: Pipeline):
    return asdict(rag.ask(req.question))
