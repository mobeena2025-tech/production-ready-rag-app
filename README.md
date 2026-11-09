# Document Q&A with RAG

![CI](https://github.com/mobeena2025-tech/production-ready-rag-app/actions/workflows/ci.yml/badge.svg)

Upload PDFs, text or Markdown files and ask questions about them. Every answer
comes **only** from your documents and cites the file and page it came from.
If the documents don't contain the answer, the app says so instead of guessing.

It ships as a **REST API** (FastAPI) and a **chat UI** (Streamlit), runs on
**Google Gemini** (`gemini-2.5-flash` + `gemini-embedding-001`), stores vectors in
**ChromaDB**, and has an **offline mode** so you can try it, test it and run CI
without an API key.

## How it works

```
 upload ──► extract text per page ──► split into overlapping chunks ──► embed ──► ChromaDB
                                                                                    │
 question ──► embed ──► top-k similar chunks ──► drop weak matches ──► Gemini ──► answer + [1][2] citations
```

| Step | Where | Notes |
|---|---|---|
| Load | `rag/loaders.py` | PDF (keeps page numbers), TXT, MD |
| Chunk | `rag/chunking.py` | ~1000 chars, 150 overlap, splits on paragraphs → lines → sentences → words |
| Embed | `rag/providers.py` | Separate *document* and *query* embedding modes; batched |
| Store | `rag/store.py` | Persistent Chroma collection, cosine similarity |
| Answer | `rag/pipeline.py` | Numbered context, strict "only use the passages" prompt, returns only the sources the model cited |

Design choices worth knowing:

- **Duplicate uploads are skipped.** Each file is identified by a hash of its contents.
- **No relevant context → no model call.** If nothing passes the similarity cutoff, the app
  returns "I cannot find the answer…" directly, so the model has nothing to make things up from.
- **Swappable providers.** The pipeline only depends on two small interfaces (`Embedder`,
  `ChatModel`), so adding OpenAI, Ollama or a local model is one class each.

## Quick start

```bash
git clone https://github.com/mobeena2025-tech/production-ready-rag-app
cd production-ready-rag-app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # paste your key from https://aistudio.google.com/apikey
```

Chat UI:

```bash
streamlit run app.py
```

REST API (interactive docs at http://localhost:8000/docs):

```bash
uvicorn api:app --reload
```

No API key? Set `PROVIDER=local` in `.env` to run fully offline with simple keyword
embeddings.

### With Docker

```bash
docker compose up --build
```

UI on http://localhost:8501, API on http://localhost:8000. The index is kept in a Docker volume.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents` | Upload a file (multipart `file`). Returns `doc_id`, pages, chunks |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{doc_id}` | Remove a document from the index |
| `POST` | `/ask` | `{"question": "..."}` → `{"answer": "...", "citations": [...]}` |
| `GET` | `/health` | Status and active model |

```bash
curl -F file=@paper.pdf localhost:8000/documents
curl localhost:8000/ask -H 'content-type: application/json' -d '{"question": "What dataset did they use?"}'
```

```json
{
  "answer": "They trained on the CIFAR-10 dataset [1].",
  "citations": [{"n": 1, "source": "paper.pdf", "page": 4, "score": 0.71, "snippet": "..."}]
}
```

## Evaluating retrieval

`scripts/evaluate.py` measures how often the right page is retrieved, using a file of
labelled questions (`eval/questions.jsonl`). It reports **recall@k** and **MRR**.

```bash
python scripts/evaluate.py eval/questions.jsonl --docs eval/docs            # with Gemini
python scripts/evaluate.py eval/questions.jsonl --docs eval/docs --provider local
```

Use it to compare chunk sizes, `TOP_K`, or embedding models on your own documents.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests run offline (no API key) and cover chunking, PDF loading, retrieval, citation
filtering, refusal behaviour and every API endpoint. GitHub Actions runs lint, tests,
the retrieval eval and a Docker build on every push.

## Configuration

All settings are environment variables (see `.env.example`):

| Variable | Default | |
|---|---|---|
| `GOOGLE_API_KEY` | – | Required for `PROVIDER=gemini` |
| `PROVIDER` | `gemini` | `gemini` or `local` |
| `CHAT_MODEL` | `gemini-2.5-flash` | |
| `EMBED_MODEL` | `gemini-embedding-001` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Characters |
| `TOP_K` | `4` | Chunks sent to the model |
| `MIN_SCORE` | per embedder | Similarity cutoff (0–1) |
| `CHROMA_DIR` | `./data/chroma` | Where the index is saved |

## Project layout

```
rag/            core library (loaders, chunking, providers, store, pipeline)
api.py          FastAPI app
app.py          Streamlit UI
scripts/        retrieval evaluation
eval/           sample documents and labelled questions
tests/          pytest suite
```

## Ideas for next steps

- Hybrid search (BM25 + vectors) and a re-ranker
- Streaming answers
- OCR for scanned PDFs
- Conversation memory (follow-up questions)
