from pathlib import Path

import pytest

from rag.pipeline import NO_ANSWER, RAGPipeline
from rag.providers import HashEmbedder
from tests.conftest import RecordingChat

DOCS = Path(__file__).resolve().parent.parent / "eval" / "docs"


def ingest_samples(pipeline):
    for f in DOCS.iterdir():
        pipeline.ingest(f.read_bytes(), f.name)


def test_ingest_counts_chunks(pipeline):
    r = pipeline.ingest((DOCS / "solar_system.md").read_bytes(), "solar_system.md")
    assert r.chunks >= 2 and r.pages == 1 and not r.skipped


def test_same_file_is_not_indexed_twice(pipeline):
    data = (DOCS / "solar_system.md").read_bytes()
    pipeline.ingest(data, "solar_system.md")
    again = pipeline.ingest(data, "copy.md")
    assert again.skipped
    assert len(pipeline.store.documents()) == 1


def test_retrieval_finds_right_document(pipeline):
    ingest_samples(pipeline)
    hits = pipeline.retrieve("Where does the Calvin cycle happen?")
    assert hits[0].source == "photosynthesis.md"


def test_answer_has_citations(pipeline):
    ingest_samples(pipeline)
    ans = pipeline.ask("Which planet has the Great Red Spot storm?")
    assert ans.answer != NO_ANSWER
    assert ans.citations[0].source == "solar_system.md"


def test_no_documents_means_no_answer_and_no_llm_call(settings, store):
    llm = RecordingChat()
    rag = RAGPipeline(settings, HashEmbedder(), llm, store)
    assert rag.ask("anything?").answer == NO_ANSWER
    assert llm.calls == []


def test_prompt_contains_numbered_context(settings, store):
    llm = RecordingChat()
    rag = RAGPipeline(settings, HashEmbedder(), llm, store)
    ingest_samples(rag)
    rag.ask("Which planet is largest?")
    system, prompt = llm.calls[0]
    assert "[1] (solar_system.md, page 1)" in prompt
    assert "Question: Which planet is largest?" in prompt
    assert NO_ANSWER in system


def test_only_cited_sources_are_returned(settings, store):
    rag = RAGPipeline(settings, HashEmbedder(), RecordingChat("Answer [2]."), store)
    ingest_samples(rag)
    ans = rag.ask("Jupiter planet storm Mars volcano")
    assert [c.n for c in ans.citations] == [2]


def test_model_refusal_drops_citations(settings, store):
    rag = RAGPipeline(settings, HashEmbedder(), RecordingChat(NO_ANSWER), store)
    ingest_samples(rag)
    ans = rag.ask("Which planet is largest?")
    assert ans.answer == NO_ANSWER and ans.citations == []


def test_delete_document(pipeline):
    r = pipeline.ingest((DOCS / "solar_system.md").read_bytes(), "solar_system.md")
    assert pipeline.store.delete_document(r.doc_id)
    assert pipeline.store.documents() == []
    assert not pipeline.store.delete_document(r.doc_id)


def test_empty_question_rejected(pipeline):
    with pytest.raises(ValueError):
        pipeline.ask("   ")
