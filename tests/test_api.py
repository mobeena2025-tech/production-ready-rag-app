from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api

DOCS = Path(__file__).resolve().parent.parent / "eval" / "docs"


@pytest.fixture
def client(pipeline):
    api.app.dependency_overrides[api.pipeline_dep] = lambda: pipeline
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def upload(client, name):
    return client.post("/documents", files={"file": (name, (DOCS / name).read_bytes())})


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_upload_list_ask_delete(client):
    r = upload(client, "photosynthesis.md")
    assert r.status_code == 201
    doc_id = r.json()["doc_id"]

    assert [d["source"] for d in client.get("/documents").json()] == ["photosynthesis.md"]

    ans = client.post("/ask", json={"question": "What pigment do chloroplasts contain?"}).json()
    assert "chlorophyll" in ans["answer"].lower()
    assert ans["citations"][0]["source"] == "photosynthesis.md"

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get("/documents").json() == []


def test_unsupported_file(client):
    r = client.post("/documents", files={"file": ("x.png", b"\x89PNG")})
    assert r.status_code == 415


def test_empty_file(client):
    assert client.post("/documents", files={"file": ("x.txt", b"")}).status_code == 400


def test_blank_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_delete_missing(client):
    assert client.delete("/documents/nope").status_code == 404
