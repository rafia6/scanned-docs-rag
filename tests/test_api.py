import json

import pytest
from fastapi.testclient import TestClient

from scanned_docs_rag.api import create_app


@pytest.fixture
def client(pipeline):
    return TestClient(create_app(lambda: pipeline))


def _upload(client, path):
    with open(path, "rb") as f:
        return client.post("/documents", files={"file": (path.name, f, "application/pdf")})


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_upload_list_and_delete(client, text_pdf):
    response = _upload(client, text_pdf)
    assert response.status_code == 201
    assert response.json()["document"] == "contract.pdf"
    assert client.get("/documents").json()["documents"] == ["contract.pdf"]

    assert client.delete("/documents/contract.pdf").status_code == 204
    assert client.get("/documents").json()["documents"] == []


def test_rejects_non_pdf(client):
    response = client.post("/documents", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert response.status_code == 400


def test_ask_returns_answer_and_sources(client, text_pdf):
    _upload(client, text_pdf)
    body = client.post("/ask", json={"question": "What is the rent?"}).json()
    assert body["cited"] == [1]
    assert body["sources"][0]["document"] == "contract.pdf"


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_stream_sends_sources_then_tokens_then_done(client, text_pdf):
    _upload(client, text_pdf)
    with client.stream("POST", "/ask/stream", json={"question": "rent?"}) as response:
        raw = "".join(response.iter_text())
    events = [block.split("\n") for block in raw.strip().split("\n\n")]
    names = [e[0].removeprefix("event: ") for e in events]
    assert names[0] == "sources" and names[-1] == "done"
    tokens = [json.loads(e[1].removeprefix("data: ")) for e in events if e[0] == "event: token"]
    assert "".join(tokens).startswith("The invoice total")
