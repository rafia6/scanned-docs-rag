"""FastAPI service: upload PDFs, ask questions, stream answers as server-sent events."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .pipeline import RAGPipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(pipeline_factory: Callable[[], RAGPipeline] = RAGPipeline.from_settings) -> FastAPI:
    app = FastAPI(
        title="scanned-docs-rag",
        version=__version__,
        description="Question answering over scanned PDFs with page-level citations.",
    )

    def get_pipeline(request: Request) -> RAGPipeline:
        # Built on first use so the app can start (and /health works) before any request.
        if getattr(request.app.state, "pipeline", None) is None:
            request.app.state.pipeline = pipeline_factory()
        return request.app.state.pipeline

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/documents")
    def list_documents(request: Request) -> dict:
        pipeline = get_pipeline(request)
        return {"documents": pipeline.store.documents(), "chunks": pipeline.store.count()}

    @app.post("/documents", status_code=201)
    def upload_document(
        request: Request, file: Annotated[UploadFile, File()], force_ocr: bool = False
    ) -> dict:
        name = Path(file.filename or "").name
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / name
            with target.open("wb") as out:
                shutil.copyfileobj(file.file, out)
            result = get_pipeline(request).ingest(target, force_ocr=force_ocr)
        return asdict(result)

    @app.delete("/documents/{name}", status_code=204)
    def delete_document(request: Request, name: str) -> None:
        get_pipeline(request).store.delete_document(name)

    @app.post("/ask")
    def ask(request: Request, body: AskRequest) -> dict:
        return asdict(get_pipeline(request).ask(body.question, body.top_k))

    @app.post("/ask/stream")
    def ask_stream(request: Request, body: AskRequest) -> StreamingResponse:
        sources, tokens = get_pipeline(request).ask_stream(body.question, body.top_k)

        def events() -> Iterator[str]:
            yield _sse("sources", [asdict(s) for s in sources])
            for token in tokens:
                yield _sse("token", token)
            yield _sse("done", {})

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
