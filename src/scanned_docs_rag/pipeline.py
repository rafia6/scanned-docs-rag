"""The RAG pipeline: ingest PDFs, retrieve relevant chunks, answer with citations."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_pages
from .config import Settings, get_settings
from .embeddings import Embedder, OpenAIEmbedder
from .ingest import load_pdf
from .llm import ChatModel, OpenAIChat
from .models import Answer, Source
from .store import VectorStore

SYSTEM_PROMPT = """You answer questions using only the numbered context passages provided.
Rules:
- Cite every claim with the passage number in square brackets, e.g. [1] or [2][3].
- If the passages do not contain the answer, say you could not find it in the documents.
- The passages may contain OCR errors; read through obvious typos but do not invent text.
- Answer in the same language as the question."""

NO_CONTEXT_ANSWER = "I could not find anything relevant in the ingested documents."


def format_context(sources: list[Source]) -> str:
    return "\n\n".join(
        f"[{s.number}] (document: {s.document}, page {s.page})\n{s.text}" for s in sources
    )


def build_messages(question: str, sources: list[Source]) -> list[dict]:
    user = f"Context passages:\n\n{format_context(sources)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def extract_citations(answer: str, max_number: int) -> list[int]:
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return sorted(n for n in numbers if 1 <= n <= max_number)


@dataclass
class IngestResult:
    document: str
    pages: int
    ocr_pages: int
    chunks: int


class RAGPipeline:
    def __init__(self, store: VectorStore, chat: ChatModel, settings: Settings | None = None):
        self.store = store
        self.chat = chat
        self.settings = settings or get_settings()

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        chat: ChatModel | None = None,
    ):
        settings = settings or get_settings()
        embedder = embedder or OpenAIEmbedder(settings.embedding_model, settings.openai_api_key)
        chat = chat or OpenAIChat(settings.chat_model, settings.openai_api_key)
        store = VectorStore(embedder, settings.chroma_path, settings.collection)
        return cls(store, chat, settings)

    # --- ingestion -------------------------------------------------------------

    def ingest(self, path: str | Path, force_ocr: bool = False) -> IngestResult:
        s = self.settings
        pages = load_pdf(
            path,
            ocr_languages=s.ocr_languages,
            ocr_dpi=s.ocr_dpi,
            min_text_chars=s.min_text_chars,
            force_ocr=force_ocr,
        )
        chunks = chunk_pages(pages, s.chunk_size, s.chunk_overlap)
        document = Path(path).name
        self.store.delete_document(document)  # replace older versions of the same file
        self.store.add(chunks)
        return IngestResult(
            document=document,
            pages=len(pages),
            ocr_pages=sum(p.ocr for p in pages),
            chunks=len(chunks),
        )

    # --- question answering ----------------------------------------------------

    def retrieve(self, question: str, k: int | None = None) -> list[Source]:
        return self.store.search(question, k or self.settings.top_k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        sources = self.retrieve(question, k)
        if not sources:
            return Answer(question=question, answer=NO_CONTEXT_ANSWER)
        text = self.chat.complete(build_messages(question, sources))
        return Answer(question, text, sources, extract_citations(text, len(sources)))

    def ask_stream(self, question: str, k: int | None = None) -> tuple[list[Source], Iterator[str]]:
        """Return the sources right away and a token stream for the answer."""
        sources = self.retrieve(question, k)
        if not sources:
            return [], iter([NO_CONTEXT_ANSWER])
        return sources, self.chat.stream(build_messages(question, sources))
