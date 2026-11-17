"""Persistent vector store backed by ChromaDB."""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from .embeddings import Embedder
from .models import Chunk, Source


class VectorStore:
    def __init__(self, embedder: Embedder, path: str = "chroma_db", collection: str = "documents"):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk]) -> int:
        """Embed and store chunks. Ids are content-based, so re-ingesting a file is a no-op."""
        if not chunks:
            return 0
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=self.embedder.embed([c.text for c in chunks]),
        )
        return len(chunks)

    def search(self, query: str, k: int = 5) -> list[Source]:
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=min(k, self.count()),
        )
        rows = zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        )
        sources = []
        for i, (text, meta, distance) in enumerate(rows, start=1):
            sources.append(
                Source(
                    number=i,
                    document=meta["document"],
                    page=int(meta["page"]),
                    text=text,
                    score=round(1 - distance, 4),  # cosine similarity
                )
            )
        return sources

    def documents(self) -> list[str]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["document"] for m in metas})

    def delete_document(self, document: str) -> None:
        self.collection.delete(where={"document": document})

    def count(self) -> int:
        return self.collection.count()
