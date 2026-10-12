import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from agentic_rag.documents import Chunk, chunk_document


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class SearchHit:
    id: str
    text: str
    source: str
    page: int
    distance: float


@dataclass(frozen=True)
class IngestResult:
    source: str
    chunks_added: int
    chunks_total: int


class VectorStore:
    def __init__(self, path: Path, embedder: Embedder, *, namespace: str = "documents") -> None:
        self.embedder = embedder
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        model_hash = hashlib.sha256(embedder.model.encode()).hexdigest()[:12]
        self.collection = client.get_or_create_collection(
            f"{namespace}-{model_hash}",
            metadata={"hnsw:space": "cosine", "embedding_model": embedder.model},
            embedding_function=None,
        )

    def count(self) -> int:
        return self.collection.count()

    def sources(self) -> list[str]:
        metadata = self.collection.get(include=["metadatas"])["metadatas"] or []
        return sorted({str(item["source"]) for item in metadata})

    def ingest(self, name: str, content: bytes) -> IngestResult:
        chunks = chunk_document(name, content)
        existing = set(self.collection.get(ids=[c.id for c in chunks], include=[])["ids"])
        missing = [c for c in chunks if c.id not in existing]
        vectors: list[list[float]] = []
        for offset in range(0, len(missing), 64):
            vectors.extend(self.embedder.embed([c.text for c in missing[offset : offset + 64]]))
        if len(vectors) != len(missing):
            raise ValueError("Embedding service returned the wrong number of vectors.")
        # Finish external calls before writing so a failed request cannot leave a partial document.
        for offset in range(0, len(missing), 64):
            batch: list[Chunk] = missing[offset : offset + 64]
            self.collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=np.asarray(vectors[offset : offset + 64], dtype=np.float32),
                metadatas=[
                    {
                        "source": c.source,
                        "page": c.page,
                        "chunk": c.index,
                        "document": c.document_id,
                    }
                    for c in batch
                ],
            )
        return IngestResult(chunks[0].source, len(missing), len(chunks))

    def search(self, query: str, *, top_k: int = 4) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        count = self.count()
        if not count:
            return []
        result = self.collection.query(
            query_embeddings=np.asarray(self.embedder.embed([query]), dtype=np.float32),
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        documents = (result["documents"] or [[]])[0]
        metadata = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        return [
            SearchHit(id_, text, str(meta["source"]), int(str(meta["page"])), float(distance))
            for id_, text, meta, distance in zip(
                result["ids"][0], documents, metadata, distances, strict=True
            )
        ]
