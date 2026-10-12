from pathlib import Path

import pytest

from agentic_rag.store import VectorStore


class FakeEmbedder:
    model = "offline-keyword-v1"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float("refund" in t.lower()), float("support" in t.lower()), 0.1] for t in texts]


def test_persistence_deduplication_and_search(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    store = VectorStore(tmp_path, embedder)
    assert store.search("refund") == []
    assert store.ingest("policy.txt", b"Refund window is 30 days").chunks_added == 1
    store.ingest("support.txt", b"Support is available weekdays")
    calls = embedder.calls
    assert store.ingest("policy.txt", b"Refund window is 30 days").chunks_added == 0
    assert embedder.calls == calls
    assert store.sources() == ["policy.txt", "support.txt"]
    reopened = VectorStore(tmp_path, embedder)
    assert reopened.count() == 2
    assert reopened.search("refund", top_k=10)[0].source == "policy.txt"
    assert reopened.search("refund")[0].page == 1


def test_model_namespaces_and_invalid_queries(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    store = VectorStore(tmp_path, embedder)
    store.ingest("policy.txt", b"refund")
    embedder.model = "different-model"
    assert VectorStore(tmp_path, embedder).count() == 0
    with pytest.raises(ValueError):
        store.search(" ")
    with pytest.raises(ValueError):
        store.search("refund", top_k=0)


def test_embedding_failure_does_not_index(tmp_path: Path) -> None:
    class BrokenEmbedder(FakeEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            return []

    store = VectorStore(tmp_path, BrokenEmbedder())
    with pytest.raises(ValueError, match="number of vectors"):
        store.ingest("policy.txt", b"refund")
    assert store.count() == 0
