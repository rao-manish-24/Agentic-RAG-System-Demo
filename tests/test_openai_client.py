from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_rag.config import Settings
from agentic_rag.openai_client import OpenAIEmbedder


def test_embedding_order_and_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0.0, 1.0]),
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
        ]
    )
    monkeypatch.setattr("agentic_rag.openai_client.OpenAI", lambda **_: client)
    embedder = OpenAIEmbedder(Settings(api_key="unit-test-placeholder"))
    assert embedder.embed([]) == []
    client.embeddings.create.assert_not_called()
    assert embedder.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input=["a", "b"]
    )
