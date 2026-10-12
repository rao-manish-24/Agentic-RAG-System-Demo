from pathlib import Path

import pytest

from agentic_rag.config import Settings


def test_defaults_and_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAIAPIKEY", raising=False)
    settings = Settings.from_env()
    assert settings.data_dir == Path("data/chroma")
    with pytest.raises(ValueError, match="Set OPENAI"):
        settings.require_api_key()


def test_legacy_key_alias_is_private(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAIAPIKEY", "unit-test-placeholder")
    settings = Settings.from_env()
    assert settings.require_api_key() == "unit-test-placeholder"
    assert "unit-test-placeholder" not in repr(settings)


def test_standard_key_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "standard-placeholder")
    monkeypatch.setenv("OPENAIAPIKEY", "alias-placeholder")
    monkeypatch.setenv("RAG_CHAT_MODEL", "custom-model")
    assert Settings.from_env().require_api_key() == "standard-placeholder"
    assert Settings.from_env().chat_model == "custom-model"
