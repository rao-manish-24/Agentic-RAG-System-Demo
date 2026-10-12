from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from agentic_rag.agent import Answer, RetrievalAgent, Turn
from tests.test_agent import FakeModel, FakeRetriever

APP = Path(__file__).parents[1] / "app.py"


def test_missing_key_disables_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAIAPIKEY", raising=False)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert app.get("file_uploader")[0].proto.disabled
    assert app.button[0].disabled
    assert "Set OPENAI_API_KEY" in app.warning[0].value


def test_chat_rendering_and_clear_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStore(FakeRetriever):
        def count(self) -> int:
            return 1

        def sources(self) -> list[str]:
            return ["policy.txt"]

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    app = AppTest.from_file(str(APP))
    model = FakeModel()
    store = FakeStore()
    app.session_state["workspace"] = (store, RetrievalAgent(store, model))
    app.run()
    app.chat_input[0].set_value("How long does processing take?").run()
    assert not app.exception
    messages: list[Turn] = app.session_state["messages"]
    answers: list[Answer] = app.session_state["answers"]
    assert len(messages) == 2 and len(answers) == 1
    assert any("policy.txt" in expander.label for expander in app.expander)
    assert app.metric[1].value == "1"
    app.button[1].click().run()
    assert not app.session_state["messages"]
    assert app.metric[1].value == "0"
