import pytest
from pydantic import ValidationError

from agentic_rag.agent import (
    ABSTENTION,
    AnswerDraft,
    DraftCitation,
    RetrievalAgent,
    RetrievalPlan,
    Turn,
    validate_citations,
)
from agentic_rag.store import SearchHit

HIT = SearchHit("chunk-1", "Atlas refunds take 5 business days.", "policy.txt", 2, 0.1)


class FakeRetriever:
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self.hits = [HIT] if hits is None else hits
        self.queries: list[str] = []

    def search(self, query: str, *, top_k: int = 4) -> list[SearchHit]:
        self.queries.append(query)
        return self.hits


class FakeModel:
    def __init__(self) -> None:
        self.plan_calls = 0
        self.answer_calls = 0
        self.history: list[Turn] = []
        self.plan_result = RetrievalPlan(
            route="retrieve",
            queries=["Atlas refund processing"],
            direct_answer="",
            reason="followup",
        )
        self.draft = AnswerDraft(
            status="answered",
            answer="It takes 5 business days [1].",
            citations=[DraftCitation(label=1, quote="5 business days")],
            follow_up_query=None,
        )

    def plan(self, question: str, history: list[Turn]) -> RetrievalPlan:
        self.plan_calls += 1
        self.history = history
        return self.plan_result

    def answer(self, question: str, history: list[Turn], hits: list[SearchHit]) -> AnswerDraft:
        self.answer_calls += 1
        return self.draft


def test_conversation_planning_and_source_citations() -> None:
    store, model = FakeRetriever(), FakeModel()
    history = [Turn(role="user", content="Atlas refunds?")]
    answer = RetrievalAgent(store, model).ask("How long does processing take?", history)
    assert store.queries == ["Atlas refund processing"]
    assert model.history == history
    assert answer.citations[0].source == "policy.txt"
    assert answer.citations[0].page == 2
    assert not answer.abstained
    assert answer.latency_ms >= answer.retrieval_ms >= 0


def test_direct_route_does_not_retrieve_or_generate_grounded_answer() -> None:
    store, model = FakeRetriever(), FakeModel()
    model.plan_result = RetrievalPlan(
        route="direct", queries=[], direct_answer="Hello!", reason="greeting"
    )
    answer = RetrievalAgent(store, model).ask("Hi")
    assert answer.text == "Hello!"
    assert not answer.retrieved
    assert store.queries == []
    assert model.answer_calls == 0


def test_baseline_is_one_raw_question_search_without_planner() -> None:
    store, model = FakeRetriever(), FakeModel()
    answer = RetrievalAgent(store, model).ask("Raw question?", baseline=True)
    assert store.queries == ["Raw question?"]
    assert model.plan_calls == 0
    assert model.answer_calls == 1
    assert answer.retrieved


def test_empty_store_abstains_without_answer_call() -> None:
    model = FakeModel()
    answer = RetrievalAgent(FakeRetriever([]), model).ask("Unknown fact?")
    assert answer.text == ABSTENTION
    assert answer.abstained
    assert model.answer_calls == 0


def test_retry_is_bounded_and_queries_deduplicated() -> None:
    store, model = FakeRetriever(), FakeModel()
    model.plan_result.queries = ["same", "same", "second"]
    model.draft = AnswerDraft(
        status="insufficient", answer="Unknown", citations=[], follow_up_query="third"
    )
    answer = RetrievalAgent(store, model).ask("Unknown fact?")
    assert store.queries == ["same", "second", "third"]
    assert model.answer_calls == 2
    assert answer.abstained
    assert not answer.citations


@pytest.mark.parametrize(
    ("text", "label", "quote"),
    [
        ("5 days [2]", 2, "5 business days"),
        ("5 days [1]", 1, "invented quote"),
        ("5 days [1] [2]", 1, "5 business days"),
    ],
)
def test_fabricated_or_mismatched_citations_fail_closed(text: str, label: int, quote: str) -> None:
    draft = AnswerDraft(
        status="answered",
        answer=text,
        citations=[DraftCitation(label=label, quote=quote)],
        follow_up_query=None,
    )
    assert validate_citations(draft, [HIT]) == ()
    model = FakeModel()
    model.draft = draft
    assert RetrievalAgent(FakeRetriever(), model).ask("Question").abstained


def test_question_and_plan_limits() -> None:
    with pytest.raises(ValueError):
        RetrievalAgent(FakeRetriever(), FakeModel()).ask(" ")
    with pytest.raises(ValueError):
        RetrievalAgent(FakeRetriever(), FakeModel()).ask("a" * 4001)
    with pytest.raises(ValidationError):
        RetrievalPlan(route="retrieve", queries=["a"] * 4, direct_answer="", reason="invalid")
    with pytest.raises(ValueError):
        RetrievalAgent(FakeRetriever(), FakeModel(), top_k=0)


def test_verified_structured_citations_render_when_inline_labels_are_missing() -> None:
    model = FakeModel()
    model.draft.answer = "It takes 5 business days."
    answer = RetrievalAgent(FakeRetriever(), model).ask("Question")
    assert answer.text == "It takes 5 business days. [1]"
    assert answer.citations[0].quote == "5 business days"


def test_blank_quote_or_missing_structured_citations_abstain() -> None:
    model = FakeModel()
    model.draft.citations = [DraftCitation(label=1, quote="   ")]
    assert RetrievalAgent(FakeRetriever(), model).ask("Question").abstained
    model.draft.citations = []
    assert RetrievalAgent(FakeRetriever(), model).ask("Question").abstained


def test_history_is_bounded_and_empty_plan_falls_back() -> None:
    store, model = FakeRetriever(), FakeModel()
    model.plan_result.queries = []
    history = [Turn(role="user", content="a" * 5000) for _ in range(12)]
    RetrievalAgent(store, model).ask("Fallback?", history)
    assert len(model.history) == 8
    assert len(model.history[0].content) == 4000
    assert store.queries == ["Fallback?"]
