import json
from pathlib import Path

import pytest

from agentic_rag.agent import Answer, Citation, RetrievalAgent
from agentic_rag.evaluation import EvalCase, run_evaluation, score_answer
from tests.test_agent import FakeModel, FakeRetriever


def case() -> EvalCase:
    return EvalCase(
        id="sample",
        question="Refund?",
        kind="lookup",
        expected_terms=[["30 days", "thirty days"]],
        expected_sources=["policy.txt"],
        should_retrieve=True,
        should_abstain=False,
        history=[],
    )


def answer(text: str, *, abstained: bool = False) -> Answer:
    return Answer(
        text,
        (Citation(1, "id", "policy.txt", 1, "30 days", "30 days"),),
        True,
        ("query",),
        (),
        100,
        20,
        abstained,
    )


def test_keyword_boundaries_and_source_recall() -> None:
    assert score_answer(case(), answer("30 days [1]")).correct
    assert score_answer(case(), answer("thirty days [1]")).correct
    assert not score_answer(case(), answer("130 days [1]")).correct
    assert score_answer(case(), answer("30 days [1]")).source_recall == 1
    assert not score_answer(case(), answer("30 days [1]", abstained=True)).correct


def test_abstention_scoring() -> None:
    unknown = case().model_copy(update={"should_abstain": True, "expected_sources": []})
    assert score_answer(unknown, answer("I cannot answer", abstained=True)).correct
    assert not score_answer(unknown, answer("Made up")).correct
    assert score_answer(unknown, answer("I cannot answer", abstained=True)).source_recall is None


def test_fixed_dataset_valid_and_has_diverse_cases() -> None:
    raw = json.loads((Path(__file__).parents[1] / "eval/questions.json").read_text())
    cases = [EvalCase.model_validate(item) for item in raw]
    assert len({c.id for c in cases}) == len(cases) == 10
    assert {c.kind for c in cases} >= {"lookup", "multi-source", "conversation", "unanswerable"}


def test_comparison_runs_both_systems_and_rejects_empty_set() -> None:
    report = run_evaluation(RetrievalAgent(FakeRetriever(), FakeModel()), [case()])
    samples, summary = report["samples"], report["summary"]
    assert isinstance(samples, list) and len(samples) == 2
    assert isinstance(summary, dict) and set(summary) == {"agentic", "baseline"}
    with pytest.raises(ValueError):
        run_evaluation(RetrievalAgent(FakeRetriever(), FakeModel()), [])
