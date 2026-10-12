import re
from dataclasses import asdict, dataclass
from statistics import mean

from pydantic import BaseModel

from agentic_rag.agent import Answer, RetrievalAgent, Turn


class EvalCase(BaseModel):
    id: str
    question: str
    kind: str
    expected_terms: list[list[str]]
    expected_sources: list[str]
    should_retrieve: bool
    should_abstain: bool
    history: list[Turn]


@dataclass(frozen=True)
class Score:
    correct: bool
    source_recall: float | None
    route_correct: bool
    latency_ms: float
    searches: int


def score_answer(case: EvalCase, answer: Answer) -> Score:
    normalized = " ".join(answer.text.casefold().split())
    terms_match = all(
        any(
            re.search(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)", normalized)
            for term in aliases
        )
        for aliases in case.expected_terms
    )
    correct = answer.abstained if case.should_abstain else terms_match and not answer.abstained
    cited_sources = {citation.source for citation in answer.citations}
    recall = (
        len(cited_sources & set(case.expected_sources)) / len(case.expected_sources)
        if case.expected_sources
        else None
    )
    return Score(
        correct,
        recall,
        answer.retrieved == case.should_retrieve,
        answer.latency_ms,
        len(answer.queries),
    )


def run_evaluation(agent: RetrievalAgent, cases: list[EvalCase]) -> dict[str, object]:
    if not cases:
        raise ValueError("Evaluation requires at least one question.")
    samples: list[dict[str, object]] = []
    scores: dict[str, list[Score]] = {"agentic": [], "baseline": []}
    for index, case in enumerate(cases):
        # Alternate order to reduce systematic warm-cache bias.
        order = ("agentic", "baseline") if index % 2 == 0 else ("baseline", "agentic")
        for system in order:
            answer = agent.ask(case.question, case.history, baseline=system == "baseline")
            score = score_answer(case, answer)
            scores[system].append(score)
            samples.append(
                {
                    "case_id": case.id,
                    "kind": case.kind,
                    "system": system,
                    "answer": asdict(answer),
                    "score": asdict(score),
                }
            )
    summary: dict[str, dict[str, float]] = {}
    for system, values in scores.items():
        recalls = [s.source_recall for s in values if s.source_recall is not None]
        summary[system] = {
            "correct_pct": 100 * mean(s.correct for s in values),
            "source_recall_pct": 100 * mean(recalls) if recalls else 0,
            "route_accuracy_pct": 100 * mean(s.route_correct for s in values),
            "mean_latency_ms": mean(s.latency_ms for s in values),
            "mean_searches": mean(s.searches for s in values),
        }
    return {"summary": summary, "samples": samples}
