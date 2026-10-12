import re
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from agentic_rag.store import SearchHit

ABSTENTION = "I don't have enough evidence in the indexed documents to answer that question."


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class RetrievalPlan(BaseModel):
    route: Literal["retrieve", "direct"]
    queries: list[str] = Field(max_length=3)
    direct_answer: str
    reason: str


class DraftCitation(BaseModel):
    label: int = Field(ge=1)
    quote: str = Field(min_length=1)


class AnswerDraft(BaseModel):
    status: Literal["answered", "insufficient"]
    answer: str
    citations: list[DraftCitation] = Field(max_length=8)
    follow_up_query: str | None


class LanguageModel(Protocol):
    def plan(self, question: str, history: list[Turn]) -> RetrievalPlan: ...

    def answer(self, question: str, history: list[Turn], hits: list[SearchHit]) -> AnswerDraft: ...


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 4) -> list[SearchHit]: ...


@dataclass(frozen=True)
class Citation:
    label: int
    chunk_id: str
    source: str
    page: int
    quote: str
    text: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    retrieved: bool
    queries: tuple[str, ...]
    trace: tuple[str, ...]
    latency_ms: float
    retrieval_ms: float
    abstained: bool


def validate_citations(draft: AnswerDraft, hits: list[SearchHit]) -> tuple[Citation, ...]:
    if draft.status != "answered" or not draft.answer.strip():
        return ()
    labels = {int(n) for n in re.findall(r"\[(\d+)\]", draft.answer)}
    if not draft.citations or not labels.issubset({c.label for c in draft.citations}):
        return ()
    citations = []
    for citation in draft.citations:
        if citation.label > len(hits):
            return ()
        hit = hits[citation.label - 1]
        quote = " ".join(citation.quote.split())
        if not quote or quote not in " ".join(hit.text.split()):
            return ()
        citations.append(
            Citation(citation.label, hit.id, hit.source, hit.page, citation.quote, hit.text)
        )
    return tuple(citations)


def safe_direct_response(question: str) -> str | None:
    normalized = question.casefold().strip().rstrip(".!?")
    if normalized in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}:
        return "Hello! Upload a document and ask me a question about it."
    if normalized in {"thanks", "thank you", "thanks a lot"}:
        return "You're welcome! Ask another document question whenever you're ready."
    if normalized in {"what can you do", "how can you help", "who are you"}:
        return "I answer questions about your uploaded documents with source citations."
    return None


class RetrievalAgent:
    def __init__(self, store: Retriever, model: LanguageModel, *, top_k: int = 4) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        self.store, self.model, self.top_k = store, model, top_k

    def ask(
        self, question: str, history: list[Turn] | None = None, *, baseline: bool = False
    ) -> Answer:
        question = question.strip()
        if not question or len(question) > 4000:
            raise ValueError("Questions must contain 1–4000 characters.")
        history = [Turn(role=t.role, content=t.content[:4000]) for t in (history or [])[-8:]]
        started = perf_counter()
        trace: list[str] = []
        if baseline:
            queries = [question]
            trace.append("Baseline: one search using the original question")
        else:
            plan = self.model.plan(question, history)
            trace.append(f"Planner: {plan.reason}")
            direct_response = safe_direct_response(question) if plan.route == "direct" else None
            if direct_response is not None:
                return Answer(
                    direct_response,
                    (),
                    False,
                    (),
                    tuple(trace),
                    (perf_counter() - started) * 1000,
                    0,
                    False,
                )
            if plan.route == "direct":
                trace.append("Unsafe direct classification overridden: factual evidence required")
                queries = [question]
            else:
                queries = list(dict.fromkeys(q.strip()[:1000] for q in plan.queries if q.strip()))
            if not queries:
                queries = [question]
        hits: dict[str, SearchHit] = {}
        executed: list[str] = []
        retrieval_ms = 0.0

        def retrieve(query: str) -> None:
            nonlocal retrieval_ms
            retrieval_started = perf_counter()
            for hit in self.store.search(query, top_k=self.top_k):
                hits.setdefault(hit.id, hit)
            retrieval_ms += (perf_counter() - retrieval_started) * 1000
            executed.append(query)

        for query in queries:
            retrieve(query)
        draft: AnswerDraft | None = None
        citations: tuple[Citation, ...] = ()
        if hits:
            draft = self.model.answer(question, history, list(hits.values()))
            citations = validate_citations(draft, list(hits.values()))
            if not baseline and not citations and draft.follow_up_query:
                retry = draft.follow_up_query.strip()[:1000]
                if retry and retry not in executed:
                    trace.append("Evidence insufficient: one bounded follow-up search")
                    retrieve(retry)
                    draft = self.model.answer(question, history, list(hits.values()))
                    citations = validate_citations(draft, list(hits.values()))
        text = draft.answer if draft and citations else ABSTENTION
        if citations:
            rendered_labels = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
            missing_labels = sorted({citation.label for citation in citations} - rendered_labels)
            if missing_labels:
                text += " " + " ".join(f"[{label}]" for label in missing_labels)
        trace.append(f"Retrieved {len(hits)} unique chunks in {len(executed)} searches")
        trace.append(
            "Source quotes verified" if citations else "Abstained: insufficient valid evidence"
        )
        return Answer(
            text,
            citations,
            True,
            tuple(executed),
            tuple(trace),
            (perf_counter() - started) * 1000,
            retrieval_ms,
            not bool(citations),
        )
