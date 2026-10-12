import argparse
import hashlib
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_rag.agent import RetrievalAgent  # noqa: E402
from agentic_rag.config import Settings  # noqa: E402
from agentic_rag.evaluation import EvalCase, run_evaluation  # noqa: E402
from agentic_rag.openai_client import OpenAIEmbedder, OpenAILanguageModel  # noqa: E402
from agentic_rag.store import VectorStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-set agentic vs single-shot RAG evaluation")
    parser.add_argument("--output", type=Path, default=ROOT / "eval/results.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval/questions.json")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.require_api_key()
    question_bytes = args.questions.read_bytes()
    cases = [EvalCase.model_validate(item) for item in json.loads(question_bytes)]
    with tempfile.TemporaryDirectory(prefix="rag-eval-") as directory:
        store = VectorStore(Path(directory), OpenAIEmbedder(settings), namespace="evaluation")
        for source in sorted((ROOT / "eval/data").glob("*.txt")):
            store.ingest(source.name, source.read_bytes())
        agent = RetrievalAgent(store, OpenAILanguageModel(settings), top_k=args.top_k)
        report = run_evaluation(agent, cases)
    report["metadata"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "top_k": args.top_k,
        "questions_sha256": hashlib.sha256(question_bytes).hexdigest(),
        "cases": len(cases),
        "notes": "Keyword correctness and source recall are not semantic entailment scores. "
        "One run; timings include remote API variability. Same corpus/models/answer "
        "prompt/grounding validation; baseline always does one raw-question search.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = report["summary"]
    assert isinstance(summary, dict)
    rows = [
        [system, *(round(value, 2) for value in metrics.values())]
        for system, metrics in summary.items()
    ]
    print(
        tabulate(
            rows,
            headers=[
                "System",
                "Correct %",
                "Source recall %",
                "Route %",
                "Mean latency ms",
                "Mean searches",
            ],
            tablefmt="github",
        )
    )
    print(f"Per-question evidence: {args.output}")
    print("Keyword scoring is a smoke benchmark, not proof of semantic correctness.")


if __name__ == "__main__":
    main()
