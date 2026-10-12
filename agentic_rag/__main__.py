import argparse
from pathlib import Path

from agentic_rag.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Agentic RAG utilities")
    commands = parser.add_subparsers(dest="command")
    ingest = commands.add_parser("ingest", help="Index local documents")
    ingest.add_argument("files", nargs="+", type=Path)
    ask = commands.add_parser("ask", help="Ask a grounded document question")
    ask.add_argument("question")
    ask.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.command in {"ingest", "ask"}:
        from agentic_rag.agent import RetrievalAgent
        from agentic_rag.openai_client import OpenAIEmbedder, OpenAILanguageModel
        from agentic_rag.store import VectorStore

        store = VectorStore(settings.data_dir, OpenAIEmbedder(settings))
        if args.command == "ask":
            agent = RetrievalAgent(store, OpenAILanguageModel(settings))
            answer = agent.ask(args.question, baseline=args.baseline)
            print(answer.text)
            for citation in answer.citations:
                print(
                    f"[{citation.label}] {citation.source}, page {citation.page}: {citation.quote}"
                )
            print(f"Latency: {answer.latency_ms:.0f} ms; searches: {len(answer.queries)}")
            return
        for path in args.files:
            result = store.ingest(path.name, path.read_bytes())
            print(f"{result.source}: {result.chunks_added} new / {result.chunks_total} chunks")
        return
    print(f"Chat model: {settings.chat_model}; embeddings: {settings.embedding_model}")
    print(f"Local index: {settings.data_dir}")
    print(f"OpenAI key configured: {bool(settings.api_key)}")


if __name__ == "__main__":
    main()
