import argparse
from pathlib import Path

from agentic_rag.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Agentic RAG utilities")
    commands = parser.add_subparsers(dest="command")
    ingest = commands.add_parser("ingest", help="Index local documents")
    ingest.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.command == "ingest":
        from agentic_rag.openai_client import OpenAIEmbedder
        from agentic_rag.store import VectorStore

        store = VectorStore(settings.data_dir, OpenAIEmbedder(settings))
        for path in args.files:
            result = store.ingest(path.name, path.read_bytes())
            print(f"{result.source}: {result.chunks_added} new / {result.chunks_total} chunks")
        return
    print(f"Chat model: {settings.chat_model}; embeddings: {settings.embedding_model}")
    print(f"Local index: {settings.data_dir}")
    print(f"OpenAI key configured: {bool(settings.api_key)}")


if __name__ == "__main__":
    main()
