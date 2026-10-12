import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False, default="")
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    data_dir: Path = Path("data/chroma")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAIAPIKEY") or "",
            chat_model=os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
            data_dir=Path(os.getenv("RAG_DATA_DIR", "data/chroma")),
        )

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY or OPENAIAPIKEY to enable OpenAI requests.")
        return self.api_key
