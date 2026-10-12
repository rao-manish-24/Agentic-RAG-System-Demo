from openai import OpenAI

from agentic_rag.config import Settings


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.embedding_model
        self.client = OpenAI(api_key=settings.require_api_key(), timeout=45, max_retries=2)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
