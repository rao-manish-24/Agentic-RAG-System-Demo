from agentic_rag.config import Settings

settings = Settings.from_env()
print(f"Chat model: {settings.chat_model}; embeddings: {settings.embedding_model}")
print(f"Local index: {settings.data_dir}")
print(f"OpenAI key configured: {bool(settings.api_key)}")
