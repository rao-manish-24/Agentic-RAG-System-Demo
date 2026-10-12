# Agentic RAG System Demo

Local document question answering with OpenAI and persistent Chroma. Python 3.11+.

## Setup
```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# Provide OPENAI_API_KEY (or OPENAIAPIKEY) in your shell environment.
.venv/bin/python -m agentic_rag
```

No key is required for offline tests. Legacy DOC support requires the `antiword` system
package; DOCX, PDF and text are parsed directly. Scanned PDFs require OCR externally.

## Checks
```sh
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy agentic_rag tests
.venv/bin/pytest -q
```

See `AGENTS.md` for contributor conventions and `.agents/skills/test-app/SKILL.md`
for the required pre-PR testing procedure. UI, ingestion and evaluation will arrive in
separate working-slice PRs. Never commit API keys or uploaded documents.
