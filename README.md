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
for the required pre-PR testing procedure. Never commit API keys or uploaded documents.

## Ingestion
```sh
.venv/bin/python -m agentic_rag ingest ./policy.pdf ./notes.docx ./legacy.doc ./faq.txt
```
OpenAI embeds page-preserving, overlapping 1,800-character chunks; local Chroma persists
them in `data/chroma` (override with `RAG_DATA_DIR`). Identical filename/content is deduplicated
before any embedding request. Changed contents are indexed as a distinct version, not silently
overwritten. Collections are separated by embedding model to avoid mixing vector spaces.
PDF page numbers are physical pages; DOC/DOCX/TXT citations use logical page 1. DOCX tables
are included. Uploads are capped at 20 MB, with a 50 MB expanded-DOCX safety limit.
