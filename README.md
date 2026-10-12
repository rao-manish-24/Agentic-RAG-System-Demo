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

## Ask and compare
```sh
.venv/bin/python -m agentic_rag ask "What does the refund policy say?"
.venv/bin/python -m agentic_rag ask --baseline "What does the refund policy say?"
.venv/bin/python scripts/evaluate.py --output eval/results.json
```
The agent plans up to 3 context-aware queries (or skips retrieval for small-talk), searches,
generates a cited answer, and may perform one evidence-repair search. Citations must refer to
retrieved labels and contain actual source quotes; invalid/missing evidence fails closed to an
abstention. This mechanical check does **not** guarantee semantic entailment of every claim.
Document instructions are explicitly untrusted in prompts, not a complete prompt-injection defense.

The fixed 10-question evaluation uses a fresh temporary Chroma index of synthetic Atlas/Beacon
policies, including lookups, multi-source questions, follow-ups, unknown facts and greetings.
Both systems use the same corpus, models, answer prompt and citation checks. The baseline always
does one search of the original question, without planning or repair. The table reports keyword
correctness, expected-source recall, routing, end-to-end latency and search count. JSON contains
all per-question answers/scores. This is a small smoke benchmark, not a claim of agent superiority.
Live evaluations incur OpenAI costs. Override models with `RAG_CHAT_MODEL`/`RAG_EMBEDDING_MODEL`.
