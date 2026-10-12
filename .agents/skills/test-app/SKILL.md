---
name: test-app
description: Install and validate the Agentic RAG demo before every PR, including recorded browser checks for UI changes.
---

# Test the Agentic RAG app

Run commands at the repository root. Never print the API key or include it in recordings.

## Install
1. Use Python 3.11 or newer: `python3.11 -m venv .venv`.
2. `.venv/bin/python -m pip install -r requirements.txt`.
3. Legacy binary `.doc` files require `antiword`: on Debian/Ubuntu install with `sudo apt-get install antiword` (or ask the machine administrator if unavailable). DOCX, PDF and TXT do not need it.
4. Provide `OPENAI_API_KEY` or `OPENAIAPIKEY` in the process environment. Do not write credentials into repository files. Offline tests do not require a key.

## Mandatory checks before every PR
1. `.venv/bin/ruff check .`
2. `.venv/bin/ruff format --check .`
3. `.venv/bin/mypy agentic_rag tests` (include `app.py scripts` once present).
4. `.venv/bin/pytest -q`
5. `.venv/bin/python -m agentic_rag` verifies configuration without printing credentials.
6. `git diff --check`; ensure no uploads, vector indexes, credentials, virtual environments or recordings are staged.

## Run and test UI changes
The bootstrap slice has no UI. Once `app.py` exists, the following is mandatory **before opening any UI PR**:
1. Generate fixtures: `.venv/bin/python scripts/make_fixtures.py`.
2. Start: `.venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true`.
3. Open port 8501 in the browser; record the real browser interaction (not shell output). If using Devin, hand off UI testing to the testing agent; it owns install/start/browser/recording.
4. Upload the fixture PDF, DOCX and TXT; click **Index documents**; verify chunk/document counts and successful indexing. Upload legacy DOC when antiword is installed.
5. Ask `What is the Atlas refund window?`; verify 30 days, clickable/expandable source chunks, filename/page citations and latency.
6. Ask a follow-up `And how long does processing take?`; verify conversational context and 5 business days.
7. Ask `Hello`; verify no retrieval. Ask an unsupported factual question; verify explicit insufficient evidence, not fabrication.
8. Verify duplicate upload/index is idempotent, clear-chat works, unsupported/empty files show actionable errors, missing-key startup is safe and no secrets appear.
9. Run `.venv/bin/python scripts/evaluate.py --output eval/results.json` for the fixed-set live agent/baseline comparison when a valid key is available. Costs money; skip only with a documented blocker.
10. Stop recording and attach one primary video plus a key screenshot in the UI PR body. Document unavailable coverage honestly. Never claim offline fakes validate live OpenAI.
