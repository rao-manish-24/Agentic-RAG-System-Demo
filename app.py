from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import uuid4

import streamlit as st

from agentic_rag.agent import Answer, RetrievalAgent, Turn
from agentic_rag.config import Settings
from agentic_rag.documents import MAX_FILE_BYTES
from agentic_rag.openai_client import OpenAIEmbedder, OpenAILanguageModel
from agentic_rag.presentation import safe_error
from agentic_rag.store import VectorStore

st.set_page_config(page_title="Groundwork · Agentic RAG", page_icon="G", layout="wide")
st.html("""
<style>
  .stApp { background: #f7f8fc; }
  .block-container { max-width: 1100px; padding-top: 2.5rem; }
  h1, h2, h3 { letter-spacing: -0.04em; }
  [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e8eaf1; }
  [data-testid="stMetric"] { background: #fff; border: 1px solid #e8eaf1;
    border-radius: 14px; padding: 16px 20px; }
  [data-testid="stChatMessage"] { background: #fff; border: 1px solid #e8eaf1;
    border-radius: 16px; margin-top: 12px; }
  .eyebrow { color: #5c52c4; font-size: .75rem; font-weight: 700;
    letter-spacing: .15em; margin-bottom: .6rem; }
  .intro { color: #687086; max-width: 680px; font-size: 1.05rem; }
  .note { padding: 18px 22px; border-left: 3px solid #7865dc; background: #f0eefb;
    border-radius: 8px; margin: 18px 0; color: #4f5267; }
</style>
""")


def get_workspace(settings: Settings) -> tuple[VectorStore, RetrievalAgent]:
    if "workspace" not in st.session_state:
        session_id = st.session_state.setdefault("workspace_id", uuid4().hex)
        path: Path = settings.data_dir / "sessions" / session_id
        store = VectorStore(path, OpenAIEmbedder(settings))
        agent = RetrievalAgent(store, OpenAILanguageModel(settings))
        st.session_state.workspace = (store, agent)
    workspace: tuple[VectorStore, RetrievalAgent] = st.session_state.workspace
    return workspace


def render_answer(answer: Answer) -> None:
    st.markdown(answer.text)
    route = "Retrieved documents" if answer.retrieved else "Direct response · no retrieval"
    st.caption(
        f"{route} · {answer.latency_ms / 1000:.2f}s total · "
        f"{answer.retrieval_ms / 1000:.2f}s retrieval · {len(answer.queries)} searches"
    )
    for citation in answer.citations:
        with st.expander(f"[{citation.label}] {citation.source} · page {citation.page}"):
            st.text(citation.quote)
            st.caption(f"Source chunk {citation.chunk_id[:12]}")
            st.text(citation.text)
    with st.expander("Agent trace"):
        for step in answer.trace:
            st.text(step)
        for query in answer.queries:
            st.code(query, language=None)


settings = Settings.from_env()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "answers" not in st.session_state:
    st.session_state.answers = []

store: VectorStore | None = None
agent: RetrievalAgent | None = None
if settings.api_key:
    try:
        store, agent = get_workspace(settings)
    except Exception as error:
        st.error(safe_error(error))

with st.sidebar:
    st.markdown("### Groundwork")
    st.caption("YOUR DOCUMENTS, GROUNDED ANSWERS")
    st.divider()
    st.markdown("#### Document library")
    uploads = st.file_uploader(
        "Add documents",
        type=["pdf", "doc", "docx", "txt"],
        accept_multiple_files=True,
        disabled=store is None,
        help="PDF, DOC, DOCX, UTF-8 TXT · up to 20 MB each · PDFs must contain selectable text",
    )
    st.caption("Text and questions are sent to OpenAI. Your vector index stays on this machine.")
    if st.button(
        "Index documents",
        type="primary",
        use_container_width=True,
        disabled=not uploads or store is None,
    ):
        assert store is not None
        started = perf_counter()
        with st.status("Reading, chunking and embedding…", expanded=True) as status:
            successes = 0
            for upload in uploads:
                try:
                    if upload.size > MAX_FILE_BYTES:
                        st.error(f"{upload.name}: Documents must be 20 MB or smaller.")
                        continue
                    result = store.ingest(upload.name, upload.getvalue())
                    st.write(
                        f"{result.source}: {result.chunks_added} new / {result.chunks_total} chunks"
                    )
                    successes += 1
                except Exception as error:
                    st.error(f"{upload.name}: {safe_error(error)}")
            status.update(
                label=f"Indexed {successes}/{len(uploads)} files in "
                f"{perf_counter() - started:.2f}s",
                state="complete" if successes == len(uploads) else "error",
            )
    if store is not None:
        sources = store.sources()
        st.caption(f"{len(sources)} documents · {store.count()} chunks")
        for source in sources:
            st.text(source)
    st.divider()
    mode = st.radio(
        "Retrieval mode",
        ["Agentic", "Single-shot baseline"],
        help="Baseline always searches once; agent plans and can repair evidence.",
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.answers = []
        st.rerun()
    st.caption(f"Model: {settings.chat_model}\n\nEmbeddings: {settings.embedding_model}")
    st.caption("Local demo · browser sessions use isolated indexes. No authentication.")

st.html('<div class="eyebrow">AGENTIC KNOWLEDGE WORKSPACE</div>')
st.title("Ask your documents. Trust the sources.")
st.html(
    '<p class="intro">Upload a policy, handbook or report. Groundwork decides what to '
    "retrieve, finds the evidence, and shows exactly where each answer comes from.</p>"
)

answers: list[Answer] = st.session_state.answers
columns = st.columns(4)
columns[0].metric("Indexed chunks", store.count() if store is not None else 0)
columns[1].metric("Questions answered", len(answers))
columns[2].metric("Last response", f"{answers[-1].latency_ms / 1000:.2f}s" if answers else "—")
columns[3].metric(
    "Average latency", f"{mean(a.latency_ms for a in answers) / 1000:.2f}s" if answers else "—"
)

if not settings.api_key:
    st.warning(
        "Set OPENAI_API_KEY or OPENAIAPIKEY in the environment, then restart the app. "
        "Uploads and chat are disabled until a key is configured."
    )
elif store is not None and not store.count():
    st.info(
        "Start by uploading documents in the library, then click Index documents. "
        "You can still say hello before indexing."
    )

messages: list[Turn] = st.session_state.messages
if not messages:
    st.html(
        '<div class="note"><strong>A good place to start</strong><br>'
        "“What is the refund window?” · “Compare these two policies”<br>"
        "Follow-up questions work too. If there is no evidence, the agent will say so.</div>"
    )
answer_index = 0
for message in messages:
    with st.chat_message(message.role):
        if message.role == "assistant":
            render_answer(answers[answer_index])
            answer_index += 1
        else:
            st.markdown(message.content)

question = st.chat_input(
    "Ask a question about your documents…", disabled=agent is None, max_chars=4000
)
if question and agent is not None:
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.spinner("Planning retrieval and checking evidence…"):
            answer = agent.ask(question, messages, baseline=mode == "Single-shot baseline")
        messages.extend(
            [Turn(role="user", content=question), Turn(role="assistant", content=answer.text)]
        )
        answers.append(answer)
        st.rerun()
    except Exception as error:
        st.error(safe_error(error))
