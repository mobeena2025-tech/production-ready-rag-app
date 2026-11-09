"""Streamlit chat UI. Run: streamlit run app.py"""

import streamlit as st

from rag.config import Settings
from rag.loaders import SUPPORTED_TYPES, UnsupportedFileError
from rag.pipeline import build_pipeline
from rag.providers import MissingAPIKeyError

st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def load_pipeline(api_key: str | None, provider: str):
    overrides = {"provider": provider}
    if api_key:
        overrides["google_api_key"] = api_key
    return build_pipeline(Settings(**overrides))


with st.sidebar:
    st.header("Settings")
    defaults = Settings()
    provider = st.radio(
        "Model provider",
        ["gemini", "local"],
        index=0 if defaults.provider == "gemini" else 1,
        help="'local' runs fully offline with simple keyword embeddings, for trying the app without a key.",
    )
    api_key = None
    if provider == "gemini" and not defaults.google_api_key:
        api_key = st.text_input(
            "Google Gemini API key", type="password", help="Free key: https://aistudio.google.com/apikey"
        )

try:
    rag = load_pipeline(api_key, provider)
except MissingAPIKeyError:
    st.title("📚 Document Q&A")
    st.info("Enter a Gemini API key in the sidebar, or switch the provider to 'local'.")
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader("Documents")
    files = st.file_uploader(
        "Add PDF, TXT or Markdown files",
        type=[t.lstrip(".") for t in SUPPORTED_TYPES],
        accept_multiple_files=True,
    )
    for f in files or []:
        key = f"ingested:{f.name}:{f.size}"
        if key in st.session_state:
            continue
        with st.spinner(f"Indexing {f.name}..."):
            try:
                r = rag.ingest(f.getvalue(), f.name)
                st.session_state[key] = True
                if r.skipped:
                    st.caption(f"{f.name} is already indexed.")
                elif r.chunks == 0:
                    st.warning(f"No text found in {f.name}.")
                else:
                    st.success(f"{f.name}: {r.pages} pages, {r.chunks} chunks")
            except UnsupportedFileError as e:
                st.error(str(e))

    for doc in rag.store.documents():
        c1, c2 = st.columns([4, 1])
        c1.caption(f"📄 {doc['source']} ({doc['chunks']} chunks)")
        if c2.button("✕", key=f"del-{doc['doc_id']}", help="Remove from index"):
            rag.store.delete_document(doc["doc_id"])
            st.rerun()

st.title("📚 Document Q&A")
model_name = rag.settings.chat_model if provider == "gemini" else "local (offline)"
st.caption(f"Answers come only from your documents, with sources. Model: {model_name}")

st.session_state.setdefault("messages", [])
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for c in m.get("citations", []):
            with st.expander(f"[{c.n}] {c.source}, page {c.page} (similarity {c.score})"):
                st.write(c.snippet + "…")

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not rag.store.documents():
            st.warning("Upload a document first.")
            st.stop()
        with st.spinner("Searching your documents..."):
            ans = rag.ask(question)
        st.markdown(ans.answer)
        for c in ans.citations:
            with st.expander(f"[{c.n}] {c.source}, page {c.page} (similarity {c.score})"):
                st.write(c.snippet + "…")
    st.session_state.messages.append({"role": "assistant", "content": ans.answer, "citations": ans.citations})
