"""Streamlit UI for the College Knowledge Assistant."""
import os
import streamlit as st
from rag_engine import (KnowledgeBase, answer, DOCS_DIR, INDEX_PATH,
                        TOP_K, MIN_SCORE)

st.set_page_config(page_title="College Knowledge Assistant", page_icon="🎓")
st.title("🎓 College Knowledge Assistant")
st.caption("Answers come only from the official college documents, with sources.")


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_kb(force_rebuild=False):
    kb = KnowledgeBase()
    if force_rebuild or not os.path.exists(INDEX_PATH):
        kb.build(DOCS_DIR)
    else:
        kb.load()
    return kb


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Add PDF / TXT files", type=["pdf", "txt", "md"],
                               accept_multiple_files=True)
    if st.button("Save uploads and rebuild index"):
        os.makedirs(DOCS_DIR, exist_ok=True)
        for f in uploads or []:
            with open(os.path.join(DOCS_DIR, f.name), "wb") as out:
                out.write(f.getbuffer())
        get_kb.clear()
        get_kb(force_rebuild=True)
        st.success("Index rebuilt.")

    st.write("**Indexed files:**")
    for name in sorted(os.listdir(DOCS_DIR)):
        st.write("•", name)

    st.header("Settings")
    top_k = st.slider("Chunks retrieved (top-k)", 1, 8, TOP_K)
    min_score = st.slider("Minimum similarity", 0.0, 0.8, MIN_SCORE, 0.05)
    use_llm = st.checkbox("Use LLM (Ollama gemma:2b)", value=True)

kb = get_kb()

# ---------- Chat ----------
if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

question = st.chat_input("Ask about fees, attendance, hostel, library, placements...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            res = answer(question, kb, top_k, min_score, use_llm)
        st.write(res["answer"])
        if res["found"]:
            with st.expander(f"Sources ({len(res['sources'])})"):
                for i, s in enumerate(res["sources"], 1):
                    st.markdown(f"**[{i}] {s['source']}** - page {s['page']} - "
                                f"last updated: {s['date']} - similarity {s['score']:.2f}")
                    st.caption(s["text"])
        else:
            st.info("No supporting source found, so no answer was generated.")
    st.session_state.history.append({"role": "assistant", "content": res["answer"]})
