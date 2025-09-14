import os
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv

from rag import RAGPipeline


load_dotenv()


st.set_page_config(page_title="Legal RAG (Georgian)", layout="wide")
st.title("Legal RAG • Georgian Civil Code")


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(index_name=os.getenv("PINECONE_INDEX_NAME", "legal-rag-index"))


with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K passages", 1, 20, 6)
    st.caption("Embeddings: intfloat/multilingual-e5-large-instruct; LLM: Gemini")
    st.divider()
    st.markdown("- Index: `" + os.getenv("PINECONE_INDEX_NAME", "legal-rag-index") + "`")
    st.markdown("- Provider: `" + os.getenv("EMBEDDING_PROVIDER", "sentence-transformers") + "`")
    st.markdown("- Gemini model: `" + os.getenv("GEMINI_MODEL", "gemini-1.5-flash") + "`")


if "messages" not in st.session_state:
    st.session_state.messages = []


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(m["sources"], 1):
                    meta = f"[{i}] {os.path.basename(s.get('source',''))} | მუხლი: {s.get('article','')}"
                    st.markdown(f"**{meta}**\n\n{(s.get('text') or '')[:600]}")


prompt = st.chat_input("დასვი კითხვა სამოქალაქო კოდექსზე…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("ძებნა და პასუხის აგება…"):
            try:
                rag = get_pipeline()
                answer, contexts = rag.ask(prompt, top_k=top_k)
            except Exception as e:
                st.error(f"შეცდომა: {e}")
                st.stop()

        st.markdown(answer or "")
        if contexts:
            with st.expander("Sources"):
                for i, s in enumerate(contexts, 1):
                    meta = f"[{i}] {os.path.basename(s.get('source',''))} | მუხლი: {s.get('article','')}"
                    st.markdown(f"**{meta}**\n\n{(s.get('text') or '')[:600]}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer or "", "sources": contexts}
        )
