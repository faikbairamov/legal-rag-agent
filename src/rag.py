import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from retriever import Retriever
from gemini_llm import GeminiLLM


load_dotenv()


class RAGPipeline:
    def __init__(self, index_name: str | None = None):
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "legal-rag-index")
        self.retriever = Retriever(index_name=self.index_name)
        self.llm = GeminiLLM()

    def _prepare_contexts(self, hits: List[Dict]) -> List[Dict]:
        contexts: List[Dict] = []
        for h in hits:
            md = h.get("metadata", {})
            contexts.append(
                {
                    "text": h.get("text", ""),
                    "article": md.get("article", ""),
                    "section_title": md.get("section_title", ""),
                    "source": md.get("source", ""),
                    "score": h.get("score", 0.0),
                }
            )
        return contexts

    def ask(self, question: str, top_k: int = 6) -> Tuple[str, List[Dict]]:
        hits = self.retriever.query(question, top_k=top_k)
        contexts = self._prepare_contexts(hits)
        answer = self.llm.answer(question, contexts)
        return answer, contexts

