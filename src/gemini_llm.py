import os
from typing import List, Dict

import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()


def _build_prompt(question: str, contexts: List[Dict]) -> str:
    """Build a single text prompt for Gemini including instructions and contexts."""
    system = (
        "You are a legal assistant answering strictly from provided Georgian Civil Code excerpts. "
        "Answer concisely in Georgian. Cite relevant articles (e.g., 'მუხლი 12') in-line where appropriate. "
        "If unsure or the answer is not in the context, say so and do not fabricate."
    )

    context_header = "ქვემოთ მოცემულია შესაბამისი ამონაწერები (წყარო/მუხლი):\n\n"
    context_blocks = []
    for i, c in enumerate(contexts, 1):
        title = c.get("section_title") or ""
        article = c.get("article") or ""
        src = c.get("source") or ""
        snippet = c.get("text") or ""
        label = f"[{i}] წყარო: {os.path.basename(src)} | მუხლი: {article} | {title}".strip()
        block = f"{label}\n{snippet}\n"
        context_blocks.append(block)

    context_text = context_header + "\n\n".join(context_blocks)

    user = (
        f"კითხვა: {question}\n\n"
        "პასუხი უნდა ემყარებოდეს მხოლოდ კონტექსტს. მიუთითე მუხლები, როცა საჭიროა."
    )

    return system + "\n\n" + context_text + "\n\n" + user


class GeminiLLM:
    def __init__(self, model_name: str | None = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.model = genai.GenerativeModel(self.model_name)

    def answer(self, question: str, contexts: List[Dict]) -> str:
        prompt = _build_prompt(question, contexts)
        resp = self.model.generate_content(prompt)
        return getattr(resp, "text", "") or ""
