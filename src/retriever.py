import os
import sys
from pathlib import Path
from typing import List, Dict

# Ensure local imports work when running from repo root
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from embeddings import Embeddings
from dotenv import load_dotenv


class Retriever:
    def __init__(self, index_name: str):
        from pinecone import Pinecone  # type: ignore

        load_dotenv()
        self.index_name = index_name
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)
        self.embed = Embeddings(provider=os.getenv("EMBEDDING_PROVIDER", "openai"))

    def query(self, q: str, top_k: int = 8) -> List[Dict]:
        vec = self.embed.embed([q], mode="query")[0]
        res = self.index.query(vector=vec, top_k=top_k, include_metadata=True)
        out: List[Dict] = []
        for m in getattr(res, "matches", []) or []:
            out.append(
                {
                    "id": m.id,
                    "score": m.score,
                    "text": m.metadata.get("text", "") if m.metadata else "",
                    "metadata": dict(m.metadata) if m.metadata else {},
                }
            )
        return out
