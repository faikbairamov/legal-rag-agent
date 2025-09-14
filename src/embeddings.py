import os
import logging
from typing import List, Literal, Optional

import numpy as np
from dotenv import load_dotenv


load_dotenv()


class Embeddings:
    def __init__(
        self,
        provider: Literal["openai", "sentence-transformers"] = "openai",
        model: Optional[str] = None,
    ) -> None:
        self.provider = provider
        if provider == "openai":
            from openai import OpenAI  # type: ignore

            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
            self._dimension = 1536 if "3-small" in self.model else 3072
        else:
            # sentence-transformers local model (multilingual; supports E5 family)
            from sentence_transformers import SentenceTransformer  # type: ignore
            try:
                import torch  # type: ignore
            except Exception:
                torch = None  # type: ignore

            self.model = model or os.getenv(
                "ST_EMBED_MODEL", "intfloat/multilingual-e5-large-instruct"
            )
            # Device selection and batch size for memory control
            env_device = os.getenv("ST_EMBED_DEVICE")
            if not env_device:
                if torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    env_device = "mps"
                elif torch and torch.cuda.is_available():
                    env_device = "cuda"
                else:
                    env_device = "cpu"
            self.device = env_device
            self.batch_size = int(os.getenv("ST_EMBED_BATCH_SIZE", "8"))

            self.client = SentenceTransformer(self.model, device=self.device)
            # Infer dimension dynamically
            try:
                vec = self.client.encode(
                    ["dim-probe"], normalize_embeddings=False, batch_size=self.batch_size
                )
                self._dimension = int(np.array(vec).shape[-1])
            except Exception:
                # Sensible fallback for many multilingual models
                self._dimension = 1024
            logging.info(
                "Embeddings init | provider=sentence-transformers model=%s device=%s batch_size=%s dim=%s",
                self.model,
                self.device,
                self.batch_size,
                self._dimension,
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    def _prefix_for_mode(self, mode: Literal["document", "query"]) -> str:
        if self.provider == "sentence-transformers":
            # E5-style prefixes; work well across multilingual E5 variants
            return "passage: " if mode == "document" else "query: "
        return ""

    def embed(
        self, texts: List[str], mode: Literal["document", "query"] = "document"
    ) -> List[List[float]]:
        if self.provider == "openai":
            out = self.client.embeddings.create(model=self.model, input=texts)  # type: ignore
            return [d.embedding for d in out.data]
        else:
            prefix = self._prefix_for_mode(mode)
            inputs = [prefix + t for t in texts]
            vecs = self.client.encode(
                inputs,
                normalize_embeddings=False,
                batch_size=getattr(self, "batch_size", 8),
                show_progress_bar=False,
            )  # type: ignore
            return np.asarray(vecs, dtype=float).tolist()
