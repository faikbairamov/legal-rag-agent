import glob
import hashlib
import json
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List, Dict

# Ensure local imports work when running from repo root
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from chunking import split_into_chunks, iter_chunks
from embeddings import Embeddings
from dotenv import load_dotenv


def _setup_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _ensure_index(pc, index_name: str, dimension: int) -> None:
    from pinecone import ServerlessSpec  # type: ignore

    # Create if missing; validate dims otherwise
    existing = pc.list_indexes()
    existing_names = [ix["name"] if isinstance(ix, dict) else getattr(ix, "name", None) for ix in existing]
    if index_name in existing_names:
        return
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=os.getenv("PINECONE_METRIC", "cosine"),
        spec=ServerlessSpec(cloud=cloud, region=region),
        deletion_protection=os.getenv("PINECONE_DELETION_PROTECTION", "disabled"),
    )


def _batch(iterable, size: int):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _embed_parallel(embed: Embeddings, chunks: List[Dict], workers: int) -> List[List[float]]:
    """Embed chunks concurrently, preserving order."""
    texts = [c["content"] for c in chunks]
    if workers <= 1:
        return embed.embed(texts, mode="document")
    vectors: List[List[float]] = [None] * len(texts)  # type: ignore
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(embed.embed, [txt], mode="document"): i for i, txt in enumerate(texts)}
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                vectors[idx] = fut.result()[0]
            except Exception as exc:
                logging.error("Embedding failed for chunk %d: %s", idx, exc)
                raise
    return vectors


def _start_upsert_worker(index, q: Queue) -> Thread:
    def _worker():
        while True:
            item = q.get()
            if item is None:
                break
            try:
                index.upsert(vectors=item)
            finally:
                q.task_done()

    t = Thread(target=_worker, daemon=True)
    t.start()
    return t


def build_index(
    data_glob: str = "data/processed/*.txt",
    provider: str = os.getenv("EMBEDDING_PROVIDER", "openai"),
    index_name: str = os.getenv("PINECONE_INDEX_NAME", "legal-rag-index"),
    batch_size: int = int(os.getenv("INDEX_BATCH_SIZE", "32")),
):
    load_dotenv()
    _setup_logging()
    from pinecone import Pinecone  # type: ignore

    # 1) Prepare embeddings
    t0 = time.perf_counter()
    embed = Embeddings(provider=provider)  # model via env
    logging.info(
        "Embeddings ready | provider=%s model=%s dim=%s",
        provider,
        getattr(embed, "model", ""),
        embed.dimension,
    )

    # 2) Prepare Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    _ensure_index(pc, index_name, dimension=embed.dimension)
    index = pc.Index(index_name)
    logging.info("Connected to Pinecone index '%s'", index_name)

    # 3) Load manifest for caching
    manifest_path = Path(os.getenv("INDEX_MANIFEST", "data/index_manifest.json"))
    manifest: Dict[str, Dict[str, str]] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
        except Exception:
            manifest = {}

    # 4) Collect documents
    files = sorted(glob.glob(data_glob))
    logging.info("Found %d files to index (glob=%s)", len(files), data_glob)

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if manifest.get(doc_id, {}).get("hash") == doc_hash:
            logging.info("%s unchanged; skipping", doc_id)
            continue
        # Stream chunks to avoid holding everything in memory
        target_tokens = int(os.getenv("CHUNK_TARGET_TOKENS", "300"))
        overlap_tokens = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))
        logging.info(
            "Chunking params | target_tokens=%d overlap_tokens=%d batch_size=%d",
            target_tokens,
            overlap_tokens,
            batch_size,
        )
        gen = iter_chunks(
            text,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            use_sections=True,
        )

        max_chunks_env = os.getenv("MAX_CHUNKS_PER_DOC")
        try:
            max_chunks = int(max_chunks_env) if max_chunks_env else None
        except ValueError:
            max_chunks = None

        produced = 0
        batch_idx = 0
        # Start upsert worker
        q: Queue = Queue(maxsize=2)
        worker = _start_upsert_worker(index, q)

        # 5) Embed and upsert in batches
        for batch in _batch(gen, batch_size):
            if max_chunks is not None and produced >= max_chunks:
                break
            if max_chunks is not None and produced + len(batch) > max_chunks:
                batch = batch[: max_chunks - produced]
            produced += len(batch)
            batch_idx += 1
            logging.info("Embedding batch %d | size=%d", batch_idx, len(batch))
            t_embed = time.perf_counter()
            vectors = _embed_parallel(embed, batch, int(os.getenv("EMBED_CONCURRENCY", "4")))
            logging.info("Embed done in %.2fs", time.perf_counter() - t_embed)
            payloads: List[Dict] = []
            for i, (chunk, vec) in enumerate(zip(batch, vectors)):
                uid = f"{doc_id}-{chunk['start']}-{chunk['end']}"
                metadata = {
                    "doc_id": doc_id,
                    "source": file_path,
                    "section_title": chunk.get("section_title", ""),
                    "article": chunk.get("article", ""),
                    "start": chunk.get("start", 0),
                    "end": chunk.get("end", 0),
                    "text": chunk["content"],
                }
                payloads.append({"id": uid, "values": vec, "metadata": metadata})

            q.put(payloads)
        q.put(None)
        worker.join()
        logging.info("%s: indexed %d chunks in %.2fs", doc_id, produced, time.perf_counter() - t0)
        manifest[doc_id] = {"hash": doc_hash}
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)
    logging.info("Indexing complete.")


if __name__ == "__main__":
    build_index()
