# legal-rag-agent

Georgian-language RAG agent that answers civil-law questions strictly from the Civil Code of Georgia with precise article/paragraph citations.

## Indexing Pipeline

- Chunking: Splits processed text into logical sections by article headers ("მუხლი N.") and then into overlapping chunks sized for embeddings.
- Embeddings: Configurable between OpenAI (`text-embedding-3-small` by default) and local sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`).
- Vector DB: Stores chunks in a Pinecone serverless index with rich metadata.

### Setup

- Create and populate `.env` (see `.env.example`). Required:
  - `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`
  - `OPENAI_API_KEY` (if using OpenAI embeddings)
- Install deps: `pip install -r requirements.txt`

### Prepare Text

- Extract and clean PDF text into `data/processed/*.txt` using `src/extract_pdf.py` (edit paths inside as needed).

### Build Index

- Run: `python src/build_index.py`
- This will:
  - Read `data/processed/*.txt`
  - Chunk the documents
  - Create Pinecone index if missing (dims auto-match embedding model)
  - Upsert vectors with metadata (`doc_id`, `article`, `section_title`, `start`, `end`, `text`)

### Query

- Use `src/retriever.py` programmatically:
  ```python
  from src.retriever import Retriever
  r = Retriever(index_name="legal-rag-index")
  hits = r.query("რა წერია მუხლი 12-ზე?", top_k=5)
  for h in hits:
      print(h["score"], h["metadata"].get("article"), h["text"][:120])
  ```

### Notes

- The chunker recognizes Georgian article headers and tries to end chunks on natural boundaries. Adjust `target_tokens`/`overlap_tokens` in `src/chunking.py` if needed.
- To run without OpenAI, set `EMBEDDING_PROVIDER=sentence-transformers` in `.env`.

## Gemini RAG + Chat UI

- LLM: Google Gemini (`GEMINI_MODEL`, default `gemini-1.5-flash`)
- Embeddings: `intfloat/multilingual-e5-large-instruct` via sentence-transformers with E5-style prefixes.
- Vector DB: Pinecone
- UI: Streamlit chat

### Setup

- Fill `.env`:
  - `GEMINI_API_KEY`
  - `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`
  - `EMBEDDING_PROVIDER=sentence-transformers`
  - `ST_EMBED_MODEL=intfloat/multilingual-e5-large-instruct`

### Run

- Ensure you've indexed your `data/processed/*.txt` with `python src/build_index.py`.
- Start the chat UI: `streamlit run src/app.py`
- Ask questions in Georgian; the app retrieves from Pinecone and composes an answer with Gemini, showing sources.
