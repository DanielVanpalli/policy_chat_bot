"""
Stage 3: chunks.json -> FAISS index (HNSW).

Reads the layout-aware chunks produced by ``ingestion/chunk_doc.py``,
embeds each chunk's content with a HuggingFace sentence-transformer,
and writes an HNSW-indexed FAISS store that the retriever loads at
serve time.

HNSW params:
  M = 32                — neighbors per graph node. Higher = more recall,
                          larger index. 32 is a solid default.
  efConstruction = 200  — build-time exploration width. Higher = better
                          graph quality, slower build.
  efSearch = 64         — query-time exploration width. Higher = better
                          recall, slower query. Tunable after load.

Usage:
    python -m ingestion.build_index
    python -m ingestion.build_index --in dataset/chunks.json --out faiss_index
"""
import argparse
import json
from pathlib import Path

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# Fixed paths — scripts resolve relative to themselves so `cd` doesn't matter.
_PKG_DIR = Path(__file__).resolve().parent              # .../policy_bot_rag/ingestion
_PROJECT_ROOT = _PKG_DIR.parent                         # .../policy_bot_rag
CHUNKS_JSON = _PKG_DIR / "dataset" / "chunks.json"
FAISS_DIR = _PROJECT_ROOT / "faiss_index"

EMBED_MODEL = "all-MiniLM-L6-v2"

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64


def build(chunks_path: Path, index_dir: Path) -> None:
    if not chunks_path.exists():
        raise SystemExit(f"Chunks not found: {chunks_path} — run chunk_doc first")

    raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(raw)} chunks from {chunks_path}")

    docs = [
        Document(
            page_content=c["content"],
            metadata={"chunk_id": c["chunk_id"], **c["metadata"]},
        )
        for c in raw
    ]

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # Infer embedding dimension from the model rather than hardcoding it —
    # so swapping to bge-small / bge-large just works.
    dim = len(embeddings.embed_query("dimension probe"))
    print(f"Embedding model: {EMBED_MODEL} (dim={dim})")

    index = faiss.IndexHNSWFlat(dim, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    print(f"FAISS index: HNSW (M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}, efSearch={HNSW_EF_SEARCH})")

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vectorstore.add_documents(docs)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"Saved FAISS index -> {index_dir}/  (index type: {type(vectorstore.index).__name__})")


def main() -> None:
    argparse.ArgumentParser(description="Stage 3 — embed chunks and build the HNSW FAISS index.").parse_args()
    build(CHUNKS_JSON, FAISS_DIR)


if __name__ == "__main__":
    main()
