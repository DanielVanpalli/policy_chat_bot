"""
Hybrid retrieval (BM25 + dense FAISS) + cross-encoder reranking.

Loads a pre-built FAISS index from disk on first use; the same documents
feed an in-memory BM25 index. An ensemble retriever blends keyword and
semantic scores; a cross-encoder then reranks the merged candidates.

The FAISS index is expected to live in settings.FAISS_INDEX_DIR.
"""
import asyncio
import pybreaker

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

from graph.state import SupportBotState
from observability.logging import get_logger
from resilience.breakers import pageindex_breaker
from config import settings


_EMBED_MODEL = "all-MiniLM-L6-v2"
_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_BM25_K = 4
_SEMANTIC_K = 4
_RERANK_TOP_K = 4
_ENSEMBLE_WEIGHTS = [0.4, 0.6]

_ensemble_retriever: EnsembleRetriever | None = None
_cross_encoder: CrossEncoder | None = None


def _build_retriever() -> EnsembleRetriever:
    embeddings = HuggingFaceEmbeddings(model_name=_EMBED_MODEL)
    vectorstore = FAISS.load_local(
        settings.FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    all_docs = list(vectorstore.docstore._dict.values())

    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = _BM25_K
    semantic = vectorstore.as_retriever(search_kwargs={"k": _SEMANTIC_K})

    return EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=_ENSEMBLE_WEIGHTS,
    )


def _get_retriever() -> EnsembleRetriever:
    global _ensemble_retriever
    if _ensemble_retriever is None:
        _ensemble_retriever = _build_retriever()
    return _ensemble_retriever


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(_RERANK_MODEL)
    return _cross_encoder


def _retrieve_and_rerank_sync(query: str) -> list[str]:
    retriever = _get_retriever()
    candidates = retriever.invoke(query)
    if not candidates:
        return []

    encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = encoder.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc.page_content for _, doc in ranked[:_RERANK_TOP_K]]


async def context_retrieval_node(state: SupportBotState) -> dict:
    log = get_logger(state["request_id"], node="context_retrieval")

    try:
        with pageindex_breaker:
            loop = asyncio.get_event_loop()
            context = await loop.run_in_executor(
                None, _retrieve_and_rerank_sync, state["scrubbed_query"]
            )
            log.info("retrieval_complete", num_chunks=len(context))
            return {"retrieved_context": context}

    except pybreaker.CircuitBreakerError:
        log.warning("retriever_circuit_open")
        return {"retrieved_context": []}
    except Exception as exc:
        log.warning("retrieval_failed", error=str(exc))
        return {"retrieved_context": []}
