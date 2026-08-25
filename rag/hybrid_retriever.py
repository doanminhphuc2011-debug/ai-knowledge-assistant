"""
hybrid_retriever.py
Pipeline: Query -> Dense + BM25 Sparse -> RRF Fusion -> Cross-Encoder Reranker -> Top-K.
- Dense + BM25: Tạo ứng viên rộng (kết hợp ngữ nghĩa và từ khóa chính xác).
- RRF (Reciprocal Rank Fusion): Hợp nhất thứ hạng mà không bị lệch thang điểm thô.
- Cross-Encoder: Rerank chuyên sâu cặp (query, document) trước khi trả kết quả Top-K.
- Giữ nguyên public API qua hàm get_hybrid_retriever().
"""
from __future__ import annotations
import os
from functools import lru_cache
from typing import Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .retriever import SCORE_THRESHOLD, TOP_K
from .reranker import get_reranker
from .sparse_retriever import bm25_search
from .vector_store import get_vector_store

# Candidate pool trước khi fusion/reranking.
# Corpus hiện tại chỉ khoảng 70-90 chunks nên 100 là đủ để lấy gần toàn bộ
# corpus từ mỗi retriever mà không làm thay đổi public TOP_K.
DENSE_POOL_K = int(os.getenv("RAG_DENSE_POOL_K", "100"))
SPARSE_POOL_K = int(os.getenv("RAG_SPARSE_POOL_K", "100"))

# Khi BM25 không có bằng chứng lexical đủ mạnh, dùng ngưỡng dense chặt hơn
# để giảm false positive ngoài phạm vi knowledge base.
STRICT_SCORE_THRESHOLD = float(os.getenv("RAG_STRICT_SCORE_THRESHOLD", "0.55"))
MIN_BM25_SCORE_FOR_SUPPORT = float(os.getenv("RAG_MIN_BM25_SCORE_FOR_SUPPORT", "1.0"))

# Hằng số chuẩn của Reciprocal Rank Fusion.
RRF_K = int(os.getenv("RAG_RRF_K", "60"))

def _doc_key(doc: Document) -> tuple[str, str]:
    return (doc.metadata.get("source", ""), doc.page_content)

class HybridRetriever(BaseRetriever):
    """Dense + BM25 + RRF + Cross-Encoder reranking."""

    k: int = TOP_K
    score_threshold: float = SCORE_THRESHOLD
    strict_score_threshold: float = STRICT_SCORE_THRESHOLD
    dense_pool_k: int = DENSE_POOL_K
    sparse_pool_k: int = SPARSE_POOL_K

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        vector_store = get_vector_store()

        # 1. Candidate generation
        dense_hits = vector_store.similarity_search_with_score(query, k=self.dense_pool_k)
        sparse_hits = bm25_search(query, k=self.sparse_pool_k)

        # 2. Dense threshold có điều kiện theo sparse corroboration
        has_sparse_support = (bool(sparse_hits) and sparse_hits[0][1] >= MIN_BM25_SCORE_FOR_SUPPORT)
        effective_dense_threshold = (self.score_threshold if has_sparse_support else self.strict_score_threshold)

        dense_filtered = [(doc, score) for doc, score in dense_hits if score >= effective_dense_threshold]

        # 3. RRF Fusion trên rank, không cộng raw score
        fused_scores: dict[tuple[str, str], float] = {}
        doc_by_key: dict[tuple[str, str], Document] = {}

        for rank, (doc, _score) in enumerate(dense_filtered, start=1):
            key = _doc_key(doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (RRF_K + rank))
            doc_by_key[key] = doc

        for rank, (doc, _score) in enumerate(sparse_hits, start=1):
            key = _doc_key(doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (RRF_K + rank))
            doc_by_key[key] = doc

        ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)

        candidates = [doc_by_key[key] for key in ranked_keys]

        if not candidates:
            return []

        # 4. Cross-Encoder reranking
        # Chỉ sau khi đã fusion mới dùng model cross-encoder, vì bước này
        # đắt hơn Dense/BM25 và cần đọc cả query + document.
        reranker = get_reranker()
        return reranker.rerank(query, candidates, top_k=self.k)

@lru_cache(maxsize=8)
def get_hybrid_retriever(top_k: int = TOP_K, score_threshold: float = SCORE_THRESHOLD) -> BaseRetriever:
    """Giữ nguyên interface cũ của HybridRetriever."""
    return HybridRetriever(
        k=top_k,
        score_threshold=score_threshold,
        strict_score_threshold=STRICT_SCORE_THRESHOLD,
        dense_pool_k=DENSE_POOL_K,
        sparse_pool_k=SPARSE_POOL_K,
    )
