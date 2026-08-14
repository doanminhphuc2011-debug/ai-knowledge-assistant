"""
hybrid_retriever.py
Giai đoạn 1 của pipeline retrieval hiện đại:

    Query -> [Dense Embedding & BM25 Sparse] -> Fusion -> Top-K

(CHƯA có Cross-Encoder Reranker - xem giải thích lý do trong PHẦN CUỐI file
này. Nếu sau khi đo bằng eval_retrieval.py vẫn còn thiếu, thêm reranker sau
như 1 bước riêng, không phải bây giờ.)

KHÔNG sửa retriever.py (giữ nguyên _ThresholdRetriever/get_retriever() cũ,
dùng nếu cần fallback/so sánh), KHÔNG sửa vector_store.py/ingest.py. File
này CHỈ đọc (get_vector_store(), bm25_search()), không ghi gì thêm.

VẤN ĐỀ THỰC TẾ CẦN GIẢI (đo được từ retrieval_eval_dense_baseline.csv):
Dense-only đang có False Positive Rate 100% trên câu hỏi ngoài phạm vi
(vd. "Quán có bán bia hoặc rượu không?") - vì các chunk FAQ/promotion CÙNG
CHỦ ĐỀ (quán cà phê, đồ uống, khuyến mãi) vẫn đạt raw cosine score >=
SCORE_THRESHOLD dù nội dung không hề nhắc "bia"/"rượu"/"sinh viên". Dense
embedding nắm được sự liên quan VỀ CHỦ ĐỀ nhưng không phân biệt được
"cùng chủ đề" với "đúng nội dung được hỏi".

CÁCH GIẢI: dùng BM25 làm tín hiệu CORROBORATION (xác nhận chéo), không chỉ
để tăng hạng:
1. Lấy pool ứng viên rộng hơn từ CẢ 2 nguồn (DENSE_POOL_K, SPARSE_POOL_K).
2. Nếu BM25 KHÔNG tìm được bất kỳ từ khóa nào khớp trong TOÀN CORPUS cho
   câu hỏi này (max điểm BM25 = 0) - đây là dấu hiệu mạnh cho thấy câu hỏi
   có thể nằm ngoài phạm vi dữ liệu - áp dụng NGƯỠNG DENSE CAO HƠN
   (STRICT_SCORE_THRESHOLD) để chỉ giữ lại match THỰC SỰ rất gần nghĩa,
   loại các match chỉ "chung chủ đề". Ngược lại, dùng ngưỡng gốc
   SCORE_THRESHOLD như cũ (không đổi hành vi khi BM25 đã có corroboration).
3. RRF (Reciprocal Rank Fusion) trên phần đã qua ngưỡng ở bước 2, để các
   chunk được CẢ 2 phương pháp cùng chọn được xếp hạng cao hơn.

CÁCH GIẢI THAY THẾ đã cân nhắc và bị loại: đơn giản nâng SCORE_THRESHOLD
cho MỌI câu hỏi. Bị loại vì true positive trong baseline cũng có
context_precision thấp (0.2-0.6, vd. "Thêm topping trân châu đen") ở vùng
điểm tương đương false positive - nâng ngưỡng toàn cục rủi ro cắt luôn
true positive mà không chắc lọc được false positive (xem phân tích đã
thảo luận). Ngưỡng ĐIỀU KIỆN theo corroboration của BM25 (chỉ ở đây) tránh
được đánh đổi đó.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .retriever import SCORE_THRESHOLD, TOP_K
from .sparse_retriever import bm25_search
from .vector_store import get_vector_store

# Số ứng viên lấy TRƯỚC KHI lọc/fusion, rộng hơn TOP_K để fusion có gì đó
# để chọn lọc. Với corpus ~70-90 chunk, 15-20 là đủ rộng (không cần Top-100
# như pipeline cho corpus lớn - xem giải thích đã trao đổi).
DENSE_POOL_K = 15
SPARSE_POOL_K = 15

# Ngưỡng dense ÁP DỤNG KHI BM25 KHÔNG tìm được corroboration ĐỦ MẠNH cho
# câu hỏi (xem MIN_BM25_SCORE_FOR_SUPPORT ngay dưới) - cao hơn hẳn
# SCORE_THRESHOLD gốc (0.4) để chỉ giữ match near-paraphrase thực sự, loại
# match "chung chủ đề". Giá trị này CẦN HIỆU CHỈNH bằng eval_retrieval.py -
# 0.55 là điểm khởi đầu hợp lý (cao hơn 0.4 một khoảng đáng kể), không phải
# số đã kiểm chứng cuối cùng.
STRICT_SCORE_THRESHOLD = 0.55

# Điểm BM25 TỐI THIỂU để coi là "có corroboration thật" - PHÁT HIỆN QUA
# TEST THỰC TẾ: chỉ kiểm tra "có sparse_hits hay không" (> 0) là CHƯA ĐỦ.
# Vd. câu "Quán có giảm giá riêng cho sinh viên không?" (đáng lẽ phải bị
# coi là KHÔNG có corroboration) vẫn khớp điểm BM25 thấp (~0.6-1.8) với
# các chunk promotion/menu khác chỉ vì có từ chung chung "giảm"/"giá" -
# không phải vì "sinh viên" thực sự xuất hiện trong corpus. Yêu cầu điểm
# BM25 cao nhất phải vượt ngưỡng này mới tính là corroboration thật, tránh
# bị "lừa" bởi match yếu/generic. CẦN HIỆU CHỈNH bằng eval_retrieval.py
# tương tự STRICT_SCORE_THRESHOLD - 1.0 là điểm khởi đầu (BM25Okapi mặc
# định thường cho match 1 từ nội dung rõ ràng ở khoảng 1.5-3+, xem log test
# "Cà phê muối" = 6.575, "bãi đậu xe" = 5.628 - match generic 1 từ yếu như
# "giảm"/"giá" chỉ quanh 0.5-1.8).
MIN_BM25_SCORE_FOR_SUPPORT = 1.0

# Hằng số k trong công thức RRF (1 / (RRF_K + rank)) - dùng giá trị chuẩn
# phổ biến trong literature (Cormack et al., 2009), không cần tự tune.
RRF_K = 60


def _doc_key(doc: Document) -> tuple[str, str]:
    """Định danh 1 Document để so khớp giữa danh sách dense và sparse - vì
    cả 2 đều build từ CÙNG corpus (_build_corpus() trong sparse_retriever.py
    dùng đúng các hàm chunk_*() mà ingest.py dùng để nạp vào Redis), nên
    (source, page_content) là đủ để định danh duy nhất 1 chunk mà không cần
    thêm ID số riêng."""
    return (doc.metadata.get("source", ""), doc.page_content)


class HybridRetriever(BaseRetriever):
    """Dense (Redis/cosine) + Sparse (BM25) + RRF Fusion, có ngưỡng dense
    điều kiện theo corroboration của BM25 (xem giải thích đầu file)."""

    k: int = TOP_K
    score_threshold: float = SCORE_THRESHOLD
    strict_score_threshold: float = STRICT_SCORE_THRESHOLD
    dense_pool_k: int = DENSE_POOL_K
    sparse_pool_k: int = SPARSE_POOL_K

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        vector_store = get_vector_store()
        dense_hits = vector_store.similarity_search_with_score(query, k=self.dense_pool_k)
        sparse_hits = bm25_search(query, k=self.sparse_pool_k)

        has_sparse_support = bool(sparse_hits) and sparse_hits[0][1] >= MIN_BM25_SCORE_FOR_SUPPORT
        effective_dense_threshold = (
            self.score_threshold if has_sparse_support else self.strict_score_threshold
        )

        dense_filtered = [(doc, score) for doc, score in dense_hits if score >= effective_dense_threshold]

        # RRF fusion trên RANK (không phải raw score - dense/BM25 không
        # cùng thang đo, cộng thẳng raw score sẽ bị lệch bởi phương pháp có
        # biên độ lớn hơn). Không xuất hiện trong 1 danh sách -> không cộng
        # gì cho danh sách đó (không giả định rank cuối bảng).
        fused_scores: dict[tuple[str, str], float] = {}
        doc_by_key: dict[tuple[str, str], Document] = {}

        for rank, (doc, _score) in enumerate(dense_filtered, start=1):
            key = _doc_key(doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            doc_by_key[key] = doc

        for rank, (doc, _score) in enumerate(sparse_hits, start=1):
            key = _doc_key(doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            doc_by_key[key] = doc

        ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)
        return [doc_by_key[key] for key in ranked_keys[: self.k]]


@lru_cache(maxsize=8)
def get_hybrid_retriever(top_k: int = TOP_K, score_threshold: float = SCORE_THRESHOLD) -> BaseRetriever:
    """Trả về 1 HybridRetriever sẵn sàng dùng qua `.invoke(query)` - cùng
    interface với get_retriever() cũ trong retriever.py, để rag.py chỉ cần
    đổi 1 chỗ import/gọi hàm, không đổi gì khác."""
    return HybridRetriever(
        k=top_k,
        score_threshold=score_threshold,
        strict_score_threshold=STRICT_SCORE_THRESHOLD,
        dense_pool_k=DENSE_POOL_K,
        sparse_pool_k=SPARSE_POOL_K,
    )
