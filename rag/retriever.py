"""
retriever.py
Tạo LangChain Retriever từ vector_store.get_vector_store() - đây là thứ mà
rag.py gọi bằng `retriever.invoke(query)` thay vì tự viết code gọi thẳng
`QdrantClient.query()` như trước.

Ngưỡng lọc "chỉ giữ chunk đủ liên quan" (trước đây là SCORE_THRESHOLD lọc
thủ công bằng tay trong rag.py, so sánh r.score >= SCORE_THRESHOLD) giờ
được cấu hình NGAY TRONG retriever thông qua
search_type="similarity_score_threshold" - đây là cách làm chuẩn của
LangChain. Nhờ vậy rag.py không cần tự tính/so sánh điểm số nữa, retriever
tự lo phần đó và chỉ trả về những chunk đã đạt ngưỡng.

Giá trị TOP_K và SCORE_THRESHOLD GIỮ NGUYÊN 100% so với trước khi refactor.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.retrievers import BaseRetriever

from .vector_store import get_vector_store

TOP_K = 5
# Điểm similarity tối thiểu để 1 chunk được coi là "liên quan". Chunk có
# điểm thấp hơn ngưỡng này sẽ bị loại để tránh nhồi context không liên quan
# vào prompt (giảm nguy cơ model trả lời sai / bịa). Giá trị không đổi so
# với bản Qdrant thuần trước đây.
SCORE_THRESHOLD = 0.4


@lru_cache(maxsize=8)
def get_retriever(top_k: int = TOP_K, score_threshold: float = SCORE_THRESHOLD) -> BaseRetriever:
    """Trả về 1 Retriever sẵn sàng dùng qua `.invoke(query)`.

    Cache theo (top_k, score_threshold) vì trong project hiện chỉ dùng đúng
    1 bộ tham số mặc định ở mọi nơi gọi - tránh tạo lại object retriever ở
    mỗi lượt hỏi mà không có lý do."""
    return get_vector_store().as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": score_threshold},
    )
