"""
retriever.py
Tạo LangChain Retriever từ vector_store.get_vector_store() - đây là thứ mà
rag.py gọi bằng `retriever.invoke(query)` thay vì tự viết code gọi thẳng
`QdrantClient.query()` như trước.

 BUG ĐÃ SỬA 
Bản trước dùng `vector_store.as_retriever(search_type="similarity_score_threshold",
search_kwargs={"score_threshold": SCORE_THRESHOLD})`. Cách này để LangChain tự lọc
theo ngưỡng, nhưng LangChain KHÔNG so ngưỡng trực tiếp trên raw score của Qdrant -
nó gọi `_select_relevance_score_fn()`, với QdrantVectorStore (distance=COSINE) là:

    relevance_score = (raw_score + 1.0) / 2.0

rồi mới so `relevance_score >= score_threshold`. SCORE_THRESHOLD = 0.4 của project
được tune cho RAW SCORE (y hệt cách rag.py bản Qdrant thuần so sánh trước đây:
`r.score >= SCORE_THRESHOLD`), KHÔNG phải cho relevance_score đã biến đổi. Vì
`(raw + 1) / 2` luôn lớn hơn raw khá nhiều với các giá trị dương thường gặp, ngưỡng
0.4 trở nên gần như vô tác dụng (retriever trả về hầu hết mọi thứ, kể cả chunk
không liên quan) -> đây chính là nguyên nhân evaluate.py cho kết quả kém/sai sau khi
migrate sang LangChain (đặc biệt các câu hỏi "expected_source": "none" luôn bị lấy
nhầm context, và context_precision giảm mạnh).

CÁCH SỬA:
Tự viết 1 Retriever nhỏ (`_ThresholdRetriever`), lọc thẳng trên RAW SCORE trả về từ
`vector_store.similarity_search_with_score()` - giống 100% cách so sánh cũ, không đi
qua bất kỳ hàm chuẩn hoá relevance-score nào của LangChain. Vẫn giữ được API
`retriever.invoke(query)` như yêu cầu kiến trúc trước đó, chỉ khác cách tính điểm
bên trong.

Giá trị TOP_K và SCORE_THRESHOLD GIỮ NGUYÊN 100% so với trước khi refactor.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from .vector_store import get_vector_store

TOP_K = 5
# Điểm similarity tối thiểu để 1 chunk được coi là "liên quan". Chunk có
# điểm thấp hơn ngưỡng này sẽ bị loại để tránh nhồi context không liên quan
# vào prompt (giảm nguy cơ model trả lời sai / bịa). Giá trị không đổi so
# với bản Qdrant thuần trước đây - đây LÀ NGƯỠNG CHO RAW SCORE của Qdrant.
SCORE_THRESHOLD = 0.4


class _ThresholdRetriever(BaseRetriever):
    """Retriever lọc theo NGƯỠNG ĐIỂM SỐ THÔ (raw similarity score do chính
    Qdrant trả về qua `similarity_search_with_score()`), KHÔNG đi qua hàm
    chuẩn hoá relevance-score nội bộ của LangChain (xem giải thích ở đầu
    file). Nhờ vậy SCORE_THRESHOLD giữ nguyên đúng ý nghĩa như bản Qdrant
    thuần trước khi refactor."""

    vector_store: VectorStore
    k: int = TOP_K
    score_threshold: float = SCORE_THRESHOLD

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        # Truyền score_threshold thẳng cho Qdrant lọc ở phía server (native
        # param của similarity_search_with_score) - tương đương kết quả với
        # cách lọc client-side cũ (`r.score >= SCORE_THRESHOLD`), chỉ hiệu
        # quả hơn vì Qdrant không cần trả về những điểm chắc chắn bị loại.
        results = self.vector_store.similarity_search_with_score(
            query, k=self.k,
        )
        docs = []

        for doc, score in results:
            if score >= self.score_threshold:
                docs.append(doc)

        return docs
    #    return [doc for doc, _score in results]


@lru_cache(maxsize=8)
def get_retriever(top_k: int = TOP_K, score_threshold: float = SCORE_THRESHOLD) -> BaseRetriever:
    """Trả về 1 Retriever sẵn sàng dùng qua `.invoke(query)`.

    Cache theo (top_k, score_threshold) vì trong project hiện chỉ dùng đúng
    1 bộ tham số mặc định ở mọi nơi gọi - tránh tạo lại object retriever ở
    mỗi lượt hỏi mà không có lý do."""
    return _ThresholdRetriever(
        vector_store=get_vector_store(), k=top_k, score_threshold=score_threshold,
    )
