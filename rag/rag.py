"""
rag.py
Module truy vấn (retrieval) - dùng LangChain Retriever (retriever.py) thay
vì gọi thẳng QdrantClient như trước. Trả về context liên quan nhất tới câu
hỏi của user.

Interface công khai (retrieve(), retrieve_context()) GIỮ NGUYÊN 100% so với
trước khi refactor - chatbot.py và evaluate.py không cần sửa gì.
"""
from __future__ import annotations

from langchain_core.documents import Document

from .retriever import SCORE_THRESHOLD, TOP_K, get_retriever

# Re-export để code cũ (nếu có) đang `from rag import SCORE_THRESHOLD/TOP_K`
# vẫn import được - dù giá trị thật giờ được định nghĩa ở retriever.py.
__all__ = ["retrieve", "retrieve_context", "TOP_K", "SCORE_THRESHOLD"]


def retrieve(query: str, top_k: int = TOP_K) -> dict:
    """Truy vấn qua LangChain Retriever (retriever.invoke) và trả về CẢ
    context đã format lẫn kết quả thô.

    Trả về:
        {
            "context": str,        # các chunk đã qua score_threshold, ghép thành 1 chuỗi
            "results": list[Document],  # danh sách Document đã lọc (mỗi Document có
                                         # .page_content và .metadata) - dùng để đo
                                         # retrieval metadata (vd. trong evaluate.py)
        }

    LƯU Ý so với bản Qdrant thuần trước đây: việc lọc theo SCORE_THRESHOLD
    giờ nằm NGAY TRONG retriever (search_type="similarity_score_threshold"),
    nên "results" ở đây chỉ còn các chunk ĐÃ ĐẠT ngưỡng liên quan - khác với
    trước kia "results" là TOÀN BỘ top_k thô (kể cả chunk bị loại). Đây là
    hệ quả tất yếu khi việc lọc được giao cho retriever theo đúng cách làm
    chuẩn của LangChain (yêu cầu dùng retriever.invoke() thay vì tự so sánh
    điểm số bằng tay) - retriever.invoke() không trả kèm điểm số thô ra
    ngoài nên không thể tách riêng "toàn bộ top_k" và "phần qua ngưỡng"
    như trước nữa.
    """
    documents: list[Document] = get_retriever(top_k=top_k).invoke(query)
    context = "\n".join(f"- {doc.page_content}" for doc in documents)
    return {"context": context, "results": documents}


def retrieve_context(query: str, top_k: int = TOP_K) -> str:
    """Giữ nguyên interface cũ để chatbot.py (và mọi code khác) không cần
    sửa gì - chỉ là lớp mỏng gọi lại retrieve()."""
    return retrieve(query, top_k)["context"]


if __name__ == "__main__":
    # Test nhanh: python rag.py "câu hỏi test"
    import sys
    q = " ".join(sys.argv[1:]) or "Quán có món cà phê muối không, giá bao nhiêu?"
    print(f"Query: {q}\n")
    ctx = retrieve_context(q)
    print("Context tìm được:\n" + (ctx or "(không có gì đủ liên quan)"))
