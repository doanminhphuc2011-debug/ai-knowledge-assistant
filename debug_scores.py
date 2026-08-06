"""
debug_scores.py
Script DEBUG TẠM THỜI - KHÔNG dùng trong luồng chạy chính thức của chatbot,
KHÔNG import bởi bất kỳ file nào khác (chatbot.py, evaluate.py, rag.py...).

Mục đích DUY NHẤT: in ra RAW SIMILARITY SCORE thật mà Redis (qua
`vector_store.similarity_search_with_score()`) trả về cho 1 số câu hỏi cụ
thể, để có SỐ LIỆU THẬT trước khi quyết định có cần chỉnh SCORE_THRESHOLD
(retriever.py) hay không - tránh đoán mò.

KHÔNG thay đổi logic retrieval: gọi thẳng `get_vector_store()` (hàm có sẵn,
không sửa) rồi gọi `similarity_search_with_score(query, k=10)` - giống hệt
cách `_ThresholdRetriever` trong retriever.py đang gọi bên trong nó, CHỈ
KHÁC 1 điểm: lấy k=10 (thay vì TOP_K=5) và KHÔNG lọc theo SCORE_THRESHOLD,
để nhìn được cả những chunk đang bị loại (ở dưới 0.4) lẫn những chunk có
điểm cao nhưng có thể sai loại (context precision thấp).

Chạy: python debug_scores.py
(không có tham số dòng lệnh - danh sách câu hỏi debug khai báo cố định bên
dưới, sửa trực tiếp trong DEBUG_QUESTIONS nếu muốn thử câu khác)
"""

from __future__ import annotations

from rag.vector_store import get_vector_store

# Số chunk tối đa lấy ra mỗi câu hỏi để debug - CAO HƠN TOP_K=5 thật (retriever.py)
# để nhìn được cả những chunk gần ngưỡng nhưng bị loại bởi TOP_K lẫn bởi
# SCORE_THRESHOLD trong luồng thật.
DEBUG_K = 10

# Preview text: chỉ in 100 ký tự đầu của mỗi chunk, đủ để nhận diện nội
# dung mà không làm rối output khi so sánh nhiều chunk cùng lúc.
PREVIEW_CHARS = 100

DEBUG_QUESTIONS = [
    "Cách tích điểm thành viên?",
    "Hạng Kim Cương có ưu đãi gì?",
    "Quán có bán bia không?",
    "Quán có giảm giá sinh viên không?",
]


def _preview(text: str, n: int = PREVIEW_CHARS) -> str:
    """Rút gọn nội dung chunk về n ký tự đầu, thay xuống dòng bằng khoảng
    trắng để mỗi chunk in gọn trên 1 dòng, dễ so sánh giữa các dòng."""
    flat = " ".join(text.split())
    return flat[:n] + ("..." if len(flat) > n else "")


def debug_query(vector_store, question: str) -> None:
    """In Top DEBUG_K chunk kèm raw score + metadata cho 1 câu hỏi."""
    print("=" * 100)
    print(f"CÂU HỎI: {question}")
    print("=" * 100)

    # Gọi ĐÚNG method mà retriever.py dùng bên trong (similarity_search_with_score),
    # chỉ khác k lớn hơn và không lọc theo score_threshold ở đây - để tự
    # xem toàn bộ điểm số thô, không bị che bởi ngưỡng 0.4 đang có sẵn.
    results = vector_store.similarity_search_with_score(question, k=DEBUG_K)

    if not results:
        print("(Không có chunk nào được trả về - có thể do k=0 hoặc lỗi kết nối)")
        print()
        return

    header = f"{'#':>3}  {'score':>8}  {'type':<12}  {'source':<16}  preview"
    print(header)
    print("-" * len(header))
    for rank, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata or {}
        chunk_type = str(meta.get("type", "unknown"))
        source = str(meta.get("source", "unknown"))
        preview = _preview(doc.page_content)
        # Đánh dấu trực quan chunk nào đang NẰM DƯỚI SCORE_THRESHOLD=0.4
        # hiện tại (tức bị retriever.py thật loại bỏ) - chỉ để đọc dễ hơn,
        # KHÔNG import/dùng lại hằng số SCORE_THRESHOLD từ retriever.py để
        # tránh mọi phụ thuộc/side-effect vào file không được sửa.
        flag = "  <- dưới ngưỡng 0.4 hiện tại" if score < 0.4 else ""
        print(f"{rank:>3}  {score:>8.4f}  {chunk_type:<12}  {source:<16}  {preview}{flag}")

    print()


def main() -> None:
    vector_store = get_vector_store()
    for question in DEBUG_QUESTIONS:
        debug_query(vector_store, question)


if __name__ == "__main__":
    main()
