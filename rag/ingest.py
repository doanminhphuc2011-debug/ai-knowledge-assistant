"""
ingest.py
Đọc dữ liệu (menu.json, menu.md, faq.md, promotions.md), chunk theo cấu trúc
ngữ nghĩa, và nạp vào Vector Database THÔNG QUA LangChain VectorStore
(vector_store.py) - không còn gọi thẳng QdrantClient ở file này. Vector DB
cụ thể đang dùng (hiện tại: Qdrant) chỉ vector_store.py cần biết.

Chạy: python ingest.py
Chạy lại mỗi khi nội dung menu/faq/promotions thay đổi (script sẽ xóa và
tạo lại collection từ đầu để tránh dữ liệu cũ còn sót lại).
"""

import os
import re
import json

from langchain_core.documents import Document

from .vector_store import get_vector_store, reset_collection

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)

# 1. CHUNKING - mỗi loại file có cấu trúc riêng nên tách riêng logic
def chunk_menu_items(filename: str = "menu.json") -> list[dict]:
    """Mỗi món trong menu.json -> 1 chunk, kèm metadata giá/danh mục
    để sau này có thể lọc (filter) theo category hoặc khoảng giá."""
    items = json.load(open(_path(filename), encoding="utf-8"))
    chunks = []
    for it in items:
        if it["price_m"] == it["price_l"]:
            text = (
                f"Món: {it['name']} (Danh mục: {it['category']}). "
                f"Giá: {it['price_m']:,} VNĐ. "
                f"Mô tả: {it['description']}")
        else:
            text = (
                f"Món: {it['name']} (Danh mục: {it['category']}). "
                f"Giá Size M: {it['price_m']:,} VNĐ, "
                f"Size L: {it['price_l']:,} VNĐ. "
                f"Mô tả: {it['description']}")
        chunks.append({
            "text": text,
            "source": "menu.json",
            "type": "menu_item",
            "name": it["name"],
            "category": it["category"],
            "price_m": it["price_m"],
            "price_l": it["price_l"],
        })
    return chunks


def chunk_menu_customization(filename: str = "menu.md") -> list[dict]:
    """Chỉ lấy phần VII (tùy chọn đường/đá/topping) trong menu.md, vì phần
    này KHÔNG có trong menu.json - tránh trùng lặp các món đã chunk ở trên."""
    text = open(_path(filename), encoding="utf-8").read()
    match = re.search(r"## VII\..*", text, re.S)
    if not match:
        return []
    return [{
        "text": match.group(0).strip(),
        "source": "menu.md",
        "type": "menu_option",
    }]


def chunk_faq(filename: str = "faq.md") -> list[dict]:
    """Tách theo từng cặp Q:/A: - mỗi cặp là 1 chunk độc lập, đủ ngữ cảnh."""
    text = open(_path(filename), encoding="utf-8").read()
    blocks = re.split(r"\n(?=Q:)", text)
    chunks = []
    for b in blocks:
        b = b.strip()
        if not b.startswith("Q:"):
            continue
        chunks.append({"text": b, "source": "faq.md", "type": "faq"})
    return chunks


def chunk_promotions(filename: str = "promotions.md") -> list[dict]:
    """Tách theo từng section (##), rồi tách nhỏ tiếp theo từng mục khuyến
    mãi/hạng thành viên (số thứ tự + **tiêu đề**), giữ tên section làm
    ngữ cảnh đi kèm để model không bị mất ngữ cảnh khi chunk bị tách rời."""
    text = open(_path(filename), encoding="utf-8").read()
    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        heading_match = re.match(r"## (.+)", sec)
        heading = heading_match.group(1) if heading_match else ""

        items = re.split(r"\n(?=\d+\.\s+\*\*)", sec)
        for item in items:
            item = item.strip()
            # Bỏ các mẩu quá ngắn (vd. chỉ có tiêu đề H1 của cả file, không mang thông tin gì để retrieve)
            if not item or (item.startswith("# ") and len(item) < 80):
                continue
            # Tránh lặp heading 2 lần khi item chính là đoạn mở đầu section
            prefixed = item if item.startswith("##") else (
                f"[{heading}]\n{item}" if heading else item
            )
            chunks.append({
                "text": prefixed,
                "source": "promotions.md",
                "type": "promotion",
            })
    return chunks

# 2. CHUYỂN CHUNK -> LANGCHAIN DOCUMENT
def _chunks_to_documents(chunks: list[dict]) -> list[Document]:
    """Chuyển từng chunk dict (như trả về bởi các hàm chunk_* ở trên) thành
    1 `langchain_core.documents.Document` - đơn vị dữ liệu chuẩn mà mọi
    LangChain VectorStore đều hiểu.

    Giữ NGUYÊN toàn bộ metadata cũ (source, type, name, category, price_m,
    price_l) - chỉ khác 1 điểm: nội dung "text" giờ nằm ở `page_content`
    thay vì vừa nằm trong page_content vừa lặp lại thêm 1 lần trong
    metadata như payload cũ của Qdrant. Đây là cách tổ chức dữ liệu chuẩn
    của LangChain, không làm mất bất kỳ field metadata nào."""
    documents = []
    for chunk in chunks:
        metadata = {key: value for key, value in chunk.items() if key != "text"}
        documents.append(Document(page_content=chunk["text"], metadata=metadata))
    return documents


# 3. NẠP VÀO VECTOR STORE (qua LangChain VectorStore, không gọi QdrantClient trực tiếp)
def build_index() -> None:
    all_chunks = (
        chunk_menu_items()
        + chunk_menu_customization()
        + chunk_faq()
        + chunk_promotions()
    )
    documents = _chunks_to_documents(all_chunks)

    # Xoá sạch collection cũ trước khi nạp lại - tương đương hành vi cũ
    # (client.delete_collection() rồi add() lại), tránh dữ liệu cũ sót lại.
    reset_collection()
    vector_store = get_vector_store()

    # add_documents() của LangChain tự động: embed từng Document bằng
    # embedding model đã cấu hình trong vector_store.py, rồi upsert vào
    # Vector DB - tương đương những gì client.add() làm trước đây, chỉ khác
    # là đi qua interface chuẩn của LangChain thay vì API riêng của Qdrant.
    vector_store.add_documents(documents, ids=list(range(len(documents))))

    print(f"Đã nạp {len(all_chunks)} chunks vào Vector Store:")
    by_type: dict[str, int] = {}
    for c in all_chunks:
        by_type[c["type"]] = by_type.get(c["type"], 0) + 1
    for t, n in by_type.items():
        print(f"  - {t}: {n} chunks")


if __name__ == "__main__":
    build_index()
