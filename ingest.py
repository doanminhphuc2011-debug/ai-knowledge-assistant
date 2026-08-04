"""
ingest.py
Đọc dữ liệu (menu.json, menu.md, faq.md, promotions.md), chunk theo cấu trúc
ngữ nghĩa, và nạp (embed + upsert) vào Qdrant để phục vụ RAG cho chatbot Ori.

Chạy: python ingest.py
Chạy lại mỗi khi nội dung menu/faq/promotions thay đổi (script sẽ xóa và
tạo lại collection từ đầu để tránh dữ liệu cũ còn sót lại).
"""

import os
import re
import json

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # để trống nếu chạy Qdrant local

COLLECTION_NAME = "dmp_knowledge"

# Model embedding đa ngôn ngữ (chạy local qua FastEmbed, không tốn API call).
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

# 2. NẠP VÀO QDRANT
def build_index() -> None:
    all_chunks = (
        chunk_menu_items()
        + chunk_menu_customization()
        + chunk_faq()
        + chunk_promotions()
    )

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    client.set_model(EMBED_MODEL)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    # client.add() tự động: embed từng "document" bằng FastEmbed, tạo
    # collection với đúng kích thước vector, và upsert - không cần tự
    # gọi model embedding hay tự tạo collection thủ công.
    client.add(
        collection_name=COLLECTION_NAME,
        documents=[c["text"] for c in all_chunks],
        metadata=all_chunks,
        ids=list(range(len(all_chunks))),
    )

    print(f"Đã nạp {len(all_chunks)} chunks vào collection '{COLLECTION_NAME}':")
    by_type: dict[str, int] = {}
    for c in all_chunks:
        by_type[c["type"]] = by_type.get(c["type"], 0) + 1
    for t, n in by_type.items():
        print(f"  - {t}: {n} chunks")


if __name__ == "__main__":
    build_index()
