"""
sparse_retriever.py
Sparse retrieval (BM25) - phần "Sparse" trong pipeline Dense + Sparse +
Fusion. Chạy HOÀN TOÀN in-memory, KHÔNG cần thêm Vector DB / index mới,
KHÔNG động vào vector_store.py hay ingest.py (2 file này giữ nguyên).

TẠI SAO IN-MEMORY, KHÔNG DÙNG REDISEARCH FULL-TEXT:
Corpus hiện tại rất nhỏ (~70-90 chunk, xem build_index() trong ingest.py) -
BM25Okapi (thư viện `rank_bm25`, thuần Python) tính điểm cho toàn bộ corpus
trong <1ms, không cần thêm 1 tầng hạ tầng full-text search riêng của Redis
(schema mới, đồng bộ index...). Nếu sau này corpus lớn hơn nhiều (hàng
nghìn+ chunk), nên chuyển sang RediSearch FT.SEARCH native để tận dụng
index đĩa/phân trang thay vì giữ toàn bộ trong RAM.

TẠI SAO XÂY CORPUS TỪ chunk_*() CỦA ingest.py, KHÔNG ĐỌC LẠI TỪ REDIS:
Các hàm chunk_menu_items/chunk_menu_customization/chunk_faq/chunk_promotions
trong ingest.py CHÍNH LÀ nguồn tạo ra dữ liệu đã nạp vào Redis (build_index()
gọi đúng các hàm này). Gọi lại các hàm đó ở đây đảm bảo BM25 index có
NỘI DUNG GIỐNG HỆT 100% với những gì đã embed vào Redis, mà không cần đọc
ngược từ Redis (vốn không tối ưu cho việc "lấy toàn bộ document ra ngoài").
Nhược điểm duy nhất: nếu build_index() được chạy lại với data mới mà chatbot
không restart, BM25 index (cache trong RAM) sẽ lệch với Redis - chấp nhận
được vì đây cũng là giả định sẵn có của _menu_cache trong tools.py (menu
hiếm khi đổi lúc chatbot đang chạy).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .ingest import (
    _chunks_to_documents,
    chunk_faq,
    chunk_menu_customization,
    chunk_menu_items,
    chunk_promotions,
)

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover - cùng fallback với tools.py, không
    # bắt buộc phải cài thêm package nếu môi trường chưa kịp cập nhật.
    def unidecode(text: str) -> str:
        text = text.replace("đ", "d").replace("Đ", "D")
        decomposed = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


# Stopword tiếng Việt (đã bỏ dấu, khớp với output của unidecode()) - PHÁT
# HIỆN QUA TEST THỰC TẾ: nếu không loại các từ chức năng cực phổ biến này,
# BM25 sẽ báo "khớp từ khóa" giả cho câu hỏi hoàn toàn không liên quan (vd.
# "Quán có bán bia hoặc rượu không?" khớp điểm cao với chunk FAQ về chỗ đậu
# xe chỉ vì cả 2 cùng chứa "quán"/"có"/"không") - phá vỡ đúng cơ chế dùng
# BM25 làm corroboration mà hybrid_retriever.py dựa vào. Danh sách này CHỈ
# gồm từ chức năng (không mang nghĩa nội dung) - không chứa từ nào có thể
# là 1 phần tên món/khái niệm cụ thể (vd. KHÔNG loại "ban" vì đây là "bán"
# - động từ nội dung quan trọng, "quán có BÁN bia không" cần giữ "ban").
_STOPWORDS = {
    "quan", "co", "khong", "la", "cua", "va", "cho", "duoc", "nay", "do",
    "cac", "nhung", "mot", "gi", "sao", "nhu", "the", "nao", "a", "de",
    "tai", "voi", "toi", "minh", "ban", "neu", "thi", "rat", "cung",
    "hay", "hoac", "vay", "ma", "nen", "khi", "nhe", "vui", "long", "xin",
}

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")


def _tokenize(text: str) -> list[str]:
    """Tokenize cho BM25: lower-case + bỏ dấu tiếng Việt (unidecode) + bỏ
    dấu câu + tách khoảng trắng + loại stopword.

    Bỏ dấu câu (không chỉ bỏ dấu thanh điệu) là bước BẮT BUỘC, không phải
    tuỳ chọn - nếu thiếu, 2 câu cùng nghĩa nhưng khác dấu câu (vd. có "?"
    hay không) sẽ tạo ra token khác nhau ("khong" != "khong?"), làm BM25
    match/miss không ổn định giữa các câu hỏi.

    Loại stopword bằng danh sách domain-specific ở trên (không dùng thư
    viện stopword tiếng Việt tổng quát, vì corpus quá nhỏ và đặc thù -
    chỉ cần loại đúng nhóm từ chức năng lặp lại khắp mọi chunk)."""
    folded = unidecode(text.lower())
    cleaned = _NON_ALNUM.sub(" ", folded)
    return [tok for tok in cleaned.split() if tok not in _STOPWORDS]


@lru_cache(maxsize=1)
def _build_corpus() -> list[Document]:
    """Xây corpus Document giống hệt build_index() trong ingest.py (cùng
    thứ tự, cùng nội dung) - để index_position ở đây khớp 1-1 với dữ liệu
    thật đã nạp vào Redis."""
    all_chunks = (
        chunk_menu_items()
        + chunk_menu_customization()
        + chunk_faq()
        + chunk_promotions()
    )
    return _chunks_to_documents(all_chunks)


@lru_cache(maxsize=1)
def _get_bm25_index() -> tuple[BM25Okapi, list[Document]]:
    """Build BM25 index 1 lần, cache lại (giống get_vector_store() /
    _menu_cache - tránh build lại mỗi câu hỏi)."""
    documents = _build_corpus()
    tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
    index = BM25Okapi(tokenized_corpus)
    return index, documents


def bm25_search(query: str, k: int) -> list[tuple[Document, float]]:
    """Trả về top-k (Document, bm25_score) theo điểm giảm dần. CHỈ giữ lại
    kết quả có score > 0 (tức có ít nhất 1 từ khớp) - loại bỏ "khớp giả"
    (BM25Okapi vẫn trả điểm 0 cho mọi document nếu không match từ nào, giữ
    lại chỉ làm nhiễu bước fusion phía sau)."""
    index, documents = _get_bm25_index()
    scores = index.get_scores(_tokenize(query))

    ranked = sorted(
        ((doc, float(score)) for doc, score in zip(documents, scores)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [(doc, score) for doc, score in ranked[:k] if score > 0.0]
