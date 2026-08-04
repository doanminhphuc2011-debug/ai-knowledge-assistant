"""
vector_store.py
Điểm truy cập DUY NHẤT tới Vector Database của project, thông qua interface
chuẩn `langchain_core.vectorstores.VectorStore` của LangChain - thay vì
ingest.py/rag.py tự gọi thẳng QdrantClient như trước.

Đây CHÍNH LÀ điểm "cắm/rút" Vector DB: sau này muốn đổi Qdrant -> Redis
(hoặc bất kỳ Vector DB nào khác LangChain hỗ trợ), CHỈ CẦN SỬA FILE NÀY
(đổi QdrantVectorStore -> RedisVectorStore, đổi các hằng số kết nối tương
ứng). ingest.py, retriever.py, rag.py đều KHÔNG cần sửa vì chúng chỉ biết
tới interface VectorStore chung (add_documents, as_retriever...), không
biết (và không cần biết) đang chạy trên Qdrant hay Redis.

Hiện tại: vẫn dùng Qdrant (CHƯA migrate sang Redis) - đúng theo yêu cầu.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from .embeddings import get_embeddings

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # để trống nếu chạy Qdrant local
COLLECTION_NAME = "dmp_knowledge"


@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    """Client Qdrant thô - CHỈ dùng nội bộ trong file này (để quản lý vòng
    đời collection). Các module khác không import trực tiếp từ đây, đúng
    tinh thần "chỉ vector_store.py biết đang chạy trên Qdrant"."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _vector_size() -> int:
    """Suy ra kích thước vector từ chính embedding model đang dùng (embed
    thử 1 câu ngắn) thay vì hardcode con số cụ thể (384) - để không phải
    sửa lại nếu sau này lỡ đổi model."""
    return len(get_embeddings().embed_query("probe"))


def ensure_collection() -> None:
    """Tạo collection nếu CHƯA tồn tại (idempotent - gọi lại nhiều lần vẫn
    an toàn). Trước đây Qdrant tự suy ra vector_size khi gọi client.add()
    (do tích hợp sẵn FastEmbed); dùng LangChain VectorStore thuần thì phải
    tự tạo collection với đúng kích thước vector trước khi ghi dữ liệu."""
    client = _get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=_vector_size(), distance=Distance.COSINE),
        )


def reset_collection() -> None:
    """Xoá sạch collection cũ (nếu có) - dùng trong ingest.py để đảm bảo
    dữ liệu cũ không còn sót lại mỗi lần chạy lại `python ingest.py`,
    tương đương hành vi cũ (client.delete_collection() trước khi add())."""
    client = _get_qdrant_client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Trả về 1 LangChain VectorStore đã sẵn sàng dùng (collection chắc
    chắn tồn tại nhờ ensure_collection()).

    ĐÂY LÀ HÀM DUY NHẤT mà ingest.py/retriever.py/rag.py cần gọi - chúng
    hoàn toàn không biết bên trong đang là QdrantVectorStore.
    """
    ensure_collection()
    return QdrantVectorStore(
        client=_get_qdrant_client(),
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
    )
