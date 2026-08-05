"""
vector_store.py
Điểm truy cập DUY NHẤT tới Vector Database của project, thông qua interface
chuẩn `langchain_core.vectorstores.VectorStore` của LangChain - thay vì
ingest.py/rag.py tự gọi thẳng RedisClient như trước.

Đây CHÍNH LÀ điểm "cắm/rút" Vector DB: sau này muốn đổi Vector DB khác
(được LangChain hỗ trợ), CHỈ CẦN SỬA FILE NÀY. ingest.py, retriever.py,
rag.py đều KHÔNG cần sửa vì chúng chỉ biết tới interface VectorStore chung
(add_documents, as_retriever, similarity_search_with_score...), không biết
(và không cần biết) đang chạy trên Qdrant hay Redis.

ĐÃ MIGRATE: Qdrant Cloud -> Redis Cloud (Redis Stack / RediSearch), dùng
package chính thức `langchain-redis` (KHÔNG tự viết FT.SEARCH bằng tay,
KHÔNG dùng redis-py trực tiếp để làm vector search).

QUAN TRỌNG - 2 điểm hành vi phải giữ đúng vì retriever.py và ingest.py
KHÔNG được sửa (xem chi tiết ở lớp _ScoreAdjustedRedisVectorStore bên dưới):
1. similarity_search_with_score() phải trả về ĐIỂM CÀNG CAO CÀNG GIỐNG
   (cosine similarity thô), y hệt quy ước cũ của QdrantVectorStore -
   RediSearch mặc định trả về "distance" (CÀNG THẤP CÀNG GIỐNG), ngược dấu
   hoàn toàn nên phải tự đảo lại.
2. add_documents(ids=...) phải chấp nhận ids kiểu int (ingest.py gọi
   `ids=list(range(len(documents)))`) - Redis key bắt buộc là chuỗi nên
   cần tự ép kiểu str() trước khi giao cho RedisVectorStore gốc.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStore
from langchain_redis import RedisConfig, RedisVectorStore

from .embeddings import get_embeddings

load_dotenv()  # đọc .env để lấy REDIS_URL, REDIS_INDEX_NAME

REDIS_URL = os.getenv("REDIS_URL")
REDIS_INDEX_NAME = os.getenv("REDIS_INDEX_NAME", "dmp_knowledge")

if not REDIS_URL:
    raise ValueError("Thiếu REDIS_URL trong file .env")


def _build_config() -> RedisConfig:
    """Cấu hình index Redis (RediSearch) dùng chung cho MỌI thao tác (đọc
    lẫn ghi) - phải luôn tạo ra config giống hệt nhau ở mọi nơi gọi, để
    không vô tình tạo 2 index khác schema cho cùng 1 index_name.

    distance_metric="COSINE": GIỮ NGUYÊN cùng 1 phép đo tương đồng như
    Qdrant (Distance.COSINE) trước đây - không đổi ý nghĩa SCORE_THRESHOLD
    đang được tune sẵn ở retriever.py (file không được sửa).

    metadata_schema: khai báo tường minh các field metadata đang có trong
    ingest.py (source/type/name/category/price_m/price_l) để RediSearch
    lưu và index đúng kiểu dữ liệu, tránh mất metadata. Các field này
    KHÔNG bắt buộc phải có mặt ở MỌI document - menu_item có price_m/
    price_l, còn faq/menu_option/promotion thì không, RediSearch cho phép
    field vắng mặt ở 1 số document, không lỗi.
    """
    return RedisConfig(
        index_name=REDIS_INDEX_NAME,
        redis_url=REDIS_URL,
        distance_metric="COSINE",
        metadata_schema=[
            {"name": "source", "type": "tag"},
            {"name": "type", "type": "tag"},
            {"name": "name", "type": "text"},
            {"name": "category", "type": "tag"},
            {"name": "price_m", "type": "numeric"},
            {"name": "price_l", "type": "numeric"},
        ],
    )


class _ScoreAdjustedRedisVectorStore(RedisVectorStore):
    """Bọc RedisVectorStore gốc để giữ đúng 2 hành vi mà retriever.py và
    ingest.py đang phụ thuộc vào (2 file này nằm trong danh sách KHÔNG
    ĐƯỢC SỬA) - xem giải thích chi tiết ở docstring đầu file.

    Không đổi interface: vẫn là 1 `VectorStore` chuẩn, các method vẫn
    cùng tên/cùng chữ ký/cùng kiểu trả về như RedisVectorStore gốc - chỉ
    khác GIÁ TRỊ trả về của 2 method dưới đây để tương thích ngược với
    phần code đã viết sẵn cho Qdrant.
    """

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        """RediSearch (distance_metric="COSINE") trả "vector_distance" =
        1 - cosine_similarity (CÀNG THẤP CÀNG GIỐNG). retriever.py lọc
        bằng `score >= SCORE_THRESHOLD` (quy ước CÀNG CAO CÀNG GIỐNG kiểu
        Qdrant, không được sửa) - nên phải đảo ngược lại thành cosine
        similarity thô (`1.0 - distance`) trước khi trả ra ngoài, để ý
        nghĩa ngưỡng 0.4 không bị đảo lộn."""
        results = super().similarity_search_with_score(query, k=k, **kwargs)
        return [(doc, 1.0 - distance) for doc, distance in results]

    def add_documents(self, documents, **kwargs):
        """ingest.py gọi `add_documents(documents, ids=list(range(len(documents))))`
        - ids là số nguyên. Redis key bắt buộc là chuỗi, nên ép kiểu str()
        ở đây thay vì bắt ingest.py (file không được sửa) tự làm."""
        ids = kwargs.get("ids")
        if ids is not None:
            kwargs["ids"] = [str(i) for i in ids]
        return super().add_documents(documents, **kwargs)


def reset_collection() -> None:
    """Xoá sạch index Redis cũ (nếu có) - dùng trong ingest.py để đảm bảo
    dữ liệu cũ không còn sót lại mỗi lần chạy lại `python -m rag.ingest`,
    tương đương hành vi cũ (client.delete_collection() trước khi add() ở
    bản Qdrant)."""
    store = _ScoreAdjustedRedisVectorStore(get_embeddings(), config=_build_config())
    if store.index.exists():
        store.index.delete(drop=True)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Trả về 1 LangChain VectorStore (Redis) đã sẵn sàng dùng - index sẽ
    tự được tạo (nếu chưa tồn tại) bởi chính RedisVectorStore, dùng đúng
    schema/metric khai báo ở _build_config().

    ĐÂY LÀ HÀM DUY NHẤT mà ingest.py/retriever.py/rag.py cần gọi - chúng
    hoàn toàn không biết bên trong đang là RedisVectorStore."""
    return _ScoreAdjustedRedisVectorStore(get_embeddings(), config=_build_config())
