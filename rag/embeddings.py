"""
embeddings.py
Định nghĩa 1 nơi DUY NHẤT để khởi tạo embedding model, dùng chung cho cả
ingest.py (embed document lúc nạp dữ liệu) và retriever.py (embed câu hỏi
lúc truy vấn) - đảm bảo 2 bên LUÔN dùng cùng 1 model. Trước đây model cũng
được dùng chung, nhưng qua 2 hằng số EMBED_MODEL khai báo LẶP LẠI độc lập ở
cả ingest.py lẫn rag.py - gộp về đây để chỉ có 1 nơi duy nhất cần đúng.

Model GIỮ NGUYÊN so với trước khi refactor:
"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", chạy local
qua FastEmbed (không tốn API call, không đổi). Trước đây Qdrant Client tự
dùng model này nội bộ qua `client.set_model()`; giờ được bọc qua lớp
`Embeddings` chuẩn của LangChain (FastEmbedEmbeddings) để tương thích với
VectorStore/Retriever - đây là thay đổi bắt buộc để dùng được API chuẩn
của LangChain, KHÔNG phải đổi model.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Trả về embedding model dùng chung cho toàn project.

    Cache 1 instance duy nhất (lru_cache) vì khởi tạo FastEmbedEmbeddings sẽ
    load model ONNX từ đĩa - chỉ nên làm 1 lần cho cả tiến trình, không phải
    load lại ở mỗi câu hỏi/mỗi chunk."""
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)
