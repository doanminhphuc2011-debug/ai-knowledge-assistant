"""
rag.py
Module truy vấn (retrieval) từ Qdrant - dùng lại collection đã tạo bởi
ingest.py. Trả về context liên quan nhất tới câu hỏi của user.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "dmp_knowledge"

# Phải TRÙNG với model dùng lúc ingest, nếu không vector sẽ không tương thích.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 7
# Điểm similarity tối thiểu để 1 chunk được coi là "liên quan". Chunk có
# điểm thấp hơn ngưỡng này sẽ bị loại để tránh nhồi context không liên quan
# vào prompt (giảm nguy cơ model trả lời sai / bịa).
SCORE_THRESHOLD = 0.2

_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
_client.set_model(EMBED_MODEL)


def retrieve_context(query: str, top_k: int = TOP_K) -> str:
    """Tìm các đoạn văn liên quan nhất tới câu hỏi trong Qdrant.
    Trả về chuỗi rỗng nếu không có gì đủ liên quan."""
    results = _client.query(
        collection_name=COLLECTION_NAME,
        query_text=query,
        limit=top_k,
    )
    relevant = [r for r in results if r.score >= SCORE_THRESHOLD]
    if not relevant:
        return ""

    return "\n".join(f"- {r.document}" for r in relevant)


if __name__ == "__main__":
    # Test nhanh: python rag.py "câu hỏi test"
    import sys
    q = " ".join(sys.argv[1:]) or "Quán có món cà phê muối không, giá bao nhiêu?"
    print(f"Query: {q}\n")
    ctx = retrieve_context(q)
    print("Context tìm được:\n" + (ctx or "(không có gì đủ liên quan)"))
