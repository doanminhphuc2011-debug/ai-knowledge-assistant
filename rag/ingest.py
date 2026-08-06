"""
ingest.py
Đọc dữ liệu (menu.json, menu.md, faq.md, promotions.md), chunk theo cấu trúc
ngữ nghĩa và nạp vào Vector Database thông qua LangChain VectorStore.

Vector Database cụ thể (Redis Cloud, Qdrant, ...) được cấu hình hoàn toàn
trong vector_store.py. File này không phụ thuộc vào bất kỳ Vector DB cụ thể nào.
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
    """Lấy các section trong menu.md KHÔNG liệt kê món ăn theo món (vì các
    section liệt kê món đã trùng với menu.json, chunk_menu_items() lo phần
    đó rồi) - phần còn lại là thông tin cấu hình/tuỳ chọn chung (đường, đá,
    topping...) không có mặt trong menu.json.

    QUAN TRỌNG: việc phân biệt "section liệt kê món" vs "section còn lại"
    dựa HOÀN TOÀN vào CẤU TRÚC danh sách Markdown, KHÔNG dựa vào số thứ tự
    La Mã hay tiêu đề section cụ thể nào:
    - Mọi section liệt kê món trong menu.md đều dùng list ĐÁNH SỐ dạng
      "N. **Tên món**" (vd. "1. **Cà Phê Đen Đá**") - đây chính là format
      dùng để liệt kê 40 món trong file, khớp 1-1 với các record trong
      menu.json.
    - Section còn lại (hiện tại là "VII. TÙY CHỌN...") không dùng list
      đánh số kiểu đó mà dùng bullet "* **Nhãn:** giá trị".
    Nhờ dựa vào cấu trúc thay vì "## VII" cố định: nếu sau này thêm/bớt/
    đổi thứ tự section món ăn trong menu.md, hoặc đổi tên section tuỳ
    chọn, hàm này vẫn nhận đúng section - miễn các section liệt kê món
    tiếp tục theo đúng convention đánh số hiện có của toàn bộ file.

    Trong mỗi section được nhận, tách tiếp theo từng bullet cấp cha (dạng
    "* **Tiêu đề:** ...") thay vì gộp NGUYÊN section thành 1 chunk - nếu
    không, các khái niệm độc lập như "Mức Đường", "Mức Đá", "Topping" sẽ
    bị nhét chung vào 1 vector duy nhất, làm loãng embedding (cùng loại
    vấn đề mà chunk_promotions() bên dưới đã xử lý cho section các hạng
    thành viên). Giữ heading cha làm tiền tố cho mỗi bullet để không mất
    ngữ cảnh khi tách nhỏ. Dòng con thụt lề (vd. "  - Trân châu đen") KHÔNG
    khớp pattern "* **" (có khoảng trắng đầu dòng) nên vẫn được giữ
    nguyên trong chunk cha của nó, không bị tách vụn."""
    text = open(_path(filename), encoding="utf-8").read()
    sections = re.split(r"\n(?=## )", text)

    chunks = []
    for sec in sections:
        sec = sec.strip()
        heading_match = re.match(r"## (.+)", sec)
        # Bỏ phần mở đầu file trước heading "## " đầu tiên (H1 title + đoạn
        # giới thiệu chung) - không phải 1 section thực sự, không có gì để
        # trích xuất theo cấu trúc "tuỳ chọn". Điều kiện này chỉ dựa vào
        # việc có/không có heading cấp "## ", không dựa vào nội dung.
        if not sec or heading_match is None:
            continue
        # Bỏ section liệt kê món (xem giải thích ở docstring) - phát hiện
        # thuần theo cấu trúc list đánh số, không hardcode số La Mã.
        if re.search(r"\n\d+\.\s+\*\*", sec):
            continue

        heading = heading_match.group(1)
        items = re.split(r"\n(?=\*\s+\*\*)", sec)
        for item in items:
            item = item.strip()
            # Bỏ mẩu chỉ gồm đúng 1 dòng heading Markdown, không có nội dung
            # theo sau (cùng logic lọc heading-only đã dùng ở chunk_promotions).
            if not item or re.fullmatch(r"#{1,6}\s+.+", item):
                continue
            prefixed = item if item.startswith("##") else (
                f"[{heading}]\n{item}" if heading else item
            )
            chunks.append({
                "text": prefixed,
                "source": "menu.md",
                "type": "menu_option",
            })
    return chunks


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
    mãi/hạng thành viên, giữ tên section làm ngữ cảnh đi kèm để model
    không bị mất ngữ cảnh khi chunk bị tách rời.

    Mỗi mục con có thể được viết theo 1 trong 2 kiểu list Markdown:
    - Đánh số:      "1. **Tiêu đề**"   (vd. section I - các chương trình KM)
    - Gạch đầu dòng: "- **Tiêu đề**"   (vd. section II - các hạng thành viên)
    Trước đây chỉ tách theo kiểu đánh số, nên cả section II bị dồn thành 1
    chunk lớn (4 hạng thành viên gộp chung), làm loãng vector embedding và
    khiến câu hỏi hẹp (vd. "Hạng Kim Cương") không tìm thấy context. Các
    dòng con lồng bên trong (thụt lề, vd. "  - Giảm 5%...") KHÔNG khớp
    pattern này (thiếu "**" ngay sau dấu "-", hoặc có khoảng trắng đầu
    dòng) nên vẫn được giữ nguyên trong chunk cha, không bị tách vụn."""
    text = open(_path(filename), encoding="utf-8").read()
    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        heading_match = re.match(r"## (.+)", sec)
        heading = heading_match.group(1) if heading_match else ""

        items = re.split(r"\n(?=\d+\.\s+\*\*|-\s+\*\*)", sec)
        for item in items:
            item = item.strip()
            # Bỏ các mẩu KHÔNG mang thông tin gì để retrieve: item chỉ gồm
            # ĐÚNG 1 dòng heading Markdown (#, ##, ###...), không có nội
            # dung nào theo sau. Trước đây chỉ check `item.startswith("# ")`
            # (heading cấp H1) nên bỏ sót heading cấp H2/H3 (## , ###...) -
            # các section trong promotions.md đều dùng "## " nên chunk
            # heading-only (vd. "## I. CHƯƠNG TRÌNH KHUYẾN MÃI CỐ ĐỊNH...")
            # bị lọt vào vector store. re.fullmatch không có flag re.S nên
            # "." không khớp xuống dòng - CHỈ khớp khi TOÀN BỘ item đúng 1
            # dòng heading, không khớp nếu có nội dung ở dòng sau (vẫn giữ
            # nguyên các chunk hợp lệ như "## II. ...\n\nKhách hàng...").
            if not item or re.fullmatch(r"#{1,6}\s+.+", item):
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
