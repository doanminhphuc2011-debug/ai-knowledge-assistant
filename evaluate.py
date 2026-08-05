"""
evaluate.py
Đánh giá tự động chất lượng chatbot RAG (retrieval + generation), dựa trên
bộ test case trong test_cases.json.

Script này KHÔNG viết lại logic chatbot - chỉ tái sử dụng:
- retrieve()  từ rag.py      (đo retrieval)
- ask()       từ chatbot.py  (đo end-to-end: retrieval + generation)

Chạy: python evaluate.py
Kết quả:
- In tóm tắt các metric ra terminal.
- Ghi chi tiết từng câu hỏi vào evaluation_report.csv
"""

from __future__ import annotations

import csv
import json
import re
import time
from typing import Any

from rag import retrieve
from chatbot import ask, reset_history

TEST_CASES_PATH = "test_cases.json"
REPORT_PATH = "evaluation_report.csv"

# Giá trị "expected_source" đặc biệt: câu hỏi KHÔNG kỳ vọng tìm thấy
# context liên quan trong knowledge base (câu hỏi ngoài phạm vi quán).
NO_SOURCE = "none"

# 1. LOAD DỮ LIỆU TEST

def load_test_cases(path: str = TEST_CASES_PATH) -> list[dict]:
    """Đọc bộ câu hỏi đánh giá từ file JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# 2. CÁC HÀM ĐO / TÍNH METRIC CHO TỪNG CÂU HỎI

def get_retrieved_sources(results: list[Any]) -> list[str]:
    """Lấy field 'type' (menu_item / menu_option / faq / promotion) từ
    metadata của từng chunk mà Qdrant trả về."""
    sources = []
    for r in results:
        meta = getattr(r, "metadata", None) or {}
        sources.append(meta.get("type", "unknown"))
    return sources


def check_retriever_correct(
    expected_source: str, retrieved_sources: list[str], context: str
) -> bool:
    """Retriever Accuracy (cho 1 câu hỏi):
    - Nếu expected_source == "none": retriever ĐÚNG khi không có chunk nào
      vượt score_threshold (tức context rỗng) - nghĩa là hệ thống không cố
      "ép" trả về thông tin không liên quan.
    - Ngược lại: retriever ĐÚNG khi expected_source xuất hiện trong danh
      sách source đã lấy về (so khớp metadata)."""
    if expected_source == NO_SOURCE:
        return context.strip() == ""
    return expected_source in retrieved_sources


def check_context_precision(
    expected_source: str, retrieved_sources: list[str]
) -> float | None:
    """Context Precision = (số chunk có metadata source == expected_source)
    / (tổng số chunk được retrieve).
    Trả về None cho câu hỏi "none" vì không có source đúng để so khớp -
    các câu này bị loại khỏi khi tính trung bình, không tính là 0."""
    if expected_source == NO_SOURCE:
        return None
    if not retrieved_sources:
        return 0.0
    relevant = sum(1 for s in retrieved_sources if s == expected_source)
    return relevant / len(retrieved_sources)


def _normalize_for_match(text: str) -> str:
    """Chuẩn hoá text trước khi so khớp keyword, để không bị lệch điểm vì
    khác biệt HÌNH THỨC chứ không phải nội dung sai:

    1. Gộp mọi loại khoảng trắng - kể cả narrow no-break space (\\u202f) mà
       model (Groq) tự chèn trước số/đơn vị theo quy ước đánh máy tiếng
       Việt - về 1 khoảng trắng chuẩn (\\s của Python ở chế độ Unicode mặc
       định đã khớp sẵn \\u202f, không cần liệt kê riêng).
    2. Bỏ khoảng trắng ngay trước dấu % (vd. "0 %" -> "0%").
    3. Coi dấu chấm và dấu phẩy đứng giữa số là NHƯ NHAU (đều là dấu phân
       cách hàng nghìn) - vì SYSTEM_PROMPT yêu cầu bot luôn trả lời số theo
       định dạng dấu chấm (86.000 VNĐ) trong khi test case có thể viết
       theo định dạng dấu phẩy (86,000) - cả hai cùng biểu diễn 1 con số,
       không nên bị tính là "trả lời sai" chỉ vì khác quy ước hiển thị."""
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"(\d)\s+%", r"\1%", t)
    t = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", t)
    return t


def check_answer_correct(
    answer: str, expected_keywords: list[str], match_ratio_threshold: float = 0.5
) -> bool:
    """Answer Accuracy (cho 1 câu hỏi): ĐÚNG khi tỉ lệ expected_keywords xuất
    hiện (không phân biệt hoa/thường, đã chuẩn hoá khoảng trắng và định
    dạng số) trong câu trả lời đạt >= 50%.

    Trước đây yêu cầu TẤT CẢ keyword phải khớp - quá cứng nhắc vì LLM có
    thể diễn đạt lại (đổi định dạng số, bỏ bớt chi tiết phụ...) mà vẫn trả
    lời đúng về mặt nội dung. Dùng tỉ lệ khớp giúp đánh giá công bằng hơn
    trong khi vẫn giữ được tín hiệu "trả lời có đúng trọng tâm hay không".

    So khớp trên bản đã chuẩn hoá (_normalize_for_match), không so trực
    tiếp trên answer/kw gốc - tránh chấm sai các câu trả lời ĐÚNG nội dung
    nhưng khác định dạng số hoặc khoảng trắng so với keyword kỳ vọng."""
    if not expected_keywords:
        return True
    answer_norm = _normalize_for_match(answer)
    matched = sum(
        1 for kw in expected_keywords if _normalize_for_match(kw) in answer_norm
    )
    match_ratio = matched / len(expected_keywords)
    return match_ratio >= match_ratio_threshold


# Các cụm từ cho thấy bot đang THÀNH THẬT nói không biết / không có thông
# tin - đây là hành vi ĐÚNG theo SYSTEM_PROMPT, không phải hallucination.
DONT_KNOW_PHRASES = [
    "không có thông tin",
    "chưa có thông tin",
    "không biết",
    "tôi không biết",
    "không rõ",
    "không nắm được",
    "chưa cập nhật",
    "liên hệ hotline",
    "liên hệ nhân viên",
]


def is_dont_know_response(answer: str) -> bool:
    """Kiểm tra câu trả lời có phải dạng 'thành thật từ chối vì thiếu dữ
    liệu' hay không, dựa trên các cụm từ đặc trưng mà SYSTEM_PROMPT yêu
    cầu bot dùng khi [THÔNG TIN THAM KHẢO] không đủ."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in DONT_KNOW_PHRASES)


def check_hallucination(retriever_correct: bool, answer_correct: bool, answer: str) -> bool:
    """Hallucination: retriever đã tìm ĐÚNG context, câu trả lời KHÔNG khớp
    expected_keywords, VÀ câu trả lời đó không phải là một lời từ chối
    trung thực (không biết / không có thông tin).

    Trước đây: hallucination = retriever_correct AND NOT answer_correct.
    Vấn đề: nếu retriever tìm đúng context nhưng bot vẫn (đúng theo
    persona) trả lời "tôi không có thông tin này" vì câu hỏi hỏi về 1
    khía cạnh khác không có trong context đó, bot đang hành xử ĐÚNG chứ
    không phải bịa - không nên bị tính là hallucination."""
    if not retriever_correct or answer_correct:
        return False
    return not is_dont_know_response(answer)

# 3. VÒNG LẶP ĐÁNH GIÁ CHÍNH
def evaluate_single_case(case: dict) -> dict:
    """Chạy đánh giá cho 1 test case, trả về 1 dòng kết quả đầy đủ."""
    question = case["question"]
    expected_source = case["expected_source"]
    expected_keywords = case["expected_keywords"]

    # Step 1: Retrieve context
    # Step 2: Đo retrieval metadata (source của từng chunk trả về)
    retrieval = retrieve(question)
    retrieved_sources = get_retrieved_sources(retrieval["results"])
    retriever_correct = check_retriever_correct(
        expected_source, retrieved_sources, retrieval["context"]
    )
    precision = check_context_precision(expected_source, retrieved_sources)

    # Mỗi câu hỏi đánh giá ĐỘC LẬP - reset lịch sử hội thoại trước khi hỏi,
    # tránh câu trả lời bị ảnh hưởng bởi các câu hỏi trước đó trong vòng lặp.
    reset_history()

    # Step 3: Gọi ask() - pipeline đầy đủ (retrieval + generation)
    # Step 4: Đo latency
    start = time.perf_counter()
    answer = ask(question)
    latency = time.perf_counter() - start

    # Step 5: Đánh giá answer + hallucination
    answer_correct = check_answer_correct(answer, expected_keywords)
    hallucinated = check_hallucination(retriever_correct, answer_correct, answer)

    return {
        "question": question,
        "expected_source": expected_source,
        "retrieved_sources": retrieved_sources,
        "retriever_correct": retriever_correct,
        "context_precision": precision,
        "answer_correct": answer_correct,
        "hallucinated": hallucinated,
        "latency": latency,
        "answer": answer,
    }


def run_evaluation() -> None:
    test_cases = load_test_cases()
    eval_results = [evaluate_single_case(case) for case in test_cases]

    write_report(eval_results)
    print_summary(eval_results)


# 4. OUTPUT: CSV + TERMINAL SUMMARY
def write_report(eval_results: list[dict], path: str = REPORT_PATH) -> None:
    """Ghi chi tiết từng câu hỏi ra CSV để review thủ công."""
    fieldnames = [
        "Question",
        "Expected Source",
        "Retrieved Sources",
        "Retriever Correct",
        "Answer Correct",
        "Hallucination",
        "Latency",
        "Answer",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in eval_results:
            writer.writerow({
                "Question": r["question"],
                "Expected Source": r["expected_source"],
                "Retrieved Sources": ", ".join(r["retrieved_sources"]) or "(none)",
                "Retriever Correct": r["retriever_correct"],
                "Answer Correct": r["answer_correct"],
                "Hallucination": r["hallucinated"],
                "Latency": round(r["latency"], 3),
                "Answer": r["answer"],
            })
    print(f"Đã ghi báo cáo chi tiết vào {path}\n")


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def print_summary(eval_results: list[dict]) -> None:
    total = len(eval_results)

    retriever_accuracy = _avg([1.0 if r["retriever_correct"] else 0.0 for r in eval_results])
    precisions = [r["context_precision"] for r in eval_results if r["context_precision"] is not None]
    context_precision = _avg(precisions)
    answer_accuracy = _avg([1.0 if r["answer_correct"] else 0.0 for r in eval_results])
    hallucination_rate = _avg([1.0 if r["hallucinated"] else 0.0 for r in eval_results])
    avg_latency = _avg([r["latency"] for r in eval_results])

    print("=" * 48)
    print("RAG Evaluation Report")
    print("=" * 48)
    print(f"Total Questions        : {total}")
    print(f"Retriever Accuracy      : {retriever_accuracy:.1%}")
    print(f"Context Precision       : {context_precision:.1%}")
    print(f"Answer Accuracy         : {answer_accuracy:.1%}")
    print(f"Hallucination Rate      : {hallucination_rate:.1%}")
    print(f"Average Response Time   : {avg_latency:.2f}s")
    print("=" * 48)


if __name__ == "__main__":
    run_evaluation()
