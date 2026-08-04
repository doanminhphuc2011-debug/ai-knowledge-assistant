"""
memory.py
Quản lý conversation memory (lịch sử hội thoại).

LLM không tự nhớ gì cả - mỗi lần gọi là 1 request độc lập. Muốn "nhớ" thì
phải tự lưu lịch sử và gửi kèm mỗi lần gọi. Module này gói gọn TOÀN BỘ
trách nhiệm đó (lưu, trim, ghép context RAG cho 1 lượt gọi) để chatbot.py
không cần biết chi tiết cấu trúc list message được quản lý ra sao.

Tách riêng khỏi chatbot.py vì đây là 1 trách nhiệm rõ ràng, độc lập với
việc gọi LLM/tool - dễ test riêng, và nếu sau này muốn đổi cách lưu trữ
(vd. Redis/DB theo session thay vì biến global trong RAM) thì chỉ cần sửa
file này, không đụng vào chatbot.py hay tool_executor.py.

Hành vi giữ NGUYÊN so với trước khi tách: cùng MAX_HISTORY, cùng cách trim,
cùng cách ghép context RAG vào câu hỏi.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from prompts import SYSTEM_PROMPT

MAX_HISTORY = 20  # số message tối đa giữ lại (không tính system prompt)

# State của phiên chat hiện tại. Luôn có SystemMessage ở vị trí đầu tiên.
conversation_history: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]


def _trim() -> None:
    """Chỉ giữ N tin nhắn gần nhất, luôn giữ system prompt ở đầu.
    Tránh context bị tràn / tốn token khi hội thoại quá dài."""
    global conversation_history
    system_msg = conversation_history[0]
    recent = conversation_history[1:][-MAX_HISTORY:]
    conversation_history = [system_msg] + recent


def append_user_message(question: str) -> None:
    """Lưu câu hỏi GỐC (không kèm context RAG) vào lịch sử rồi trim ngay -
    để lịch sử gọn, tránh phình to / lặp lại context RAG nhiều lần qua các
    lượt chat (context RAG chỉ dùng cho ĐÚNG lượt gọi hiện tại, xem
    build_augmented_message() + build_llm_messages() bên dưới)."""
    conversation_history.append(HumanMessage(content=question))
    _trim()


def append_ai_message(answer: str) -> None:
    """Lưu câu trả lời CUỐI CÙNG của AI vào lịch sử để nhớ cho lượt sau.
    Không lưu các bước tool_call trung gian vào đây (xem tool_executor.py)
    để lịch sử gọn, tránh phình to."""
    conversation_history.append(AIMessage(content=answer))


def build_augmented_message(user_input: str, context: str) -> str:
    """Ghép câu hỏi user với context RAG lấy được từ Qdrant."""
    if context:
        return f"{user_input}\n\n[THÔNG TIN THAM KHẢO]\n{context}"
    return f"{user_input}\n\n[THÔNG TIN THAM KHẢO]\n(không tìm thấy thông tin liên quan trong dữ liệu quán)"


def build_llm_messages(augmented_input: str) -> list[BaseMessage]:
    """Ghép lịch sử hội thoại hiện có (KHÔNG tính câu hỏi gốc vừa append,
    vì bản GỐC không có context) với bản câu hỏi ĐÃ AUGMENT context RAG
    cho lượt gọi hiện tại. Bản augment này chỉ dùng để gửi cho LLM ở lượt
    này, KHÔNG được lưu vào conversation_history."""
    return conversation_history[:-1] + [HumanMessage(content=augmented_input)]


def reset() -> None:
    """Đưa lịch sử hội thoại về trạng thái ban đầu (chỉ còn system prompt).

    LƯU Ý: module này KHÔNG reset giỏ hàng - giỏ hàng là state riêng do
    tools.py quản lý. chatbot.reset_history() gọi cả memory.reset() lẫn
    tools.reset_cart() để 2 module không cần phụ thuộc chéo vào nhau."""
    global conversation_history
    conversation_history = [SystemMessage(content=SYSTEM_PROMPT)]
