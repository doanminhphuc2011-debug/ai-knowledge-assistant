"""
Chatbot RAG - Python + LangChain + Groq + Qdrant
Gồm: System Prompt (Persona Pattern) + RAG (Retrieval-Augmented Generation)
+ Conversation Memory (có trim history) + Tool Calling (giỏ hàng)

File này CHỈ còn đóng vai trò ĐIỀU PHỐI (entry point) - toàn bộ logic chi
tiết đã được tách sang các module chuyên trách, mỗi module 1 trách nhiệm:
- llm.py           : khởi tạo LLM (Groq/Gemini/Local) + fallback + bind tool
- prompts.py       : system prompt (persona + quy tắc RAG + quy tắc tool)
- memory.py        : lịch sử hội thoại (lưu/trim/ghép context RAG)
- tool_executor.py : vòng lặp gọi tool khi model yêu cầu
- rag.py           : truy xuất context từ Qdrant (không đổi từ trước)
- tools.py         : định nghĩa tool + giỏ hàng (không đổi từ trước)
"""
import warnings
warnings.filterwarnings("ignore")  # Ẩn toàn bộ warning trong hệ thống

import memory
from llm import llm
from rag import retrieve_context
from tool_executor import generate_with_tools
from tools import reset_cart


# 1. GỌI LLM (RAG + MEMORY + TOOL CALLING)
def ask(question: str) -> str:
    """Hàm chính: nhận câu hỏi, trả về câu trả lời của chatbot.
    Đây là entrypoint mà cả CLI lẫn evaluate.py đều dùng chung, để không
    có 2 nơi implement logic RAG + generate khác nhau.

    Luồng xử lý (giữ nguyên thứ tự như trước khi tách module, chỉ khác là
    mỗi bước giờ được ủy quyền cho đúng module chuyên trách):
        a) RETRIEVE  : lấy context liên quan từ Qdrant           -> rag.py
        b) MEMORY    : lưu câu hỏi gốc (không kèm context)       -> memory.py
        c) GENERATE  : gọi LLM, tự xử lý Tool Calling nếu cần    -> llm.py + tool_executor.py
        d) MEMORY    : lưu câu trả lời cuối cùng                 -> memory.py
    """
    # a) RETRIEVE: lấy context liên quan nhất từ Qdrant
    context = retrieve_context(question)
    augmented_input = memory.build_augmented_message(question, context)

    # b) Lưu câu hỏi GỐC (không kèm context) vào lịch sử - để lịch sử gọn,
    # tránh phình to / lặp lại context nhiều lần qua các lượt chat.
    memory.append_user_message(question)

    # c) GENERATE + TOOL CALLING: gửi lịch sử hội thoại (giữ nguyên context
    # các lượt trước) + câu hỏi hiện tại đã augment context RAG (chỉ dùng
    # bản augment cho lượt gọi này, không lưu vào history). Nếu model cần
    # gọi tool (add_to_cart, checkout...), generate_with_tools() tự lo toàn
    # bộ vòng lặp thực thi tool và trả về câu trả lời cuối cùng.
    messages_for_llm = memory.build_llm_messages(augmented_input)
    answer = generate_with_tools(llm, messages_for_llm)

    # d) Lưu câu trả lời CUỐI CÙNG (đã tổng hợp từ tool, nếu có) vào memory để nhớ cho lượt sau.
    memory.append_ai_message(answer)

    return answer


def reset_history() -> None:
    """Đưa lịch sử hội thoại về trạng thái ban đầu (chỉ còn system prompt)
    VÀ làm trống giỏ hàng - vì giỏ hàng cũng là state của phiên chat, nếu
    không reset cùng lúc thì 1 phiên mới có thể "thừa hưởng" giỏ hàng của
    phiên trước (vd. giữa các câu hỏi độc lập trong evaluate.py)."""
    memory.reset()
    reset_cart()


# Giữ tên hàm cũ để tương thích ngược - bất kỳ code nào đang gọi chat()
chat = ask


# 2. VÒNG LẶP CHAT (CLI)
if __name__ == "__main__":
    print("☕ Ori - Trợ lý quán cà phê DMP")
    print("Nhập 'exit' để thoát.")
    while True:
        user_input = input("👤 Bạn: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        answer = ask(user_input)
        print(f"Ori: {answer}\n")
