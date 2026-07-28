"""
Chatbot RAG - Python + LangChain + Groq + Qdrant
Gồm: System Prompt (Persona Pattern) + RAG (Retrieval-Augmented Generation)
+ Conversation Memory (có trim history)
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from rag import retrieve_context

# 1. LOAD API KEY TỪ .env
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Thiếu GROQ_API_KEY trong file .env")

# 2. KHỞI TẠO LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)

# 3. ROOT PROMPT / SYSTEM PROMPT
# Xác định vai trò (persona) của chatbot.
# Mọi câu trả lời phải ưu tiên dựa trên context do RAG truy xuất.
# Nếu context không đủ thông tin thì trả lời không biết, không suy đoán.
SYSTEM_PROMPT = """Bạn là Ori, trợ lý bán hàng thân thiện của quán cà phê DMP.

Nhiệm vụ:
- Trả lời ngắn gọn, tự nhiên, thân thiện.
- Xưng "tôi", gọi khách là "bạn".
- Chỉ trả lời các câu hỏi liên quan đến quán, menu, giá, khuyến mãi, thông tin cửa hàng và dịch vụ.

Quy tắc:
- Mỗi câu hỏi sẽ đi kèm phần [THÔNG TIN THAM KHẢO].
- Ưu tiên sử dụng thông tin trong [THÔNG TIN THAM KHẢO] để trả lời.
- Có thể diễn đạt lại bằng lời văn tự nhiên, nhưng không được thay đổi ý nghĩa hoặc bịa thêm thông tin.
- Nếu [THÔNG TIN THAM KHẢO] có đủ thông tin, hãy trả lời trực tiếp, không nói rằng "theo tài liệu" hay "theo thông tin tham khảo".
- Nếu [THÔNG TIN THAM KHẢO] không có hoặc không đủ thông tin để trả lời, hãy nói rõ rằng bạn không có thông tin và gợi ý khách liên hệ hotline hoặc nhân viên của quán.
- Không suy đoán, không tự tạo thông tin ngoài dữ liệu được cung cấp.
- Nếu khách hỏi về một chương trình khuyến mãi cụ thể (ví dụ: sinh viên, học sinh, sinh nhật, GrabFood...)
  nhưng trong [THÔNG TIN THAM KHẢO] không có chương trình đó,
  hãy nói rằng hiện tại chưa có thông tin về chương trình đó.

- Nếu [THÔNG TIN THAM KHẢO] có các chương trình khuyến mãi khác,
  hãy giới thiệu những chương trình đang áp dụng thay vì chỉ trả lời "không biết".
"""

# 4. CONVERSATION MEMORY
# LLM không tự nhớ gì cả - mỗi lần gọi là 1 request độc lập.
# Muốn "nhớ" thì phải tự lưu lịch sử và gửi kèm mỗi lần gọi.
conversation_history = [SystemMessage(content=SYSTEM_PROMPT)]

MAX_HISTORY = 20  # số message tối đa giữ lại (không tính system prompt)


def trim_history() -> None:
    """Chỉ giữ N tin nhắn gần nhất, luôn giữ system prompt ở đầu.
    Tránh context bị tràn / tốn token khi hội thoại quá dài."""
    global conversation_history
    system_msg = conversation_history[0]
    recent = conversation_history[1:][-MAX_HISTORY:]
    conversation_history = [system_msg] + recent


def build_augmented_message(user_input: str, context: str) -> str:
    """Ghép câu hỏi user với context RAG lấy được từ Qdrant."""
    if context:
        return f"{user_input}\n\n[THÔNG TIN THAM KHẢO]\n{context}"
    return f"{user_input}\n\n[THÔNG TIN THAM KHẢO]\n(không tìm thấy thông tin liên quan trong dữ liệu quán)"


# 5. GỌI LLM (RAG + MEMORY)
def ask(question: str) -> str:
    """Hàm chính: nhận câu hỏi, trả về câu trả lời của chatbot.
    Đây là entrypoint mà cả CLI lẫn evaluate.py đều dùng chung, để không
    có 2 nơi implement logic RAG + generate khác nhau."""
    # a) RETRIEVE: lấy context liên quan nhất từ Qdrant
    context = retrieve_context(question)
    augmented_input = build_augmented_message(question, context)

    # b) Lưu câu hỏi GỐC (không kèm context) vào lịch sử - để lịch sử gọn,
    # tránh phình to / lặp lại context nhiều lần qua các lượt chat.
    conversation_history.append(HumanMessage(content=question))
    trim_history()

    # c) GENERATE: gửi cho LLM lịch sử hội thoại + câu hỏi hiện tại đã
    # được augment với context (chỉ dùng bản augment cho lượt gọi này).
    messages_for_llm = conversation_history[:-1] + [HumanMessage(content=augmented_input)]
    response = llm.invoke(messages_for_llm)

    # d) Lưu câu trả lời của AI vào memory để nhớ cho lượt sau
    conversation_history.append(AIMessage(content=response.content))

    return response.content


def reset_history() -> None:
    """Đưa lịch sử hội thoại về trạng thái ban đầu (chỉ còn system prompt).
    Dùng khi bắt đầu phiên chat mới, hoặc khi evaluate.py cần chạy từng
    câu hỏi ĐỘC LẬP mà không bị ảnh hưởng bởi các câu hỏi trước đó."""
    global conversation_history
    conversation_history = [SystemMessage(content=SYSTEM_PROMPT)]


# Giữ tên hàm cũ để tương thích ngược - bất kỳ code nào đang gọi chat()
# (thay vì ask()) vẫn chạy đúng như trước, không cần sửa gì thêm.
chat = ask


# 6. VÒNG LẶP CHAT (CLI)
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
