"""
Chatbot cơ bản - Python + LangChain + Groq
Gồm: System Prompt (Persona Pattern) + Conversation Memory (có trim history)
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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

# 3. ROOT PROMPT / SYSTEM PROMPT (Persona Pattern)
# Đây là nơi định hình "nhân cách" của bot: vai trò, tính cách,
# giới hạn được phép trả lời. Đổi nội dung ở đây để đổi persona.
SYSTEM_PROMPT = """Bạn là Ori, trợ lý bán thân thiện dữ của quán cà phê DMP.
- Trả lời ngắn gọn, thân thiện, xưng "tôi" gọi khách là "bạn".
- Chỉ tư vấn về menu, giá cả, khuyến mãi.
- Nếu không biết thông tin, "thành thật nói không biết", không bịa đặt.
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


# 5. GỌI LLM
def chat(user_input: str) -> str:
    # Thêm câu hỏi user vào memory
    conversation_history.append(HumanMessage(content=user_input))
    trim_history()

    # Gửi TOÀN BỘ lịch sử cho LLM, không chỉ câu hỏi hiện tại
    response = llm.invoke(conversation_history)

    # Lưu câu trả lời của AI vào memory để nhớ cho lượt sau
    conversation_history.append(AIMessage(content=response.content))

    return response.content


# 6. VÒNG LẶP CHAT (CLI)
if __name__ == "__main__":
    print("Chat với Ori (gõ 'exit' để thoát)\n")
    while True:
        user_input = input("Bạn: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        answer = chat(user_input)
        print(f"Ori: {answer}\n")
