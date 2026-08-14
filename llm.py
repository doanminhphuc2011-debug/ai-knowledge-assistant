"""
llm.py
Khởi tạo các LLM (Groq, Gemini, Ollama local), bind Tool Calling cho từng
model, rồi ghép thành 1 chuỗi fallback duy nhất: Groq -> Gemini -> Local.

Tách riêng khỏi chatbot.py vì đây là logic KHỞI TẠO HẠ TẦNG (đọc API key,
cấu hình provider/model/temperature) - khác hẳn trách nhiệm "điều phối 1
lượt hỏi-đáp" của chatbot.py. Tách ra giúp:
- chatbot.py không bị dài dòng bởi phần cấu hình model.
- Muốn đổi provider/model/temperature chỉ cần sửa đúng 1 file này.
- Có thể import `llm` từ đây để dùng ở nơi khác (vd. script test riêng)
  mà không phải import cả chatbot.py (kéo theo cả RAG, tool executor...).

Hành vi giữ NGUYÊN so với trước khi tách: cùng model, cùng temperature,
cùng thứ tự bind_tools() trước rồi mới with_fallbacks() (bind trước để
đảm bảo dù model nào trả lời cũng có khả năng gọi tool).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.runnables import Runnable

from tools import ALL_TOOLS

# LOAD API KEY TỪ .env
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Thiếu GROQ_API_KEY trong file .env")

# KHỞI TẠO CÁC MODEL CHUỖI FALLBACK
_llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7)

_llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7)

_llm_local = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0.7)

# BIND TOOL CALLING
# Bind tool cho TỪNG model rồi mới ghép fallback (thay vì bind sau khi ghép)
# vì mỗi provider tự implement tool-calling theo cách riêng - bind trước
# đảm bảo dù model nào trả lời (groq/gemini/local) cũng gọi được tool.
_llm_groq = _llm_groq.bind_tools(ALL_TOOLS)
_llm_gemini = _llm_gemini.bind_tools(ALL_TOOLS)
_llm_local = _llm_local.bind_tools(ALL_TOOLS)

# GHÉP CHUỖI FALLBACK: Groq lỗi -> thử Gemini -> thử model local.
llm: Runnable = _llm_groq.with_fallbacks([_llm_gemini, _llm_local])
