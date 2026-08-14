"""
tool_executor.py
Vòng lặp thực thi Tool Calling: gọi LLM; nếu LLM yêu cầu gọi tool thì thực
thi tool THẬT rồi gửi kết quả lại cho LLM; lặp lại cho đến khi LLM trả về
câu trả lời cuối cùng (không còn tool_calls) hoặc chạm giới hạn số vòng lặp.

Tách riêng khỏi chatbot.py vì đây là 1 trách nhiệm khá phức tạp và độc lập
(xử lý lỗi tool, giới hạn vòng lặp, map tên tool -> function) - gộp chung
vào ask() sẽ làm hàm đó quá dài và trộn lẫn nhiều mối quan tâm khác nhau
(RAG, memory, tool calling) trong cùng 1 hàm.

Hành vi giữ NGUYÊN so với trước khi tách: cùng MAX_TOOL_ITERATIONS, cùng
cách xử lý 3 loại lỗi tool (unknown_tool / invalid_arguments /
tool_execution_error), cùng câu trả lời fallback khi chạm giới hạn.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from tools import ALL_TOOLS, error_response

MAX_TOOL_ITERATIONS = 5  # chặn vòng lặp tool-call vô hạn nếu model "loop"

# Map tên tool -> function thật, dùng để thực thi khi model trả về tool_calls
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

_FALLBACK_ANSWER = (
    "Xin lỗi, hiện tôi đang xử lý yêu cầu này chưa xong. "
    "Bạn vui lòng thử lại hoặc nói rõ hơn giúp tôi nhé."
)


def _extract_text(content: Any) -> str:
    """Chuẩn hoá `AIMessage.content` về `str` thuần trước khi trả ra ngoài
    cho người dùng/memory/evaluate.py.

    TẠI SAO CẦN HÀM NÀY: `response.content` KHÔNG LUÔN LÀ str như type hint
    của generate_with_tools() khai báo - đây là giả định ĐÚNG với Groq
    (luôn trả content dạng str), nhưng SAI với Gemini
    (`ChatGoogleGenerativeAI`) - khi with_fallbacks() rơi từ Groq xuống
    Gemini, `response.content` có thể là `list[dict]` (nhiều content block,
    vd. `[{"type": "text", "text": "..."}]`) thay vì string thuần. Nếu
    không chuẩn hoá, người dùng cuối có nguy cơ nhận nguyên str() của cả
    list (vd. "[{'type': 'text', 'text': '...'}]") thay vì câu trả lời tự
    nhiên - đã tái hiện được lỗi này qua evaluate.py
    (AttributeError: 'list' object has no attribute 'lower').

    Xử lý cả 2 dạng block phổ biến: block là str thuần, hoặc block là dict
    có field "text" (chuẩn content block của Gemini/Anthropic-style)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _execute_tool_call(tool_call: dict) -> str:
    """Thực thi 1 tool_call và LUÔN trả về một chuỗi JSON hợp lệ (không bao
    giờ raise exception ra ngoài) - vì kết quả này sẽ được đưa thẳng vào
    ToolMessage gửi cho LLM, nên phải là dữ liệu model đọc được để tự phục
    hồi (recover) thay vì làm sập cả chương trình.

    3 loại lỗi được xử lý riêng biệt:
    - unknown_tool: model gọi tên tool không tồn tại (hiếm khi xảy ra vì
      TOOLS_BY_NAME được build từ đúng ALL_TOOLS đã bind, nhưng vẫn nên
      phòng hờ).
    - invalid_arguments: tool tồn tại nhưng argument model truyền vào sai
      kiểu / thiếu field / vi phạm ràng buộc (vd. quantity <= 0 ở schema
      OrderItem) -> pydantic raise ValidationError khi LangChain validate
      args trước khi gọi hàm thật.
    - tool_execution_error: lỗi phát sinh trong lúc tool đang chạy (vd. lỗi
      đọc file menu.json) - không nên để crash cả server chatbot.
    """
    tool_name = tool_call["name"]
    tool_fn = TOOLS_BY_NAME.get(tool_name)

    if tool_fn is None:
        return error_response("unknown_tool", f"Không tìm thấy tool '{tool_name}'.")

    try:
        return tool_fn.invoke(tool_call["args"])
    except ValidationError as e:
        # Sai kiểu / thiếu field / vi phạm ràng buộc (vd. quantity <= 0 ở
        # schema OrderItem) - LangChain validate args bằng pydantic TRƯỚC
        # khi gọi hàm thật, nên lỗi này luôn là ValidationError.
        return error_response(
            "invalid_arguments",
            f"Tham số truyền vào tool '{tool_name}' không hợp lệ: {e.errors()}",
        )
    except Exception as e:  # noqa: BLE001 - cố tình bắt rộng để không bao giờ crash vòng lặp chat
        return error_response("tool_execution_error", f"Lỗi khi thực thi tool '{tool_name}': {e}")


def generate_with_tools(llm: Runnable, messages: list[BaseMessage]) -> str:
    """Gửi `messages` cho `llm`, tự động thực thi các tool được yêu cầu
    (nếu có) và lặp lại cho đến khi nhận được câu trả lời cuối cùng.

    Đây chính là 2 bước GENERATE + TOOL CALLING trước đây nằm trong ask()
    của chatbot.py - tách ra để chatbot.py chỉ còn đóng vai trò điều phối
    (orchestration), không chứa logic chi tiết của vòng lặp tool.

    Lặp (không phải if) vì model có thể cần vài vòng gọi tool trước khi ra
    câu trả lời cuối (vd. gọi add_to_cart rồi tự gọi tiếp view_cart để xác
    nhận lại với khách). Giới hạn MAX_TOOL_ITERATIONS để tránh loop vô hạn
    nếu model bị kẹt.

    Trả về: chuỗi câu trả lời cuối cùng cho khách (response.content), hoặc
    câu trả lời fallback an toàn nếu chạm giới hạn số vòng lặp.
    """

    response: AIMessage = llm.invoke(messages)

    iterations = 0
    while response.tool_calls and iterations < MAX_TOOL_ITERATIONS:
        iterations += 1
        messages = messages + [response]
        for tool_call in response.tool_calls:
            tool_result = _execute_tool_call(tool_call)
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
        response = llm.invoke(messages)

    if iterations >= MAX_TOOL_ITERATIONS and response.tool_calls:
        # Model vẫn muốn gọi tool tiếp sau khi đã chạm giới hạn - dừng lại
        # và trả 1 câu xin lỗi an toàn thay vì lặp vô hạn hoặc trả rỗng.
        return _FALLBACK_ANSWER

    return _extract_text(response.content)
