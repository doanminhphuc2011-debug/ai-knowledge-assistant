from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"


def _load_environment() -> None:
    if not ENV_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy file .env: {ENV_PATH}")
    load_dotenv(ENV_PATH, override=False)


def _extract_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts).strip()

    return str(content).strip() if content else ""


def main() -> None:
    _load_environment()

    gateway_url = os.getenv("LLM_GATEWAY_URL", "http://localhost:4000/v1")
    model = os.getenv("LLM_GATEWAY_MODEL", "dmp-chat")
    key = os.getenv("LITELLM_MASTER_KEY")

    if not key:
        raise RuntimeError("Thiếu LITELLM_MASTER_KEY trong .env")

    client = ChatOpenAI(
        model=model,
        api_key=key,
        base_url=gateway_url,
        temperature=0,
        max_tokens=256,
    )

    response = client.invoke(
        [HumanMessage(content="Trả lời đúng một từ: OK")]
    )

    text = _extract_text(response.content)

    print(f"✓ Gateway reachable")
    print(f"✓ Model alias: {model}")
    print(f"✓ Raw content type: {type(response.content).__name__}")
    print(f"✓ Response: {text or '<empty>'}")

    if not text:
        raise RuntimeError(
            "Gateway trả về message không có text. "
            "Kiểm tra terminal LiteLLM để xem provider/model đã trả gì."
        )


if __name__ == "__main__":
    main()
