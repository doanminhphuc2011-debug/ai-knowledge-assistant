from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from tools import ALL_TOOLS


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"


def _load_environment() -> None:
    load_dotenv(ENV_PATH, override=False)


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
    ).bind_tools(ALL_TOOLS)

    response = client.invoke(
        [HumanMessage(content="Cho tôi 2 ly Bạc Xỉu size M.")]
    )

    if not response.tool_calls:
        raise RuntimeError("Gateway không phát sinh tool call.")

    names = [call["name"] for call in response.tool_calls]

    if "add_to_cart" not in names:
        raise RuntimeError(f"Tool call không đúng: {names}")

    print("✓ Gateway tool-calling: PASS")
    print(f"✓ Tool calls: {names}")


if __name__ == "__main__":
    main()
