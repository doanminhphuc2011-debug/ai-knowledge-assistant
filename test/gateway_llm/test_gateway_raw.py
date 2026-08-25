from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"


def main() -> None:
    load_dotenv(ENV_PATH, override=False)

    client = OpenAI(
        api_key=os.environ["LITELLM_MASTER_KEY"],
        base_url=os.getenv("LLM_GATEWAY_URL", "http://localhost:4000/v1"),
    )

    response = client.chat.completions.create(
        model=os.getenv("LLM_GATEWAY_MODEL", "dmp-chat"),
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=256,
    )

    message = response.choices[0].message

    print("✓ Gateway HTTP request: PASS")
    print(f"✓ model: {response.model}")
    print(f"✓ content: {message.content!r}")
    print(f"✓ finish_reason: {response.choices[0].finish_reason}")

    if getattr(message, "reasoning", None):
        print("✓ reasoning field: present")

    if not message.content:
        raise RuntimeError(
            "Gateway trả về content rỗng. "
            "Nếu provider là GPT-OSS, kiểm tra include_reasoning=false "
            "và giới hạn output trong config."
        )


if __name__ == "__main__":
    main()
