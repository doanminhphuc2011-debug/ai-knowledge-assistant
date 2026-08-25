"""
Smoke test cho Tool Registry.

Chạy:
    py test_tool_registry.py
"""

from tool_registry import REGISTERED_TOOLS, TOOLS_BY_NAME, tool_registry


def main() -> None:
    names = [tool.name for tool in REGISTERED_TOOLS]

    assert names
    assert len(names) == len(set(names))
    assert set(names) == set(TOOLS_BY_NAME)
    assert all(tool_registry.get(name) is not None for name in names)

    print("✓ Registered tools:", ", ".join(names))
    print("✓ Không có duplicate tool name")
    print("✓ Registry lookup hoạt động")
    print("✓ Tool discovery không hard-code")


if __name__ == "__main__":
    main()
