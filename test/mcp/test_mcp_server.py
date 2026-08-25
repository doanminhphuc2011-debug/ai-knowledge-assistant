from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "mcp_server.py"


async def main() -> None:
    if not SERVER.exists():
        raise FileNotFoundError(f"Không tìm thấy MCP server: {SERVER}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]

            expected = {
                "add_to_cart",
                "view_cart",
                "remove_from_cart",
                "update_cart",
                "clear_cart",
                "checkout",
            }

            assert expected.issubset(set(tool_names))

            print(f"✓ MCP tools: {', '.join(tool_names)}")

            schemas = {
                tool.name: tool.inputSchema
                for tool in tools_result.tools
            }

            assert schemas["add_to_cart"]
            assert schemas["view_cart"]

            print("✓ Tool schemas hợp lệ")

            result = await session.call_tool(
                "view_cart",
                arguments={},
            )

            if result.isError:
                raise RuntimeError(
                    f"view_cart lỗi: {result.content}"
                )

            print("✓ MCP call_tool(view_cart) thành công")

            result = await session.call_tool(
                "add_to_cart",
                arguments={
                    "items": [
                        {
                            "product_name": "Bạc Xỉu",
                            "size": "M",
                            "quantity": 1,
                        }
                    ]
                },
            )

            if result.isError:
                raise RuntimeError(
                    f"add_to_cart lỗi: {result.content}"
                )

            print("✓ MCP call_tool(add_to_cart) thành công")
            print("✓ MCP Server hoạt động đúng")


if __name__ == "__main__":
    asyncio.run(main())