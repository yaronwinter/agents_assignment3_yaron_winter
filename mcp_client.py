"""
Minimal MCP client that connects to the BiText FastMCP server over stdio,
lists the exposed tools, and calls a few of them.

This demonstrates the full client side of the connection:
  1. Launch the server as a subprocess via stdio transport.
  2. Open an MCP ``ClientSession`` and run the initialize handshake.
  3. Discover the available tools (``list_tools``).
  4. Invoke tools by name with arguments (``call_tool``).

Run it with the same interpreter that has the ``mcp`` package installed:

    python mcp_client.py
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = Path(__file__).resolve().parent / "mcp_server.py"


def _text(result) -> str:
    """Flatten a CallToolResult's content blocks into plain text."""
    parts = []
    for block in result.content:
        parts.append(getattr(block, "text", str(block)))
    return "\n".join(parts)


async def main() -> None:
    # Launch the server with the SAME interpreter running this client so it
    # inherits the environment where `mcp` and the langgraph stack are installed.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Handshake.
            await session.initialize()

            # 2. Discover tools.
            tools = (await session.list_tools()).tools
            print(f"Connected. Server exposes {len(tools)} tools:")
            for t in tools:
                print(f"  - {t.name}: {t.description.splitlines()[0]}")

            # 3. Call a no-argument tool.
            print("\n> list_categories()")
            res = await session.call_tool("list_categories", {})
            print(_text(res))

            # 4. Call a tool with arguments.
            print("\n> count_category({'category': 'REFUND'})")
            res = await session.call_tool("count_category", {"category": "REFUND"})
            print(_text(res))

            print("\n> get_distribution_of_intents_for_category({'category': 'ACCOUNT'})")
            res = await session.call_tool(
                "get_distribution_of_intents_for_category", {"category": "ACCOUNT"}
            )
            print(_text(res))


if __name__ == "__main__":
    asyncio.run(main())
