"""
Interactive MCP client for the BiText FastMCP server (stdio transport).

It launches the server as a subprocess, runs the initialize handshake, then
drops into a REPL where you type a tool name + JSON arguments and see the result.

Run it with the interpreter that has the ``mcp`` package installed:

    python mcp_client.py            # interactive REPL
    python mcp_client.py --demo     # run a few canned calls then exit

REPL usage:
    mcp> list                                   # re-list available tools
    mcp> help count_category                    # show a tool's input schema
    mcp> list_categories                        # call a no-arg tool
    mcp> count_category {"category": "REFUND"}  # call with JSON arguments
    mcp> count_category category=REFUND         # ...or key=value shorthand
    mcp> exit
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = Path(__file__).resolve().parent / "mcp_server.py"


def _text(result) -> str:
    """Flatten a CallToolResult's content blocks into plain text."""
    parts = [getattr(block, "text", str(block)) for block in result.content]
    out = "\n".join(parts)
    if getattr(result, "isError", False):
        return f"[tool error] {out}"
    return out


def _print_tools(tools) -> None:
    print(f"\n{len(tools)} tools available:")
    for t in tools:
        params = list((t.inputSchema or {}).get("properties", {}).keys())
        sig = ", ".join(params) if params else ""
        first_line = (t.description or "").splitlines()[0] if t.description else ""
        print(f"  - {t.name}({sig}) — {first_line}")


def _parse_args(arg_str: str) -> dict:
    """Parse the argument portion of a REPL line into a dict.

    Accepts either a JSON object (`{"category": "REFUND"}`) or simple
    whitespace-separated key=value pairs (`category=REFUND limit=5`).
    """
    arg_str = arg_str.strip()
    if not arg_str:
        return {}
    if arg_str.startswith("{"):
        return json.loads(arg_str)

    args = {}
    for token in arg_str.split():
        if "=" not in token:
            raise ValueError(f"expected key=value, got {token!r}")
        key, value = token.split("=", 1)
        # Coerce ints so e.g. limit=5 arrives as a number, not a string.
        try:
            value = int(value)
        except ValueError:
            pass
        args[key] = value
    return args


async def run_demo(session: ClientSession) -> None:
    """A few canned calls, useful as a smoke test (`--demo`)."""
    for name, args in [
        ("list_categories", {}),
        ("count_category", {"category": "REFUND"}),
        ("get_distribution_of_intents_for_category", {"category": "ACCOUNT"}),
    ]:
        print(f"\n> {name}({args})")
        print(_text(await session.call_tool(name, args)))


async def run_repl(session: ClientSession, tools) -> None:
    tool_names = {t.name for t in tools}
    by_name = {t.name: t for t in tools}
    print("\nType a tool name to call it. 'list' to see tools, 'help <tool>' "
          "for a tool's schema, 'exit' to quit.")

    while True:
        # Read input off the event loop so the stdio reader tasks keep running.
        try:
            line = (await asyncio.to_thread(input, "\nmcp> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue

        head, _, rest = line.partition(" ")

        if head in ("exit", "quit"):
            return
        if head == "list":
            _print_tools(tools)
            continue
        if head == "help":
            target = rest.strip()
            if target in by_name:
                schema = by_name[target].inputSchema or {}
                print(json.dumps(schema.get("properties", {}), indent=2))
            else:
                print(f"Unknown tool: {target!r}")
            continue

        if head not in tool_names:
            print(f"Unknown tool: {head!r}  (type 'list' to see available tools)")
            continue

        try:
            args = _parse_args(rest)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Could not parse arguments: {e}")
            continue

        try:
            result = await session.call_tool(head, args)
        except Exception as e:  # noqa: BLE001 - surface any protocol/validation error
            print(f"Call failed: {e}")
            continue
        print(_text(result))


async def main() -> None:
    demo = "--demo" in sys.argv[1:]

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print(f"Connected to '{SERVER_PATH.name}'.")
            _print_tools(tools)

            if demo:
                await run_demo(session)
            else:
                await run_repl(session, tools)


if __name__ == "__main__":
    asyncio.run(main())
