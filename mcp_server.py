"""
FastMCP server exposing the BiText dataset tools over the Model Context Protocol.

This server reuses the deterministic ``BiTextDataset`` helpers directly (the same
ones backing the LangChain ReAct tools in ``agent/struct_tools.py`` and
``agent/unstruct_tools.py``) and re-exposes them as MCP tools so any MCP client
(Claude Desktop, the bundled ``mcp_client.py``, MCP Inspector, etc.) can discover
and call them.

Run it over stdio (the default MCP transport):

    python mcp_server.py

Or, equivalently, with the MCP CLI / Inspector:

    mcp dev mcp_server.py

The tool functions are thin wrappers: all real work happens in ``BiTextDataset``,
keeping the logic deterministic and Python-side rather than delegating to an LLM.
"""

from pathlib import Path
from typing import Dict, List

from mcp.server.fastmcp import FastMCP

from agent.dataset import BiTextDataset

# Load the dataset once at startup and share it across all tool calls.
DATA_PATH = Path(__file__).resolve().parent / "data" / "bitext.csv"
dataset = BiTextDataset(str(DATA_PATH))

# The MCP server instance. The name is what clients see during the handshake.
mcp = FastMCP("bitext-dataset")


@mcp.tool()
def list_categories() -> List[str]:
    """Return all unique categories present in the BiText dataset."""
    return dataset.get_all_categories()


@mcp.tool()
def list_intents() -> List[str]:
    """Return all unique intents present in the BiText dataset."""
    return dataset.get_all_intents()


@mcp.tool()
def count_intent(intent: str) -> int:
    """Count how many rows belong to a specific intent (case-insensitive)."""
    return dataset.count_by_intent(intent)


@mcp.tool()
def count_category(category: str) -> int:
    """Count how many rows belong to a specific category (case-insensitive)."""
    return dataset.count_by_category(category)


@mcp.tool()
def get_examples_by_intent(intent: str, limit: int = 3) -> List[Dict[str, str]]:
    """Return up to ``limit`` example rows (instruction/response pairs) for an intent."""
    return dataset.get_examples_by_intent(intent, limit)


@mcp.tool()
def get_category_distribution() -> Dict[str, int]:
    """Return the number of rows per category across the whole dataset."""
    return dataset.get_category_distribution()


@mcp.tool()
def get_intent_distribution() -> Dict[str, int]:
    """Return the number of rows per intent across the whole dataset."""
    return dataset.get_intent_distribution()


@mcp.tool()
def get_distribution_of_intents_for_category(category: str) -> Dict[str, int]:
    """Return the intent distribution within a single category."""
    return dataset.get_distribution_of_intents_for_category(category)


if __name__ == "__main__":
    # Default MCP transport is stdio: the client launches this process and
    # communicates over stdin/stdout.
    mcp.run()
