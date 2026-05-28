# agents_assignment3_yaron_winter
Assignment 3 repo of the "From AI model to AI Product" class

## MCP server (FastMCP)

`mcp_server.py` exposes the BiText dataset operations as MCP tools using
[FastMCP](https://github.com/modelcontextprotocol/python-sdk). It reuses the
deterministic `BiTextDataset` helpers (the same ones backing the LangChain ReAct
tools), so any MCP client can discover and call them.

### Tools exposed

- `list_categories` — all dataset categories
- `list_intents` — all dataset intents
- `count_intent` — row count for an intent
- `count_category` — row count for a category
- `get_examples_by_intent` — example instruction/response pairs for an intent
- `get_category_distribution` — rows per category
- `get_intent_distribution` — rows per intent
- `get_distribution_of_intents_for_category` — intent breakdown within a category

### Run the server

The MCP SDK needs Python ≥ 3.10. This project's `agentic_task` conda env (Python
3.11) has everything installed:

```bash
/home/yaron/miniconda3/envs/agentic_task/bin/python mcp_server.py   # stdio transport
```

Or inspect it interactively with the MCP Inspector:

```bash
mcp dev mcp_server.py
```

### Connect a client

`mcp_client.py` launches the server over stdio, runs the initialize handshake,
lists the tools, and then drops into an interactive REPL so you can send your own
requests:

```bash
/home/yaron/miniconda3/envs/agentic_task/bin/python mcp_client.py          # interactive REPL
/home/yaron/miniconda3/envs/agentic_task/bin/python mcp_client.py --demo   # canned calls, then exit
```

REPL commands:

```text
mcp> list                                   # re-list available tools
mcp> help count_category                    # show a tool's input schema
mcp> list_categories                        # call a no-arg tool
mcp> count_category {"category": "REFUND"}  # call with JSON arguments
mcp> count_category category=REFUND         # ...or key=value shorthand
mcp> exit
```

To register the server with an MCP host (e.g. Claude Desktop), merge the
`mcpServers` block from `mcp_config.example.json` into the host's config file and
restart it.
