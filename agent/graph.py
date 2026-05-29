import os
import sqlite3
from typing import Dict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from agent.state import AgentState
from agent.react_nodes import (
    structured_react_node,
    unstructured_react_node,
    personal_node
)
from agent import router

MAX_ITERATIONS = 10

CLI_MODE = os.environ.get("CLI_MODE") == "1"

STRUCTURED_NODE = "structured_react"
UNSTRUCTURED_NODE = "unstructured_react"
PERSONAL_NODE = "personal"
OUT_OF_SCOPE_NODE = "oos"
ROUTER_NODE = router.ROUTER

# How many of the most recent messages to surface to the router so it can
# resolve follow-ups like "what about refunds?" against the prior turn.
ROUTER_HISTORY_WINDOW = 6


def _format_history(messages) -> str:
    lines = []
    for m in messages[-ROUTER_HISTORY_WINDOW:]:
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        # Trim each message so the router prompt stays small.
        lines.append(f"{getattr(m, 'type', 'msg')}: {content[:300]}")
    return "\n".join(lines)


def route_question(state: AgentState) -> Dict[str, str]:
    """The router node, which routes the question to the appropriate agent based on the question content."""
    history = state.get("messages") or []
    if history:
        human_input = (
            f"Recent conversation:\n{_format_history(history)}\n\n"
            f"Current question: {state['question']}"
        )
    else:
        human_input = state["question"]

    result = router.router_llm.invoke([
        ("system", router.ROUTER_PROMPT),
        ("human", human_input)
    ])

    if CLI_MODE:
        print(f"Router: (result: {result.route})")

    return {
        "route": result.route
    }


def out_of_scope_node(state: AgentState) -> Dict[str, str]:
    """The out-of-scope node, which is returned when the question is outside the scope of the dataset."""
    return {
        "answer": (
            "This question is outside the scope "
            "of the BiText dataset."
        )
    }

def should_continue(state: AgentState) -> str:
    """A function that determines whether the agent should continue iterating or stop."""
    if state["iterations"] >= MAX_ITERATIONS:
        return "stop"

    return "end"


"""
The state graph definition, which defines the flow of the agent's reasoning process.
The agent starts at the router node, which routes the question to either the react node or the out-of-scope node.
The react node processes the question using the ReAct agent, and then determines whether to continue iterating or stop based on the number of iterations.
The out-of-scope node returns a polite refusal message when the question is outside the scope of the dataset.
"""
graph = StateGraph(AgentState)
graph.add_node(ROUTER_NODE, route_question)
graph.add_node(STRUCTURED_NODE, structured_react_node)
graph.add_node(UNSTRUCTURED_NODE, unstructured_react_node)
graph.add_node(OUT_OF_SCOPE_NODE, out_of_scope_node)
graph.add_node(PERSONAL_NODE, personal_node)

graph.set_entry_point(ROUTER_NODE)

graph.add_conditional_edges(
    ROUTER_NODE,
    lambda s: s["route"],
    {
        router.STRUCTURED: STRUCTURED_NODE,
        router.UNSTRUCTURED: UNSTRUCTURED_NODE,
        router.OUT_OF_SCOPE: OUT_OF_SCOPE_NODE,
        router.PERSONAL: PERSONAL_NODE,
    }
)

graph.add_conditional_edges(
    STRUCTURED_NODE,
     should_continue,
    {
        "end": END,
        "stop": END
    }
)

graph.add_conditional_edges(
    UNSTRUCTURED_NODE,
    should_continue,
    {
        "end": END,
        "stop": END
    }
)


graph.add_edge(OUT_OF_SCOPE_NODE, END)
graph.add_edge(PERSONAL_NODE, END)


def build_app(persist: bool = False):
    """Compile the graph.

    persist=True attaches a SQLite checkpointer scoped per thread_id
    (.sessions/agent.sqlite), so a conversation can be restored across runs.
    persist=False compiles without a checkpointer, so each invocation is
    stateless (no session memory).
    """
    if persist:
        _sessions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".sessions")
        os.makedirs(_sessions_dir, exist_ok=True)
        _conn = sqlite3.connect(
            os.path.join(_sessions_dir, "agent.sqlite"),
            check_same_thread=False,
        )
        return graph.compile(checkpointer=SqliteSaver(_conn))
    return graph.compile()


# Module-level app for LangGraph Studio / langgraph-api, which injects its own
# checkpointer. The CLI builds its own app via build_app() based on --session.
app = graph.compile()
