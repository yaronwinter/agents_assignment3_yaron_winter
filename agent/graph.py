from typing import Dict
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent import router
from agent.react_nodes import structured_react_node, unstructured_react_node

MAX_ITERATIONS = 10

def route_question(state: AgentState) -> Dict[str, str]:
    """The router node, which routes the question to the appropriate agent based on the question content."""
    result = router.router_llm.invoke([
        ("system", router.ROUTER_PROMPT),
        ("human", state["question"])
    ])

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
graph.add_node("router", route_question)
graph.add_node("structured_react", structured_react_node)
graph.add_node("unstructured_react", unstructured_react_node)
graph.add_node("oos", out_of_scope_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda s: s["route"],
    {
        router.STRUCTURED: "structured_react",
        router.UNSTRUCTURED: "unstructured_react",
        router.OUT_OF_SCOPE: "oos"
    }
)

graph.add_conditional_edges(
    "structured_react",
     should_continue,
    {
        "end": END,
        "stop": END
    }
)

graph.add_conditional_edges(
    "unstructured_react",
    should_continue,
    {
        "end": END,
        "stop": END
    }
)


graph.add_edge("oos", END)

app = graph.compile()
