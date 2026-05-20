from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from langchain_core.messages import HumanMessage

from state import AgentState
from router import router_llm, ROUTER_PROMPT
from prompts import SYSTEM_PROMPT
from tools import (
    list_categories,
    count_intent,
    get_examples,
    get_intent_distribution,
    get_category_documents,
    get_agent_responses,
)

MAX_ITERATIONS = 10

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

tools = [
    list_categories,
    count_intent,
    get_examples,
    get_intent_distribution,
    get_category_documents,
    get_agent_responses,
]


react_agent = create_react_agent(
    llm=llm,
    tools=tools,
)


# ---------------------------------------------------
# ROUTER NODE
# ---------------------------------------------------

def route_question(state: AgentState):

    result = router_llm.invoke([
        ("system", ROUTER_PROMPT),
        ("human", state["question"])
    ])

    return {
        "route": result.route
    }


# ---------------------------------------------------
# OOS NODE
# ---------------------------------------------------

def out_of_scope_node(state: AgentState):
    return {
        "answer": (
            "This question is outside the scope "
            "of the BiText dataset."
        )
    }


# ---------------------------------------------------
# REACT NODE
# ---------------------------------------------------

def react_node(state: AgentState):

    result = react_agent.invoke({
        "messages": [
            HumanMessage(content=state["question"])
        ]
    })

    return {
        "answer": result["messages"][-1].content,
        "iterations": state["iterations"] + 1
    }


# ---------------------------------------------------
# ITERATION GUARD
# ---------------------------------------------------

def should_continue(state: AgentState):

    if state["iterations"] >= MAX_ITERATIONS:
        return "stop"

    return "end"


# ---------------------------------------------------
# BUILD GRAPH
# ---------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("router", route_question)
graph.add_node("react", react_node)
graph.add_node("oos", out_of_scope_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda s: s["route"],
    {
        "structured": "react",
        "unstructured": "react",
        "out_of_scope": "oos"
    }
)

graph.add_conditional_edges(
    "react",
    should_continue,
    {
        "end": END,
        "stop": END
    }
)

graph.add_edge("oos", END)

app = graph.compile()