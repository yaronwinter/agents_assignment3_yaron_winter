import os
from typing import Dict
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent.state import AgentState
from agent.router import router_llm, ROUTER_PROMPT
from agent.prompts import STRUCTURED_SYSTEM_PROMPT, UNSTRUCTURED_SYSTEM_PROMPT
from agent import struct_tools, unstruct_tools

"""
The ReAct LLM.
Used as the LLM for both the structured and unstructured agents.
I Chose nemotron-3-super-120b-a12b since it's a strong model, which designed for
instructions and reasoning tasks, and it is relatively not too expensive.
"""
llm = ChatOpenAI(
    base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
    api_key=os.environ["NEBIUS_API_KEY"],
    model="nvidia/nemotron-3-super-120b-a12b",
    temperature=0
)

structured_tools = [
    struct_tools.list_categories,
    struct_tools.list_intents,
    struct_tools.count_intent,
    struct_tools.count_category,
    struct_tools.get_intent_distribution,
    struct_tools.get_category_distribution,
    struct_tools.get_examples_by_intent,
    struct_tools.get_examples_by_category,
]

unstructured_tools = [
    unstruct_tools.get_docs_by_category,
    unstruct_tools.get_docs_by_intent,
    unstruct_tools.get_responses_by_intent,
    unstruct_tools.get_instructions_by_intent,
    unstruct_tools.get_responses_by_intent,
    unstruct_tools.get_responses_by_category,
    unstruct_tools.get_instructions_by_category
]

"""
The structured ReAct agent.
Provided with the system prompt and the tools defined above.
"""
structured_react_agent = create_react_agent(
    model=llm,
    tools=structured_tools,
    prompt=STRUCTURED_SYSTEM_PROMPT
)

"""
The unstructured ReAct agent.
Provided with the system prompt and the tools defined above.
"""
unstructured_react_agent = create_react_agent(
    model=llm,
    tools=unstructured_tools,
    prompt=UNSTRUCTURED_SYSTEM_PROMPT
)

CLI_MODE = os.environ.get("CLI_MODE") == "1"

def structured_react_node(state: AgentState) -> Dict[str, str]:
    """The structured React node, which invokes the ReAct agent to process the question."""
    result = structured_react_agent.invoke({
        "messages": [
            HumanMessage(content=state["question"])
        ]
    })

    messages = result["messages"]
    if CLI_MODE:
        display_reasoning(messages)

    return {
        "answer": messages[-1].content,
        "messages": messages,
        "iterations": state["iterations"] + 1
    }

def unstructured_react_node(state: AgentState) -> Dict[str, str]:
    """The unstructured React node, which invokes the ReAct agent to process the question."""
    result = unstructured_react_agent.invoke({
        "messages": [
            HumanMessage(content=state["question"])
        ]
    })

    messages = result["messages"]
    if CLI_MODE:
        display_reasoning(messages)

    return {
        "answer": messages[-1].content,
        "messages": messages,
        "iterations": state["iterations"] + 1
    }

def display_reasoning(messages):
    """Helper function to display the reasoning steps taken by the agent."""
    for m in messages:
        if isinstance(m, AIMessage):
            if m.tool_calls:
                for tc in m.tool_calls:
                    print(f"\nThought: Need tool {tc['name']}")
                    print(f"Action: {tc['name']}")
                    print(f"Input: {tc['args']}")
            elif m.content:
                print(f"\nFinal Answer: {m.content}")
        elif isinstance(m, ToolMessage):
            print(f"Observation: {m.content}")
