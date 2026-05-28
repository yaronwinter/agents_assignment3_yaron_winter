import os
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent 
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState
from agent.prompts import (
    STRUCTURED_SYSTEM_PROMPT,
    UNSTRUCTURED_SYSTEM_PROMPT,
    PERSONAL_SYSTEM_PROMPT
)
from agent import struct_tools, unstruct_tools, display

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
    struct_tools.get_distribution_of_intents_for_category,
    struct_tools.get_distribution_of_categories_for_intent,
]

unstructured_tools = [
    unstruct_tools.get_docs_by_category,
    unstruct_tools.get_docs_by_intent,
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

"""
The personal agent.
"""
personal_react_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt=PERSONAL_SYSTEM_PROMPT
)

CLI_MODE = os.environ.get("CLI_MODE") == "1"

def _run_react(agent, state: AgentState, label: str) -> Dict[str, object]:
    """Invoke a ReAct sub-agent with accumulated history and return only the delta."""
    prior = state.get("messages") or []
    user_msg = HumanMessage(content=state["question"])

    # Surface the per-user profile to the sub-agent as an extra system
    # message. The agent's built-in system prompt (dataset instructions)
    # remains in place; this is appended context, not a replacement.
    result = agent.invoke({"messages": list(prior) + [user_msg]})
    all_messages = result["messages"]

    # Persist only the conversation thread (user turn + final answer). Tool
    # calls and intermediate reasoning are scratch work — keeping them in
    # state would inflate every subsequent sub-agent prompt for no gain on
    # follow-up resolution. CLI display still shows the full chain below.
    final_ai = all_messages[-1]
    new_messages = [user_msg, final_ai]

    if CLI_MODE:
        display.display_reasoning(label, all_messages)

    return {
        "answer": final_ai.content,
        "messages": new_messages,
        "iterations": state["iterations"] + 1
    }


def structured_react_node(state: AgentState) -> Dict[str, object]:
    """The structured React node, which invokes the ReAct agent to process the question."""
    return _run_react(structured_react_agent, state, "Structured")


def unstructured_react_node(state: AgentState) -> Dict[str, object]:
    """The unstructured React node, which invokes the ReAct agent to process the question."""
    return _run_react(unstructured_react_agent, state, "Unstructured")

def personal_node(state: AgentState) -> Dict[str, object]:
    """The personal node, which invokes the personal agent to answer questions about the user."""
    profile = (state.get("profile") or "").strip()
    if len(profile) == 0:
        return {
            "answer": (
                "I don't have a profile for you yet — tell me about yourself, "

            ),
            "messages": "",
            "iterations": state["iterations"] + 1
        }
    user_msg = HumanMessage(content=f"{state["question"]}\n\n{profile}")
    result = personal_react_agent.invoke({"messages": [user_msg]})
    messages = result["messages"]
    if CLI_MODE:
        display.display_reasoning("Personal", messages)

    return {
        "answer": messages[-1].content,
        "messages": messages,
        "iterations": state["iterations"] + 1
    }
