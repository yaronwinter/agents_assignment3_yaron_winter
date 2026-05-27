from langchain_core.messages import AIMessage, ToolMessage
from typing import Iterable, Union

ANSWER_PREFIX_LENGTH = 50

def display_reasoning(node_name: str,messages: Iterable[Union[AIMessage, ToolMessage]]) -> None:
    """Helper function to display the reasoning steps taken by the agent."""
    print(f"\n--- Reasoning for {node_name} Node ---")
    for m in messages:
        if isinstance(m, AIMessage):
            if m.tool_calls:
                for tc in m.tool_calls:
                    print(f"\nThought: Need tool {tc['name']}")
                    print(f"Action: {tc['name']}")
                    print(f"Input: {tc['args']}")
            elif m.content:
                answer = m.content.strip()
                print(f"\nFinal Answer: {answer[:ANSWER_PREFIX_LENGTH]}...")
        elif isinstance(m, ToolMessage):
            content = f"{m.content}"
            print(f"Observation: {content[:ANSWER_PREFIX_LENGTH]}...")
