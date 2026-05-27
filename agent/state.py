from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    question: str
    route: str
    messages: Annotated[List, add_messages]
    iterations: int
    answer: str
