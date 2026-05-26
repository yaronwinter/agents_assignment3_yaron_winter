from typing import TypedDict, List, Any

class AgentState(TypedDict):
    question: str
    route: str
    messages: List[Any]
    iterations: int
    answer: str
