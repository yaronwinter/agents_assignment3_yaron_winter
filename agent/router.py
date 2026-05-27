import os
from typing import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

STRUCTURED = "structured"
UNSTRUCTURED = "unstructured"
OUT_OF_SCOPE = "out_of_scope"
PERSONAL = "personal"

"""
The router module, which routes the given question to the appropriate agent based on the question content.
I chose Nemotron-3-Nano-Omni for this task, as it is designed for instructions and reasoning tasks,
and it is relatively not too expensive.
"""
llm = ChatOpenAI(base_url="https://api.tokenfactory.nebius.com/v1/",
                 api_key=os.environ["NEBIUS_API_KEY"],
                 model="nvidia/Nemotron-3-Nano-Omni")


class RouteDecision(BaseModel):
    route: Literal[
        "structured",
        "unstructured",
        "out_of_scope",
        "personal",
    ]


router_llm = llm.with_structured_output(RouteDecision)


ROUTER_PROMPT = """
You are a query classifier.

Classify user questions into:

structured:
- factual dataset queries
- counting
- distributions
- examples
- categories
- intents

unstructured:
- summarization
- behavioral analysis
- qualitative insights
- pattern analysis

personal:
- questions about what the assistant knows or remembers about the USER
  themselves (not the dataset)
- examples: "what do you remember about me?", "who am I?",
  "what do you know about me?", "what are my preferences?"
- statements where the user is telling the assistant facts about themselves
  (name, role, preferences) with no dataset question attached

out_of_scope:
- unrelated to dataset
- creative writing
- world knowledge

Return only the route.
"""
