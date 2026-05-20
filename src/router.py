import os
from typing import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="https://api.tokenfactory.nebius.com/v1/",
                 api_key=os.environ["NEBIUS_API_KEY"],
                 model="Qwen/Qwen3-235B-A22B-Instruct-2507")


class RouteDecision(BaseModel):
    route: Literal[
        "structured",
        "unstructured",
        "out_of_scope"
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

unstructured:
- summarization
- behavioral analysis
- qualitative insights

out_of_scope:
- unrelated to dataset
- creative writing
- world knowledge

Return only the route.
"""