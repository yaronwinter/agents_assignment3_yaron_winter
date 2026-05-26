from typing import List, Dict
from pydantic import BaseModel, Field
from langchain.tools import tool
from agent.dataset import BiTextDataset

dataset = BiTextDataset("data/bitext.csv")

class CategoryDocsInput(BaseModel):
    category: str = Field(
        description="Category to summarize"
    )


@tool(args_schema=CategoryDocsInput)
def get_docs_by_category(category: str) -> List[Dict[str, str]]:
    """
    Return raw records from a category.
    Used for summarization tasks.
    """
    return dataset.summarize_by_category(category)

class IntentDocsInput(BaseModel):
    intent: str = Field(
        description="Intent to summarize"
    )


@tool(args_schema=IntentDocsInput)
def get_docs_by_intent(intent: str) -> List[Dict[str, str]]:
    """
    Return raw records from an intent.
    Used for summarization tasks.
    """
    return dataset.summarize_by_intent(intent)


class ResponsePatternByIntentInput(BaseModel):
    intent: str = Field(
        description="Intent to analyze"
    )


@tool(args_schema=ResponsePatternByIntentInput)
def get_responses_by_intent(intent: str) -> List[Dict[str, str]]:
    """
    Return agent responses for a specific intent.
    Useful for response-pattern analysis.
    """
    return dataset.get_responses_by_intent(intent)

class InstructionsByIntentInput(BaseModel):
    intent: str = Field(
        description="Intent to analyze"
    )

@tool(args_schema=InstructionsByIntentInput)
def get_instructions_by_intent(intent: str) -> List[str]:
    """
    Return instructions for a specific intent.
    Useful for instruction-pattern analysis.
    """
    return dataset.get_instructions_by_intent(intent)

class ResponsePatternByCategoryInput(BaseModel):
    category: str = Field(
        description="Category to analyze"
    )

@tool(args_schema=ResponsePatternByCategoryInput)
def get_responses_by_category(category: str) -> List[Dict[str, str]]:
    """
    Return agent responses for a specific category.
    Useful for response-pattern analysis.
    """
    return dataset.get_responses_by_category(category)

class InstructionsByCategoryInput(BaseModel):
    category: str = Field(
        description="Category to analyze"
    )

@tool(args_schema=InstructionsByCategoryInput)
def get_instructions_by_category(category: str) -> List[str]:
    """
    Return instructions for a specific category.
    Useful for instruction-pattern analysis.
    """
    return dataset.get_instructions_by_category(category)
