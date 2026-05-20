from typing import List
from pydantic import BaseModel, Field
from langchain.tools import tool
from dataset import BiTextDataset

dataset = BiTextDataset("data/bitext.csv")


# ---------------------------------------------------
# LIST CATEGORIES
# ---------------------------------------------------

class ListCategoriesInput(BaseModel):
    """No input required."""


@tool(
    args_schema=ListCategoriesInput,
    return_direct=False
)
def list_categories() -> List[str]:
    """
    Return all dataset categories.
    Useful when user asks about available categories.
    """
    return dataset.categories()


# ---------------------------------------------------
# COUNT INTENT
# ---------------------------------------------------

class CountIntentInput(BaseModel):
    intent: str = Field(
        description="Intent name to count."
    )


@tool(args_schema=CountIntentInput)
def count_intent(intent: str) -> int:
    """
    Count rows belonging to a specific intent.
    """
    return dataset.count_by_intent(intent)


# ---------------------------------------------------
# EXAMPLES
# ---------------------------------------------------

class ExamplesInput(BaseModel):
    intent: str = Field(description="Intent name")
    limit: int = Field(
        default=3,
        description="Number of examples"
    )


@tool(args_schema=ExamplesInput)
def get_examples(intent: str, limit: int = 3):
    """
    Return example conversations for an intent.
    """
    return dataset.examples_by_intent(intent, limit)


# ---------------------------------------------------
# DISTRIBUTION
# ---------------------------------------------------

class DistributionInput(BaseModel):
    category: str = Field(
        description="Category name"
    )


@tool(args_schema=DistributionInput)
def get_intent_distribution(category: str):
    """
    Return intent distribution inside a category.
    """
    return dataset.intent_distribution(category)


# ---------------------------------------------------
# CATEGORY DOCUMENTS
# ---------------------------------------------------

class CategoryDocsInput(BaseModel):
    category: str = Field(
        description="Category to summarize"
    )


@tool(args_schema=CategoryDocsInput)
def get_category_documents(category: str):
    """
    Return raw records from a category.
    Used for summarization tasks.
    """
    return dataset.summarize_category(category)


# ---------------------------------------------------
# RESPONSES
# ---------------------------------------------------

class ResponsePatternInput(BaseModel):
    intent: str = Field(
        description="Intent to analyze"
    )


@tool(args_schema=ResponsePatternInput)
def get_agent_responses(intent: str):
    """
    Return agent responses for a specific intent.
    Useful for response-pattern analysis.
    """
    return dataset.responses_for_intent(intent)