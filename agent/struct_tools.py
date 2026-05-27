from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain.tools import tool
from agent.dataset import BiTextDataset

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bitext.csv"
dataset = BiTextDataset(str(DATA_PATH))

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
    return dataset.get_all_categories()


class ListIntentsInput(BaseModel):
    """No input required."""

@tool(
    args_schema=ListIntentsInput,
    return_direct=False
)
def list_intents() -> List[str]:
    """
    Return all dataset intents.
    Useful when user asks about available intents.
    """
    return dataset.get_all_intents()


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

class CountCategoryInput(BaseModel):
    category: str = Field(
        description="Category name to count."
    )


@tool(args_schema=CountCategoryInput)
def count_category(category: str) -> int:
    """
    Count rows belonging to a specific category.
    """
    return dataset.count_by_category(category)


class ExamplesByIntentInput(BaseModel):
    intent: str = Field(description="Intent name")
    limit: int = Field(
        default=3,
        description="Number of examples"
    )


@tool(args_schema=ExamplesByIntentInput)
def get_examples_by_intent(intent: str, limit: int = 3) -> List[Dict[str, str]]:
    """
    Return example conversations for an intent.
    """
    return dataset.get_examples_by_intent(intent, limit)

class ExamplesByCategoryInput(BaseModel):
    category: str = Field(description="Category name")
    limit: int = Field(
        default=3,
        description="Number of examples"
    )

@tool(args_schema=ExamplesByCategoryInput)
def get_examples_by_category(category: str, limit: int = 3) -> List[Dict[str, str]]:
    """
    Return example conversations for a category.
    """
    return dataset.get_examples_by_category(category, limit)

class CategoryDistributionInput(BaseModel):
    """ no input required """


@tool(args_schema=CategoryDistributionInput)
def get_category_distribution() -> Dict[str, int]:
    """
    Return intent distribution inside a category.
    """
    return dataset.get_category_distribution()


class IntentDistributionInput(BaseModel):
    """ no input required """


@tool(args_schema=IntentDistributionInput)
def get_intent_distribution() -> Dict[str, int]:
    """
    Return category distribution for a specific intent.
    """
    return dataset.get_intent_distribution()

class IntentDistributionForCategoryInput(BaseModel):
    category: str = Field(description="Category name")

@tool(args_schema=IntentDistributionForCategoryInput)
def get_distribution_of_intents_for_category(category: str) -> Dict[str, int]:
    """
    Return intent distribution inside a category.
    """
    return dataset.get_distribution_of_intents_for_category(category)

class CategoryDistributionForIntentInput(BaseModel):
    intent: str = Field(description="Intent name")

@tool(args_schema=CategoryDistributionForIntentInput)
def get_distribution_of_categories_for_intent(intent: str) -> Dict[str, int]:
    """
    Return category distribution for a specific intent.
    """
    return dataset.get_distribution_of_categories_for_intent(intent)
