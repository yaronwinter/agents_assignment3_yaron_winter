import pandas as pd
from typing import List, Dict

CATEGORY = "category"
INTENT = "intent"
INSTRUCTION = "instruction"
RESPONSE = "response"

class BiTextDataset:
    """
    This class provides methods that might be necessary for answering questions about the BiText dataset, such as:
    - What categories exist in the dataset?
    - How many refund requests did we get?
    - What is the distribution of intents in the ACCOUNT category?
    - Summarize the conversations of the SHIPPING category.
    """

    def __init__(self, path: str):
        self.df = pd.read_csv(path)

    def get_categories(self) -> List[str]:
        """Returns a list of unique categories in the dataset."""
        return sorted(self.df[CATEGORY].unique().tolist())

    def get_intents(self) -> List[str]:
        """Returns a list of unique intents in the dataset."""
        return sorted(self.df[INTENT].unique().tolist())

    def count_by_intent(self, intent: str) -> int:
        """Returns the count of rows for a given intent."""
        return len(
            self.df[self.df[INTENT].str.lower() == intent.lower()]
        )
    
    def count_by_category(self, category: str) -> int:
        """Returns the count of rows for a given category."""
        return len(
            self.df[self.df[CATEGORY].str.lower() == category.lower()]
        )

    def get_examples_by_intent(self, intent: str, limit: int = 3) -> List[Dict[str, str]]:
        """"""
        rows = self.df[
            self.df[INTENT].str.lower() == intent.lower()
        ].head(limit)

        return rows.to_dict(orient="records")
    
    def get_examples_by_category(self, category: str, limit: int = 3) -> List[Dict[str, str]]:
        """Returns example rows for a given category."""
        rows = self.df[
            self.df[CATEGORY].str.lower() == category.lower()
        ].head(limit)

        return rows.to_dict(orient="records")

    def intent_distribution(self, category: str) -> Dict[str, int]:
        """Gets the distribution of intents for a given category."""
        subset = self.df[
            self.df[CATEGORY].str.lower() == category.lower()
        ]

        return (
            subset[INTENT]
            .value_counts()
            .to_dict()
        )
    
    def category_distribution(self) -> Dict[str, int]:
        """Gets the distribution of categories in the dataset."""
        return (
            self.df[CATEGORY]
            .value_counts()
            .to_dict()
        )

    def summarize_by_category(self, category: str) -> List[Dict[str, str]]:
        """Provides all rows for a given category, for summarization purposes."""
        subset = self.df[
            self.df[CATEGORY].str.lower() == category.lower()
        ]

        return subset.to_dict(orient="records")
    
    def summarize_by_intent(self, intent: str) -> List[Dict[str, str]]:
        """Provides all rows for a given intent, for summarization purposes."""
        subset = self.df[
            self.df[INTENT].str.lower() == intent.lower()
        ]

        return subset.to_dict(orient="records")

    def get_instructions_by_category(self, category: str) -> List[str]:
        """Returns the instructions for a given category."""
        subset = self.df[
            self.df[CATEGORY].str.lower() == category.lower()
        ]

        return subset[INSTRUCTION].tolist()
    
    def get_responses_by_category(self, category: str) -> List[str]:
        """Returns the agent responses for a given category."""
        subset = self.df[
            self.df[CATEGORY].str.lower() == category.lower()
        ]

        return subset[RESPONSE].tolist()
    
    def get_responses_by_intent(self, intent: str) -> List[str]:
        """Returns the agent responses for a given intent."""
        subset = self.df[
            self.df[INTENT].str.lower() == intent.lower()
        ]

        return subset[RESPONSE].tolist()
    
    def get_instructions_by_intent(self, intent: str) -> List[str]:
        """Returns the instructions for a given intent."""
        subset = self.df[
            self.df[INTENT].str.lower() == intent.lower()
        ]

        return subset[INSTRUCTION].tolist() 
