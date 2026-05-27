import pandas as pd
import numpy as np
from typing import List, Dict

CATEGORY = "category"
INTENT = "intent"
INSTRUCTION = "instruction"
RESPONSE = "response"

MAX_DOCS_TO_SUMMARIZE = 45 # To avoid overloading the LLM with too much information,
                           # which may exceed its context window, harm performance, increase costs
                           # and exceeds the quota of the used model.

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

    def get_all_categories(self) -> List[str]:
        """Returns a list of unique categories in the dataset."""
        return sorted(self.df[CATEGORY].unique().tolist())

    def get_all_intents(self) -> List[str]:
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

    def get_distribution_of_intents_for_category(self, category: str) -> Dict[str, int]:
        """Gets the distribution of intents for a given category."""
        subset = self.df[
            self.df[CATEGORY].str.lower() == category.lower()
        ]

        return (
            subset[INTENT]
            .value_counts()
            .to_dict()
        )
    
    def get_distribution_of_categories_for_intent(self, intent: str) -> Dict[str, int]:
        """Gets the distribution of categories for a given intent."""
        subset = self.df[
            self.df[INTENT].str.lower() == intent.lower()
        ]

        return (
            subset[CATEGORY]
            .value_counts()
            .to_dict()
        )
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Gets the distribution of categories in the dataset."""
        return (
            self.df[CATEGORY]
            .value_counts()
            .to_dict()
        )
    
    def get_intent_distribution(self) -> Dict[str, int]:
        """Gets the distribution of intents in the dataset."""
        return (
            self.df[INTENT]
            .value_counts()
            .to_dict()
        )

    def summarize_by_category(self, category: str) -> List[Dict[str, str]]:
        """Provides all rows for a given category, for summarization purposes."""
        subset = self.df[
            self.df[CATEGORY].str.lower() == category.lower()
        ]

        subset = reduce_docs_number(subset)

        return subset.to_dict(orient="records")
    
    def summarize_by_intent(self, intent: str) -> List[Dict[str, str]]:
        """Provides all rows for a given intent, for summarization purposes."""
        subset = self.df[
            self.df[INTENT].str.lower() == intent.lower()
        ]

        subset = reduce_docs_number(subset)

        return subset.to_dict(orient="records")

    def get_instructions_by_category(self, category: str) -> List[str]:
        """Returns the instructions for a given category."""
        subset = self.df[
            self.df[CATEGORY].str.lower() == category.lower()
        ]

        subset = reduce_docs_number(subset)

        return subset[INSTRUCTION].tolist()
    
    def get_responses_by_category(self, category: str) -> List[str]:
        """Returns the agent responses for a given category."""
        subset = self.df[
            self.df[CATEGORY].str.lower() == category.lower()
        ]

        subset = reduce_docs_number(subset)
        return subset[RESPONSE].tolist()
    
    def get_responses_by_intent(self, intent: str) -> List[str]:
        """Returns the agent responses for a given intent."""
        subset = self.df[
            self.df[INTENT].str.lower() == intent.lower()
        ]

        subset = reduce_docs_number(subset)
        return subset[RESPONSE].tolist()
    
    def get_instructions_by_intent(self, intent: str) -> List[str]:
        """Returns the instructions for a given intent."""
        subset = self.df[
            self.df[INTENT].str.lower() == intent.lower()
        ]

        subset = reduce_docs_number(subset)
        return subset[INSTRUCTION].tolist() 

def reduce_docs_number(df: pd.DataFrame) -> pd.DataFrame:
    """Reduces the number of documents in a dataframe to avoid overloading the LLM."""
    if len(df) == 0:
        return df
    
    threshold = MAX_DOCS_TO_SUMMARIZE / len(df)
    rands = np.random.rand(len(df)).tolist()
    subset = df[[x < threshold for x in rands]].reset_index(drop=True)

    return subset
