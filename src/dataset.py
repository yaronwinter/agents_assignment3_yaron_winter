import pandas as pd


class BiTextDataset:
    def __init__(self, path: str):
        self.df = pd.read_csv(path)

    def categories(self):
        return sorted(self.df["category"].unique().tolist())

    def intents(self):
        return sorted(self.df["intent"].unique().tolist())

    def count_by_intent(self, intent: str):
        return len(
            self.df[self.df["intent"].str.lower() == intent.lower()]
        )

    def examples_by_intent(self, intent: str, limit: int = 3):
        rows = self.df[
            self.df["intent"].str.lower() == intent.lower()
        ].head(limit)

        return rows.to_dict(orient="records")

    def intent_distribution(self, category: str):
        subset = self.df[
            self.df["category"].str.lower() == category.lower()
        ]

        return (
            subset["intent"]
            .value_counts()
            .to_dict()
        )

    def summarize_category(self, category: str):
        subset = self.df[
            self.df["category"].str.lower() == category.lower()
        ]

        return subset.to_dict(orient="records")

    def responses_for_intent(self, intent: str, limit: int = 20):
        subset = self.df[
            self.df["intent"].str.lower() == intent.lower()
        ].head(limit)

        return subset["agent"].tolist()