from functools import reduce
import pandas as pd
import ast

class MoviePreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def extract_year(self):
        if "release_date" in self.df.columns:
            self.df["release_date"] = pd.to_datetime(self.df["release_date"], errors="coerce")
            self.df["year"] = self.df["release_date"].dt.year.astype("Int64")
        else:
            self.df["year"] = pd.NA
        return self

    def filter_recent(self, year_threshold=None):
        if "year" not in self.df.columns:
            return self
        if year_threshold is None:
            year_threshold = int(self.df["year"].min())
        self.df = self.df[self.df["year"] >= year_threshold]
        return self

    def extract_categorical_values(self, column):
        if column not in self.df.columns:
            return []

        if column == "genres":
            values = self.df["genres"].dropna().apply(
                lambda x: [g['name'] for g in ast.literal_eval(x)] if x != '[]' else []
            )
        elif column == "production_companies":
            values = self.df["production_companies"].dropna().apply(
                lambda x: [c['name'] for c in ast.literal_eval(x)] if x != '[]' else []
            )
        elif column == "production_countries":
            values = self.df["production_countries"].dropna().apply(
                lambda x: [c['name'] for c in ast.literal_eval(x)] if x != '[]' else []
            )
        elif column == "year":
            return self.df["year"].dropna().astype(int).tolist()
        else:
            return self.df[column].dropna().astype(str).tolist()

        flat_values = [v for sublist in values for v in sublist]
        return flat_values

    def get_df(self):
        return self.df
