from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

class BaseCluster:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.clustered_df = pd.DataFrame()

    def preprocess(self):
        raise NotImplementedError

    def fit_predict(self, features, n_clusters=4, random_state=42, remove_zero=True):
        X = self.df[features].copy()
        if remove_zero:
            X = X[(X > 0).all(axis=1)]
        X = X.dropna()
        if X.empty:
            self.clustered_df = pd.DataFrame()
            return self
        if len(X) < n_clusters:
            n_clusters = len(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        X["cluster"] = kmeans.fit_predict(X_scaled)
        self.clustered_df = self.df.loc[X.index].copy()
        self.clustered_df["cluster"] = X["cluster"]
        return self


    def get_clustered_df(self):
        return self.clustered_df


class BudgetRevenueCluster(BaseCluster):
    def preprocess(self, remove_zero=True):
        self.df["budget"] = pd.to_numeric(self.df["budget"], errors="coerce")
        self.df["revenue"] = pd.to_numeric(self.df["revenue"], errors="coerce")
        if remove_zero:
            self.df = self.df[(self.df["budget"] > 0) & (self.df["revenue"] > 0)]
        self.df = self.df.dropna(subset=["budget","revenue"])
        return self

    def run_kmeans(self, n_clusters=4, random_state=42, remove_zero=True):
        return self.fit_predict(["budget","revenue"], n_clusters, random_state, remove_zero)


class PopularityRatingCluster(BaseCluster):
    def preprocess(self, remove_zero=True):
        self.df["popularity"] = pd.to_numeric(self.df["popularity"], errors="coerce")
        self.df["vote_average"] = pd.to_numeric(self.df["vote_average"], errors="coerce")
        if remove_zero:
            self.df = self.df[(self.df["popularity"] > 0) & (self.df["vote_average"] > 0)]
        self.df = self.df.dropna(subset=["popularity","vote_average"])
        return self

    def run_kmeans(self, n_clusters=4, random_state=42, remove_zero=True):
        return self.fit_predict(["popularity","vote_average"], n_clusters, random_state, remove_zero)
