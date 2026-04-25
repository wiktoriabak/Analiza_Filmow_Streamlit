import pandas as pd
from src.cluster import BudgetRevenueCluster, PopularityRatingCluster

def sample_cluster_df():
    return pd.DataFrame({
        "budget": [0, 100, 200, 300, 400],
        "revenue": [0, 1000, 1500, 2000, 2500],
        "popularity": [0, 10, 20, 30, 40],
        "vote_average": [0, 6, 7, 8, 9]
    })

# Testowanie klastrowania
def test_budget_revenue_cluster_adds_cluster_column():
    df = sample_cluster_df()
    clustered = BudgetRevenueCluster(df).preprocess().run_kmeans(n_clusters=2).get_clustered_df()

    assert "cluster" in clustered.columns
    assert clustered["cluster"].nunique() == 2

# Testowanie klastrowania v2
def test_popularity_rating_cluster():
    df = sample_cluster_df()
    clustered = PopularityRatingCluster(df).preprocess().run_kmeans(n_clusters=2).get_clustered_df()

    assert "cluster" in clustered.columns
    assert clustered["cluster"].nunique() == 2

# Testowanie usuwania wartości zerowych
def test_cluster_removes_zero_values():
    df = sample_cluster_df()
    clustered = BudgetRevenueCluster(df).preprocess(remove_zero=True).run_kmeans().get_clustered_df()

    assert len(clustered) == 4
