import pandas as pd
from src.processors import MoviePreprocessor
from src.analiza import MovieAnalyzer

# Testowanie zliczania filmów w roku oraz najczęściej występujących gatunków (klasa MoviePreprocessor i MovieAnalyzer)
def test_full_analysis_pipeline():
    df = pd.DataFrame({
        "release_date": ["2020-01-01", "2020-05-01"],
        "vote_average": [7.0, 8.0],
        "genres": [
            "[{'id': 1, 'name': 'Drama'}]",
            "[{'id': 1, 'name': 'Drama'}]"
        ]
    })

    df_clean = MoviePreprocessor(df).extract_year().get_df()
    analyzer = MovieAnalyzer(df_clean)

    movies_per_year = analyzer.movies_per_year()
    top_genres = analyzer.top_genres()

    assert movies_per_year.loc[2020] == 2
    assert top_genres["Drama"] == 2
