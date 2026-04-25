import pandas as pd
from src.analiza import MovieAnalyzer

def sample_df():
    return pd.DataFrame({
        "year": [2020, 2020, 2021],
        "vote_average": [7.0, 8.0, 6.0],
        "genres": [
            "[{'id': 1, 'name': 'Drama'}]",
            "[{'id': 2, 'name': 'Comedy'}]",
            "[{'id': 1, 'name': 'Drama'}]"
        ]
    })

# Testowanie agregacji filmów według roku
def test_movies_per_year():
    analyzer = MovieAnalyzer(sample_df())
    result = analyzer.movies_per_year()

    assert result.loc[2020] == 2
    assert result.loc[2021] == 1

# Testowanie średniej oceny filmów w danym roku
def test_average_rating_per_year():
    analyzer = MovieAnalyzer(sample_df())
    result = analyzer.average_rating_per_year()

    assert result.loc[2020] == 7.5
    assert result.loc[2021] == 6.0

# Testowanie agregowania gatunków i zwracania najpopularniejszych
def test_top_genres():
    analyzer = MovieAnalyzer(sample_df())
    genres = analyzer.top_genres(top_n=2)

    assert genres["Drama"] == 2
    assert genres["Comedy"] == 1


# Testowanie zliczanie filmów: obsługa przypadku gdy nie ma kolumny year
def test_movies_per_year_no_year_column():
    df = pd.DataFrame({"vote_average": [5, 6]})
    analyzer = MovieAnalyzer(df)

    result = analyzer.movies_per_year()
    assert result.empty
