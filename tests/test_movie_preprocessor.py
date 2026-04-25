import pandas as pd
from src.processors import MoviePreprocessor


# Testowanie wyciągania roku z daty
def test_extract_year_correct():
    df = pd.DataFrame({
        "release_date": ["2020-01-01", "2021-05-10"]
    })
    processed = MoviePreprocessor(df).extract_year().get_df()

    assert processed.loc[0, "year"] == 2020
    assert processed.loc[1, "year"] == 2021

# Testowanie wyciągania roku z daty - przypadek niepoprawnych danych
def test_extract_year_invalid_date():
    df = pd.DataFrame({
        "release_date": ["invalid-date"]
    })
    processed = MoviePreprocessor(df).extract_year().get_df()

    assert pd.isna(processed.loc[0, "year"])

# Testowanie filtrowania po roku
def test_filter_recent():
    df = pd.DataFrame({"year": [1990, 2000, 2010]})
    processed = MoviePreprocessor(df).filter_recent(2000).get_df()

    assert processed["year"].min() >= 2000

# Testowanie wyciągania gatunków z danych
def test_extract_genres():
    df = pd.DataFrame({
        "genres": [
            "[{'id': 1, 'name': 'Drama'}]",
            "[{'id': 2, 'name': 'Comedy'}]"
        ]
    })
    values = MoviePreprocessor(df).extract_categorical_values("genres")

    assert "Drama" in values
    assert "Comedy" in values
