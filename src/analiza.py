import pandas as pd
from collections import Counter
import ast

COLUMN_DESCRIPTIONS = {
    "adult": "informuje, czy film jest przeznaczony tylko dla dorosłych",
    "belongs_to_collection": "informacja, czy film należy do serii lub kolekcji",
    "budget": "budżet produkcji filmu w dolarach",
    "genres": "lista gatunków filmu",
    "homepage": "oficjalna strona internetowa filmu",
    "id": "unikalny identyfikator filmu w bazie TMDB",
    "imdb_id": "identyfikator filmu w serwisie IMDb",
    "original_language": "język oryginalny filmu",
    "original_title": "oryginalny tytuł filmu",
    "overview": "krótki opis fabuły filmu",
    "popularity": "wskaźnik popularności filmu",
    "poster_path": "ścieżka do plakatu filmu",
    "production_companies": "firmy produkujące film",
    "production_countries": "kraje produkcji filmu",
    "release_date": "data premiery filmu",
    "revenue": "przychód filmu w dolarach",
    "runtime": "długość filmu w minutach",
    "spoken_languages": "języki używane w filmie",
    "status": "status produkcji filmu",
    "tagline": "hasło promocyjne filmu",
    "title": "oficjalny tytuł filmu",
    "video": "informacja, czy wpis zawiera materiał wideo",
    "vote_average": "średnia ocena filmu",
    "vote_count": "liczba oddanych głosów",
}

class MovieAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def movies_per_year(self):
        if "year" not in self.df.columns:
            return pd.Series(dtype=int)
        return self.df["year"].value_counts().sort_index()

    def average_rating_per_year(self):
        if "year" not in self.df.columns or "vote_average" not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df.groupby("year")["vote_average"].mean()

    def top_genres(self, top_n=10):
        if "genres" not in self.df.columns:
            return {}
        genres_list = self.df["genres"].dropna().apply(
            lambda x: [g['name'] for g in ast.literal_eval(x)] if x != '[]' else []
        )
        all_genres = [g for sublist in genres_list for g in sublist]
        return dict(Counter(all_genres).most_common(top_n))
