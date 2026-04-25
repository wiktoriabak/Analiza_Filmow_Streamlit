import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from collections import Counter

from src.loader import KaggleMoviesLoader
from src.processors import MoviePreprocessor
from src.analiza import MovieAnalyzer, COLUMN_DESCRIPTIONS
from src.cluster import BudgetRevenueCluster, PopularityRatingCluster

class MovieApp:
    def __init__(self):
        st.set_page_config(page_title="Analiza filmów", layout="wide")
        st.title("Analiza The Movies Dataset")

        self.df = None
        self.df_clean = None

    def load_data(self):
        if "df" not in st.session_state:
            loader = KaggleMoviesLoader()
            st.session_state.df = loader.load()
            if st.session_state.df is None:
                st.stop()

        self.df = st.session_state.df
        return self.df

    def clean_data(self):
        self.df_clean = MoviePreprocessor(self.df).extract_year().filter_recent().get_df()
        return self.df_clean


    def tab_exploration(self, df):
        st.header("Podstawowa eksploracja danych")
        cols = st.columns(4)
        for i, col_name in enumerate(df.columns):
            cols[i % 4].write(col_name)

        st.divider()
        st.header("Opis kolumn")
        with st.expander("Kliknij, aby rozwinąć opis kolumn"):
            for col in df.columns:
                desc = COLUMN_DESCRIPTIONS.get(col, "Brak opisu")
                st.markdown(f"**{col}** – {desc}")

        st.divider()
        st.header("Pierwsze wiersze danych")
        st.dataframe(df.head(5))

        numeric_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "vote_count"]
        default_index = numeric_cols.index("runtime")
        num_col = st.selectbox("Wybierz kolumnę numeryczną", numeric_cols, index=default_index)
        col_data = pd.to_numeric(df[num_col], errors='coerce').dropna()
        if len(col_data) > 0:
            stats = col_data.describe()[["mean", "std", "min", "max"]].to_frame().T.rename(index={0:num_col}).round(2)
            st.table(stats)
            st.subheader(f"Rozkład wartości – {num_col}")

            q01, q99 = col_data.quantile([0.01, 0.99])
            trimmed_data = col_data[(col_data >= q01) & (col_data <= q99)]

            iqr = trimmed_data.quantile(0.75) - trimmed_data.quantile(0.25)
            bin_width = 2 * iqr / (len(trimmed_data) ** (1 / 3))
            nbins = int((trimmed_data.max() - trimmed_data.min()) / bin_width) if bin_width > 0 else 30
            nbins = max(15, min(nbins, 60))

            fig = px.histogram(
                trimmed_data,
                nbins=nbins,
                opacity=0.8,
                labels={"value": num_col},
                title=f"Histogram wartości ({num_col})",
            )
            
            mean_val = trimmed_data.mean()
            fig.add_vline(
                x=mean_val,
                line_width=2,
                line_dash="dash",
                annotation_text=f"Średnia: {mean_val:.2f}",
                annotation_position="top right",
            )

            fig.update_layout(
                height=450,
                bargap=0.04,
                title_x=0.5,
                margin=dict(l=20, r=20, t=60, b=40),
                xaxis_title=num_col,
                yaxis_title="Liczba obserwacji",
            )

            st.plotly_chart(fig, width='stretch')

        categorical_cols = ["original_language","genres","production_companies","production_countries","status","video"]
        default_index = categorical_cols.index("production_companies")
        cat_col = st.selectbox("Wybierz kolumnę kategoryczną", categorical_cols, index=default_index)
        preprocessor = MoviePreprocessor(df)
        cat_data = preprocessor.extract_categorical_values(cat_col)
        counts = Counter(cat_data)
        top_counts = dict(counts.most_common(20))
        df_top = pd.DataFrame({"Kategoria": list(top_counts.keys()), "Liczba": list(top_counts.values())})
        fig = px.bar(df_top, x="Kategoria", y="Liczba", text="Liczba", color="Liczba",
                     color_continuous_scale="Blues", title=f"Top 20 wartości w kolumnie {cat_col}")
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, height=600)
        st.plotly_chart(fig)

    def tab_filters(self, df):
        st.header("Filtry i wykresy")
        col1, col2 = st.columns([0.25, 0.75])
        with col1:
            year_slider = st.slider("Minimalny rok wydania filmu:",
                                    int(df["year"].min()), int(df["year"].max()), int(df["year"].min()))
            rating_slider = st.slider("Minimalna ocena:", 0.0, 10.0, 0.0)
            title_filter = st.text_input("Filtr tytułów (zawiera):", "")
            top_genre_count = st.selectbox("Ile najpopularniejszych gatunków pokazać?", [5,10,15,20], index=1)
        with col2:
            df_filtered = df[df["year"] >= year_slider]
            df_filtered = df_filtered[df_filtered["vote_average"] >= rating_slider] if "vote_average" in df_filtered.columns else df_filtered
            if title_filter:
                df_filtered = df_filtered[df_filtered["title"].str.contains(title_filter, case=False, na=False)]

            analyzer = MovieAnalyzer(df_filtered)
            st.subheader("Liczba filmów w danym roku")
            st.bar_chart(analyzer.movies_per_year())

            st.subheader("Średnia ocena filmów w danym roku")
            st.line_chart(analyzer.average_rating_per_year())

            st.subheader(f"Najpopularniejsze {top_genre_count} gatunków filmów")
            genres = analyzer.top_genres(top_genre_count)
            if genres:
                df_genres = pd.DataFrame({"Gatunek": list(genres.keys()), "Liczba filmów": list(genres.values())})
                fig = px.bar(df_genres, x="Gatunek", y="Liczba filmów", text="Liczba filmów",
                             title=f"Top {top_genre_count} gatunków filmów")
                fig.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig)

                
    def tab_cluster(self, df):
        st.header("Klasteryzacja filmów")
        col1, col2 = st.columns([0.25, 0.75])
        with col1:
            cluster_type = st.selectbox("Wybierz typ klasteryzacji:",
                                        ["Budżet vs Przychód", "Popularność vs Ocena"])
            n_clusters = st.slider("Liczba klastrów:", 2, 6, 4)
            remove_zero = st.checkbox("Usuń filmy z zerowymi wartościami", value=True)

        with col2:
            if cluster_type == "Budżet vs Przychód":
                clusterer = BudgetRevenueCluster(df).preprocess(remove_zero).run_kmeans(n_clusters)
                x_col, y_col = "budget", "revenue"
            else:
                clusterer = PopularityRatingCluster(df).preprocess(remove_zero).run_kmeans(n_clusters)
                x_col, y_col = "popularity", "vote_average"

            clustered_df = clusterer.get_clustered_df()
            if clustered_df.empty:
                st.warning("Brak danych do klasteryzacji")
            else:
                import plotly.express as px
                fig = px.scatter(clustered_df, x=x_col, y=y_col, color="cluster",
                                hover_data=["title", "year"], 
                                title=f"Klasteryzacja filmów: {cluster_type} (k={n_clusters})")
                st.plotly_chart(fig, width='stretch')


    def tab_top_movies(self, df):
        st.header("Top 10 filmów wg roku i gatunku")
        years = sorted(df['year'].dropna().unique().astype(int))
        selected_year = st.selectbox("Wybierz rok", years, index=len(years)-1)
        preprocessor = MoviePreprocessor(df)
        all_genres = preprocessor.extract_categorical_values("genres")

        flat_genres = sorted(set(all_genres))
        selected_genres = st.multiselect("Wybierz gatunek(i)", flat_genres)

        df_filtered = df[df['year'] == selected_year].copy()
        df_filtered['genres_name'] = df_filtered['genres'].dropna().apply(
            lambda x: [g['name'] for g in eval(x)] if x != '[]' else []
        )
        if selected_genres:
            df_filtered = df_filtered[df_filtered['genres_name'].apply(lambda gs: any(g in gs for g in selected_genres))]
            

        df_filtered = df_filtered[df_filtered['vote_count'] >= 100]


        df_top10 = df_filtered.sort_values(['vote_average','vote_count'], ascending=[False,False]).head(10)
        if df_top10.empty:
            st.write("Brak filmów spełniających kryteria")
        else:
            st.subheader(f"Top 10 filmów w roku {selected_year}")
            st.dataframe(df_top10[['title','year','genres_name','vote_average','vote_count']])

    def run(self):
        self.load_data()
        self.clean_data()
        max_rows = st.slider("Ile wierszy brać pod uwagę?", 100, len(self.df_clean), 5000, step=100)
        df_limited = self.df_clean.head(max_rows)

        tab1, tab2, tab3, tab4 = st.tabs([
            "Podstawowa eksploracja danych",
            "Filtry i wykresy",
            "Klasteryzacja",
            "Najlepiej oceniane filmy"
        ])

        with tab1:
            self.tab_exploration(df_limited)
        with tab2:
            self.tab_filters(df_limited)
        with tab3:
            self.tab_cluster(df_limited)
        with tab4:
            self.tab_top_movies(df_limited)
