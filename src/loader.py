import os
import pandas as pd
import streamlit as st
from src.decorators import with_spinner, measure_time

class KaggleMoviesLoader:
    def __init__(self, dataset="rounakbanik/the-movies-dataset", file_name="movies_metadata.csv"):
        self.dataset = dataset
        self.file_name = file_name
        self.data_dir = "data"

    @with_spinner("Pobieranie i wczytywanie danych z Kaggle...")
    @measure_time
    def load(self):
        os.makedirs(self.data_dir, exist_ok=True)
        csv_path = os.path.join(self.data_dir, self.file_name)
        kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")

        if os.path.exists(kaggle_json_path):
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                api.dataset_download_files(self.dataset, path=self.data_dir, unzip=True)
                df = pd.read_csv(csv_path, low_memory=False)
                st.success("Pobrano dane z Kaggle i wczytano CSV")
                return df
            except Exception as e:
                st.warning(f"Nie udało się pobrać danych z Kaggle, używam lokalnego pliku: {e}")

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                st.info("Używam lokalnego CSV (offline)")
                return df
            except Exception as e:
                st.error(f"Błąd przy wczytywaniu CSV: {e}")
        else:
            st.error("Brak danych CSV i brak tokena Kaggle")
        return None
