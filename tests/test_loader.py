import pandas as pd
from unittest.mock import patch, MagicMock
from src.loader import KaggleMoviesLoader

# Testowanie wczytywania danych z pliku CSV
@patch("os.path.exists")
@patch("pandas.read_csv")
def test_loader_fallback_to_local_csv(mock_read_csv, mock_exists):
    mock_exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame({"a": [1, 2, 3]})

    loader = KaggleMoviesLoader()
    df = loader.load()

    assert not df.empty
    assert "a" in df.columns