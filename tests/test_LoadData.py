import pytest
from unittest.mock import patch
from pathlib import Path
import pandas as pd
from model.data import LoadData


def test_get_path_valid_file():
    file_name = "valid_file"
    with patch('pathlib.Path.exists', return_value=True):
        load_data = LoadData(file_name)
        expected_path = str(Path(__file__).resolve().parent.parent / 'raw_data' / f"{file_name}.csv")
        assert load_data.path == expected_path

def test_get_path_invalid_file():
    file_name = "invalid_file"
    with patch('pathlib.Path.exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            LoadData(file_name)

def test_load_csv():
    file_name = "valid_file"
    mock_data = {"column1": [1, 2], "column2": [3, 4]}
    mock_df = pd.DataFrame(mock_data)
    with patch('pandas.read_csv', return_value=mock_df):
        with patch.object(LoadData, 'get_path', return_value='/mock/path/to/valid_file.csv'):
            load_data = LoadData(file_name)
            load_data.data = load_data.load_csv(load_data.path)
            assert load_data.data.equals(mock_df)

def test_init():
    file_name = "valid_file"
    with patch.object(LoadData, 'get_path', return_value="mock/path/to/file.csv"):
        load_data = LoadData(file_name)
        assert load_data.path == "mock/path/to/file.csv"
