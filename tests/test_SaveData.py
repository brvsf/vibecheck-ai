import os
from pathlib import Path
import pandas as pd
from model.data import SaveData

def test_prepare_to_save():
    """Tests if prepare_to_save function correctly prepares the DataFrame before saving."""
    mock_data = {"sentence": ["Hello", "World"],
                 "cleaned_sentence": ["hello", "world"],
                 "label": [1, 0]}
    mock_df = pd.DataFrame(mock_data).drop(columns = ["sentence", "cleaned_sentence"])

    saver = SaveData(mock_df, "processed_file")
    processed_data = saver.prepare_to_save()

    assert processed_data.equals(mock_df)

def test_save_csv():
    """Tests if save_csv function correctly saves the processed DataFrame as a CSV file."""
    mock_data = {"sentence": ["Hello", "World"],
                 "cleaned_sentence": ["hello", "world"],
                 "label": [1, 0]}
    mock_df = pd.DataFrame(mock_data)

    saver = SaveData(mock_df, "processed_file")
    saver.save_csv()

    expected_path = Path(__file__).resolve().parent.parent / 'processed_data' / 'processed_file.csv'
    assert Path(expected_path).exists()

    os.remove(expected_path) # Removing the file after the test
