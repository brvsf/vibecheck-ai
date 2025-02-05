import os
from pathlib import Path
import pandas as pd

class LoadData:
    """
    A class to handle loading data from CSV files. It retrieves the file path
    and provides a method to load the CSV content as a pandas DataFrame.

    Example:
        >>> loader = LoadData("df_name")
        >>> df = loader.load_csv(loader.path)
        >>> print(df.shape) # Output: (x, y)
    """

    def __init__(self, file_name: str):
        """
        Initializes the LoadData object, retrieves the file path for the given file name.

        Args:
            file_name (str): The name of the CSV file (without extension) to load.
        """
        self.file_name = file_name
        self.path = self.get_path(file_name)  # Retrieves the path for the file

    def get_path(self, file_name: str) -> str:
        """
        Constructs the full path to the CSV file based on the base directory of the script.
        It checks if the file exists and raises a FileNotFoundError if it doesn't.

        Args:
            file_name (str): The name of the CSV file (without extension) to find.

        Returns:
            str: The full path to the CSV file.

        Raises:
            FileNotFoundError: If the file does not exist at the given location.
        """
        base_dir = Path(__file__).resolve().parent
        data_dir = (base_dir / '..' / '..' / 'raw_data').resolve()
        full_path = data_dir / f"{file_name}.csv"

        if not full_path.exists():
            raise FileNotFoundError(f"File {full_path} not found.")

        return str(full_path)

    def load_csv(self, path: str) -> pd.DataFrame:
        """
        Loads the CSV file from the provided path into a pandas DataFrame.

        Args:
            path (str): The path to the CSV file to load.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the data from the CSV file.
        """
        return pd.read_csv(path)


class SaveData:
    """
    A class to handle saving processed data to a CSV file.

    This class takes a pandas DataFrame, processes it by removing unwanted columns,
    shuffling rows, and eliminating duplicates or NaN values before saving it to a CSV file.

    The processed CSV file will be stored in the 'processed_data' directory.

    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({'sentence': ['Hello', 'World'],
        >>>                      'cleaned_sentence': ['hello', 'world'],
        >>>                      'label': [1, 0]})
        >>> saver = SaveData(data, "processed_file")
        >>> processed_data = saver.prepare_to_save(data)
        >>> saver.save_csv("processed_file", processed_data)
    """

    def __init__(self, data: pd.DataFrame, file_name: str):
        """
        Initializes the SaveData object.

        Args:
            data (pd.DataFrame): The DataFrame containing the processed data to be saved.
            file_name (str): The name of the CSV file (without extension) where data will be saved.

        Example:
            >>> import pandas as pd
            >>> data = pd.DataFrame({'sentence': ['Hello'], 'cleaned_sentence': ['hello'], 'label': [1]})
            >>> saver = SaveData(data, "processed_file")
        """
        self.data = data
        self.file_name = file_name

    def prepare_to_save(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the DataFrame before saving by:
        - Randomly shuffling the rows.
        - Dropping specific unwanted columns (`sentence`, `cleaned_sentence`, and `Unnamed: 0` if present).
        - Removing duplicate and NaN values.

        Args:
            data (pd.DataFrame): The input DataFrame to be processed.

        Returns:
            pd.DataFrame: The cleaned and processed DataFrame ready for saving.

        Example:
            >>> import pandas as pd
            >>> data = pd.DataFrame({'sentence': ['Hello', 'World'],
            >>>                      'cleaned_sentence': ['hello', 'world'],
            >>>                      'label': [1, 0]})
            >>> saver = SaveData(data, "processed_file")
            >>> processed_data = saver.prepare_to_save(data)
            >>> print(processed_data)
        """
        processed_data = data.sample(frac=1).reset_index(drop=True)  # Shuffle
        processed_data = processed_data.drop(columns=['sentence', 'cleaned_sentence'], errors='ignore')

        if 'Unnamed: 0' in processed_data.columns:
            processed_data = processed_data.drop(columns=['Unnamed: 0'], errors='ignore')

        return processed_data.drop_duplicates().dropna()

    def save_csv(self, data: pd.DataFrame) -> None:
        """
        Saves the processed DataFrame as a CSV file in the 'processed_data' directory.

        If the directory does not exist, it is created automatically.

        Args:
            file_name (str): The name of the CSV file (without extension) where data will be saved.
            data (pd.DataFrame): The DataFrame to be saved.

        Returns:
            None

        Example:
            >>> import pandas as pd
            >>> data = pd.DataFrame({'label': [1, 0]})
            >>> saver = SaveData(data, "processed_file")
            >>> saver.save_csv("processed_file", data)
        """
        base_dir = Path(__file__).resolve().parent
        data_dir = (base_dir / '..' / '..' / 'processed_data').resolve()

        os.makedirs(data_dir, exist_ok=True)

        full_path = data_dir / f"{self.file_name}.csv"
        data.to_csv(full_path, index=False)
