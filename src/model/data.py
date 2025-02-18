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


class DataAdjustment:
    """
    A class for adjusting and balancing data by sampling emotions.

    Attributes:
        data (pd.DataFrame): The dataframe containing the data to be adjusted.

    Methods:
        sample_all_emotions(sample_size: int = 14959) -> dict:
            Samples each emotion from the data to balance the dataset.

        balancing_data() -> pd.DataFrame:
            Balances the dataset by sampling emotions and concatenating them into a single dataframe.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DataAdjustment object with the given dataframe.

        Args:
            data (pd.DataFrame): The input dataframe containing the data to be adjusted.
        """
        self.data = data

    def sample_all_emotions(self, sample_size: int = 14959) -> dict:
        """
        Samples each emotion from the data to balance the dataset.

        Args:
            sample_size (int, optional): The number of samples to take for each emotion (default is 14959).

        Returns:
            dict: A dictionary with emotion as keys and sampled dataframes as values.

        Example:
            >>> df = pd.DataFrame({'emotion': ['joy', 'sad', 'joy', 'fear'], 'text': ['happy', 'sad', 'excited', 'scared']})
            >>> data_adjustment = DataAdjustment(df)
            >>> samples = data_adjustment.sample_all_emotions(2)
            >>> print(samples['joy'])
            emotion   text
            0     joy  happy
            2     joy  excited
        """
        emotions = list(self.data['emotion'].unique())
        samples = {}

        for emotion in emotions:
            if emotion == 'suprise':
                samples[emotion] = self.data[self.data['emotion'] == emotion]
            else:
                samples[emotion] = self.data[self.data['emotion'] == emotion].sample(sample_size)

        return samples

    def balancing_data(self) -> pd.DataFrame:
        """
        Balances the dataset by sampling emotions and concatenating them into a single dataframe.

        Returns:
            pd.DataFrame: A balanced dataframe with samples of various emotions.

        Example:
            >>> df = pd.DataFrame({'emotion': ['joy', 'sad', 'joy', 'fear'], 'text': ['happy', 'sad', 'excited', 'scared']})
            >>> data_adjustment = DataAdjustment(df)
            >>> balanced_df = data_adjustment.balancing_data()
            >>> print(balanced_df)
            emotion   text
            0     sad    sad
            1     love   happy
            2     joy    excited
            3     fear   scared
            ...
        """
        samples = self.sample_all_emotions()

        return pd.concat([
            samples['sad'], samples['love'], samples['joy'],
            samples['fear'], samples['anger'], samples['suprise']
            ]).sample(frac=1).reset_index(drop=True)
