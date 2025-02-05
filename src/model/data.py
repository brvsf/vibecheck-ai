from pathlib import Path
import pandas as pd

class LoadData:
    """
    A class to handle loading data from CSV files. It retrieves the file path
    and provides a method to load the CSV content as a pandas DataFrame.
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



# Example of usage:
# def main():
#     loader = LoadData("combined_emotion")
#     df = loader.load_csv(loader.path)
#     print(df.shape)

# if __name__ == '__main__':
#     main()
