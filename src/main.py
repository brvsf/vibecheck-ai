import shutil
from pathlib import Path
import pandas as pd
from model.data import LoadData
from model.model import ModelCreator
from model.preprocessing import TextCleaning, Preprocessing, DataTransformers

from sklearn.model_selection import train_test_split

def main():

    # Getting path to preprocessed data directory
    main_dir = Path(__file__).resolve().parent.parent
    processed_data_dir = (main_dir / 'processed_data').resolve()

    # If processed data has files skips preprocessing steps
    if not any(processed_data_dir.iterdir()):
        print("Processed data directory is empty, preprocessing data ⌛\n")
        # Loading data from the csv named 'combined_emotion' from raw_data directory

        loader = LoadData("combined_emotion")
        data = loader.load_csv(loader.path)

        print(f"Data loaded ✅\nShape = {data.shape}\nColumns = {list(data.columns)}\n")

        # Basic cleaning processing
        print("Cleaning data ⌛\n")

        cleaner = TextCleaning()
        cleaned_data = data.copy()

        cleaned_data['sentence'] = data['sentence'].apply(cleaner.full_cleaning)

        print("Cleaning completed ✅\n")

        # Preprocessing data
        print("Preprocessing data, this can take a while ⌛\n")

        preprocessor = Preprocessing(remove_stopwords=False)
        preprocessed_data = cleaned_data.copy()

        preprocessed_data['sentence'] = cleaned_data['sentence'].apply(preprocessor.full_preprocessor)

        print("Preprocessing completed ✅\n")

        # Saving data in processed_data
        print("Saving preprocessed data in the correct directory ⌛\n")
        preprocessed_data.to_csv(processed_data_dir / 'processed_data.csv')

        print("Data saved ✅\n")

        # Removing useless dataframes from memory
        del(data)
        del(cleaned_data)
        del(preprocessed_data)

    print("Loading data ⌛\n")
    data = pd.read_csv(processed_data_dir / 'processed_data.csv').drop(columns='Unnamed: 0')

    if data.empty:
        # Deleting all files from processed_data_dir
        for child in processed_data_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)  # Deleting subdirectory

        # Deleting directory
        processed_data_dir.rmdir()

        # Re-creating directory again
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        raise ValueError("The data is empty, resetting directory")

    # Checking for duplicated and nA values
    if data.duplicated().sum():
        data.drop_duplicates(inplace=True)

    if data.isna().any().any():
        data.dropna(inplace=True)

    print("Data loaded ✅\n")

    # Encoding + vectorizing
    print("Enconding and vectorizing data ⌛\n")

    transformer = DataTransformers()
    y, mapping_dict = transformer.encode_emotions(data)
    X, tf_idf_vectorizer = transformer.generate_tfidf_features(data)

    # Holdout method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Data transformed ✅")



if __name__ == '__main__':
    main()
