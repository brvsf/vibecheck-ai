import shutil
from pathlib import Path
import pandas as pd
from model.data import LoadData, DataAdjustment
from model.model import ModelCreator, ModelPredictor
from model.preprocessing import TextCleaning, Preprocessing, DataTransformers


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

        # Balancing classes
        balanced_data = data.copy()

        balancer = DataAdjustment(data)
        balanced_data = balancer.balancing_data()

        # Basic cleaning processing
        print("Cleaning data ⌛\n") ## ta errado aqui

        cleaner = TextCleaning()
        cleaned_data = balanced_data.copy()

        cleaned_data['sentence'] = balanced_data['sentence'].apply(cleaner.full_cleaning)

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
        del(balanced_data)
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

    X = data[["sentence"]]
    y = data["emotion"]

    train_size = int(0.8 * len(X))

    # Holdout method
    X_train = X[:train_size]
    y_train = y[:train_size]

    X_test = X[train_size:]
    y_test = y[train_size:]

    transformer = DataTransformers()
    y_train, y_test, mapping_dict = transformer.encode_emotions(y_train, y_test)
    X_train, X_test, tf_idf_vectorizer = transformer.generate_tfidf_features(X_train, X_test)
    # Removing useless data from memory
    del(data)
    del(X)
    del(y)
    print("Data transformed ✅\n")

    # Model
    print("Training model ⌛\n")

    model = ModelCreator.build_model()
    model.fit(X_train, y_train)
    cross_val_acc, precision, recall, f1 = ModelCreator.evaluate(
        model, X_train, X_test, y_train, y_test
    )

    print("-----Model Scoring-----")
    print(f"Accuracy: {round(cross_val_acc, 2)}\nPrecision: {round(precision, 2)}\
            \nRecall: {round(recall, 2)}\nF1-Score: {round(f1, 2)}")
    print("-----------------------\n")
    print("Model trained ✅\n")

    # Saving model
    ModelCreator.save(model=model)

    # Loading model
    model = ModelCreator.load()

    # Testing a sentence
    print("Test sentence ⌛\n")
    test_sentence='something'
    while(test_sentence != '0'):
        test_sentence = input("Escreva uma frase: \n")

        # Ensure the prediction is aligned with the model
        print(f"Sentence: {str(test_sentence)}\n")

        prediction = ModelPredictor.prediction(model, [test_sentence], tf_idf_vectorizer)

        translation = ModelPredictor.translate(mapping=mapping_dict, y_pred=prediction)

        proba = ModelPredictor.probabilities(model, [test_sentence], tf_idf_vectorizer)

        print(mapping_dict)

        print(f"Prediction for test sentence: {prediction}, {translation}")
        print(f"Proba for each class sentence: {proba}")


if __name__ == '__main__':
    main()
