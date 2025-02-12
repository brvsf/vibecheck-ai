import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate, train_test_split


class ModelCreator:
    """
    A class to handle the creation, training, and evaluation of a Naive Bayes model.
    It provides methods to build the model, train it on the data, and evaluate its performance.

    Methods:
        build_model(): Builds a Multinomial Naive Bayes model with custom hyperparameters.
        train(): Trains the model on the training data.
        evaluate(): Evaluates the model on the test data and returns the performance metrics.
        save(): Saves the trained model to a file.
        load(): Loads a trained model from a file.

    Example:
        >>> creator = ModelCreator(X, y)
        >>> creator.train()
        >>> cross_val_acc, precision, recall, f1 = creator.evaluate()
        >>> creator.save()
        >>> model = creator.load()
    """

    def __init__(self, X : pd.DataFrame, y : pd.Series):
        """
        Initializes the ModelCreator with the input features and target labels.

        Args:
            X (pd.DataFrame): The input features for the model (already vectorized).
            y (pd.Series): The target labels for the model (already encoded).

        Examples:
            >>> creator = ModelCreator(X, y)
        """
        self.model = self.build_model()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def build_model(self) -> MultinomialNB:
        """
        Builds a Multinomial Naive Bayes model with custom hyperparameters.

        Returns:
            MultinomialNB: A Naive Bayes model with custom hyperparameters.

        Examples:
            >>> model = build_model()
        """
        return MultinomialNB(
            alpha=1.99,
            class_prior=[0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]
        )

    def train(self) -> MultinomialNB:
        """
        Trains the model on the training data.

        Returns:
            MultinomialNB: The trained Naive Bayes model.

        Examples:
            >>> model.train()
        """
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self) -> tuple[float, float, float, float]:
        """
        Scores the model on the test data and returns the performance metrics.

        Returns:
            tuple[float, float, float, float]: The cross-validation accuracy, precision, recall, and F1 score.

        Examples:
            >>> cross_val_acc, precision, recall, f1 = model.evaluate()
        """
        model = self.model
        cross_val_acc = cross_validate(model, self.X_train, self.y_train, cv=5, scoring='accuracy')['test_score'].mean()
        y_pred = model.predict(self.X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='macro')
        return cross_val_acc, precision, recall, f1

    def save(self):
        """
        Saves the trained model to a file.

        Examples:
            >>> model.save()
        """
        with open("../checkpoints/model.pkl", "wb") as f:
            pkl.dump(self.model, f)

    def load(self):
        """
        Loads a trained model from a file.

        Returns:
            MultinomialNB: The loaded Naive Bayes model.

        Raises:
            FileNotFoundError: If the model file is not found.
        Examples:
            >>> model = ModelCreator().load()
        """
        try:
            with open("../checkpoints/model.pkl", "rb") as f:
                model = pkl.load(f)
                return model
        except FileNotFoundError:
            print("Model file not found. Please train the model first.")
            return None


class ModelPredictor:
    """
    A class to handle prediction tasks using a trained model and TF-IDF vectorizer.
    It provides methods to predict classes, obtain class probabilities, and map predicted
    class labels back to their original text representations.

    Methods:
        predict(X_new): Predicts the class label for a new input based on the trained model.
        probabilities(X_new): Returns the class probabilities for a new input.
        translate(y_pred): Translates the predicted class label to its corresponding string label.

    Example:
        >>> predictor = ModelPredictor(model, tfidf, mapping)
        >>> prediction = predictor.predict("This is a new sentence.")
        >>> probabilities = predictor.probabilities("This is a new sentence.")
        >>> translated_label = predictor.translate(prediction)
    """

    def __init__(self, model : MultinomialNB, tfidf : TfidfVectorizer, mapping : dict):
        """
        Initializes the ModelPredictor with a trained model, TF-IDF vectorizer, and a mapping of class labels.

        Args:
            model (MultinomialNB): A trained Naive Bayes model.
            tfidf (TfidfVectorizer): A TF-IDF vectorizer fitted on the training data.
            mapping (dict): A mapping of class labels to their corresponding string representations.

        Examples:
            >>> predictor = ModelPredictor(model, tfidf, mapping)
        """
        self.model = model
        self.tfidf = tfidf
        self.mapping = mapping

    def predict(self, X_new : list) -> int:
        """
        Predicts the class label for a new input based on the trained model.

        The input is first transformed using the TF-IDF vectorizer and then passed to the model for prediction.

        Returns:
            int: The predicted class label.

        Examples:
            >>> prediction = predictor.predict("This is a new sentence.")
        """
        X_new_tfidf = self.tfidf.transform(X_new)
        return self.model.predict(X_new_tfidf)[0]

    def probabilities(self, X_new : list) -> np.ndarray[float]:
        """
        Predicts the class probabilities for a new input based on the trained model.

        The input is first transformed using the TF-IDF vectorizer and then passed to the model for prediction.

        Returns:
            np.ndarray[float]: The predicted class probabilities.

        Examples:
            >>> probabilities = predictor.probabilities("This is a new sentence.")
        """
        X_new = self.tfidf.transform(X_new)
        return self.model.predict_proba(X_new)

    def translate(self, y_pred : int) -> str:
        """
        Translates the predicted class label to its corresponding string label.

        Args:
            y_pred (int): The predicted class label.

        Returns:
            str: The corresponding string label.

        Examples:
            >>> translated_label = predictor.translate(prediction)
        """

        return self.mapping.get(y_pred, "Unknown")
