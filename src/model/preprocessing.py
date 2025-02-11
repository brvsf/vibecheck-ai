import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

class TextCleaning:
    """
    A utility class for cleaning textual data by applying various cleaning operations.

    Example:
        >>> cleaner = TextCleaning()
        >>> sentence = " Hello, World! 123 "
        >>> cleaned_sentence = cleaner.full_cleaning(sentence)
        >>> print(cleaned_sentence)  # Output: "hello world"
    """
    def __init__(self):
        """
        Initializes the Cleaning object with punctuation data for later use in cleaning operations.
        """
        self.PUNCTUATION = string.punctuation

    def to_lowercase(self, sentence: str) -> str:
        """
        Convert all characters in the sentence to lowercase.

        Args:
            sentence (str): The input sentence.

        Returns:
            str: The sentence with all characters in lowercase.

        Example:
            >>> sentence = "Hello World"
            >>> result = cleaner.to_lowercase(sentence)
            >>> print(result)  # Output: "hello world"
        """
        if not isinstance(sentence, str):
            raise ValueError("Input must be a string")
        return sentence.lower()

    def remove_numbers(self, sentence: str) -> str:
        """
        Remove all numeric characters from the sentence.

        Args:
            sentence (str): The input sentence.

        Returns:
            str: The sentence without numeric characters.

        Example:
            >>> sentence = "Hello 123"
            >>> result = cleaner.remove_numbers(sentence)
            >>> print(result)  # Output: "Hello "
        """
        if not isinstance(sentence, str):
            raise ValueError("Input must be a string")
        return ''.join(char for char in sentence if not char.isdigit())

    def remove_punctuation(self, sentence: str) -> str:
        """
        Remove all punctuation characters from the sentence.

        Args:
            sentence (str): The input sentence.

        Returns:
            str: The sentence without punctuation.

        Example:
            >>> sentence = "Hello, World!"
            >>> result = cleaner.remove_punctuation(sentence)
            >>> print(result)  # Output: "Hello World"
        """
        if not isinstance(sentence, str):
            raise ValueError("Input must be a string")
        return ''.join(char for char in sentence if char not in self.PUNCTUATION)

    def strip_spaces(self, sentence: str) -> str:
        """
        Remove leading and trailing spaces from the sentence.

        Args:
            sentence (str): The input sentence.

        Returns:
            str: The sentence without leading and trailing spaces.

        Example:
            >>> sentence = "   Hello World   "
            >>> result = cleaner.strip_spaces(sentence)
            >>> print(result)  # Output: "Hello World"
        """
        if not isinstance(sentence, str):
            raise ValueError("Input must be a string")
        return sentence.strip()

    def full_cleaning(self, sentence: str) -> str:
        """
        Apply all preprocessing steps: lowercase conversion,
        removal of numbers, punctuation, and extra spaces.

        Args:
            sentence (str): The input sentence.

        Returns:
            str: The fully cleaned sentence.

        Example:
            >>> sentence = " Hello, World! 123 "
            >>> cleaned_sentence = cleaner.full_cleaning(sentence)
            >>> print(cleaned_sentence)  # Output: "hello world"
        """
        if not isinstance(sentence, str):
            raise ValueError("Input must be a string")

        sentence = self.to_lowercase(sentence)
        sentence = self.remove_numbers(sentence)
        sentence = self.remove_punctuation(sentence)
        sentence = self.strip_spaces(sentence)

        return sentence


class Preprocessing:
    """
    A utility class for preprocessing textual data by applying various operations, including
    tokenization, removal of stopwords, and lemmatization.

    This class provides methods to:
    - Tokenize a sentence and remove stopwords.
    - Lemmatize tokens with part-of-speech tagging.
    - Combine tokenization and lemmatization into a single preprocessing step.

    Example:
        >>> preprocessor = Preprocessing(remove_stopwords=True)
        >>> sentence = "The cats are running quickly towards the garden."
        >>> preprocessor.preprocessor(sentence)
        'cat run quickly toward garden'
    """

    def __init__(self, remove_stopwords: bool = True):
        """
        Initializes the Preprocessing class.

        Args:
            remove_stopwords (bool, optional): Whether to remove stopwords. Defaults to True.

        Example:
            >>> preprocessor = Preprocessing(remove_stopwords=False)
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()

    def tokenizer(self, sentence: str) -> list:
        """
        Tokenize a sentence and remove stopwords.

        Args:
            sentence (str): The input sentence.

        Returns:
            list: A list of words from the sentence with stopwords removed.

        Raises:
            ValueError: If the input is not a string.

        Example:
            >>> preprocessor = Preprocessing()
            >>> preprocessor.tokenizer("The quick brown fox jumps over the lazy dog.")
            ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
        """
        if not isinstance(sentence, str):
            raise ValueError("Input must be a string")

        # Tokenize the sentence, preserving line breaks
        tokens = word_tokenize(sentence, preserve_line=True)

        # Remove stopwords from the tokenized list
        sentence = [w for w in tokens if w.lower() not in self.stop_words]

        return sentence

    def get_wordnet_pos(self, word: str) -> dict:
        """
        Assigns a part-of-speech (POS) tag to a word in WordNet format.

        WordNet uses the following categories:
        - 'n' = Noun
        - 'v' = Verb
        - 'a' = Adjective
        - 'r' = Adverb
        - 's' = Satellite Adjective

        This function retrieves the word's POS tag using NLTK's `pos_tag`
        (which uses the Averaged Perceptron Tagger) and maps it to the WordNet format.

        Args:
            word (str): The word to be classified.

        Returns:
            dict: A dictionary mapping the word to its WordNet POS tag.

        Raises:
            ValueError: If the input is not a string.
        """
        if not isinstance(word, str):
            raise ValueError("Input must be a string.")

        # Get the POS tag using NLTK's pos_tag function
        nltk_tag = pos_tag([word])[0][1]

        if word.endswith("ing"):
            return {word: "v"}

        # Mapping from NLTK POS tags to WordNet POS tags
        tag_mapping = {
            "J": "a",  # Adjective
            "V": "v",  # Verb
            "N": "n",  # Noun
            "R": "r"   # Adverb
        }

        # Get the first letter of the NLTK POS tag and map it
        pos = tag_mapping.get(nltk_tag[0], "n")  # Default to 'n' if unknown

        return {word: pos}

    def lemmatizing(self, sentence: list) -> str:
        """
        Lemmatize the words in a list of tokens and join them into a sentence.

        This method reduces words to their base or dictionary form (lemmas), using the part-of-speech
        tags to improve accuracy. For example, turning "running" into "run" or "better" into "good".

        Args:
            sentence (list): A list of words (tokens) to be lemmatized.

        Returns:
            str: A sentence formed by joining the lemmatized words.

        Raises:
            ValueError: If the input is not a list.

        Example:
            >>> preprocessor = Preprocessing()
            >>> preprocessor.lemmatize(['running', 'quickly', 'towards', 'the', 'garden'])
            'run quickly toward garden'
        """
        if not isinstance(sentence, list):
            raise ValueError("Input must be a list")

        # Get POS tags for each word in the sentence
        tags = [self.get_wordnet_pos(word) for word in sentence]

        lemmatized_sentence = []
        for d in tags:
            for word, pos in d.items():
                lemmatized_sentence.append(self.lemmatizer.lemmatize(word, pos))

        # Join the lemmatized words into a single sentence
        sentence = ' '.join(lemmatized_sentence)

        return sentence

    def preprocessor(self, sentence: str) -> str:
        """
        Apply tokenization and lemmatization on the input sentence.

        This method combines both the `tokenizer` and `lemmatizer` methods
        to preprocess the input sentence, returning the cleaned and lemmatized result.

        Args:
            sentence (str): The input sentence to be processed.

        Returns:
            str: The fully processed sentence.

        Raises:
            ValueError: If the input is not a string.

        Example:
            >>> preprocessor = Preprocessing()
            >>> preprocessor.preprocessor("The cats are running quickly towards the garden.")
            'cat run quickly toward garden'
        """
        if not isinstance(sentence, str):
            raise ValueError("Input must be a string")

        # Tokenize the sentence
        sentence = self.tokenizer(sentence)

        # Lemmatize the tokenized sentence
        sentence = self.lemmatizing(sentence)

        return sentence
