import string

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
