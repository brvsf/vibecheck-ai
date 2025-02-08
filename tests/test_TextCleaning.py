import pytest
from model.preprocessing import TextCleaning

def test_to_lowercase():
    """Tests if to_lowercase function corectly converts to lowercase."""

    cleaner = TextCleaning()

    # Testing lowercase and uppercase sentences
    assert cleaner.to_lowercase(" Hello, World! 123 ") == " hello, world! 123 "
    assert cleaner.to_lowercase("PYTHON") == "python"
    assert cleaner.to_lowercase("tEsTiNg") == "testing"

    # Testing only lowercase string
    assert cleaner.to_lowercase("already lowercase") == "already lowercase"

    # Testing special characters
    assert cleaner.to_lowercase("123!@#ABC") == "123!@#abc"

    # Testing empty strings
    assert cleaner.to_lowercase("") == ""

    # Testing expections
    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.to_lowercase(123) # Integer

    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.to_lowercase(None)

def test_remove_numbers():
    """Tests if remove_numbers function corectly remove numbers from strings."""
    cleaner = TextCleaning()

    # Testing different numers
    assert cleaner.remove_numbers(" Hello, World! 123 ") == " Hello, World!  "
    assert cleaner.remove_numbers("PYTHON 3.10.1") == "PYTHON .."
    assert cleaner.remove_numbers("tEsTiNg 1039") == "tEsTiNg "

    # Testing only string with no numbers
    assert cleaner.remove_numbers("no numbers :)") == "no numbers :)"

    # Testing only string only numbers
    assert cleaner.remove_numbers("12345") == ""

    # Testing special characters
    assert cleaner.remove_numbers("123!@#ABC") == "!@#ABC"

    # Testing empty strings
    assert cleaner.remove_numbers("") == ""

    # Testing expections
    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.remove_numbers(123) # Integer

    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.remove_numbers(None)

def test_remove_punctuation():
    """Tests if remove_punctuation function corectly remove punctuation from strings."""
    cleaner = TextCleaning()

    # Testing different punctuations
    assert cleaner.remove_punctuation(" Hello, World!? ") == " Hello World "
    assert cleaner.remove_punctuation("PYTHON 3.10.1") == "PYTHON 3101"
    assert cleaner.remove_punctuation("tEsTiNg &%(!];)") == "tEsTiNg "
    assert cleaner.remove_punctuation("123!@#ABC") == "123ABC"

    # Testing only string only special characters
    assert cleaner.remove_punctuation("!@#$(%_)&*+!_#)%*![]/}{") == ""

    # Testing expections
    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.remove_punctuation(123) # Integer

    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.remove_punctuation(None)

def test_strip_spaces():
    """Tests if strip_spaces function corectly remove leading and trailing spaces from strings."""
    cleaner = TextCleaning()

    # Testing different leading and trailing spaces
    assert cleaner.strip_spaces(" Hello, World ") == "Hello, World"
    assert cleaner.strip_spaces(" Hello, World") == "Hello, World"
    assert cleaner.strip_spaces("Hello, World ") == "Hello, World"
    assert cleaner.strip_spaces(" ") == ""

    # Testing empty string
    assert cleaner.strip_spaces("") == ""

    # Testing expections
    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.strip_spaces(123) # Integer

    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.strip_spaces(None)


def test_full_cleaning():
    """Tests if full_cleaning function corectly apply all cleaning methods above."""
    cleaner = TextCleaning()

    # Testing all in one
    assert cleaner.full_cleaning(" Hello, World! 123 ") == "hello world"
    assert cleaner.full_cleaning("  PlEaSe12345 WoRk!(#*) ") == "please work"
    assert cleaner.full_cleaning("123!@#ABC") == "abc"

    # Testing empty string
    assert cleaner.full_cleaning("") == ""

    # Testing expections
    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.full_cleaning(123) # Integer

    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.full_cleaning(None)
