import pytest
from model.preprocessing import Preprocessing

def test_tokenizer():
    """Tests if tokenizer function correctly tokenizes the sentence."""
    cleaner = Preprocessing()
    with_stopwords = Preprocessing(remove_stopwords=False)
    # Testing different sentences
    assert cleaner.tokenizer("Hello, World!") == ["Hello", ",","World", "!"]
    assert cleaner.tokenizer("Python is awesome!") == ["Python", "awesome", "!"]
    assert cleaner.tokenizer("I love Python! 1000") == ["I", "love", "Python", "!", "1000"]

    # Testing empty strings
    assert cleaner.tokenizer("") == []

    # Testing cleaner with stopwords
    assert with_stopwords.tokenizer("Python is awesome!") == ["Python", "is","awesome", "!"]
    assert with_stopwords.tokenizer("I love Python! 1000") == ["I", "love", "Python", "!", "1000"]

    # Testing expections
    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.tokenizer(123)  # Integer

    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.tokenizer(None)

def test_get_wordnet_pos():
    """Tests if get_wordnet_pos function correctly returns the POS tag."""
    cleaner = Preprocessing(remove_stopwords=False)

    # Testing different words
    assert cleaner.get_wordnet_pos("hello") == {"hello":"n"}
    assert cleaner.get_wordnet_pos("always") == {"always":"r"}
    assert cleaner.get_wordnet_pos("Python") == {"Python":"n"}
    assert cleaner.get_wordnet_pos("is") == {"is":"v"}
    assert cleaner.get_wordnet_pos("awesome") == {"awesome":"n"}
    assert cleaner.get_wordnet_pos("1000") == {"1000":"n"}
    assert cleaner.get_wordnet_pos("barking") == {"barking":"v"}
    assert cleaner.get_wordnet_pos("testing") == {"testing":"v"}

    # Testing empty strings
    assert cleaner.get_wordnet_pos("") == {"": "n"}

    # Testing expections
    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.get_wordnet_pos(123)  # Integer

    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.get_wordnet_pos(None)

def test_lemmatizing():
    """Tests if lemmatizing function correctly lemmatizes the words."""
    cleaner = Preprocessing(remove_stopwords=False)

    # Testing different words
    assert cleaner.lemmatizing(["running"]) == "run"
    assert cleaner.lemmatizing(["flies"]) == "fly"
    assert cleaner.lemmatizing(["better"]) == "well"
    assert cleaner.lemmatizing(["bigger"]) == "big"
    assert cleaner.lemmatizing(["dogs"]) == "dog"
    assert cleaner.lemmatizing(["went"]) == "go"
    assert cleaner.lemmatizing(["doing"]) == "do"
    assert cleaner.lemmatizing(["barking"]) == "bark"

    # Testing words that dont change
    assert cleaner.lemmatizing(["data"]) == "data"
    assert cleaner.lemmatizing(["music"]) == "music"
    assert cleaner.lemmatizing(["happiness"]) == "happiness"


    # Testing empty strings
    assert cleaner.lemmatizing([""]) == ""

    # Testing expections
    with pytest.raises(ValueError, match="Input must be a list"):
        cleaner.lemmatizing(123)  # Integer

    with pytest.raises(ValueError, match="Input must be a list"):
        cleaner.lemmatizing(None)

    with pytest.raises(ValueError, match="Input must be a list"):
        cleaner.lemmatizing("string")   # String


def test_full_preprocessor():
    """Tests if full_preprocessor function correctly preprocesses the sentence."""
    cleaner = Preprocessing(remove_stopwords=True)
    with_stopwords = Preprocessing(remove_stopwords=False)

    # Testing different sentences
    assert cleaner.full_preprocessor("the cats are running quickly towards the garden") == "cat run quickly towards garden"
    assert cleaner.full_preprocessor("the dogs are barking loudly in the park") == "dog bark loudly park"
    assert cleaner.full_preprocessor("the birds are chirping in the trees") == "bird chirp tree"

    # Testing empty strings
    assert cleaner.full_preprocessor("") == ""

    # Testing cleaner with stopwords
    assert with_stopwords.full_preprocessor("the cats are running quickly towards the garden") == "the cat be run quickly towards the garden"
    assert with_stopwords.full_preprocessor("the dogs are barking loudly in the park") == "the dog be bark loudly in the park"
    assert with_stopwords.full_preprocessor("the birds are chirping in the trees") == "the bird be chirp in the tree"

    # Testing expections
    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.full_preprocessor(123)  # Integer

    with pytest.raises(ValueError, match="Input must be a string"):
        cleaner.full_preprocessor(None)
