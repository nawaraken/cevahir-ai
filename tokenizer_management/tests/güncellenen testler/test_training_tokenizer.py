import pytest
from tokenizer_management.training.training_tokenizer import TrainingTokenizer

def test_tokenize_normal():
    """
    Test that a normal sentence is split into tokens correctly.
    """
    tokenizer = TrainingTokenizer()
    text = "Hello world"
    expected = ["Hello", "world"]
    result = tokenizer.tokenize(text)
    assert result == expected

def test_tokenize_extra_spaces():
    """
    Test that extra spaces are ignored and tokens are correctly split.
    """
    tokenizer = TrainingTokenizer()
    text = "   Hello    world   "
    expected = ["Hello", "world"]
    result = tokenizer.tokenize(text)
    assert result == expected

def test_tokenize_empty_string():
    """
    Test that an empty string (or string with only spaces) returns an empty list.
    """
    tokenizer = TrainingTokenizer()
    text = "     "
    expected = []
    result = tokenizer.tokenize(text)
    assert result == expected

def test_tokenize_none_input():
    """
    Test that passing None as input raises a ValueError.
    """
    tokenizer = TrainingTokenizer()
    with pytest.raises(ValueError):
        tokenizer.tokenize(None)
