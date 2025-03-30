import pytest
from tokenizer_management.training.training_preprocessor import TrainingPreprocessor

def test_preprocess_normal():
    """
    Test that the preprocessor converts text to lowercase,
    removes special characters, and normalizes extra spaces.
    """
    preprocessor = TrainingPreprocessor()
    input_text = "Hello,   World! This is   a Test."
    # Expected output: all letters in lowercase, punctuation removed,
    # and extra spaces normalized to a single space.
    expected = "hello world this is a test"
    result = preprocessor.preprocess(input_text)
    assert result == expected

def test_preprocess_no_special_characters():
    """
    Test that text without special characters remains unchanged except for lowercasing.
    """
    preprocessor = TrainingPreprocessor()
    input_text = "HELLO WORLD"
    expected = "hello world"
    result = preprocessor.preprocess(input_text)
    assert result == expected

def test_preprocess_empty_string():
    """
    Test that an empty string is handled gracefully.
    """
    preprocessor = TrainingPreprocessor()
    input_text = ""
    expected = ""
    result = preprocessor.preprocess(input_text)
    assert result == expected

def test_preprocess_none_input():
    """
    Test that passing None as input raises a ValueError.
    """
    preprocessor = TrainingPreprocessor()
    with pytest.raises(ValueError):
        preprocessor.preprocess(None)
