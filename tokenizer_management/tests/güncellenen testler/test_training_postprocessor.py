import pytest
import logging
from tokenizer_management.training.training_postprocessor import TrainingPostprocessor

logger = logging.getLogger(__name__)

def test_process_valid_tokens():
    """
    Test that a valid list of tokens is correctly joined into a single string
    with single spaces between tokens.
    """
    postprocessor = TrainingPostprocessor()
    tokens = ["hello", "world", "this", "is", "test"]
    expected = "hello world this is test"
    result = postprocessor.process(tokens)
    assert result == expected

def test_process_empty():
    """
    Test that an empty token list returns an empty string.
    """
    postprocessor = TrainingPostprocessor()
    tokens = []
    expected = ""
    result = postprocessor.process(tokens)
    assert result == expected

def test_process_single_token():
    """
    Test that a single token is returned as is.
    """
    postprocessor = TrainingPostprocessor()
    tokens = ["token"]
    expected = "token"
    result = postprocessor.process(tokens)
    assert result == expected

def test_process_extra_spaces():
    """
    Test that tokens with extra spaces or empty tokens are properly cleaned up.
    For example, tokens with leading/trailing or multiple internal spaces are reduced.
    """
    postprocessor = TrainingPostprocessor()
    tokens = ["hello", "", "world", "  test "]
    # " ".join(tokens) would yield "hello  world   test ", and then split/join cleans it up.
    expected = "hello world test"
    result = postprocessor.process(tokens)
    assert result == expected
