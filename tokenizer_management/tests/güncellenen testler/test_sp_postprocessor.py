# tests/test_sp_postprocessor.py

import pytest
from tokenizer_management.sentencepiece.tokenization.postprocessor import SPPostprocessor

@pytest.fixture
def postprocessor():
    return SPPostprocessor()

def test_process_valid_tokens(postprocessor):
    """
    Test that valid token list is processed correctly.
    """
    tokens = ["hello", "world"]
    result = postprocessor.process(tokens)
    assert result == "helloworld"

def test_process_empty_tokens(postprocessor):
    """
    Test that empty token list returns an empty string.
    """
    tokens = []
    result = postprocessor.process(tokens)
    assert result == ""

def test_process_single_token(postprocessor):
    """
    Test that a single token is processed correctly.
    """
    tokens = ["hello"]
    result = postprocessor.process(tokens)
    assert result == "hello"

def test_process_special_characters(postprocessor):
    """
    Test that special characters are handled correctly.
    """
    tokens = ["hello", "!", "world", "."]
    result = postprocessor.process(tokens)
    assert result == "hello!world."

def test_process_mixed_case_tokens(postprocessor):
    """
    Test that mixed case tokens are processed correctly.
    """
    tokens = ["HeLLo", "WoRLd"]
    result = postprocessor.process(tokens)
    assert result == "HeLLoWoRLd"
