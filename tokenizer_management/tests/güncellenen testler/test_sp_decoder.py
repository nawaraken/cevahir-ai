import pytest
from tokenizer_management.sentencepiece.sp_decoder import SPDecoder

# A helper function to provide a sample vocabulary
def sample_vocab() -> dict:
    return {
        "hello": {"id": 0, "total_freq": 5, "positions": []},
        "world": {"id": 1, "total_freq": 3, "positions": []},
    }

def test_spdecoder_initialization_valid():
    """Test that SPDecoder initializes correctly with a valid vocabulary."""
    vocab = sample_vocab()
    decoder = SPDecoder(vocab)
    # Check that the reverse mapping is created correctly.
    assert decoder.id_to_token[0] == "hello"
    assert decoder.id_to_token[1] == "world"
    # The mapping should have exactly 2 entries.
    assert len(decoder.id_to_token) == 2

def test_spdecoder_initialization_invalid():
    """Test that initializing SPDecoder with an invalid vocab raises ValueError."""
    with pytest.raises(ValueError):
        SPDecoder("not a dict")

def test_decode_valid_tokens():
    """Test that a valid token ID list is decoded correctly."""
    vocab = sample_vocab()
    decoder = SPDecoder(vocab)
    token_ids = [0, 1]  # should decode to "helloworld"
    decoded_text = decoder.decode(token_ids)
    assert decoded_text == "helloworld"

def test_decode_unknown_token():
    """Test that an unknown token ID is replaced by '<UNK>'."""
    vocab = sample_vocab()
    decoder = SPDecoder(vocab)
    token_ids = [0, 99]  # 99 is not in the vocabulary
    decoded_text = decoder.decode(token_ids)
    assert decoded_text == "hello<UNK>"

def test_decode_empty_token_ids():
    """Test that decoding an empty token ID list raises a ValueError."""
    vocab = sample_vocab()
    decoder = SPDecoder(vocab)
    with pytest.raises(ValueError):
        decoder.decode([])
