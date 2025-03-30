import pytest
from tokenizer_management.sentencepiece.sp_encoder import SPEncoder

def sample_vocab() -> dict:
    """
    Returns a sample vocabulary dictionary for testing.
    """
    return {
        "<PAD>": {"id": 0, "total_freq": 1, "positions": []},
        "<UNK>": {"id": 1, "total_freq": 1, "positions": []},
        "hello": {"id": 2, "total_freq": 5, "positions": []},
        "world": {"id": 3, "total_freq": 3, "positions": []},
    }

def test_initialization_valid():
    """
    Test that SPEncoder initializes correctly when a valid vocab dictionary is provided.
    """
    vocab = sample_vocab()
    encoder = SPEncoder(vocab)
    # Check that each token is mapped correctly to its id.
    assert encoder.token_to_id["<PAD>"] == 0
    assert encoder.token_to_id["<UNK>"] == 1
    assert encoder.token_to_id["hello"] == 2
    assert encoder.token_to_id["world"] == 3
    # The total mapping size should equal the number of items in the vocab.
    assert len(encoder.token_to_id) == len(vocab)

def test_initialization_empty_vocab():
    """
    Test that initializing SPEncoder with an empty vocabulary raises a ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        SPEncoder({})
    assert "Vocab boş olamaz" in str(exc_info.value)

def test_encode_known_tokens():
    """
    Test that SPEncoder encodes tokens present in the vocab correctly.
    """
    vocab = sample_vocab()
    encoder = SPEncoder(vocab)
    tokens = ["hello", "world"]
    encoded = encoder.encode(tokens)
    # Expected token IDs for "hello" and "world" are 2 and 3, respectively.
    assert encoded == [2, 3]

def test_encode_unknown_token():
    """
    Test that SPEncoder uses the '<UNK>' token ID for tokens not found in the vocabulary.
    """
    vocab = sample_vocab()
    encoder = SPEncoder(vocab)
    tokens = ["unknown"]
    encoded = encoder.encode(tokens)
    # Expected to use the <UNK> token id which is 1.
    assert encoded == [1]

def test_encode_empty_list():
    """
    Test that encoding an empty token list raises a ValueError.
    """
    vocab = sample_vocab()
    encoder = SPEncoder(vocab)
    with pytest.raises(ValueError) as exc_info:
        encoder.encode([])
    assert "Token listesi boş" in str(exc_info.value)
