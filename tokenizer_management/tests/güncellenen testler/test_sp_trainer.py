import pytest
from tokenizer_management.sentencepiece.sp_trainer import SPTrainer

# Fixture to provide an initial vocabulary dictionary.
@pytest.fixture
def initial_vocab():
    return {
        "<PAD>": {"id": 0, "total_freq": 1, "positions": []},
        "<UNK>": {"id": 1, "total_freq": 1, "positions": []},
    }

def test_initialization_with_valid_vocab(initial_vocab):
    trainer = SPTrainer(initial_vocab)
    # Ensure that the trainer's vocab is initialized and has the same number of tokens as the input.
    assert len(trainer.get_vocab()) == len(initial_vocab)
    # Verify that the vocab is a copy (i.e. modifying the original does not affect the trainer's vocab)
    initial_vocab["new"] = {"id": 2, "total_freq": 1, "positions": []}
    assert "new" not in trainer.get_vocab()

def test_initialization_with_invalid_vocab():
    # Passing a non-dictionary should raise a ValueError.
    with pytest.raises(ValueError):
        SPTrainer("not a dict")

def test_train_empty_corpus(initial_vocab):
    trainer = SPTrainer(initial_vocab)
    # Training with an empty corpus or invalid target_vocab_size should not alter the vocab.
    trainer.train([], target_vocab_size=5)
    assert trainer.get_vocab() == initial_vocab

def test_train_adds_new_tokens(initial_vocab):
    trainer = SPTrainer(initial_vocab)
    corpus = [
        "Hello world", 
        "Hello test", 
        "world hello"
    ]
    # With an initial vocab size of 2, setting target_vocab_size=5 allows adding 3 new tokens.
    trainer.train(corpus, target_vocab_size=5)
    vocab = trainer.get_vocab()
    # We expect the new tokens "hello", "world", and "test" to be added.
    assert len(vocab) == 5
    assert "hello" in vocab
    assert "world" in vocab
    assert "test" in vocab

def test_train_respects_target_vocab_size(initial_vocab):
    trainer = SPTrainer(initial_vocab)
    # Create a corpus with many distinct tokens.
    corpus = ["a b c d e f g h i j"]
    # With an initial vocab size of 2, setting target_vocab_size=5 means only 3 new tokens should be added.
    trainer.train(corpus, target_vocab_size=5)
    vocab = trainer.get_vocab()
    assert len(vocab) == 5
