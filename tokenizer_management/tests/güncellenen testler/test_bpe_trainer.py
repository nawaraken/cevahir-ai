import pytest
from tokenizer_management.bpe.bpe_trainer import BPETrainer, BPETrainingError

@pytest.fixture
def initial_vocab():
    return {
        "<PAD>": {"id": 0, "total_freq": 1, "positions": []},
        "<UNK>": {"id": 1, "total_freq": 1, "positions": []},
        "<BOS>": {"id": 2, "total_freq": 1, "positions": []},
        "<EOS>": {"id": 3, "total_freq": 1, "positions": []},
    }

@pytest.fixture
def trainer(initial_vocab):
    return BPETrainer(vocab=initial_vocab)

def test_initial_vocab(trainer):
    vocab = trainer.get_vocab()
    assert len(vocab) == 4
    assert "<PAD>" in vocab
    assert "<UNK>" in vocab
    assert "<BOS>" in vocab
    assert "<EOS>" in vocab

def test_train_success(trainer):
    corpus = ["abc", "ab", "bc"]
    trainer.train(corpus, target_merges=10)
    vocab = trainer.get_vocab()
    assert len(vocab) > 4
    assert "ab" in vocab or "abc" in vocab

def test_update_vocab(trainer):
    trainer.update_vocab(["yeni", "token"])
    vocab = trainer.get_vocab()
    assert "yeni" in vocab
    assert "token" in vocab
    assert vocab["yeni"]["id"] == 4
    assert vocab["token"]["id"] == 5

def test_count_tokens_after_train(trainer):
    trainer.train(["aba", "abc"], target_merges=10)
    vocab = trainer.get_vocab()
    assert len(vocab) > 4
    assert "ab" in vocab or "ba" in vocab or "abc" in vocab

def test_training_with_duplicate_tokens(trainer):
    corpus = ["abc", "abc", "abc"]
    trainer.train(corpus, target_merges=5)
    vocab = trainer.get_vocab()
    assert len(vocab) > 4
    assert "abc" in vocab or "ab" in vocab

def test_train_empty_corpus(trainer):
    with pytest.raises(ValueError):
        trainer.train([], target_merges=5)

def test_invalid_target_merges(trainer):
    with pytest.raises(ValueError):
        trainer.train(["abc"], target_merges=-1)

def test_invalid_max_iter(trainer):
    with pytest.raises(ValueError):
        trainer.train(["abc"], target_merges=10, max_iter=0)

def test_reset(trainer):
    trainer.update_vocab(["token"])
    trainer.reset()
    vocab = trainer.get_vocab()
    assert len(vocab) == 4
    assert "token" not in vocab

def test_get_vocab_error(trainer):
    trainer.vocab = {}
    with pytest.raises(BPETrainingError):
        trainer.get_vocab()

def test_init_with_invalid_vocab():
    with pytest.raises(TypeError):
        BPETrainer(vocab="invalid_vocab")

def test_init_with_empty_vocab():
    trainer = BPETrainer(vocab={})
    vocab = trainer.get_vocab()
    assert len(vocab) == 4
