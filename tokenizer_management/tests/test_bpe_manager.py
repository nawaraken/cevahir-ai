import pytest
import os
import json
from tokenizer_management.bpe.bpe_manager import BPEManager, BPETokenError

# VOCAB_PATH tanımı yapılmalı (kaydetme işlemi için)
VOCAB_PATH = "data/vocab_lib/test_vocab.json"

@pytest.fixture
def bpe_manager():
    vocab = {
        "<PAD>": {"id": 0, "total_freq": 1, "positions": []},
        "<UNK>": {"id": 1, "total_freq": 1, "positions": []},
        "<BOS>": {"id": 2, "total_freq": 1, "positions": []},
        "<EOS>": {"id": 3, "total_freq": 1, "positions": []},
        "merhaba": {"id": 4, "total_freq": 1, "positions": []},
        "dünya": {"id": 5, "total_freq": 1, "positions": []},
    }
    return BPEManager(vocab)

# ----------------- TESTLER -----------------

def test_initialize(bpe_manager):
    assert isinstance(bpe_manager.get_vocab(), dict)
    assert len(bpe_manager.get_vocab()) == 6
    assert bpe_manager.get_vocab().get("merhaba")["id"] == 4
    assert bpe_manager.get_vocab().get("dünya")["id"] == 5

def test_encode(bpe_manager):
    text = "Merhaba dünya"
    token_ids = bpe_manager.encode(text)
    assert isinstance(token_ids, list)
    assert token_ids == [4, 5]

def test_decode(bpe_manager):
    token_ids = [4, 5]
    decoded_text = bpe_manager.decode(token_ids)
    assert decoded_text == "Merhaba dünya"

def test_unknown_token_encoding(bpe_manager):
    text = "Bilim adamı"
    token_ids = bpe_manager.encode(text)
    assert token_ids == [1, 1]   # '<UNK>' token'ı dönecek

def test_update_vocab(bpe_manager):
    new_tokens = ["bilim", "adamı"]
    bpe_manager.update_vocab(new_tokens)

    vocab = bpe_manager.get_vocab()
    assert "bilim" in vocab
    assert "adamı" in vocab
    assert vocab["bilim"]["id"] == 6
    assert vocab["adamı"]["id"] == 7

def test_reset_vocab(bpe_manager):
    bpe_manager.reset()
    vocab = bpe_manager.get_vocab()
    
    assert len(vocab) == 4
    assert "<PAD>" in vocab
    assert "<UNK>" in vocab
    assert "<BOS>" in vocab
    assert "<EOS>" in vocab

def test_save_vocab(bpe_manager):
    # Update vocab
    bpe_manager.update_vocab(["bilim", "adamı"])

    # Save updated vocab
    bpe_manager.save_vocab(VOCAB_PATH)

    # Read saved file and check content
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        saved_vocab = json.load(f)

    assert "bilim" in saved_vocab
    assert "adamı" in saved_vocab
    assert saved_vocab["bilim"]["id"] == 6
    assert saved_vocab["adamı"]["id"] == 7

def test_load_vocab(bpe_manager):
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        loaded_vocab = json.load(f)

    manager = BPEManager(loaded_vocab)
    assert "bilim" in manager.get_vocab()
    assert "adamı" in manager.get_vocab()

def test_invalid_vocab_error():
    with pytest.raises(BPETokenError):
        BPEManager(vocab=None)

def test_empty_vocab_encoding(bpe_manager):
    bpe_manager.reset()
    bpe_manager._vocab.clear()  # Vocab'ı manuel olarak tamamen boşaltıyoruz

    with pytest.raises(BPETokenError):
        bpe_manager.encode("Merhaba dünya")


# ----------------- TEMİZLEME -----------------
@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    if os.path.exists(VOCAB_PATH):
        os.remove(VOCAB_PATH)

