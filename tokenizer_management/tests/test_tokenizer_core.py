import os
import pytest
from tokenizer_management.tokenizer_core import TokenizerCore

CONFIG = {
    "vocab_path": os.path.join("data", "vocab_lib", "test_vocab.json"),
    "data_directory": "education",
    "batch_size": 4,
    "training": {"epochs": 1, "learning_rate": 0.001},
    "max_seq_length": 64,
    "device": "cpu"
}

@pytest.fixture(scope="module")
def tokenizer():
    return TokenizerCore(CONFIG)

def test_vocab_loaded(tokenizer):
    assert tokenizer.vocab_manager.vocab is not None
    assert isinstance(tokenizer.vocab_manager.vocab, dict)
    assert len(tokenizer.vocab_manager.vocab) > 0

def test_data_loaded(tokenizer):
    data = tokenizer.data_loader.load_data()
    assert isinstance(data, list)
    assert len(data) > 0
    assert all(isinstance(entry, dict) for entry in data)


def test_bos_eos_presence(tokenizer):
    tokenized_data = tokenizer.load_training_data()
    for input_ids, _ in tokenized_data:
        assert input_ids[0] == 2, "BOS token eksik"
        assert input_ids[-1] == 3, "EOS token eksik"

def test_training_data_format_basic(tokenizer):
    tokenized_data = tokenizer.load_training_data()
    assert isinstance(tokenized_data, list)
    assert all(isinstance(pair, tuple) for pair in tokenized_data)
    assert all(len(pair) == 2 for pair in tokenized_data)
    assert all(isinstance(pair[0], list) and isinstance(pair[1], list) for pair in tokenized_data)

def test_training_data_structure(tokenizer):
    """
    Eğitim verisi (input, target) formatında mı?
    Her öğe bir tuple olmalı ve (input_ids, target_ids) listelerinden oluşmalı.
    """
    tokenized_data = tokenizer.load_training_data()

    assert isinstance(tokenized_data, list), "Eğitim verisi bir liste olmalı."
    assert all(isinstance(pair, tuple) for pair in tokenized_data), "Her öğe bir tuple olmalı."
    assert all(len(pair) == 2 for pair in tokenized_data), "Her tuple (input, target) şeklinde 2 eleman içermeli."
    assert all(isinstance(pair[0], list) and isinstance(pair[1], list) for pair in tokenized_data), \
        "Input ve target verileri listelerden oluşmalı."


def test_training_data_format(tokenizer):
    """
    Eğitim verisi (input, target) formatında mı?
    Her öğe bir tuple olmalı ve (input_ids, target_ids) listelerinden oluşmalı.
    """
    tokenized_data = tokenizer.load_training_data()

    assert isinstance(tokenized_data, list), "Eğitim verisi bir liste olmalı."
    assert all(isinstance(pair, tuple) for pair in tokenized_data), "Her öğe bir tuple olmalı."
    assert all(len(pair) == 2 for pair in tokenized_data), "Her tuple (input, target) şeklinde 2 eleman içermeli."
    assert all(isinstance(pair[0], list) and isinstance(pair[1], list) for pair in tokenized_data), \
        "Input ve target verileri listelerden oluşmalı."
