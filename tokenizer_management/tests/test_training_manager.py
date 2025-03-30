import pytest
import os
import json
from tokenizer_management.training.training_manager import TrainingManager, TrainingManagerError

# Yapılandırma ve test verileri
CONFIG = {
    "batch_size": 16,
    "vocab_threshold": 5
}

VALID_CORPUS = [
    "Merhaba dünya!",
    "Python harika bir programlama dili.",
    "Test verisi oluşturuyoruz."
]

INVALID_CORPUS = None
VOCAB_PATH = "data/test_training_vocab.json"

@pytest.fixture
def training_manager():
    return TrainingManager(CONFIG)

# === Başlatma Testleri ===
def test_initialize(training_manager):
    assert training_manager.config == CONFIG
    assert isinstance(training_manager.preprocessor, object)
    assert isinstance(training_manager.tokenizer, object)
    assert isinstance(training_manager.postprocessor, object)
    assert isinstance(training_manager.tensorizer, object)

def test_invalid_config():
    with pytest.raises(TypeError):
        TrainingManager(config="invalid_config")

# === Train Testleri ===
def test_train_empty_corpus(training_manager):
    training_manager.train([], target_vocab_size=10)
    assert len(training_manager.get_vocab()) == 0

def test_train_valid_corpus(training_manager):
    training_manager.train(VALID_CORPUS, target_vocab_size=10)
    vocab = training_manager.get_vocab()
    assert len(vocab) > 0
    assert "merhaba" in vocab
    assert vocab["merhaba"]["total_freq"] > 0

# === Tensorize Testleri ===
def test_tensorize_valid_input(training_manager):
    training_manager.train(VALID_CORPUS, target_vocab_size=10)
    tensor_data = training_manager.tensorize(VALID_CORPUS)
    assert tensor_data is not None

def test_tensorize_invalid_input(training_manager):
    with pytest.raises(TrainingManagerError):
        training_manager.tensorize(INVALID_CORPUS)

# === Vocab Güncelleme Testleri ===
def test_empty_vocab_update(training_manager):
    with pytest.raises(TypeError):
        training_manager.update_vocab("invalid_vocab")

def test_update_vocab(training_manager):
    new_vocab = {
        "yeni": {"id": 10, "total_freq": 5, "positions": []}
    }
    training_manager.update_vocab(new_vocab)
    vocab = training_manager.get_vocab()
    assert "yeni" in vocab
    assert vocab["yeni"]["total_freq"] == 5

# === Vocab Dosya Kaydetme Testleri ===
def test_save_vocab(training_manager):
    training_manager.train(VALID_CORPUS, target_vocab_size=10)
    training_manager.save_vocab(VOCAB_PATH)
    
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) > 0
    os.remove(VOCAB_PATH)

def test_invalid_save_vocab(training_manager):
    invalid_path = "/invalid_path/test_vocab.json"
    training_manager.train(VALID_CORPUS, target_vocab_size=10)
    with pytest.raises(OSError):
        training_manager.save_vocab(invalid_path)

# === Vocab Kontrol Testleri ===
def test_get_vocab(training_manager):
    training_manager.train(VALID_CORPUS, target_vocab_size=10)
    vocab = training_manager.get_vocab()
    assert isinstance(vocab, dict)
    assert len(vocab) > 0

# === Hatalı ve Boş Vocab Testleri ===
def test_empty_vocab(training_manager):
    with pytest.raises(TrainingManagerError):
        training_manager.tensorize([])

# === Geçersiz Vocab Dosyası Yükleme Testi ===
def test_load_invalid_vocab():
    with pytest.raises(FileNotFoundError):
        with open("invalid_path.json", "r") as f:
            json.load(f)

