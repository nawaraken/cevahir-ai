import os
import json
import pytest
from tokenizer_management.vocab.vocab_loader import VocabLoader, VocabLoadError
from tokenizer_management.config import VOCAB_PATH

@pytest.fixture
def vocab_loader():
    return VocabLoader()

@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown_vocab():
    # Orijinal vocab dosyasını yedekle
    backup_path = VOCAB_PATH + ".bak"
    if os.path.exists(VOCAB_PATH):
        os.rename(VOCAB_PATH, backup_path)

    yield
    
    # Test sonrası orijinal vocab dosyasını geri yükle
    if os.path.exists(backup_path):
        if os.path.exists(VOCAB_PATH):
            os.remove(VOCAB_PATH)
        os.rename(backup_path, VOCAB_PATH)

def create_vocab_file(data):
    with open(VOCAB_PATH, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def test_load_vocab_success(vocab_loader):
    vocab_data = {
        "<PAD>": {"id": 0, "total_freq": 0, "positions": []},
        "<UNK>": {"id": 1, "total_freq": 0, "positions": []},
        "<EOS>": {"id": 2, "total_freq": 0, "positions": []},
        "<BOS>": {"id": 3, "total_freq": 0, "positions": []},
        "merhaba": {"id": 4, "total_freq": 15, "positions": [{"sentence": 0, "pos": 0}]}
    }
    create_vocab_file(vocab_data)

    vocab = vocab_loader.load_vocab()
    assert vocab["<PAD>"]["total_freq"] == 0

def test_load_vocab_duplicate_token(vocab_loader):
    vocab_data = {
        "<PAD>": {"id": 1, "total_freq": 0, "positions": []}
    }
    create_vocab_file(vocab_data)

    vocab = vocab_loader.load_vocab()
    assert vocab["<PAD>"]["id"] == 0
    assert vocab["<PAD>"]["total_freq"] == 0




# === Dosya Bulunamadı Testi ===
def test_load_vocab_missing_file(vocab_loader):
    if os.path.exists(VOCAB_PATH):
        os.remove(VOCAB_PATH)

    with pytest.raises(VocabLoadError, match="Vocab dosyası bulunamadı"):
        vocab_loader.load_vocab()

# === Geçersiz JSON Testi ===
def test_load_vocab_invalid_json(vocab_loader):
    with open(VOCAB_PATH, 'w', encoding='utf-8') as file:
        file.write("{invalid json format}")

    with pytest.raises(VocabLoadError, match="JSON çözümleme hatası"):
        vocab_loader.load_vocab()

# === Eksik Anahtar Testi ===
def test_load_vocab_missing_keys(vocab_loader):
    vocab_data = {
        "<PAD>": {"id": 0, "total_freq": 10}  # 'positions' eksik
    }
    create_vocab_file(vocab_data)

    with pytest.raises(VocabLoadError, match="Token içinde 'positions' eksik"):
        vocab_loader.load_vocab()





# === Eksik Temel Tokenler Testi ===
def test_missing_special_tokens(vocab_loader):
    vocab_data = {
        "cevahir": {"id": 4, "total_freq": 1, "positions": [{"sentence": 0, "pos": 0}]}
    }
    create_vocab_file(vocab_data)

    vocab = vocab_loader.load_vocab()
    assert "<PAD>" in vocab
    assert "<UNK>" in vocab
    assert "<BOS>" in vocab
    assert "<EOS>" in vocab

# === Boş Dosya Yükleme Testi ===
def test_get_vocab_empty_file(vocab_loader):
    create_vocab_file({})

    vocab = vocab_loader.get_vocab()
    
    # === Temel Tokenlerin Eklendiğinden Emin Ol ===
    assert len(vocab) == 4
    assert "<PAD>" in vocab
    assert "<UNK>" in vocab
    assert "<BOS>" in vocab
    assert "<EOS>" in vocab


# === Kaydetme Testi ===
def test_save_vocab(vocab_loader):
    vocab_loader.vocab = {
        "<PAD>": {"id": 0, "total_freq": 10, "positions": []},
        "cevahir": {"id": 4, "total_freq": 1, "positions": [{"sentence": 0, "pos": 0}]}
    }
    vocab_loader.save_vocab()

    with open(VOCAB_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
        assert data["cevahir"]["id"] == 4

# === Dosya Oluşturma Testi ===
def test_initialize_vocab_file(vocab_loader):
    if os.path.exists(VOCAB_PATH):
        os.remove(VOCAB_PATH)

    vocab_loader._initialize_vocab_file()
    assert os.path.exists(VOCAB_PATH)

# === 🔥 Yeni Testler 🔥 ===

# 1️⃣ Geçersiz ID Formatı Testi
def test_invalid_token_id_format(vocab_loader):
    vocab_data = {
        "token_1": {"id": "invalid_id", "total_freq": 10, "positions": []}
    }
    create_vocab_file(vocab_data)

    with pytest.raises(VocabLoadError, match="Geçersiz token ID formatı"):
        vocab_loader.load_vocab()

# 2️⃣ Geçersiz Pozisyon Formatı Testi
def test_invalid_position_format(vocab_loader):
    vocab_data = {
        "token_2": {"id": 1, "total_freq": 10, "positions": "invalid_format"}
    }
    create_vocab_file(vocab_data)

    with pytest.raises(VocabLoadError, match="Geçersiz token positions formatı"):
        vocab_loader.load_vocab()

# 3️⃣ JSON Formatı Olmayan Dosya Testi
def test_non_json_file(vocab_loader):
    with open(VOCAB_PATH, 'w', encoding='utf-8') as file:
        file.write("Invalid data format")

    with pytest.raises(VocabLoadError, match="JSON çözümleme hatası"):
        vocab_loader.load_vocab()

# 4️⃣ Eksik Temel Tokenler Ekleme Testi
def test_missing_special_tokens_repair(vocab_loader):
    vocab_data = {
        "token_3": {"id": 1, "total_freq": 10, "positions": []}
    }
    create_vocab_file(vocab_data)

    vocab = vocab_loader.load_vocab()
    assert "<PAD>" in vocab
    assert "<UNK>" in vocab
    assert "<BOS>" in vocab
    assert "<EOS>" in vocab

# 5️⃣ Boş Pozisyon Testi
def test_empty_positions(vocab_loader):
    vocab_data = {
        "token_4": {"id": 2, "total_freq": 5, "positions": []}
    }
    create_vocab_file(vocab_data)

    vocab = vocab_loader.load_vocab()
    assert vocab["token_4"]["positions"] == []

