import os
import pytest
from tokenizer_management.sentencepiece.sentencepiece_manager import SentencePieceManager, SPTokenError

# Test için geçici dosya yolu
TEST_VOCAB_PATH = "data/test_sp_vocab.json"

@pytest.fixture(scope="function")
def sp_manager():
    """
    Test için SentencePieceManager örneğini başlatır.
    Her test öncesinde vocab dosyasını sıfırlar.
    """
    if os.path.exists(TEST_VOCAB_PATH):
        os.remove(TEST_VOCAB_PATH)
        
    manager = SentencePieceManager(vocab_path=TEST_VOCAB_PATH)
    yield manager
    if os.path.exists(TEST_VOCAB_PATH):
        os.remove(TEST_VOCAB_PATH)

# 1. Başlatma Testi
def test_initialize(sp_manager):
    assert sp_manager is not None
    assert sp_manager.get_vocab() is not None
    assert "<PAD>" in sp_manager.get_vocab()

# 2. Kodlama Testi
def test_encode(sp_manager):
    sp_manager.update_vocab(["merhaba", "dünya"])
    token_ids = sp_manager.encode("Merhaba dünya")
    assert token_ids == [4, 5]

# 3. Çözümleme Testi
def test_decode(sp_manager):
    sp_manager.update_vocab(["merhaba", "dünya"])
    token_ids = sp_manager.encode("Merhaba dünya")
    decoded_text = sp_manager.decode(token_ids)
    assert decoded_text == "merhaba dünya"

# 4. Bilinmeyen Token Testi
def test_unknown_token_encoding(sp_manager):
    token_ids = sp_manager.encode("bilinmeyen kelime")
    assert token_ids == [1, 1]  # "<UNK>" token ID'si 1 olmalı

# 5. Güncelleme Testi
def test_update_vocab(sp_manager):
    sp_manager.update_vocab(["yeni", "kelime"])
    vocab = sp_manager.get_vocab()
    assert "yeni" in vocab
    assert "kelime" in vocab

# 6. Kaydetme Testi
def test_save_vocab(sp_manager):
    sp_manager.update_vocab(["test", "kayıt"])
    sp_manager.save_vocab(TEST_VOCAB_PATH)
    assert os.path.exists(TEST_VOCAB_PATH)

# 7. Yükleme Testi
def test_load_vocab(sp_manager):
    sp_manager.update_vocab(["test", "yükleme"])
    sp_manager.save_vocab(TEST_VOCAB_PATH)

    new_manager = SentencePieceManager(vocab_path=TEST_VOCAB_PATH)
    vocab = new_manager.get_vocab()
    assert "test" in vocab
    assert "yükleme" in vocab

# 8. Sıfırlama Testi
def test_reset(sp_manager):
    sp_manager.update_vocab(["sıfırlama", "deneme"])
    sp_manager.reset()
    vocab = sp_manager.get_vocab()
    assert "sıfırlama" not in vocab
    assert "deneme" not in vocab
    assert "<PAD>" in vocab

# 9. Boş Vocab Kodlama Testi
def test_empty_vocab_encoding(sp_manager):
    sp_manager.reset()
    with pytest.raises(SPTokenError):
        sp_manager.encode("test")

# 10. JSON Format Hatası Testi
def test_invalid_json_format():
    with open(TEST_VOCAB_PATH, 'w', encoding='utf-8') as f:
        f.write("{invalid_json")

    with pytest.raises(SPTokenError):
        SentencePieceManager(vocab_path=TEST_VOCAB_PATH)

# 11. Vocab Dosyası Bulunamama Testi
def test_missing_vocab_file():
    with pytest.raises(SPTokenError):
        SentencePieceManager(vocab_path="invalid_path.json")

# 12. Başlatma Hatası Testi
def test_initialize_failure():
    with pytest.raises(SPTokenError):
        SentencePieceManager(vocab_path=None)
