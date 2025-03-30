import os
import json
import pytest
from tokenizer_management.vocab.vocab_updater import VocabUpdater, VocabUpdateError
from tokenizer_management.config import VOCAB_PATH

# === Fixture ===
@pytest.fixture
def sample_vocab():
    return {
        "<PAD>": 0,
        "<UNK>": 1,
        "<EOS>": 2,
        "<BOS>": 3,
        "merhaba": 4,
        "dünya": 5
    }

@pytest.fixture
def vocab_updater(sample_vocab):
    return VocabUpdater(sample_vocab.copy())

@pytest.fixture(autouse=True)
def setup_and_teardown_vocab():
    backup_path = VOCAB_PATH + ".bak"
    if os.path.exists(VOCAB_PATH):
        os.rename(VOCAB_PATH, backup_path)

    yield

    if os.path.exists(backup_path):
        if os.path.exists(VOCAB_PATH):
            os.remove(VOCAB_PATH)
        os.rename(backup_path, VOCAB_PATH)


# === 🧪 TESTLER ===

# ✅ Başarıyla yeni token ekleniyor mu?
def test_add_new_token(vocab_updater):
    vocab_updater.add_token("cevahir")
    assert "cevahir" in vocab_updater.vocab
    assert vocab_updater.vocab["cevahir"] == 6

# ✅ Mevcut token tekrar eklendiğinde warning veriyor mu?
def test_add_existing_token(vocab_updater, caplog):
    vocab_updater.add_token("merhaba")
    assert "Token zaten mevcut" in caplog.text

# ✅ Token ID manuel olarak verilebiliyor mu?
def test_add_token_with_specific_id(vocab_updater):
    vocab_updater.add_token("cevahir", 10)
    assert vocab_updater.vocab["cevahir"] == 10

# ✅ Token ID çakışma hatası kontrolü
def test_add_token_id_conflict(vocab_updater):
    with pytest.raises(VocabUpdateError, match="Token ID çakışması"):
        vocab_updater.add_token("cevahir", 1)

# ✅ Mevcut token ID güncellemesi başarılı mı?
def test_update_existing_token_id(vocab_updater):
    vocab_updater.update_token("merhaba", 10)
    assert vocab_updater.vocab["merhaba"] == 10

# ✅ Olmayan token güncellemesi hata fırlatıyor mu?
def test_update_nonexistent_token(vocab_updater):
    with pytest.raises(VocabUpdateError, match="Token bulunamadı"):
        vocab_updater.update_token("cevahir", 10)

# ✅ Güncellemede token ID çakışması hatası veriyor mu?
def test_update_token_id_conflict(vocab_updater):
    with pytest.raises(VocabUpdateError, match="Token ID çakışması"):
        vocab_updater.update_token("merhaba", 1)

# ✅ Başarıyla token siliniyor mu?
def test_remove_existing_token(vocab_updater):
    vocab_updater.remove_token("merhaba")
    assert "merhaba" not in vocab_updater.vocab

# ✅ Silinemeyen token için uyarı veriliyor mu?
def test_remove_nonexistent_token(vocab_updater, caplog):
    vocab_updater.remove_token("cevahir")
    assert "Token bulunamadığı için silinemedi" in caplog.text

# ✅ Vocab dosyası başarıyla kaydediliyor mu?
def test_save_vocab(vocab_updater):
    vocab_updater.add_token("cevahir")
    vocab_updater.save_vocab()

    assert os.path.exists(VOCAB_PATH)

    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
        assert data["cevahir"] == 6

# ✅ Dosya izin hatası doğru şekilde yakalanıyor mu?
def test_save_vocab_permission_error(vocab_updater, mocker):
    mocker.patch("builtins.open", side_effect=PermissionError("İzin hatası"))

    with pytest.raises(VocabUpdateError, match="Vocab dosyası yazılamadı"):
        vocab_updater.save_vocab()

# ✅ Vocab boyutunu doğru döndürüyor mu?
def test_get_vocab_size(vocab_updater):
    size = vocab_updater.get_vocab_size()
    assert size == 6  # sample_vocab'daki başlangıç tokenleri sayısı
