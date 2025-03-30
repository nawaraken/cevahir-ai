import os
import json
import pytest
from tokenizer_management.vocab.vocab_builder import VocabBuilder, VocabBuildError
from tokenizer_management.config import VOCAB_PATH

# === Pytest için test verilerini hazırlayan fixture ===
@pytest.fixture
def vocab_builder():
    special_tokens = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<EOS>": 2,
        "<BOS>": 3
    }
    return VocabBuilder(special_tokens=special_tokens)

# === VOCAB_PATH dosyasını yedekleyen fixture ===
@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown_vocab():
    backup_path = VOCAB_PATH + ".bak"
    if os.path.exists(VOCAB_PATH):
        os.rename(VOCAB_PATH, backup_path)

    yield
    
    if os.path.exists(backup_path):
        if os.path.exists(VOCAB_PATH):
            os.remove(VOCAB_PATH)
        os.rename(backup_path, VOCAB_PATH)

# === 🧪 Testler ===

# ✅ 1️⃣ Başarılı bir şekilde liste ile vocab oluşturma
def test_build_from_list(vocab_builder):
    token_list = ["merhaba", "dünya", "merhaba", "cevahir"]

    # Vocab oluştur
    vocab_builder.build_from_list(token_list)

    # Doğru tokenler var mı kontrol et
    vocab = vocab_builder.vocab
    assert vocab["merhaba"] is not None
    assert vocab["dünya"] is not None
    assert vocab["cevahir"] is not None

    # ID'ler unique olmalı
    assert len(set(vocab.values())) == len(vocab)

    # Vocab boyutu doğru mu?
    assert vocab_builder.get_vocab_size() == 7


# ✅ 2️⃣ Başarılı bir şekilde metinden vocab oluşturma
def test_build_from_text(vocab_builder):
    text = "merhaba dünya merhaba cevahir merhaba"
    
    # Vocab oluştur
    vocab_builder.build_from_text(text, min_frequency=2)

    vocab = vocab_builder.vocab

    assert "merhaba" in vocab
    assert "dünya" not in vocab   # min_frequency = 2 olduğu için eklenmemeli
    assert "cevahir" not in vocab # min_frequency = 2 olduğu için eklenmemeli

    # Vocab boyutu kontrolü
    assert vocab_builder.get_vocab_size() == 5


# ✅ 3️⃣ Token ID çakışma hatası kontrolü
def test_token_id_conflict(vocab_builder):
    vocab_builder.add_token("token_1", token_id=10)

    # Aynı ID ile başka bir token eklemeye çalışınca hata vermeli
    with pytest.raises(VocabBuildError, match="Token ID çakışması"):
        vocab_builder.add_token("token_2", token_id=10)


# ✅ 4️⃣ Geçersiz string formatı hatası kontrolü
def test_invalid_text_input(vocab_builder):
    with pytest.raises(TypeError, match="Girdi metni bir string olmalıdır"):
        vocab_builder.build_from_text(12345)


# ✅ 5️⃣ Geçersiz liste formatı hatası kontrolü
def test_invalid_list_input(vocab_builder):
    with pytest.raises(TypeError, match="Girdi bir liste olmalıdır"):
        vocab_builder.build_from_list("Bu bir liste değil")


# ✅ 6️⃣ Başarılı bir şekilde vocab dosyasını kaydetme
def test_save_vocab(vocab_builder):
    token_list = ["merhaba", "dünya", "cevahir"]
    vocab_builder.build_from_list(token_list)
    vocab_builder.save_vocab()

    assert os.path.exists(VOCAB_PATH)

    # JSON formatı doğru mu kontrol et
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["merhaba"] is not None
        assert data["dünya"] is not None
        assert data["cevahir"] is not None


# ✅ 7️⃣ Dosya yazma hatası kontrolü
def test_save_vocab_permission_error(vocab_builder, mocker):
    mocker.patch("builtins.open", side_effect=PermissionError("İzin hatası"))

    with pytest.raises(VocabBuildError, match="Vocab dosyası yazılamadı"):
        vocab_builder.save_vocab()


# ✅ 8️⃣ Token tekrar etme kontrolü
def test_add_duplicate_token(vocab_builder):
    vocab_builder.add_token("token_1")
    
    # Tekrar token eklediğinde uyarı vermeli ama exception atmamalı
    vocab_builder.add_token("token_1")
    vocab = vocab_builder.vocab

    assert vocab["token_1"] is not None
    assert len(vocab) == 5  # Özel tokenlar + token_1


# ✅ 9️⃣ Vocab boyutunu kontrol etme
def test_get_vocab_size(vocab_builder):
    token_list = ["merhaba", "dünya", "cevahir"]
    vocab_builder.build_from_list(token_list)

    size = vocab_builder.get_vocab_size()
    assert size == 7


# ✅ 🔥 EKSTRA TEST 🔥: Boş metin girişi
def test_build_from_empty_text(vocab_builder):
    vocab_builder.build_from_text("")
    vocab = vocab_builder.vocab

    assert len(vocab) == 4  # Sadece özel tokenler olmalı


# ✅ 🔥 EKSTRA TEST 🔥: Boş liste girişi
def test_build_from_empty_list(vocab_builder):
    vocab_builder.build_from_list([])
    vocab = vocab_builder.vocab

    assert len(vocab) == 4  # Sadece özel tokenler olmalı


# ✅ 🔥 EKSTRA TEST 🔥: Çift token ID hatası testi
def test_duplicate_token_id(vocab_builder):
    vocab_builder.add_token("token_1", 10)
    with pytest.raises(VocabBuildError, match="Token ID çakışması"):
        vocab_builder.add_token("token_2", 10)

