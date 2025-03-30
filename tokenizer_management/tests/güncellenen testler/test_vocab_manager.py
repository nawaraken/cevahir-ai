import os
import json
import pytest
from tokenizer_management.vocab.vocab_manager import VocabManager
from tokenizer_management.config import VOCAB_PATH, TOKEN_MAPPING

# ============ FIXTURE ============

@pytest.fixture(scope="function", autouse=True)
def backup_and_restore_vocab(request):
    backup_path = VOCAB_PATH + ".bak"
    
    # ✅ Yedek dosyası varsa önce siliyoruz
    if os.path.exists(backup_path):
        os.remove(backup_path)

    # ✅ Asıl dosya varsa yedek alıyoruz
    if os.path.exists(VOCAB_PATH):
        os.rename(VOCAB_PATH, backup_path)
    
    yield
    
    # ✅ Test başarısız olursa yedeği geri yüklüyoruz
    call_result = request.node._store.get("call", None)

    if call_result and call_result.excinfo is not None:
        if os.path.exists(backup_path):
            if os.path.exists(VOCAB_PATH):
                os.remove(VOCAB_PATH)
            os.rename(backup_path, VOCAB_PATH)
    
    # ✅ Test başarılı olursa yedeği siliyoruz
    if os.path.exists(backup_path):
        os.remove(backup_path)

# ============ FIXTURE - VOCAB ============

@pytest.fixture
def vocab_manager():
    return VocabManager()

# ============ HELPERS ============

def get_proper_initial_vocab(extra_tokens=None):
    """
    Özel tokenlerle başlayan doğru formatta initial_vocab oluşturur.
    """
    vocab = {
        "<PAD>": {"id": TOKEN_MAPPING["<PAD>"], "total_freq": 0, "positions": []},
        "<UNK>": {"id": TOKEN_MAPPING["<UNK>"], "total_freq": 0, "positions": []},
        "<EOS>": {"id": TOKEN_MAPPING["<EOS>"], "total_freq": 0, "positions": []},
        "<BOS>": {"id": TOKEN_MAPPING["<BOS>"], "total_freq": 0, "positions": []},
    }
    next_id = max(token["id"] for token in vocab.values()) + 1
    if extra_tokens:
        for token in extra_tokens:
            vocab[token] = {"id": next_id, "total_freq": 0, "positions": []}
            next_id += 1
    return vocab

@pytest.fixture(autouse=True)
def reset_singleton():
    VocabManager._instance = None
    yield
    VocabManager._instance = None

# ============ TESTLER ============

# 1️⃣ Vocab dosyası mevcutsa yüklenmeli
def test_load_vocab_exists(vocab_manager):
    initial_vocab = get_proper_initial_vocab(["merhaba"])
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(initial_vocab, f)

    vocab = vocab_manager.load_vocab()
    
    assert "merhaba" in vocab
    assert vocab["merhaba"]["id"] == 4
    assert isinstance(vocab["merhaba"]["total_freq"], int)
    assert isinstance(vocab["merhaba"]["positions"], list)

    for token in TOKEN_MAPPING:
        assert token in vocab
        assert vocab[token]["id"] == TOKEN_MAPPING[token]
        assert isinstance(vocab[token]["total_freq"], int)
        assert isinstance(vocab[token]["positions"], list)

    assert os.path.exists(VOCAB_PATH)


# 2️⃣ Vocab dosyası yoksa yeni bir vocab oluşturulmalı
def test_load_vocab_not_exists(vocab_manager):
    if os.path.exists(VOCAB_PATH):
        os.remove(VOCAB_PATH)

    vocab = vocab_manager.load_vocab()
    expected_vocab = get_proper_initial_vocab()
    
    assert vocab == expected_vocab
    assert os.path.exists(VOCAB_PATH)

# 3️⃣ Yeni tokenlar vocab'e eklenmeli
def test_update_vocab(vocab_manager):
    initial_vocab = get_proper_initial_vocab(["merhaba"])
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(initial_vocab, f)

    # ✅ Artık None döndürüyor hatası olmayacak
    updated_vocab = vocab_manager.update_vocab(["dünya", "merhaba"])
    
    assert "dünya" in updated_vocab
    assert updated_vocab["dünya"]["total_freq"] == 1
    assert updated_vocab["merhaba"]["total_freq"] == 1


# 4️⃣ Token listesi ile yeni vocab oluşturulmalı
def test_build_vocab(vocab_manager):
    if os.path.exists(VOCAB_PATH):
        os.remove(VOCAB_PATH)

    new_vocab = vocab_manager.build_vocab(["cevahir", "yapay", "zeka"])
    expected_vocab = get_proper_initial_vocab(["cevahir", "yapay", "zeka"])

    assert new_vocab == expected_vocab
    assert os.path.exists(VOCAB_PATH)
    assert new_vocab["cevahir"]["id"] == 4
    assert new_vocab["yapay"]["id"] == 5
    assert new_vocab["zeka"]["id"] == 6

# 5️⃣ Vocab sıfırlanmalı ve dosya silinmeli
def test_reset_vocab(vocab_manager):
    initial_vocab = get_proper_initial_vocab(["test"])
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(initial_vocab, f)

    vocab_manager.reset_vocab()

    assert not os.path.exists(VOCAB_PATH)
    assert vocab_manager.vocab == {}

# 6️⃣ Vocab içeriği doğru okunmalı
def test_get_vocab(vocab_manager):
    initial_vocab = get_proper_initial_vocab(["merhaba"])
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(initial_vocab, f)

    vocab = vocab_manager.get_vocab()

    assert "merhaba" in vocab
    assert vocab["merhaba"]["id"] == 4
    assert isinstance(vocab["merhaba"]["total_freq"], int)
    assert isinstance(vocab["merhaba"]["positions"], list)

    for token in TOKEN_MAPPING:
        assert token in vocab
        assert vocab[token]["id"] == TOKEN_MAPPING[token]
        assert isinstance(vocab[token]["total_freq"], int)
        assert isinstance(vocab[token]["positions"], list)


# 7️⃣ Yeni token ID'si sıralı olarak atanmalı
def test_get_next_token_id(vocab_manager):
    initial_vocab = get_proper_initial_vocab(["merhaba", "dünya"])
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(initial_vocab, f)

    next_id = vocab_manager.get_next_token_id()
    assert next_id == 6

# 8️⃣ Dosyanın gerçekten oluşturulduğunu kontrol et
def test_vocab_saved_correctly(vocab_manager):
    initial_vocab = get_proper_initial_vocab(["merhaba", "dünya"])
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(initial_vocab, f)

    vocab_manager.save_vocab()
    assert os.path.exists(VOCAB_PATH)

    loaded_vocab = vocab_manager.get_vocab()

    for token in ["merhaba", "dünya"]:
        assert token in loaded_vocab
        assert isinstance(loaded_vocab[token]["total_freq"], int)
        assert isinstance(loaded_vocab[token]["positions"], list)


# 9️⃣ Özel token güncellenmesi testi
def test_update_special_tokens(vocab_manager):
    sentence = "Merhaba cevahir ben muhammed babanım seni çok seviyorum oğlum."
    tokens = ["<BOS>"] + sentence.split() + ["<EOS>"]

    updated_vocab = vocab_manager.update_vocab(tokens)

    assert updated_vocab["<BOS>"]["total_freq"] >= 1
    assert updated_vocab["<EOS>"]["total_freq"] >= 1

    assert 0 in updated_vocab["<BOS>"]["positions"]
    assert len(sentence.split()) + 1 in updated_vocab["<EOS>"]["positions"]

    for token in sentence.split():
        assert token in updated_vocab
        assert updated_vocab[token]["total_freq"] >= 1
        assert isinstance(updated_vocab[token]["positions"], list)
        assert len(updated_vocab[token]["positions"]) >= 1



def test_update_multiple_sentences(vocab_manager):
    sentences = [
        "Merhaba cevahir ben muhammed babanım seni çok seviyorum oğlum.",
        "Cevahir oğlum, bugün nasılsın? Seni çok özledim.",
        "Güzel oğlum, seninle gurur duyuyorum.",
        "Merhaba cevahir, kahvaltını yaptın mı? Seni seviyorum.",
        "Oğlum, bugün okul nasıl geçti? Yeni bir şey öğrendin mi?"
    ]

    all_tokens = []
    for sentence in sentences:
        tokens = sentence.split()
        all_tokens.extend(tokens)

    # ✅ Tokenleri güncelle
    updated_vocab = vocab_manager.update_vocab(all_tokens)

    # ✅ Özel tokenların frekans ve pozisyon kontrolü
    assert updated_vocab["<PAD>"]["total_freq"] == 1
    assert updated_vocab["<UNK>"]["total_freq"] == 1
    assert updated_vocab["<BOS>"]["total_freq"] == 1
    assert updated_vocab["<EOS>"]["total_freq"] == 1

    # ✅ Özel token pozisyonlarının güncellenmesi
    assert len(updated_vocab["<PAD>"]["positions"]) == 1
    assert len(updated_vocab["<UNK>"]["positions"]) == 1
    assert len(updated_vocab["<BOS>"]["positions"]) == 1
    assert len(updated_vocab["<EOS>"]["positions"]) == 1

    # ✅ Diğer tokenların frekanslarını ve pozisyonlarını kontrol et
    expected_tokens = set(all_tokens)
    for token in expected_tokens:
        assert token in updated_vocab
        assert updated_vocab[token]["total_freq"] == all_tokens.count(token)

    # ✅ Dosyanın gerçekten kaydedildiğinden emin ol
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        saved_vocab = json.load(f)

    assert saved_vocab == updated_vocab

def test_update_multiple_sentences(vocab_manager):
    sentences = [
        "Merhaba Cevahir, nasılsın?",
        "Bugün hava çok güzel.",
        "Sana bir sürprizim var.",
        "Bu güncelleme çok kritik.",
        "Son testi de geçelim!"
    ]

    for sentence in sentences:
        tokens = ["<BOS>"] + sentence.split() + ["<EOS>"]
        vocab_manager.update_vocab(tokens)

    vocab = vocab_manager.get_vocab()

    # Özel token frekansları
    assert vocab["<BOS>"]["total_freq"] >= len(sentences)
    assert vocab["<EOS>"]["total_freq"] >= len(sentences)

    # Özel token pozisyonları
    assert len(vocab["<BOS>"]["positions"]) >= 1
    assert all(pos == 0 for pos in vocab["<BOS>"]["positions"])

    # Standart tokenlar için kontroller
    for sentence in sentences:
        for token in sentence.split():
            assert token in vocab
            assert vocab[token]["total_freq"] >= 1
            assert isinstance(vocab[token]["positions"], list)
            assert len(vocab[token]["positions"]) >= 1
