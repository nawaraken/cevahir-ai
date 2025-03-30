import pytest
from tokenizer_management.bpe.bpe_encoder import BPEEncoder, BPEEncodingError

@pytest.fixture
def initial_vocab():
    """
    Temel vocab örneği (özel tokenlar dahil).
    """
    return {
        "hello": {"id": 1},
        "world": {"id": 2},
        "test": {"id": 3},
        "<PAD>": {"id": 0},
        "<UNK>": {"id": 1},
        "<BOS>": {"id": 2},
        "<EOS>": {"id": 3}
    }

@pytest.fixture
def encoder(initial_vocab):
    """
    Temel vocab kullanarak encoder oluşturur.
    """
    return BPEEncoder(initial_vocab)

def test_basic_encoding(encoder):
    """
    Temel token kodlama testi.
    """
    tokens = ["hello", "world", "test"]
    result = encoder.encode(tokens)
    assert result == [1, 2, 3]

def test_unknown_token(encoder):
    """
    Bilinmeyen tokenların kodlama testi.
    """
    tokens = ["hello", "unknown", "world"]
    result = encoder.encode(tokens)
    # "unknown" token'ı için UNK token ID'si kullanılmalı
    assert result == [1, 1, 2]

def test_update_vocab():
    """
    Güncellenmiş vocab kontrolü.
    """
    initial_vocab = {
        "hello": {"id": 1},
        "world": {"id": 2},
        "test": {"id": 3},
        "<PAD>": {"id": 0},
        "<UNK>": {"id": 1},
        "<BOS>": {"id": 2},
        "<EOS>": {"id": 3}
    }
    encoder = BPEEncoder(initial_vocab)

    # Yeni vocab oluştur (test token kaldırıldı)
    new_vocab = {
        "hello": {"id": 10},
        "world": {"id": 20},
        "<PAD>": {"id": 0},
        "<UNK>": {"id": 1},
        "<BOS>": {"id": 2},
        "<EOS>": {"id": 3}
    }
    encoder.update_vocab(new_vocab)

    tokens = ["hello", "world", "test"]
    result = encoder.encode(tokens)
    # "test" kaldırıldığı için UNK token kullanılmalı
    assert result == [10, 20, 1]

def test_reset():
    """
    Reset sonrası özel tokenlar korunmalı ve kodlama doğru çalışmalı.
    """
    initial_vocab = {
        "hello": {"id": 1},
        "world": {"id": 2},
        "test": {"id": 3},
        "<PAD>": {"id": 0},
        "<UNK>": {"id": 1},
        "<BOS>": {"id": 2},
        "<EOS>": {"id": 3}
    }
    encoder = BPEEncoder(initial_vocab)

    # Vocab'ı güncelleyelim
    new_vocab = {
        "hello": {"id": 5},
        "world": {"id": 6},
        "test": {"id": 7},
        "<PAD>": {"id": 0},
        "<UNK>": {"id": 1},
        "<BOS>": {"id": 2},
        "<EOS>": {"id": 3}
    }
    encoder.update_vocab(new_vocab)

    # Reset yapalım (özel tokenlar korunmalı)
    encoder.reset()

    tokens = ["hello", "world", "test", "unknown"]
    result = encoder.encode(tokens)

    # Hello, world ve test kaldırıldığı için UNK kullanılacak (ID = 1)
    assert result == [1, 1, 1, 1]

def test_empty_vocab_error():
    """
    Boş vocab yüklenirse ValueError fırlatılmalı.
    """
    empty_vocab = {}
    with pytest.raises(ValueError) as excinfo:
        BPEEncoder(empty_vocab)
    assert "Vocab yüklenemedi veya boş." in str(excinfo.value)

def test_invalid_vocab_type():
    """
    Yanlış tipte vocab yüklenirse TypeError fırlatılmalı.
    """
    invalid_vocab = "invalid_vocab"
    with pytest.raises(TypeError) as excinfo:
        BPEEncoder(invalid_vocab)
    assert "Vocab bir sözlük olmalıdır." in str(excinfo.value)

def test_large_vocab():
    """
    Çok büyük bir vocab dosyası yükleyip test edelim.
    """
    large_vocab = {
        f"token_{i}": {"id": i} for i in range(100_000)
    }
    # Özel tokenlar ekleyelim
    large_vocab.update({
        "<PAD>": {"id": 0},
        "<UNK>": {"id": 1},
        "<BOS>": {"id": 2},
        "<EOS>": {"id": 3}
    })
    encoder = BPEEncoder(large_vocab)

    tokens = ["token_0", "token_99999", "unknown_token"]
    result = encoder.encode(tokens)
    assert result == [0, 99999, 1]  # Bilinmeyen token için UNK ID = 1 olmalı

def test_partial_vocab_update():
    """
    Güncellenen vocab içindeki token ID'lerinin bozulmaması testi.
    """
    initial_vocab = {
        "hello": {"id": 1},
        "world": {"id": 2},
        "test": {"id": 3},
        "<PAD>": {"id": 0},
        "<UNK>": {"id": 1},
        "<BOS>": {"id": 2},
        "<EOS>": {"id": 3}
    }
    encoder = BPEEncoder(initial_vocab)

    # Yeni vocab oluştur (test token kaldırıldı)
    partial_vocab = {
        "hello": {"id": 10},
        "world": {"id": 20},
        "<PAD>": {"id": 0},
        "<UNK>": {"id": 1}
    }
    encoder.update_vocab(partial_vocab)

    tokens = ["hello", "world", "test", "unknown"]
    result = encoder.encode(tokens)
    # Test kaldırıldığı için UNK ID'si kullanılmalı
    assert result == [10, 20, 1, 1]

