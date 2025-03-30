import pytest
from tokenizer_management.bpe.bpe_decoder import BPEDecoder, BPEDecodingError

@pytest.fixture
def initial_vocab():
    return {
        "<PAD>": {"id": 0},
        "<UNK>": {"id": 1},
        "<BOS>": {"id": 2},
        "<EOS>": {"id": 3},
        "hello": {"id": 4},
        "world": {"id": 5},
        ".": {"id": 6},
    }

@pytest.fixture
def decoder(initial_vocab):
    return BPEDecoder(initial_vocab)

def test_build_reverse_vocab(decoder):
    """
    Reverse vocab'ın doğru şekilde oluşturulduğunu kontrol eder.
    """
    reverse_vocab = decoder.reverse_vocab
    expected = {
        0: "<PAD>",
        1: "<UNK>",
        2: "<BOS>",
        3: "<EOS>",
        4: "hello",
        5: "world",
        6: "."
    }
    assert reverse_vocab == expected

def test_decode_success(decoder):
    """
    Kodlanmış token ID'lerini başarılı şekilde çözümlediğini test eder.
    """
    token_ids = [4, 5, 6]  # "hello world ."
    decoded = decoder.decode(token_ids)
    assert decoded == "hello world ."

def test_decode_unknown_token(decoder):
    """
    Bilinmeyen bir token ID verildiğinde <UNK> token'ini döndürmeli.
    """
    token_ids = [4, 99, 6]  # 99 vocab içinde yok, <UNK> olarak işlenecek
    decoded = decoder.decode(token_ids)
    assert decoded == "hello <UNK> ."

def test_update_vocab(decoder):
    """
    Vocab güncelleme işleminin doğru çalıştığını test eder.
    """
    new_vocab = {
        "<PAD>": {"id": 0},
        "<UNK>": {"id": 1},
        "<BOS>": {"id": 2},
        "<EOS>": {"id": 3},
        "goodbye": {"id": 4},
        "moon": {"id": 5}
    }
    decoder.update_vocab(new_vocab)

    expected_reverse = {
        0: "<PAD>",
        1: "<UNK>",
        2: "<BOS>",
        3: "<EOS>",
        4: "goodbye",
        5: "moon"
    }
    assert decoder.reverse_vocab == expected_reverse

def test_reset(decoder):
    """
    Reset sonrası özel tokenların korunup korunmadığını test eder.
    """
    new_vocab = {
        "<PAD>": {"id": 0},
        "<UNK>": {"id": 1},
        "<BOS>": {"id": 2},
        "<EOS>": {"id": 3},
        "reset": {"id": 4}
    }
    decoder.update_vocab(new_vocab)

    # Reset yapalım
    decoder.reset()

    expected = {
        0: "<PAD>",
        1: "<UNK>",
        2: "<BOS>",
        3: "<EOS>",
    }
    assert decoder.reverse_vocab == expected

def test_decode_empty_token_ids(decoder):
    """
    Boş token ID listesi verildiğinde BPEDecodingError fırlatmalı.
    """
    with pytest.raises(BPEDecodingError):
        decoder.decode([])

def test_partial_vocab_update(decoder):
    """
    Eksik vocab güncellemesinde özel tokenlar korunmalı.
    """
    new_vocab = {
        "hello": {"id": 10},
        "world": {"id": 20},
        "<PAD>": {"id": 0},
        "<UNK>": {"id": 1}
    }

    # Eksik vocab güncelle
    decoder.update_vocab(new_vocab)

    expected = {
        0: "<PAD>",
        1: "<UNK>",
        2: "<BOS>",
        3: "<EOS>",
        10: "hello",
        20: "world"
    }

    assert decoder.reverse_vocab == expected

def test_missing_special_tokens():
    """
    Eksik özel tokenlar olduğunda ValueError fırlatmalı.
    """
    invalid_vocab = {
        "hello": {"id": 10},
        "world": {"id": 20}
    }
    with pytest.raises(ValueError):
        BPEDecoder(invalid_vocab)

def test_unknown_token_handling(decoder):
    """
    Bilinmeyen token ID'sinde `'<UNK>'` dönmesini test eder.
    """
    token_ids = [4, 99, 5]
    decoded = decoder.decode(token_ids)
    assert decoded == "hello <UNK> world"
