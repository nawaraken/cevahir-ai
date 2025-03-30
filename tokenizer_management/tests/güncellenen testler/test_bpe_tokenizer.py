# tokenizer_management/tests/test_bpe_tokenizer.py

import pytest
from typing import List
from tokenizer_management.bpe.bpe_tokenizer import BPETokenizer, BPETokenizerError

# Dummy Encoder: Belirli metinler için öngörülebilir token ID listesi döner.
class DummyEncoder:
    def encode(self, text: str) -> List[int]:
        if text == "error":
            raise Exception("Dummy encoder error")
        if text == "hello world":
            return [1, 2, 3]
        # Basitçe metnin uzunluğunu döndürüyoruz (örnek amaçlı)
        return [len(text)]

# Dummy Decoder: Belirli token ID listeleri için öngörülebilir metin döner.
class DummyDecoder:
    def __init__(self):
        # Örnek ters vocab eşlemesi
        self.reverse_vocab = {1: "hello", 2: "world", 3: ".", 4: "test", 5: "case", 6: "!"}
    def decode(self, token_ids: List[int]) -> str:
        if token_ids == []:
            raise Exception("Empty token_ids error")
        # Her token ID'sini, reverse_vocab'dan getir; bulunamazsa <UNK> döner.
        return " ".join(self.reverse_vocab.get(token, "<UNK>") for token in token_ids)

@pytest.fixture
def dummy_encoder():
    return DummyEncoder()

@pytest.fixture
def dummy_decoder():
    return DummyDecoder()

@pytest.fixture
def bpe_tokenizer(dummy_encoder, dummy_decoder):
    return BPETokenizer(dummy_encoder, dummy_decoder)

def test_encode_success(bpe_tokenizer):
    # "hello world" metni için dummy encoder [1, 2, 3] dönecektir.
    token_ids = bpe_tokenizer.encode("hello world")
    assert token_ids == [1, 2, 3]

def test_encode_empty(bpe_tokenizer):
    # Boş metin verildiğinde hata fırlatılmalıdır.
    with pytest.raises(BPETokenizerError):
        bpe_tokenizer.encode("")

def test_decode_success(bpe_tokenizer):
    # Dummy decoder'a göre [1,2,3] → "hello world ."
    text = bpe_tokenizer.decode([1, 2, 3])
    assert text == "hello world ."

def test_decode_empty(bpe_tokenizer):
    # Boş token ID listesi verildiğinde hata fırlatılmalıdır.
    with pytest.raises(BPETokenizerError):
        bpe_tokenizer.decode([])

def test_get_token_ids(bpe_tokenizer):
    # get_token_ids, encode metodunu çağırır.
    token_ids = bpe_tokenizer.get_token_ids("hello world")
    assert token_ids == [1, 2, 3]

def test_get_text(bpe_tokenizer):
    # get_text, decode metodunu çağırır.
    text = bpe_tokenizer.get_text([1, 2, 3])
    assert text == "hello world ."

def test_encoder_exception(bpe_tokenizer):
    # Dummy encoder "error" metni için Exception fırlatır,
    # bu da BPETokenizerError olarak yakalanmalıdır.
    with pytest.raises(BPETokenizerError):
        bpe_tokenizer.encode("error")
