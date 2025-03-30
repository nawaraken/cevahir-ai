import pytest
from tokenizer_management.sentencepiece.sp_tokenizer import SentencePieceTokenizer

# Dummy manager sınıfı; SentencePieceManager'ın yerine geçer.
class DummySPManager:
    def __init__(self):
        # Basit bir sözlükle örnek vocab tanımı yapıyoruz.
        self.vocab = {"hello": 1, "world": 2, "<UNK>": 0}

    def encode(self, text: str) -> list:
        # Basitçe, boşlukla ayrılmış token'ları vocab üzerinden ID'ye çevirir.
        tokens = text.split()
        return [self.vocab.get(token.lower(), self.vocab["<UNK>"]) for token in tokens]

    def decode(self, token_ids: list) -> str:
        # Vocab'ı tersine çevirip, token ID'leri metne dönüştürür.
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        # Eğer token id'si vocab'da yoksa "<UNK>" döner.
        return " ".join([reverse_vocab.get(token_id, "<UNK>") for token_id in token_ids])

    def train(self, corpus: list, vocab_size: int):
        # Dummy eğitim: Vocab'a "trained" kelimesini ekler.
        self.vocab["trained"] = max(self.vocab.values()) + 1

    def update_vocab(self, new_tokens: list):
        for token in new_tokens:
            token_lower = token.lower()
            if token_lower not in self.vocab:
                self.vocab[token_lower] = max(self.vocab.values()) + 1

    def reset(self):
        self.vocab = {}

    def get_vocab(self) -> dict:
        return self.vocab

    def get_token_ids(self, text: str) -> list:
        return self.encode(text)

    def get_text(self, token_ids: list) -> str:
        return self.decode(token_ids)


@pytest.fixture
def dummy_manager():
    return DummySPManager()


@pytest.fixture
def sp_tokenizer(dummy_manager):
    # Dummy manager örneği kullanılarak SentencePieceTokenizer oluşturulur.
    return SentencePieceTokenizer(manager=dummy_manager)


def test_encode(sp_tokenizer):
    text = "Hello World"
    token_ids = sp_tokenizer.encode(text)
    # Beklenen: "hello" -> 1, "world" -> 2
    assert token_ids == [1, 2]


def test_decode(sp_tokenizer):
    token_ids = [1, 2]
    text = sp_tokenizer.decode(token_ids)
    # Beklenen: "hello world" (küçük harf olarak)
    assert text == "hello world"


def test_get_token_ids_and_text(sp_tokenizer):
    text = "Hello World"
    token_ids = sp_tokenizer.get_token_ids(text)
    assert isinstance(token_ids, list)
    decoded_text = sp_tokenizer.get_text(token_ids)
    assert isinstance(decoded_text, str)
    # Hem encode hem decode işleminin doğru çalıştığını kontrol ediyoruz.
    assert decoded_text == "hello world"


def test_train(sp_tokenizer, dummy_manager):
    corpus = ["Some text", "More text"]
    sp_tokenizer.train(corpus, vocab_size=5)
    vocab = dummy_manager.get_vocab()
    # Dummy eğitimde, "trained" kelimesi ekleniyor.
    assert "trained" in vocab


def test_update_vocab(sp_tokenizer, dummy_manager):
    sp_tokenizer.update_vocab(["NewToken"])
    vocab = dummy_manager.get_vocab()
    # Yeni eklenen token küçük harf olarak eklenmeli.
    assert "newtoken" in vocab


def test_reset(sp_tokenizer, dummy_manager):
    sp_tokenizer.update_vocab(["NewToken"])
    sp_tokenizer.reset()
    vocab = dummy_manager.get_vocab()
    assert vocab == {}
