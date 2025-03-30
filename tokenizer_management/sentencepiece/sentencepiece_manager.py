import logging
from typing import List, Dict, Any

from .sp_encoder import SPEncoder
from .sp_decoder import SPDecoder
from .sp_trainer import SPTrainer
from .tokenization.sp_pretokenizer import SPPretokenizer
from .tokenization.language_processor import LanguageProcessor
from .tokenization.postprocessor import SPPostprocessor

from tokenizer_management.base_tokenizer_manager import BaseTokenizerManager

logger = logging.getLogger(__name__)


class SPTokenError(Exception):
    """SentencePieceManager hataları için özel exception."""
    pass


class SentencePieceManager(BaseTokenizerManager):
    _instance = None

    def __new__(cls, vocab: Dict[str, dict]):
        if cls._instance is not None:
            logger.info("[+] Mevcut SentencePieceManager örneği serbest bırakılıyor...")
            cls._instance._release()
        cls._instance = super().__new__(cls)
        cls._instance._initialize(vocab)
        return cls._instance

    def _initialize(self, vocab: Dict[str, dict]):
        logger.info("[+] SentencePieceManager başlatılıyor...")

        self.vocab = vocab
        self.encoder = SPEncoder(self.vocab)
        self.decoder = SPDecoder(self.vocab)
        self.trainer = SPTrainer(self.vocab)
        self.pretokenizer = SPPretokenizer()
        self.language_processor = LanguageProcessor()
        self.postprocessor = SPPostprocessor()

        self.update_reverse_vocab()
        logger.info("[+] SentencePieceManager başarıyla başlatıldı.")

    def _release(self):
        logger.info("[+] SentencePieceManager örneği serbest bırakılıyor...")
        self.encoder = None
        self.decoder = None
        self.trainer = None

    def encode(self, text: str) -> List[int]:
        if not text:
            raise SPTokenError("Girdi metni boş olamaz.")
        tokens = self.pretokenizer.tokenize(text)
        token_ids = self.encoder.encode(tokens)
        logger.info(f"[+] Kodlanan token ID'leri: {token_ids}")
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        if not token_ids:
            raise SPTokenError("Token ID listesi boş olamaz.")
        tokens = self.decoder.decode(token_ids)
        decoded_text = self.postprocessor.process(tokens)
        logger.info(f"[+] Çözülen metin: {decoded_text}")
        return decoded_text

    def train(self, corpus: List[str], vocab_size: int):
        if not corpus or not isinstance(corpus, list):
            raise ValueError("Geçersiz eğitim verisi formatı.")
        tokens = [
            token
            for text in corpus
            for token in self.language_processor.process(self.pretokenizer.tokenize(text))
        ]
        self.trainer.train(tokens, vocab_size)
        # Eğitimin sonucunda oluşturulan yeni vocab'ı alıyoruz
        self.vocab.update(self.trainer.get_vocab())
        self.update_reverse_vocab()
        logger.info(f"[+] SentencePiece eğitimi tamamlandı. Vocab boyutu: {vocab_size}")

    def set_vocab(self, vocab: Dict[str, Any]):
        if not isinstance(vocab, dict):
            raise ValueError("Vocab formatı geçersiz, dict tipinde olmalı.")
        self.vocab = vocab
        # Aşağıdaki metodlar SPEncoder, SPDecoder, SPTrainer içinde tanımlı olmalıdır.
        self.encoder.set_vocab(vocab)
        self.decoder.set_vocab(vocab)
        self.trainer.set_vocab(vocab)
        self.update_reverse_vocab()
        logger.info("[+] Vocab başarıyla güncellendi.")

    def get_vocab(self) -> Dict[str, Any]:
        return self.vocab

    def update_reverse_vocab(self) -> None:
        self.decoder.reverse_vocab = {info["id"]: token for token, info in self.vocab.items()}
        logger.info(f"[+] Reverse vocab başarıyla güncellendi. Toplam token sayısı: {len(self.decoder.reverse_vocab)}")

    @property
    def reverse_vocab(self) -> Dict[int, str]:
        return self.decoder.reverse_vocab

    def reset(self):
        logger.warning("[!] SentencePieceManager sıfırlanıyor...")
        # Varsayılan özel tokenlarla sıfırlama
        self.vocab = {
            "<PAD>": {"id": 0, "total_freq": 0, "positions": []},
            "<UNK>": {"id": 1, "total_freq": 0, "positions": []},
            "<BOS>": {"id": 2, "total_freq": 0, "positions": []},
            "<EOS>": {"id": 3, "total_freq": 0, "positions": []},
        }
        self.encoder = SPEncoder(self.vocab)
        self.decoder = SPDecoder(self.vocab)
        self.trainer = SPTrainer(self.vocab)
        self.update_reverse_vocab()
        logger.info("[+] SentencePieceManager başarıyla sıfırlandı.")

    def auto_update_vocab(self, tokens: List[str]) -> None:
        if not tokens or not isinstance(tokens, list):
            raise SPTokenError("[X] Geçerli bir token listesi sağlanmalıdır.")
        new_tokens = [token.lower() for token in tokens if token.lower() not in self.vocab]
        if new_tokens:
            logger.info(f"[+] Yeni tokenler tespit edildi: {new_tokens}")
            existing_ids = {info["id"] for info in self.vocab.values()}
            next_available_id = max(existing_ids, default=3) + 1
            for token in new_tokens:
                if token not in self.vocab:
                    self.vocab[token] = {
                        "id": next_available_id,
                        "total_freq": 1,
                        "positions": []
                    }
                    logger.info(f"[+] Yeni token eklendi: '{token}' -> ID: {next_available_id}")
                    next_available_id += 1
                else:
                    self.vocab[token]["total_freq"] += 1
                    logger.debug(f"[~] Token zaten mevcut: '{token}'")
            self.update_reverse_vocab()
            logger.info("[✓] SentencePiece vocab güncellemesi tamamlandı.")
