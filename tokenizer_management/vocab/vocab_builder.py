import json
import logging
import os
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class VocabBuildError(Exception):
    """Vocab oluşturma sırasında oluşan hatalar için özel exception."""
    pass


class VocabBuilder:
    def __init__(self, vocab_path: str, special_tokens: Optional[Dict[str, int]] = None) -> None:
        self.vocab_path = vocab_path
        self.vocab: Dict[str, Dict[str, any]] = {}
        self.token_set = set()

        # Özel tokenları başlangıçta güvenli şekilde ekle
        if special_tokens:
            for token, token_id in special_tokens.items():
                self.add_token(token, token_id=token_id, total_freq=0, positions=[])

    def add_token(self, token: str, token_id: Optional[int] = None,
                  total_freq: int = 0, positions: Optional[List[int]] = None) -> None:
        token = token.strip()
        if not token:
            raise VocabBuildError("Token boş olamaz.")

        if token in self.token_set:
            logger.debug(f"[!] Token zaten mevcut: '{token}' -> ID: {self.vocab[token]['id']}")
            return

        if token_id is None:
            token_id = max((v["id"] for v in self.vocab.values()), default=-1) + 1

        if any(v["id"] == token_id for v in self.vocab.values()):
            raise VocabBuildError(f"Token ID çakışması: {token_id} zaten mevcut.")

        self.vocab[token] = {
            "id": token_id,
            "total_freq": total_freq,
            "positions": positions if positions is not None else []
        }
        self.token_set.add(token)

        logger.info(f"[+] Yeni token eklendi: '{token}' -> ID: {token_id}")

    def build_from_text(self, text: str, min_frequency: int = 1) -> None:
        if not isinstance(text, str):
            raise TypeError("Girdi metni bir string olmalıdır.")

        token_freq: Dict[str, int] = {}
        token_positions: Dict[str, List[int]] = {}

        tokens = text.split()
        for idx, token in enumerate(tokens):
            token = token.strip()
            if token:
                token_freq[token] = token_freq.get(token, 0) + 1
                token_positions.setdefault(token, []).append(idx)

        for token, freq in token_freq.items():
            if freq >= min_frequency:
                self.add_token(token, total_freq=freq, positions=token_positions.get(token, []))

        logger.info(f"[+] Metinden vocab oluşturuldu. Eklenen token sayısı: {len(self.vocab)}")

    def build_from_list(self, token_list: List[str]) -> None:
        if not isinstance(token_list, list):
            raise TypeError("Girdi bir liste olmalıdır.")

        token_freq: Dict[str, int] = {}
        token_positions: Dict[str, List[int]] = {}

        for idx, token in enumerate(token_list):
            if not isinstance(token, str):
                logger.warning(f"[!] Geçersiz token formatı: '{token}'")
                continue

            token = token.strip()
            if not token:
                continue

            token_freq[token] = token_freq.get(token, 0) + 1
            token_positions.setdefault(token, []).append(idx)

        for token, freq in token_freq.items():
            self.add_token(token, total_freq=freq, positions=token_positions[token])

        logger.info(f"[+] Listeden vocab oluşturuldu. Toplam token sayısı: {len(self.vocab)}")

    def build_vocab(self, token_list: List[str]) -> Dict[str, Dict[str, any]]:
        self.build_from_list(token_list)
        return self.vocab

    def save_vocab(self) -> None:
        try:
            vocab_dir = os.path.dirname(self.vocab_path)
            if not os.path.exists(vocab_dir):
                os.makedirs(vocab_dir, exist_ok=True)

            with open(self.vocab_path, 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, indent=4, ensure_ascii=False)

            logger.info(f"[+] Vocab dosyası kaydedildi: {self.vocab_path}")

        except Exception as e:
            raise VocabBuildError(f"[X] Vocab dosyası yazılamadı: {e}")

    def get_vocab_size(self) -> int:
        size = len(self.vocab)
        logger.info(f"[+] Vocab büyüklüğü: {size}")
        return size

    def load_vocab(self) -> None:
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
                self.token_set = set(self.vocab.keys())
            logger.info(f"[+] Vocab dosyası yüklendi: {self.vocab_path}")
        except FileNotFoundError:
            logger.warning(f"[!] Vocab dosyası bulunamadı: {self.vocab_path}")
        except Exception as e:
            raise VocabBuildError(f"[X] Vocab yüklenemedi: {e}")

    def reset_vocab(self, special_tokens: Optional[Dict[str, int]] = None) -> None:
        self.vocab.clear()
        self.token_set.clear()

        if special_tokens:
            for token, token_id in special_tokens.items():
                self.add_token(token, token_id=token_id, total_freq=0, positions=[])

        logger.info(f"[+] Vocab sıfırlandı ve özel tokenlar yeniden eklendi.")
