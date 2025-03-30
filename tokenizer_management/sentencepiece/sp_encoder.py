"""
sp_encoder.py

Bu modül, SentencePiece tokenizasyon sürecinde,
girdi token listesini, verilen vocab sözlüğüne göre token ID'lerine dönüştüren SPEncoder sınıfını içerir.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class SPEncodingError(Exception):
    """SPEncoder sırasında oluşan hatalar için özel exception."""
    pass


class SPEncoder:
    """
    SPEncoder, verilen bir vocab sözlüğü üzerinden (token → id eşlemesi)
    kodlama işlemi gerçekleştirir.
    """

    def __init__(self, vocab: Dict[str, dict]):
        """
        SPEncoder, bir vocab sözlüğü ile başlatılır.
        
        Args:
            vocab (Dict[str, dict]): Her token için "id" anahtarını içeren sözlük.
            
        Raises:
            ValueError: Vocab boş ise veya özel tokenlar eksikse.
        """
        if not isinstance(vocab, dict):
            raise TypeError("Vocab bir sözlük olmalıdır.")
        
        if not vocab:
            raise ValueError("Vocab yüklenemedi veya boş.")

        self.vocab = vocab
        self.token_to_id = self._build_token_to_id()
        self._validate_special_tokens()

        logger.info(f"[+] SPEncoder başarıyla başlatıldı. Toplam {len(self.token_to_id)} token yüklendi.")


    def set_vocab(self, vocab: Dict[str, dict]) -> None:
        """
        Yeni vocab sözlüğünü alır, mevcut vocab'ı günceller ve
        token-to-ID eşleşmesini yeniden oluşturur.
        """
        if not isinstance(vocab, dict):
            raise ValueError("Vocab formatı geçersiz, dict tipinde olmalı.")
        
        self.vocab = vocab
        self.token_to_id = self._build_token_to_id()
        self._validate_special_tokens()
        logger.info("[+] SPEncoder vocab başarıyla güncellendi.")


    def _build_token_to_id(self) -> Dict[str, int]:
        """
        Vocab sözlüğünü token → ID eşleşmesine çevirir.

        Returns:
            Dict[str, int]: Token → ID eşleşmesi yapan sözlük.
        """
        try:
            token_to_id = {}
            for token, info in self.vocab.items():
                token_id = info.get("id")
                if token_id is not None:
                    if token in token_to_id:
                        raise ValueError(f"ID çakışması tespit edildi: {token} ({token_id})")
                    token_to_id[token] = token_id

            if not token_to_id:
                raise ValueError("Token-to-ID eşleşmesi oluşturulamadı. Vocab boş olabilir.")

            logger.info(f"[+] Token-to-ID eşleşmesi oluşturuldu. Toplam {len(token_to_id)} token yüklendi.")
            return token_to_id
        
        except Exception as e:
            logger.error(f"[X] Token-to-ID oluşturulurken hata: {e}")
            raise SPEncodingError(f"Token-to-ID oluşturulamadı: {e}")

    def _validate_special_tokens(self) -> None:
        """
        Özel tokenların (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`) eksik olup olmadığını kontrol eder.
        Eksik ise hata fırlatır.
        """
        required_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        missing_tokens = []
        for token in required_tokens:
            if token not in self.token_to_id:
                missing_tokens.append(token)

        if missing_tokens:
            raise ValueError(f"Vocab içinde eksik özel tokenlar: {missing_tokens}")

        logger.info("[+] Özel token ID'leri doğrulandı.")

    def encode(self, tokens: List[str]) -> List[int]:
        """
        Verilen token listesini, vocab'da tanımlı token id'lerine dönüştürür.
        Eğer bir token vocab'da yoksa, "<UNK>" için tanımlı id kullanılır.
        
        Args:
            tokens (List[str]): Kodlanacak tokenlerin listesi.
        
        Returns:
            List[int]: Kodlanmış token ID'lerinin listesi.
        
        Raises:
            ValueError: Token listesi boş ise veya "<UNK>" token id bulunamazsa.
        """
        if not tokens:
            raise ValueError("Token listesi boş olamaz.")

        unk_id = self.token_to_id.get("<UNK>")
        if unk_id is None:
            raise ValueError("'<UNK>' token ID'si bulunamadı.")

        encoded_tokens = []
        for token in tokens:
            token = token.strip().lower()  # Büyük-küçük harf farkını kaldırıyoruz
            token_id = self.token_to_id.get(token, unk_id)
            encoded_tokens.append(token_id)

        logger.info(f"[+] Kodlama işlemi tamamlandı. Toplam {len(encoded_tokens)} token encode edildi.")
        return encoded_tokens

    def update_vocab(self, new_vocab: Dict[str, dict]):
        """
        Yeni bir vocab yükleyip mevcut vocab'ı günceller.
        Özel tokenları koruyarak günceller.

        Args:
            new_vocab (Dict[str, dict]): Güncellenecek yeni vocab.
        """
        try:
            if not isinstance(new_vocab, dict):
                raise TypeError("Vocab bir sözlük olmalıdır.")

            logger.info("[+] Vocab güncelleniyor...")

            preserved_special_tokens = {
                "<PAD>": self.vocab.get("<PAD>", {"id": 0}),
                "<UNK>": self.vocab.get("<UNK>", {"id": 1}),
                "<BOS>": self.vocab.get("<BOS>", {"id": 2}),
                "<EOS>": self.vocab.get("<EOS>", {"id": 3}),
            }

            self.vocab = {**preserved_special_tokens, **new_vocab}
            self.token_to_id = self._build_token_to_id()

            logger.info("[+] Vocab başarıyla güncellendi.")

        except Exception as e:
            logger.error(f"[X] Vocab güncelleme hatası: {e}")
            raise SPEncodingError(f"Vocab update error: {e}")

    def reset(self):
        """
        Tüm vocab'ı sıfırlar ve varsayılan özel tokenları ekler.
        """
        try:
            logger.warning("[!] SPEncoder sıfırlanıyor...")

            self.vocab = {
                "<PAD>": {"id": 0, "total_freq": 0, "positions": []},
                "<UNK>": {"id": 1, "total_freq": 0, "positions": []},
                "<BOS>": {"id": 2, "total_freq": 0, "positions": []},
                "<EOS>": {"id": 3, "total_freq": 0, "positions": []},
            }

            self.token_to_id = self._build_token_to_id()

            logger.info("[+] Encoder sıfırlandı.")

        except Exception as e:
            logger.error(f"[X] Reset hatası: {e}")
            raise SPEncodingError(f"Encoder reset error: {e}")

