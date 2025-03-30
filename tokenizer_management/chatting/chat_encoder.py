import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class ChatEncoder:
    """
    ChatEncoder
    -----------
    Token listesini, sözlükteki karşılıklarına göre ID listesine çevirir.
    BOS ve EOS token'larını başa ve sona ekler.
    Bilinmeyen token'lar için UNK token ID'si kullanılır.
    """

    # Özel token tanımları (sabitler)
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, vocab: Dict[str, int] = None):
        """
        Encoder başlatılırken vocab sağlanmazsa varsayılan özel tokenlarla oluşturulur.

        Args:
            vocab (Dict[str, int], optional): Token → ID eşlemesi.
        """
        self.vocab = vocab.copy() if vocab else {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }

        self._load_special_token_ids()
        logger.info("ChatEncoder başlatıldı. Token sayısı: %d", len(self.vocab))

    def _load_special_token_ids(self):
        """
        Vocab içindeki özel token ID'lerini yükler.
        Eksikse hata fırlatır.
        """
        try:
            self.pad_token_id = self.vocab[self.PAD_TOKEN]
            self.unk_token_id = self.vocab[self.UNK_TOKEN]
            self.bos_token_id = self.vocab[self.BOS_TOKEN]
            self.eos_token_id = self.vocab[self.EOS_TOKEN]
        except KeyError as e:
            raise ValueError(f"Vocab içinde özel token eksik: {e}")

    def encode(self, tokens: List[str]) -> List[int]:
        """
        Token listesi → Token ID listesi (BOS ve EOS dahil).

        Args:
            tokens (List[str]): Token dizisi.

        Returns:
            List[int]: Token ID dizisi.
        """
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise TypeError("encode(): Token listesi yalnızca str içermelidir.")

        if not tokens:
            logger.warning("Boş token listesi alındı.")
            return [self.bos_token_id, self.eos_token_id]

        encoded = [self.bos_token_id]
        for token in tokens:
            token_lower = token.lower().strip()
            token_id = self.vocab.get(token_lower, self.unk_token_id)
            encoded.append(token_id)
        encoded.append(self.eos_token_id)

        logger.debug("Encode tamamlandı: %s", encoded)
        return encoded

    def add_token(self, token: str) -> int:
        """
        Vocab'e yeni bir token ekler.

        Args:
            token (str): Eklenecek token.

        Returns:
            int: Token'ın ID'si.
        """
        token_lower = token.lower().strip()
        if token_lower in self.vocab:
            logger.debug("Token zaten mevcut: '%s'", token_lower)
            return self.vocab[token_lower]

        new_id = len(self.vocab)
        self.vocab[token_lower] = new_id
        logger.info("[+] Yeni token eklendi: '%s' → ID: %d", token_lower, new_id)
        return new_id

    def remove_token(self, token: str) -> None:
        """
        Token'ı vocab'ten çıkarır.

        Args:
            token (str): Silinecek token.
        """
        token_lower = token.lower().strip()
        if token_lower in self.vocab:
            del self.vocab[token_lower]
            logger.info("[-] Token silindi: '%s'", token_lower)
        else:
            logger.warning("[!] Token vocab'te bulunamadı: '%s'", token_lower)

    def reset_vocab(self) -> None:
        """
        Vocab'i varsayılan özel tokenlarla sıfırlar.
        """
        self.vocab = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }
        self._load_special_token_ids()
        logger.info("[✓] Vocab sıfırlandı. Özel tokenlar yüklendi.")

    def set_vocab(self, vocab: Dict[str, int]) -> None:
        """
        Yeni bir vocab atar ve özel token ID'lerini günceller.

        Args:
            vocab (Dict[str, int]): Yeni token → ID eşlemesi.

        Raises:
            ValueError: Eksik özel token varsa.
        """
        if not isinstance(vocab, dict):
            raise TypeError("Vocab bir sözlük olmalıdır.")

        self.vocab = vocab.copy()
        self._load_special_token_ids()
        logger.info("Vocab güncellendi. Token sayısı: %d", len(self.vocab))

    def get_vocab(self) -> Dict[str, int]:
        """
        Güncel vocab'i döner.

        Returns:
            Dict[str, int]: Token → ID eşlemesi.
        """
        return self.vocab.copy()

    def get_vocab_size(self) -> int:
        """
        Vocab'teki toplam token sayısını verir.

        Returns:
            int: Token sayısı.
        """
        return len(self.vocab)

    def get_token_id(self, token: str) -> int:
        """
        Belirli bir token'ın ID'sini verir.

        Args:
            token (str): Sorgulanan token.

        Returns:
            int: Token ID (veya UNK ID).
        """
        return self.vocab.get(token.lower().strip(), self.unk_token_id)

    def get_special_token_ids(self) -> Dict[str, int]:
        """
        Özel token ID'lerini döner.

        Returns:
            Dict[str, int]: {'PAD': ..., 'UNK': ..., ...}
        """
        return {
            "PAD": self.pad_token_id,
            "UNK": self.unk_token_id,
            "BOS": self.bos_token_id,
            "EOS": self.eos_token_id
        }
