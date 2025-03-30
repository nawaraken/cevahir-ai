import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ChatDecoder:
    """
    ChatDecoder
    -----------
    Token ID dizilerini çözüp okunabilir metne dönüştürür.
    BOS ve EOS token'larını göz ardı eder.
    Bilinmeyen token ID'leri için <UNK> döndürülür.
    """

    # Özel token sabitleri
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab.copy() if vocab else {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }
        self._build_reverse_vocab()
        self._load_special_token_ids()
        logger.info("ChatDecoder başlatıldı. Token sayısı: %d", len(self.vocab))

    def _build_reverse_vocab(self) -> None:
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}

    def _load_special_token_ids(self) -> None:
        try:
            self.pad_token_id = self.vocab[self.PAD_TOKEN]
            self.unk_token_id = self.vocab[self.UNK_TOKEN]
            self.bos_token_id = self.vocab[self.BOS_TOKEN]
            self.eos_token_id = self.vocab[self.EOS_TOKEN]
        except KeyError as e:
            raise ValueError(f"[!] Eksik özel token: {e}")

    def set_vocab(self, vocab: Dict[str, int]) -> None:
        if not isinstance(vocab, dict):
            raise TypeError("Vocab dict formatında olmalıdır.")
        self.vocab = vocab.copy()
        self._build_reverse_vocab()
        self._load_special_token_ids()
        logger.info("[✓] Vocab güncellendi. Token sayısı: %d", len(self.vocab))

    def set_reverse_vocab(self, reverse_vocab: Dict[int, str]) -> None:
        if not isinstance(reverse_vocab, dict):
            raise TypeError("reverse_vocab dict formatında olmalıdır.")
        self.id_to_token = reverse_vocab.copy()
        logger.info("[✓] Reverse vocab güncellendi. Token sayısı: %d", len(self.id_to_token))

    def decode(
        self,
        token_ids: List[int],
        ignore_special_tokens: bool = True,
        strict: bool = False
    ) -> str:
        """
        Token ID listesini çözümleyip metne dönüştürür.

        Args:
            token_ids (List[int]): ID dizisi
            ignore_special_tokens (bool): True ise BOS/EOS/PAD atlanır
            strict (bool): True ise bilinmeyen tokenlar loglanır, False ise sessiz geçilir

        Returns:
            str: Çözülmüş metin
        """
        if not token_ids:
            logger.warning("decode() boş liste ile çağrıldı.")
            return ""

        if not isinstance(token_ids, list) or not all(isinstance(tid, int) for tid in token_ids):
            raise TypeError("decode() sadece int türünde ID listesi kabul eder.")

        filtered_ids = []
        for tid in token_ids:
            if ignore_special_tokens and tid in {
                self.pad_token_id, self.bos_token_id, self.eos_token_id
            }:
                continue
            filtered_ids.append(tid)

        decoded_tokens = []
        for tid in filtered_ids:
            token = self.id_to_token.get(tid)
            if token is None:
                token = self.UNK_TOKEN
                if strict:
                    logger.warning(f"[!] Bilinmeyen token ID: {tid}")
            decoded_tokens.append(token)

        text = " ".join(decoded_tokens)
        logger.debug("decode() tamamlandı → '%s'", text)
        return text

    def update_vocab(self, new_tokens: List[str]) -> None:
        """
        Yeni token'ları vocab'e ekler. Tüm vocab güncellenir.
        """
        for token in new_tokens:
            token = token.strip().lower()
            if token not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[token] = new_id
                self.id_to_token[new_id] = token
                logger.info(f"[+] Yeni token eklendi: '{token}' → ID: {new_id}")
            else:
                logger.debug(f"[~] Token zaten mevcut: '{token}'")

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
        self._build_reverse_vocab()
        self._load_special_token_ids()
        logger.info("[✓] Vocab sıfırlandı ve özel tokenlar yüklendi.")

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def get_token(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, self.UNK_TOKEN)

    def get_special_token_ids(self) -> Dict[str, int]:
        return {
            "PAD": self.pad_token_id,
            "UNK": self.unk_token_id,
            "BOS": self.bos_token_id,
            "EOS": self.eos_token_id
        }

    def get_token_info(self, token_id: int) -> Dict[str, str]:
        """
        Debug amaçlı bir ID'nin detaylı bilgilerini verir.
        """
        token = self.get_token(token_id)
        info = {
            "token_id": str(token_id),
            "token": token,
            "is_special": str(token in {
                self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN
            }),
            "exists_in_vocab": str(token in self.vocab)
        }
        return info
