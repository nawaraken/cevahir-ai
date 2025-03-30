import json
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


# Özel hata sınıfı
class VocabUpdateError(Exception):
    """Vocab güncelleme sırasında oluşan hatalar için özel exception."""
    pass


class VocabUpdater:
    def __init__(self, vocab: Dict[str, dict], vocab_path: str, special_tokens: Optional[Dict[str, int]] = None) -> None:
        """
        VocabUpdater, vocab sözlüğünü güncellemek için kullanılır.

        Args:
            vocab (Dict[str, dict]): Güncellenecek vocab sözlüğü.
            vocab_path (str): Vocab dosya yolu.
            special_tokens (Optional[Dict[str, int]]): Özel tokenlar. Sağlanırsa, bu tokenlar otomatik olarak vocab'e eklenir.
        """
        if not isinstance(vocab, dict):
            raise TypeError("Vocab formatı geçersiz. Dict bekleniyor.")
        self.vocab = vocab
        self.vocab_path = vocab_path
        self.special_tokens = special_tokens or {}

        # Özel tokenları uygulayarak vocab'i güncelle
        self._apply_special_tokens()

    def _apply_special_tokens(self) -> None:
        """
        Özel tokenları vocab içine ekler; varsa ID'lerini doğrular.
        """
        for token, token_id in self.special_tokens.items():
            if token not in self.vocab:
                self.vocab[token] = {
                    "id": token_id,
                    "total_freq": 0,
                    "positions": []
                }
            elif self.vocab[token]["id"] != token_id:
                logger.warning(f"[!] Özel token '{token}' için ID çakışması tespit edildi. ID {token_id}'ye güncelleniyor.")
                self.vocab[token]["id"] = token_id

    def add_token(self, token: str, position: Optional[int] = None) -> None:
        """
        Vocab'e yeni bir token ekler veya mevcut tokenın frekansını ve pozisyon bilgisini günceller.

        Args:
            token (str): Eklenecek token.
            position (Optional[int]): Token'ın bulunduğu pozisyon.
        
        Raises:
            VocabUpdateError: Token boş ise.
        """
        token = token.strip()
        if not token:
            raise VocabUpdateError("Token boş olamaz.")

        if token in self.vocab:
            # Mevcut token için sadece frekans ve pozisyon güncellemesi yap
            self.vocab[token]["total_freq"] += 1
            if position is not None and position not in self.vocab[token]["positions"]:
                self.vocab[token]["positions"].append(position)
            logger.info(f"[+] Token güncellendi: '{token}' -> {self.vocab[token]}")
        else:
            # Yeni token ekle → ID çakışması kontrolü yap
            token_id = self.get_next_token_id()
            self.vocab[token] = {
                "id": token_id,
                "total_freq": 1,
                "positions": [position] if position is not None else []
            }
            logger.info(f"[+] Yeni token eklendi: '{token}' -> {token_id}")



    def update_token(self, token: str, token_id: int) -> None:
        """
        Var olan bir token'ın ID'sini günceller.

        Args:
            token (str): Güncellenecek token.
            token_id (int): Yeni ID.

        Raises:
            VocabUpdateError: Token mevcut değilse veya ID çakışması varsa.
        """
        if token not in self.vocab:
            raise VocabUpdateError(f"Token bulunamadı: '{token}'")
        existing_ids = {info["id"] for info in self.vocab.values()}
        if token_id in existing_ids:
            raise VocabUpdateError(f"Token ID çakışması: {token_id} zaten mevcut.")
        self.vocab[token]["id"] = token_id
        logger.info(f"[~] Token ID güncellendi: '{token}' → {token_id}")

    def remove_token(self, token: str) -> None:
        """
        Vocab sözlüğünden bir token'ı kaldırır.

        Args:
            token (str): Kaldırılacak token.
        """
        if token in self.vocab:
            del self.vocab[token]
            logger.info(f"[-] Token kaldırıldı: '{token}'")
        else:
            logger.warning(f"[!] Token bulunamadı: '{token}'")

    def update_vocab(self, tokens: List[str]) -> Dict[str, dict]:
        """
        Vocab sözlüğünü verilen token listesi ile günceller.
        Her token için, varsa frekansı artırır, yoksa yeni entry oluşturur.

        Args:
            tokens (List[str]): Güncellenecek token listesi.

        Returns:
            Dict[str, dict]: Güncellenmiş vocab.
        """
        # Her bir token için add_token çağrılır; böylece duplicate geçişler frekansı artırır.
        for index, token in enumerate(tokens):
            self.add_token(token, position=index)
        # Özel tokenların tekrar uygulanması (örneğin ID'lerin sabitlenmesi için)
        self._apply_special_tokens()
        self.save_vocab()
        return self.vocab

    def save_vocab(self) -> None:
        """
        Güncellenmiş vocab sözlüğünü JSON formatında dosyaya kaydeder.
        """
        try:
            with open(self.vocab_path, 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, indent=4, ensure_ascii=False)
            logger.info(f"[+] Vocab dosyası başarıyla güncellendi: {self.vocab_path}")
        except Exception as e:
            raise VocabUpdateError(f"Vocab dosyası kaydedilemedi: {e}")

    def get_vocab_size(self) -> int:
        """
        Vocab sözlüğündeki toplam token sayısını döndürür.
        """
        size = len(self.vocab)
        logger.info(f"[INFO] Vocab büyüklüğü: {size}")
        return size

    def get_token_info(self, token: str) -> Optional[dict]:
        """
        Belirtilen token'ın bilgilerini döndürür.
        """
        return self.vocab.get(token)

    def get_next_token_id(self) -> int:
        """
        Bir sonraki uygun token ID'sini döndürür.
        Kullanılan ID'ler taranır ve en yüksek ID'den sonra gelen sayı seçilir.
        """
        used_ids = {info["id"] for info in self.vocab.values()}
        next_id = max(used_ids, default=-1) + 1
        while next_id in used_ids:
            next_id += 1
        return next_id
