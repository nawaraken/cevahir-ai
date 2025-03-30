import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class SPDecodingError(Exception):
    pass

class SPDecoder:
    def __init__(self, vocab: Dict[str, dict]):
        if not isinstance(vocab, dict):
            raise TypeError("Vocab yüklenirken bir sözlük olmalıdır.")
        if not vocab:
            raise ValueError("Vocab yüklenemedi veya boş.")

        self.vocab = vocab
        self.id_to_token = self._build_reverse_vocab()
        self.reverse_vocab = self.id_to_token  # reverse_vocab tanımı

        self._validate_special_tokens()
        logger.info(f"[+] SPDecoder, {len(self.id_to_token)} token ile başlatıldı.")

    def _build_reverse_vocab(self) -> Dict[int, str]:
        try:
            reverse_vocab = {}
            for token, info in self.vocab.items():
                token_id = info.get("id")
                if token_id is None:
                    continue
                if token_id in reverse_vocab:
                    logger.warning(f"[!] ID çakışması tespit edildi: {token_id} ({token})")
                    token_id = max(reverse_vocab.keys(), default=-1) + 1
                    logger.warning(f"[!] ID çakışması giderildi, yeni ID: {token_id}")
                reverse_vocab[token_id] = token

            if not reverse_vocab:
                raise ValueError("Reverse vocab oluşturulamadı. Vocab boş olabilir.")

            logger.info(f"[+] Reverse vocab başarıyla oluşturuldu. Toplam {len(reverse_vocab)} token yüklendi.")
            return reverse_vocab

        except Exception as e:
            logger.error(f"[X] Reverse vocab oluşturulurken hata: {e}")
            raise SPDecodingError(f"Reverse vocab oluşturulamadı: {e}")

    def update_reverse_vocab(self):
        """
        Reverse vocab güncellemesini sağlar.
        """
        try:
            self.reverse_vocab = self._build_reverse_vocab()
            logger.info("[+] Reverse vocab başarıyla güncellendi.")
        except Exception as e:
            logger.error(f"[X] Reverse vocab güncellenemedi: {e}")
            raise SPDecodingError(f"Reverse vocab update error: {e}")

    def _validate_special_tokens(self):
        required_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        missing_tokens = []

        for token in required_tokens:
            if token not in self.vocab:
                new_id = max([info["id"] for info in self.vocab.values()], default=-1) + 1
                self.vocab[token] = {
                    "id": new_id,
                    "total_freq": 0,
                    "positions": []
                }
                missing_tokens.append(token)

            elif "id" not in self.vocab[token]:
                raise ValueError(f"{token} için ID bulunamadı.")

        if missing_tokens:
            logger.warning(f"[!] Eksik özel tokenlar oluşturuldu: {missing_tokens}")

        logger.info("[+] Özel token ID'leri doğrulandı.")

    def _handle_unknown_token(self, token_id: int) -> str:
        try:
            unk_id = self.vocab.get("<UNK>", {}).get("id")
            if unk_id is None:
                unk_id = max(self.id_to_token.keys(), default=-1) + 1
                self.vocab["<UNK>"] = {
                    "id": unk_id,
                    "total_freq": 0,
                    "positions": []
                }
                self.id_to_token[unk_id] = "<UNK>"
                logger.warning(f"[!] '<UNK>' token için yeni ID oluşturuldu: {unk_id}")

            logger.warning(f"[!] Bilinmeyen token ID'si bulundu: {token_id}, '<UNK>' ile eşleştirildi.")
            return "<UNK>"

        except Exception as e:
            logger.error(f"[X] Bilinmeyen token işleme hatası: {e}")
            raise SPDecodingError(f"Unknown token error: {e}")

    def decode(self, token_ids: List[int]) -> str:
        try:
            if not token_ids:
                raise ValueError("Token ID listesi boş olamaz.")

            logger.info("[+] Çözümleme işlemi başlatılıyor...")
            tokens = []
            for token_id in token_ids:
                token = self.reverse_vocab.get(token_id, "<UNK>")
                tokens.append(token)

            decoded_text = " ".join(tokens)
            logger.debug(f"[+] Çözümlenmiş metin: {decoded_text}")

            return decoded_text

        except Exception as e:
            logger.error(f"[X] Çözümleme sırasında hata: {e}")
            raise SPDecodingError(f"Decoding error: {e}")

    def update_vocab(self, new_vocab: Dict[str, dict]):
        try:
            if not isinstance(new_vocab, dict):
                raise TypeError("Vocab bir sözlük olmalıdır.")

            existing_ids = set(info["id"] for info in self.vocab.values())
            new_id = max(existing_ids, default=-1) + 1

            for token, info in new_vocab.items():
                if token not in self.vocab:
                    self.vocab[token] = {
                        "id": new_id,
                        "total_freq": info.get("total_freq", 0),
                        "positions": info.get("positions", [])
                    }
                    new_id += 1

            # Reverse vocab güncellemesi
            self.update_reverse_vocab()
            logger.info("[+] Vocab başarıyla güncellendi.")

        except Exception as e:
            logger.error(f"[X] Vocab güncelleme hatası: {e}")
            raise SPDecodingError(f"Vocab update error: {e}")

    def reset(self):
        try:
            logger.warning("[!] SPDecoder sıfırlanıyor...")
            self.vocab = {
                "<PAD>": {"id": 0, "total_freq": 0, "positions": []},
                "<UNK>": {"id": 1, "total_freq": 0, "positions": []},
                "<BOS>": {"id": 2, "total_freq": 0, "positions": []},
                "<EOS>": {"id": 3, "total_freq": 0, "positions": []},
            }
            self.update_reverse_vocab()
            logger.info("[+] Decoder sıfırlandı.")

        except Exception as e:
            logger.error(f"[X] Reset hatası: {e}")
            raise SPDecodingError(f"Decoder reset error: {e}")

    def set_vocab(self, new_vocab: Dict[str, dict]) -> None:
        """
        Dışarıdan yeni bir vocab sözlüğü alır, mevcut vocab'ı günceller ve reverse_vocab'i yeniden oluşturur.
        
        Args:
            new_vocab (Dict[str, dict]): Güncellenecek yeni vocab sözlüğü.
        """
        if not isinstance(new_vocab, dict):
            raise ValueError("Vocab formatı geçersiz, dict tipinde olmalı.")
        
        self.vocab = new_vocab
        self.id_to_token = self._build_reverse_vocab()
        self.reverse_vocab = self.id_to_token
        self._validate_special_tokens()
        logger.info("[+] SPDecoder için vocab başarıyla güncellendi.")
