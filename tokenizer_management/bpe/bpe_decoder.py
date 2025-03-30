import logging
from typing import List, Dict,Optional

logger = logging.getLogger(__name__)


class BPEDecodingError(Exception):
    pass


class DummyPostprocessor:
    """
    Basit bir varsayılan postprocessor; token listesini boşlukla birleştirir.
    """
    def process(self, tokens: List[str]) -> str:
        return " ".join(tokens)


class BPEDecoder:
    def __init__(self, vocab: Dict[str, dict], postprocessor: Optional[DummyPostprocessor] = None):
        """
        BPEDecoder sınıfı, kodlanmış token ID'lerini çözümleyip metne dönüştürür.
        
        Args:
            vocab (Dict[str, dict]): Vocab sözlüğü.
            postprocessor: (Optional) Token listesini metne dönüştürecek postprocessor.
                           Sağlanmazsa, varsayılan DummyPostprocessor kullanılır.
        """
        if not isinstance(vocab, dict):
            raise TypeError("Vocab bir sözlük olmalıdır.")
        if not vocab:
            raise ValueError("Vocab yüklenemedi veya boş.")

        self.vocab = vocab
        self._validate_special_tokens()
        self.reverse_vocab = self._build_reverse_vocab()
        
        # Eğer postprocessor sağlanmamışsa, varsayılanı ata.
        self.postprocessor = postprocessor if postprocessor is not None else DummyPostprocessor()

        logger.info("[+] BPEDecoder başlatıldı.")

    def _validate_special_tokens(self) -> None:
        """
        Vocab içinde gerekli özel tokenların (<PAD>, <UNK>, <BOS>, <EOS>) bulunduğunu kontrol eder.
        Eksikse, yeni ID'lerle ekler.
        """
        required_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        missing_tokens = []

        for token in required_tokens:
            if token not in self.vocab:
                new_id = max((info["id"] for info in self.vocab.values() if "id" in info), default=-1) + 1
                self.vocab[token] = {"id": new_id, "total_freq": 0, "positions": []}
                missing_tokens.append(token)
            elif "id" not in self.vocab[token]:
                raise ValueError(f"{token} için ID bulunamadı.")

        if missing_tokens:
            logger.warning(f"[!] Eksik özel tokenlar oluşturuldu: {missing_tokens}")
        logger.info("[+] Özel token ID'leri doğrulandı.")

    def _build_reverse_vocab(self) -> Dict[int, str]:
        """
        Vocab sözlüğünü ters çevirerek (ID → token) bir sözlük oluşturur.
        Aynı ID için çakışma varsa, otomatik olarak yeni ID atar.
        
        Returns:
            Dict[int, str]: Ters vocab sözlüğü.
        """
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
            raise BPEDecodingError(f"Reverse vocab oluşturulamadı: {e}")

    def set_vocab(self, new_vocab: Dict[str, dict]) -> None:
        """
        Yeni bir vocab atar ve reverse_vocab'i yeniden oluşturur.
        
        Args:
            new_vocab (Dict[str, dict]): Güncel vocab sözlüğü.
        """
        if not new_vocab:
            raise ValueError("[X] Yeni vocab boş olamaz.")
        self.vocab = new_vocab
        self.reverse_vocab = self._build_reverse_vocab()
        logger.info(f"[+] BPEDecoder için vocab başarıyla güncellendi. Toplam {len(self.reverse_vocab)} token yüklendi.")

    def update_vocab(self, new_vocab: Dict[str, dict]):
        """
        Mevcut vocab'a yeni tokenları ekler ve ters vocab'i yeniden oluşturur.
        Çakışan token ID'leri için otomatik çözüm uygulanır.
        
        Args:
            new_vocab (Dict[str, dict]): Eklenmek istenen yeni vocab sözlüğü.
        """
        try:
            if not isinstance(new_vocab, dict):
                raise TypeError("Vocab bir sözlük olmalıdır.")

            logger.info("[+] Vocab güncelleniyor...")

            existing_ids = {info["id"] for info in self.vocab.values() if "id" in info}
            new_id = max(existing_ids, default=-1) + 1

            for token, info in new_vocab.items():
                if token not in self.vocab:
                    self.vocab[token] = {
                        "id": new_id,
                        "total_freq": info.get("total_freq", 0),
                        "positions": info.get("positions", [])
                    }
                    new_id += 1

            self.reverse_vocab = self._build_reverse_vocab()
            logger.info(f"[+] Vocab başarıyla güncellendi. Toplam {len(self.reverse_vocab)} token yüklendi.")

        except Exception as e:
            logger.error(f"[X] Vocab güncelleme hatası: {e}")
            raise BPEDecodingError(f"Vocab update error: {e}")

    def decode(self, token_ids: List[int]) -> str:
        """
        Kodlanmış token ID'lerini çözümleyip metin olarak döndürür.
        Bilinmeyen ID'ler `<UNK>` token ile eşleştirilir.
        
        Args:
            token_ids (List[int]): Kodlanmış token ID listesi.
        
        Returns:
            str: Çözümlenmiş metin.
        """
        try:
            if not token_ids:
                logger.warning("[!] Boş token ID listesi alındı, boş string döndürülüyor.")
                return ""
            if not isinstance(token_ids, list) or not all(isinstance(tid, int) for tid in token_ids):
                raise TypeError("[X] Çözümleme için geçerli bir ID listesi sağlanmalıdır.")

            if not self.reverse_vocab:
                logger.warning("[!] reverse_vocab eksik veya boş. Yeniden oluşturuluyor...")
                self.reverse_vocab = self._build_reverse_vocab()

            tokens = [self.reverse_vocab.get(tid, "<UNK>") for tid in token_ids]

            filtered_tokens = [
                token for token in tokens
                if token not in {"<BOS>", "<EOS>", "<PAD>"} and not token.startswith("__tag__")
            ]

            decoded_text = self.postprocessor.process(filtered_tokens)
            decoded_text = decoded_text.strip()

            if decoded_text.endswith(".") and not decoded_text.endswith("..."):
                decoded_text = decoded_text[:-1]
            if not decoded_text:
                logger.warning("[!] Çözümleme sonucu boş çıktı. '<EMPTY>' döndürüldü.")
                decoded_text = "<EMPTY>"

            logger.info(f"[+] Çözümleme başarıyla tamamlandı: {decoded_text}")
            return decoded_text

        except Exception as e:
            logger.error(f"[X] Çözümleme sırasında hata: {e}")
            raise BPEDecodingError(f"Decoding error: {e}")

    def _handle_unknown_token(self, token_id: int) -> str:
        """
        Bilinmeyen token ID'sini `<UNK>` token ile eşleştirir.
        Eğer `<UNK>` ID'si mevcut değilse, otomatik olarak oluşturur.
        
        Args:
            token_id (int): Bilinmeyen token ID'si.
        
        Returns:
            str: Eşleşen token (varsayılan olarak "<UNK>")
        """
        try:
            unk_id = self.vocab.get("<UNK>", {}).get("id")
            if unk_id is None:
                unk_id = max(self.reverse_vocab.keys(), default=-1) + 1
                self.vocab["<UNK>"] = {"id": unk_id, "total_freq": 0, "positions": []}
                self.reverse_vocab[unk_id] = "<UNK>"
                logger.warning(f"[!] '<UNK>' token için yeni ID oluşturuldu: {unk_id}")

            logger.warning(f"[!] Bilinmeyen token ID'si bulundu: {token_id}, '<UNK>' ile eşleştirildi.")
            return "<UNK>"

        except Exception as e:
            logger.error(f"[X] Bilinmeyen token işleme hatası: {e}")
            raise BPEDecodingError(f"Unknown token error: {e}")

    def reset(self):
        """
        BPEDecoder'ı, özel tokenlar dışındaki tüm bilgileri sıfırlar.
        """
        try:
            logger.warning("[!] BPEDecoder sıfırlanıyor...")
            self.vocab = {
                "<PAD>": {"id": 0, "total_freq": 0, "positions": []},
                "<UNK>": {"id": 1, "total_freq": 0, "positions": []},
                "<BOS>": {"id": 2, "total_freq": 0, "positions": []},
                "<EOS>": {"id": 3, "total_freq": 0, "positions": []},
            }
            self.reverse_vocab = self._build_reverse_vocab()
            logger.info("[+] BPEDecoder sıfırlandı.")

        except Exception as e:
            logger.error(f"[X] Reset hatası: {e}")
            raise BPEDecodingError(f"Decoder reset error: {e}")
