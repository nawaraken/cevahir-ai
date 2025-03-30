import os
import json
import logging
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)


class VocabLoadError(Exception):
    """Vocab dosyasını yüklerken oluşan hatalar için özel exception."""
    pass


class VocabLoader:
    def __init__(self, vocab_path: str):
        self.vocab_path = vocab_path
        self.vocab = {}

    def load_vocab(self, vocab_path: str = None) -> Dict[str, Union[int, dict]]:
        """
        Vocab.json dosyasını yükler ve sözlüğü döner.

        Bu metot:
          - Dosya mevcut değilse VocabLoadError fırlatır.
          - Belirtilen kodlamalar (utf-8, utf-8-sig, latin-1) ile dosyayı yüklemeyi dener.
          - Yüklenen verinin dict formatında olduğunu doğrular; eğer token değerleri int ise,
            bunları {"id": <int>, "total_freq": 0, "positions": []} formatına dönüştürür.
          - Çift tokenleri tespit edip temizler.
          - Herhangi bir güncelleme yapıldıysa, dosya otomatik olarak kaydedilir.
          - Vocab sözlüğü boş veya beklenenden az token içeriyorsa hata fırlatır.

        Args:
            vocab_path (str, optional): Yüklemek için alternatif dosya yolu. Verilmezse self.vocab_path kullanılır.

        Returns:
            Dict[str, Union[int, dict]]: Güncellenmiş vocab sözlüğü.

        Raises:
            VocabLoadError: Dosya bulunamadı, JSON çözümleme hatası, format uyuşmazlığı veya beklenmeyen hata durumlarında.
        """
        # Eğer vocab zaten yüklenmişse, tekrar yüklemeye gerek yok.
        if self.vocab:
            logger.info("[+] Vocab zaten yüklü, tekrar yüklemeye gerek yok.")
            return self.vocab

        # Kullanılacak dosya yolunu belirle
        load_path = vocab_path if vocab_path is not None else self.vocab_path

        # Dosyanın varlığını kontrol et.
        if not os.path.exists(load_path):
            logger.warning(f"[!] Vocab dosyası bulunamadı: {load_path}")
            raise VocabLoadError(f"Vocab dosyası bulunamadı: {load_path}")

        # Farklı kodlamaları dene.
        encodings = ['utf-8', 'utf-8-sig', 'latin-1']
        vocab_data = None
        for enc in encodings:
            try:
                with open(load_path, 'r', encoding=enc) as file:
                    logger.info(f"[+] Vocab dosyası {enc} kodlamasıyla yükleniyor: {load_path}")
                    vocab_data = json.load(file)
                # Başarılı ise döngüden çık.
                break
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                logger.warning(f"{enc} kodlamasıyla hata: {e}")
                vocab_data = None

        if vocab_data is None:
            raise VocabLoadError(f"Tüm denenen kodlamalar başarısız: {load_path}")

        if not isinstance(vocab_data, dict):
            raise VocabLoadError(f"Geçersiz JSON formatı. Beklenen: dict, Alınan: {type(vocab_data)}")

        logger.debug(f"[+] Vocab dosyası {len(vocab_data)} token içeriyor.")

        # Vocab sözlüğünü temizle.
        self.vocab.clear()

        # Tokenleri yükle; eğer token değeri int ise, dict formatına çevir.
        for token, token_info in vocab_data.items():
            if isinstance(token_info, int):
                converted = {"id": token_info, "total_freq": 0, "positions": []}
                self.vocab[token] = converted
                logger.debug(f"[+] Token '{token}' int formatından dict'e dönüştürüldü: {converted}")
            else:
                self._validate_and_add_token(token, token_info)

        # Çift token kontrolü.
        try:
            num_duplicates_removed = self._remove_duplicate_tokens() or 0
            if num_duplicates_removed > 0:
                logger.info(f"[+] Çift token temizlendi: {num_duplicates_removed} adet.")
        except Exception as e:
            logger.error(f"[!] Çift token temizleme hatası: {e}")
            raise VocabLoadError(f"Çift token temizleme hatası: {e}")

        # Güncelleme yapıldıysa dosyayı kaydet.
        self.save_vocab()

        logger.info(f"[+] Vocab dosyası başarıyla yüklendi. Yüklenen token sayısı: {len(self.vocab)}")
        return self.vocab

    def _remove_duplicate_tokens(self) -> int:
        """
        Çift tokenleri kontrol eder ve kaldırır.
        """
        logger.info("[+] Çift token kontrolü başlatıldı...")

        seen_tokens = set()
        duplicates = set()

        for token in list(self.vocab.keys()):
            if token in seen_tokens:
                duplicates.add(token)
            else:
                seen_tokens.add(token)

        for token in duplicates:
            del self.vocab[token]
            logger.warning(f"[!] Çift token kaldırıldı: '{token}'")

        num_removed = len(duplicates)
        if num_removed > 0:
            logger.info(f"[+] Çift token kontrolü tamamlandı. Kaldırılan token sayısı: {num_removed}")
        else:
            logger.info("[+] Çift token bulunamadı.")
        return num_removed

    def _validate_and_add_token(self, token: str, token_info: Union[int, dict]) -> None:
        """
        Token ve metadata geçerliliğini kontrol eder ve vocab içine ekler.
        """
        if not isinstance(token, str):
            raise VocabLoadError(f"Geçersiz token formatı: '{token}' (Beklenen: str)")

        if isinstance(token_info, int):
            self.vocab[token] = {
                "id": token_info,
                "total_freq": 0,
                "positions": []
            }
            return

        if isinstance(token_info, dict):
            if "id" not in token_info or not isinstance(token_info["id"], int):
                raise VocabLoadError(f"Geçersiz token ID formatı: {token_info.get('id')}")
            self.vocab[token] = token_info
        else:
            raise VocabLoadError(f"Geçersiz token formatı: '{token_info}' (Beklenen: int veya dict)")

    def save_vocab(self) -> None:
        """
        Vocab sözlüğünü dosyaya kaydeder.
        """
        try:
            vocab_dir = os.path.dirname(self.vocab_path)
            if not os.path.exists(vocab_dir):
                os.makedirs(vocab_dir, exist_ok=True)
            with open(self.vocab_path, 'w', encoding='utf-8') as file:
                json.dump(self.vocab, file, ensure_ascii=False, indent=4)
            logger.info(f"[+] Vocab dosyası başarıyla güncellendi: {self.vocab_path}")
        except Exception as e:
            logger.error(f"[!] Vocab dosyasına yazılırken hata oluştu: {e}")
            raise VocabLoadError(f"Vocab dosyasına yazılırken hata oluştu: {e}")
