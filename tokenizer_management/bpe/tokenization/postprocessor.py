import logging
import re
from typing import List

logger = logging.getLogger(__name__)

class PostProcessingError(Exception):
    pass

class Postprocessor:
    def __init__(self):
        """
        Postprocessor sınıfı, token ID listesini düzenleyerek okunabilir metne dönüştürür.
        """
        logger.info("[+] Postprocessor başlatıldı.")

        # Özel token tanımları
        self.special_tokens = {
            "<PAD>": "",
            "<UNK>": "[BİLİNMEYEN]",
            "<BOS>": "",
            "<EOS>": "."
        }

        # Noktalama kuralları (regex ile daha esnek)
        self.punctuation_fixes = {
            r"\s+,": ",",
            r"\s+\.": ".",
            r"\s+!": "!",
            r"\s+\?": "?",
            r"\s+:": ":",
            r"\s+;": ";",
            r"\s+\)": ")",
            r"\(\s+": "("
        }

        # Cümle başını büyük harfle başlatma
        self.capitalize_sentence = True

    def process(self, tokens: List[str]) -> str:
        """
        Token listesini düzenleyip metne çevirir.

        Args:
            tokens (List[str]): Token listesi

        Returns:
            str: Düzenlenmiş metin
        """
        try:
            if not tokens:
                logger.warning("[!] Boş token listesi alındı, boş string döndürülüyor.")
                return ""

            logger.debug(f"[+] İşlenmeden önceki token listesi: {tokens}")

            # Özel tokenleri kaldır veya dönüştür
            tokens = [self._replace_special_tokens(token) for token in tokens]
            tokens = [token.strip() for token in tokens if token.strip()]  # Boş token'leri temizle

            if not tokens:
                logger.warning("[!] Özel tokenlerden sonra geçerli token kalmadı.")
                return ""

            # Token listesini kelime bazında birleştir
            text = " ".join(tokens).strip()

            # Noktalama işaretlerini düzelt
            text = self._fix_punctuation(text)

            # Cümle başını büyük harfe çevir
            if self.capitalize_sentence:
                text = self._capitalize(text)

            # Fazla boşlukları ve yanlış formatları temizle
            text = self._clean_extra_spaces(text)

            logger.debug(f"[+] İşlendikten sonraki metin: {text}")

            if not text:
                logger.warning("[!] Postprocessing sonucunda geçerli bir metin bulunamadı.")
                return ""

            return text

        except Exception as e:
            logger.error(f"[X] Postprocessing sırasında hata: {e}")
            raise PostProcessingError(f"Postprocessing Error: {e}")

    def _replace_special_tokens(self, token: str) -> str:
        """
        Özel tokenleri metin formatına dönüştürür.

        Args:
            token (str): Token string'i

        Returns:
            str: Düzenlenmiş token
        """
        try:
            if token in self.special_tokens:
                return self.special_tokens[token]
            return token
        except Exception as e:
            logger.error(f"[X] Özel token değiştirme hatası: {e}")
            raise PostProcessingError(f"Special token processing error: {e}")

    def _fix_punctuation(self, text: str) -> str:
        """
        Noktalama işaretlerini düzeltir.

        Args:
            text (str): Düzenlenecek metin

        Returns:
            str: Düzenlenmiş metin
        """
        try:
            for wrong, correct in self.punctuation_fixes.items():
                text = re.sub(wrong, correct, text)
            return text
        except Exception as e:
            logger.error(f"[X] Noktalama düzeltme hatası: {e}")
            raise PostProcessingError(f"Punctuation fixing error: {e}")

    def _capitalize(self, text: str) -> str:
        """
        Cümle başlarını büyük harfle başlatır.

        Args:
            text (str): Düzenlenecek metin

        Returns:
            str: Düzenlenmiş metin
        """
        try:
            if not text:
                return text

            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [sentence.capitalize() for sentence in sentences]
            return ' '.join(sentences)
        except Exception as e:
            logger.error(f"[X] Büyük harf başlatma hatası: {e}")
            raise PostProcessingError(f"Capitalization error: {e}")

    def _clean_extra_spaces(self, text: str) -> str:
        """
        Fazla boşlukları ve yanlış formatları düzeltir.

        Args:
            text (str): Düzenlenecek metin

        Returns:
            str: Düzenlenmiş metin
        """
        try:
            # Fazladan boşlukları kaldır
            text = re.sub(r'\s+', ' ', text).strip()

            # Noktalama sonrası boşluk kontrolü
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)

            return text
        except Exception as e:
            logger.error(f"[X] Boşluk temizleme hatası: {e}")
            raise PostProcessingError(f"Whitespace cleaning error: {e}")

    def reset(self):
        """
        Postprocessor durumunu sıfırlar.
        """
        try:
            logger.warning("[!] Postprocessor sıfırlanıyor...")

            self.special_tokens = {
                "<PAD>": "",
                "<UNK>": "[BİLİNMEYEN]",
                "<BOS>": "",
                "<EOS>": "."
            }

            self.punctuation_fixes = {
                r"\s+,": ",",
                r"\s+\.": ".",
                r"\s+!": "!",
                r"\s+\?": "?",
                r"\s+:": ":",
                r"\s+;": ";",
                r"\s+\)": ")",
                r"\(\s+": "("
            }

            self.capitalize_sentence = True
            logger.info("[+] Postprocessor sıfırlandı.")
        except Exception as e:
            logger.error(f"[X] Postprocessor sıfırlama hatası: {e}")
            raise PostProcessingError(f"Reset error: {e}")
