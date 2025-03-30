import logging
import re
from typing import List
import unicodedata

logger = logging.getLogger(__name__)

class LanguageProcessor:
    """
    LanguageProcessor

    Bu sınıf, SentencePiece tokenizasyon sürecinde ön işleme aşamasından sonra elde edilen
    token listesinin dil analizlerini gerçekleştirir. Türkçe diline özgü karakter dönüşümleri,
    noktalama işaretlerinin temizlenmesi ve genel metin normalizasyonu yaparak daha doğru
    tokenizasyon çıktıları üretir.

    Özellikler:
      - Türkçe'ye özgü büyük/küçük harf dönüşümü (case folding) gerçekleştirir.
      - Noktalama işaretlerini temizler, ancak birleşik diakritik işaretleri (combining marks)
        korunur.
      - Boş token’ları filtreler.
    """

    def __init__(self):
        # \w: kelime karakterleri, \s: boşluk karakterleri.
        # Ek olarak, U+0300-U+036F aralığı (diakritik/combining işaretleri) eklenmiştir.
        self.punctuation_pattern = re.compile(r"[^\w\s\u0300-\u036F]", re.UNICODE)
        logger.info("LanguageProcessor başlatıldı.")

    def process(self, tokens: List[str]) -> List[str]:
        """
        Verilen token listesini alır, dil analizleri gerçekleştirir ve normalize edilmiş token
        listesini döner.

        Args:
            tokens (List[str]): Ön işleme tabi tutulmuş tokenlerin listesi.

        Returns:
            List[str]: İşlenmiş, normalize edilmiş tokenlerin listesi.
        """
        if not tokens:
            logger.warning("İşlenecek token listesi boş. Boş liste döndürülüyor.")
            return []

        processed_tokens = []
        for token in tokens:
            # Case folding: Türkçe için uygun şekilde küçültme işlemi yapar.
            normalized = token.casefold()
            # Unicode normalizasyonu: Birleşik işaretlerin (ör. kombinasyon) uygun biçime getirilmesi.
            normalized = unicodedata.normalize('NFC', normalized)
            # Noktalama işaretlerini temizle (birleşik diakritik işaretler korunur).
            normalized = self.punctuation_pattern.sub('', normalized)
            # Kenardaki boşlukları temizle.
            normalized = normalized.strip()
            if normalized:
                processed_tokens.append(normalized)

        logger.debug(f"LanguageProcessor çıktısı: {processed_tokens}")
        return processed_tokens
