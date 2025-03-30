"""
sp_pretokenizer.py

Bu modül, SentencePiece tokenizasyon sürecinde kullanılacak ön işleme (pretokenization) işlemlerini gerçekleştirir.
Sınıf, girdi metnini normalize eder; Unicode normalizasyonu, küçük harf dönüşümü ve ekstra boşlukların temizlenmesi 
gibi işlemleri yapar. Çıktı olarak, normalize edilmiş metni boşluk karakterlerine göre bölerek token listesini döner.

Not: Bu sınıf, diğer modüllerle (ör. encoder, decoder vs.) doğrudan iletişime geçmeden yalnızca kendi sorumluluğunu yerine getirir.
Ana yöneticide (SentencePieceManager) diğer modüllerle entegrasyonu sağlanacaktır.
"""

import re
import unicodedata
import logging
from typing import List

logger = logging.getLogger(__name__)

class SPPretokenizer:
    """
    SPPretokenizer

    SentencePiece tokenizasyonu için metnin ön işleme tabi tutulmasını sağlar.
    - Unicode normalizasyonu (NFKC)
    - Küçük harfe çevirme
    - Fazla boşlukların temizlenmesi
    - Metni boşluk karakterlerine göre bölme

    Bu sınıf yalnızca ön işleme işlemlerine odaklanır; diğer modüllerle doğrudan iletişim kurmaz.
    """

    def __init__(self):
        logger.info("SPPretokenizer başlatıldı.")

    def normalize_text(self, text: str) -> str:
        """
        Metni Unicode NFKC standardına göre normalize eder, küçük harfe çevirir ve ekstra boşlukları temizler.

        Args:
            text (str): Girdi metni.

        Returns:
            str: Normalize edilmiş metin.
        """
        # Unicode normalizasyonu (NFKC)
        normalized = unicodedata.normalize('NFKC', text)
        # Küçük harfe çevirme
        normalized = normalized.lower()
        # Fazla boşlukları tek boşluk haline getirip, baştaki ve sondaki boşlukları kaldırma
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    def tokenize(self, text: str) -> List[str]:
        """
        Girdi metni normalize eder ve boşluk karakterlerine göre bölerek token listesini döner.

        Args:
            text (str): Girdi metni.

        Returns:
            List[str]: Token listesi.
        """
        if not text:
            logger.warning("Girdi metni boş; boş liste döndürülüyor.")
            return []

        try:
            normalized_text = self.normalize_text(text)
            tokens = normalized_text.split(' ')
            logger.debug(f"SPPretokenizer çıktı tokenleri: {tokens}")
            return tokens
        except Exception as e:
            logger.error(f"Tokenization sırasında hata oluştu: {e}")
            raise e
