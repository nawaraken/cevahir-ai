"""
training_preprocessor.py

Bu modül, eğitim verisi için metin temizleme ve normalizasyon işlemlerini gerçekleştirir.
Yapılan işlemler:
  - Metni küçük harfe çevirir.
  - Özel karakterleri (noktalama işaretleri, semboller vb.) kaldırır.
  - Fazla boşlukları tek boşluk haline getirip, baştaki ve sondaki boşlukları temizler.
Bu sayede eğitim verisi, sonraki aşamalara (tokenizasyon, tensörleştirme vb.) daha temiz bir formatta aktarılır.
"""

import re
import logging

logger = logging.getLogger(__name__)

class TrainingPreprocessor:
    """
    TrainingPreprocessor, eğitim verisi için metin temizleme ve normalizasyon işlemlerini gerçekleştirir.
    Sadece bu sorumluluğa odaklanır; diğer eğitim adımları (tokenizasyon, tensörleştirme, vs.) ayrı modüllerde ele alınır.
    """
    
    def __init__(self):
        # Özel karakterleri kaldırmak için regex ifadesi: Sadece alfanümerik karakterler ve boşluklara izin ver.
        self.special_chars_pattern = re.compile(r"[^\w\s]", re.UNICODE)
        logger.info("TrainingPreprocessor başlatıldı.")

    def preprocess(self, text: str) -> str:
        """
        Girdi metnini temizler ve normalize eder.
        
        İşlemler:
          1. Metni küçük harfe çevirir.
          2. Özel karakterleri kaldırır.
          3. Fazla boşlukları tek boşluk haline getirir ve kenardaki boşlukları temizler.
        
        Args:
            text (str): Temizlenecek metin.
        
        Returns:
            str: Normalizasyon işlemleri sonrası temizlenmiş metin.
            
        Raises:
            ValueError: Girdi metni None veya boş string ise.
        """
        if text is None:
            logger.error("Preprocess işlemi için girdi metni None olarak alındı.")
            raise ValueError("Girdi metni None olamaz.")

        # Küçük harfe çevirme
        processed = text.lower()
        # Özel karakterleri kaldırma
        processed = self.special_chars_pattern.sub("", processed)
        # Fazla boşlukları temizleme: Birden fazla boşluğu tek boşluk yap ve kenardaki boşlukları kaldır.
        processed = " ".join(processed.split())
        
        logger.debug(f"Preprocessed text: '{processed}'")
        return processed
