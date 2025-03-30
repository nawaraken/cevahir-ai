"""
training_tokenizer.py

Bu modül, eğitim verisi üzerinde tokenizasyon işlemlerini gerçekleştiren
TrainingTokenizer sınıfını içerir. Bu sınıf, verilen metni basit bir şekilde
boşluk karakterlerine göre bölerek token listesi üretir. Eğer ek tokenizasyon
işlemleri (ör. noktalama temizleme veya özel karakter filtreleme) gerekiyorsa,
bunlar da bu sınıf içerisinde uygulanabilir.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

class TrainingTokenizer:
    """
    TrainingTokenizer

    Bu sınıf, eğitim verisi üzerinde tokenizasyon işlemlerini gerçekleştirir.
    Girdi metnini boşluklara dayalı olarak bölerek basit bir token listesi oluşturur.
    Sınıf yalnızca tokenizasyon görevine odaklanır; diğer adımlarla (ön işleme,
    postprocessing, tensörleştirme) doğrudan ilgilenmez.
    """
    
    def __init__(self):
        # Tokenization için basit bir regex ifadesi: boşluk olmayan karakterlerden oluşan gruplar.
        self.pattern = re.compile(r'\S+')
        logger.info("TrainingTokenizer başlatıldı.")

    def tokenize(self, text: str) -> List[str]:
        """
        Verilen metni tokenlere ayırır.
        
        Args:
            text (str): Tokenize edilecek girdi metni.
            
        Returns:
            List[str]: Token listesini döner.
            
        Raises:
            ValueError: Eğer girdi metni None veya boş string ise.
        """
        if text is None:
            logger.error("Tokenize edilecek metin None değeri aldı.")
            raise ValueError("Girdi metni None olamaz.")
        
        if not text.strip():
            logger.warning("Tokenize edilecek metin boş veya sadece boşluklardan oluşuyor.")
            return []
        
        tokens = self.pattern.findall(text)
        logger.debug("Tokenize edilmiş metin: %s", tokens)
        return tokens
