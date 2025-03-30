"""
training_postprocessor.py

Bu modül, eğitim sürecinde tokenizasyon sonrası elde edilen token listesini alır ve
uygun biçimde işleyerek nihai metin çıktısını oluşturur. Bu işlem, tokenlerin
boşluk karakterleriyle birleştirilmesi ve gereksiz boşlukların temizlenmesi gibi adımları içerir.
"""

import logging

logger = logging.getLogger(__name__)

class TrainingPostprocessor:
    """
    TrainingPostprocessor:
    Bu sınıf, eğitim tokenizasyon sürecinde elde edilen token listesini alır ve
    bu tokenleri insan tarafından okunabilir veya eğitim süreçlerinde kullanılacak şekilde
    uygun formatta birleştirir.
    """
    
    def __init__(self):
        logger.info("TrainingPostprocessor başlatıldı.")
    
    def process(self, tokens: list) -> str:
        """
        Verilen token listesini alır, tokenleri boşluk karakterleriyle birleştirir ve
        fazladan oluşan boşlukları temizleyerek nihai metni üretir.

        Args:
            tokens (list): Token listesi.

        Returns:
            str: İşlenmiş metin.
        """
        if not tokens:
            logger.warning("İşlenecek token listesi boş. Boş metin döndürülecek.")
            return ""
        
        # Tokenleri boşluk karakteri ile birleştir
        output = " ".join(tokens)
        # Fazla boşlukları temizle
        output = " ".join(output.split())
        logger.debug(f"TrainingPostprocessor çıktı: {output}")
        return output
