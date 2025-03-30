import logging

logger = logging.getLogger(__name__)

class SPPostprocessor:
    """
    SentencePiece Postprocessor:
    Bu sınıf, sp_decoder tarafından üretilen token listesini alır ve
    nihai metni oluşturmak üzere işleme tabi tutar.
    """

    def __init__(self):
        logger.info("SPPostprocessor başlatıldı.")

    def process(self, tokens: list) -> str:
        """
        Verilen token listesini alır, birleştirir ve temizlenmiş bir metin üretir.
        
        Args:
            tokens (list): Çözümlenmiş token veya morfemlerin listesi.
        
        Returns:
            str: İşlenmiş ve biçimlendirilmiş metin.
        """
        if not tokens:
            logger.warning("İşlenecek token bulunamadı. Boş metin döndürülecek.")
            return ""
        
        # Tokenleri boşluk ile birleştir.
        output = " ".join(tokens).strip()

        # İlk harfi büyük yap (opsiyonel)
        output = output.capitalize()

        # Fazla boşlukları kaldır (örneğin çift boşluk vb.)
        output = " ".join(output.split())

        logger.debug(f"Postprocessed output: '{output}'")
        return output
