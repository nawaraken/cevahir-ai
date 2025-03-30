import logging
from typing import List

logger = logging.getLogger(__name__)


class ChatPostprocessor:
    """
    ChatPostprocessor
    -----------------
    Token listesini son kullanıcıya gösterilecek metne dönüştürür.
    - Boşluk birleştirme
    - Gereksiz boşluk temizliği
    - Noktalama, özel karakter veya dil düzeltmeleri (gerekiyorsa)
    """

    def __init__(self):
        logger.info("ChatPostprocessor başlatıldı.")

    def process(self, tokens: List[str]) -> str:
        """
        Token listesinden okunabilir düz metin üretir.

        Args:
            tokens (List[str]): Çözümlenmiş token listesi.

        Returns:
            str: Nihai düzenlenmiş metin.
        """
        if not tokens:
            logger.warning("[!] ChatPostprocessor: İşlenecek token listesi boş.")
            return ""

        if not all(isinstance(token, str) for token in tokens):
            raise TypeError("Token listesi yalnızca string içermelidir.")

        # 1. Tokenleri birleştir
        output = " ".join(tokens)

        # 2. Fazla boşlukları temizle (birden fazla → bir boşluk)
        output = " ".join(output.strip().split())

        # 3. (İsteğe Bağlı) Noktalama düzeltmeleri
        output = self._clean_punctuation(output)

        logger.debug(f"[✓] Nihai chat çıktısı: '{output}'")
        return output

    def _clean_punctuation(self, text: str) -> str:
        """
        Noktalama işaretlerinden önceki boşlukları temizler.
        Örneğin: 'Nasılsın ?' → 'Nasılsın?'

        Args:
            text (str): Girdi metni

        Returns:
            str: Düzenlenmiş metin
        """
        import re
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text
