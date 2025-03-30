import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ChatPreprocessor:
    """
    ChatPreprocessor
    ----------------
    Chat mesajları için ön işleme sağlar:

    - Tüm harfleri küçük harfe çevirir.
    - Başlangıç ve sondaki boşlukları temizler.
    - Birden fazla boşluk karakterini tek boşluk haline getirir.
    - Gereksiz noktalama ve özel karakterleri opsiyonel olarak filtreler (gerektiğinde genişletilebilir).
    """

    def __init__(self, remove_nontext_symbols: bool = False):
        """
        Args:
            remove_nontext_symbols (bool): True ise özel karakterleri filtreler.
        """
        self.remove_nontext_symbols = remove_nontext_symbols
        self.extra_spaces_pattern = re.compile(r'\s+')
        self.non_text_pattern = re.compile(r'[^a-zA-ZçÇğĞıİöÖşŞüÜ0-9\s.,!?;:]+')

        logger.info("ChatPreprocessor başlatıldı.")

    def preprocess(self, text: Optional[str]) -> str:
        """
        Verilen chat mesajını normalize eder.

        Args:
            text (str): Girdi metni.

        Returns:
            str: Temizlenmiş, normalize edilmiş metin.

        Raises:
            ValueError: Girdi boş veya None ise.
            TypeError: Girdi tipi string değilse.
        """
        if text is None:
            logger.error("[X] Preprocess hatası: Girdi metni None.")
            raise ValueError("Girdi metni None olamaz.")

        if not isinstance(text, str):
            logger.error(f"[X] Preprocess hatası: Girdi tipi geçersiz: {type(text)}")
            raise TypeError("Girdi tipi string olmalıdır.")

        # 1. Trim + lowercase
        text = text.strip().lower()

        # 2. Gereksiz karakterleri temizle (opsiyonel)
        if self.remove_nontext_symbols:
            text = self.non_text_pattern.sub('', text)

        # 3. Boşluk normalize et
        text = self.extra_spaces_pattern.sub(' ', text)

        logger.debug(f"[✓] Preprocessed text: '{text}'")
        return text
