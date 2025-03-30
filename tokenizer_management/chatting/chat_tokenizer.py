import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ChatTokenizer:
    """
    ChatTokenizer
    -------------
    Chat mesajlarını basit, hızlı ve genişletilebilir biçimde tokenlere ayırır.

    Varsayılan olarak, boşluk karakterlerine göre kelimeleri ayırır.
    İsteğe bağlı olarak noktalama işaretlerini ayrı token olarak bölebilir.
    """

    def __init__(self, split_punctuation: bool = True):
        """
        Args:
            split_punctuation (bool): True ise noktalama işaretlerini ayrı token olarak ayırır.
        """
        self.split_punctuation = split_punctuation

        # Noktalama ayırmalı ya da ayırmasız token desenleri
        self.pattern = (
            re.compile(r'\w+|[^\w\s]', re.UNICODE)  # Kelimeler ve noktalama ayırma
            if split_punctuation else
            re.compile(r'\S+')  # Sadece boşlukla ayırma
        )

        logger.info("ChatTokenizer başlatıldı. Noktalama ayırma: %s", self.split_punctuation)

    def tokenize(self, text: Optional[str]) -> List[str]:
        """
        Verilen metni token'lara ayırır.

        Args:
            text (str): Girdi chat metni.

        Returns:
            List[str]: Token listesi.

        Raises:
            ValueError: Girdi None ise.
            TypeError: Girdi tipi string değilse.
        """
        if text is None:
            logger.error("[X] ChatTokenizer: Girdi metni None.")
            raise ValueError("Girdi metni None olamaz.")

        if not isinstance(text, str):
            logger.error("[X] ChatTokenizer: Girdi tipi geçersiz: %s", type(text))
            raise TypeError("Girdi tipi string olmalıdır.")

        text = text.strip()
        if not text:
            logger.warning("[!] ChatTokenizer: Girdi boş string.")
            return []

        tokens = self.pattern.findall(text)
        logger.debug("ChatTokenizer: Tokenize edilmiş metin → %s", tokens)
        return tokens
