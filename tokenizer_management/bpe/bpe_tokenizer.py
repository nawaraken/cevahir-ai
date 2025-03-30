import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class BPETokenizerError(Exception):
    pass


class BPETokenizer:
    def __init__(self, encoder, decoder, vocab=None):
        """
        BPETokenizer sınıfı, BPE kodlama ve çözme işlemlerini başlatır.
        Encoder ve Decoder doğrudan dışarıdan bağımlılık enjeksiyonu ile alınır.

        Args:
            encoder (BPEEncoder): Kodlama işlemlerini yürüten sınıf.
            decoder (BPEDecoder): Çözme işlemlerini yürüten sınıf.
            vocab (dict): Vocab sözlüğü.
        """
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab or {}

        logger.info("[+] BPETokenizer başlatıldı.")

    def encode(self, text: str) -> List[int]:
        """
        Metni BPE formatında kodlar ve token ID'lerini döner.

        Args:
            text (str): Kodlanacak metin.

        Returns:
            List[int]: Kodlanmış token ID listesi.
        """
        try:
            if not text:
                raise ValueError("Kodlanacak metin boş olamaz.")

            if not self.vocab:
                raise BPETokenizerError("[X] Vocab boş. Kodlama yapılamaz.")  #  Hata yönetimi düzeltildi.

            logger.info(f"[+] Kodlama işlemi başlatılıyor: {text}")
            token_ids = self.encoder.encode(text)
            logger.debug(f"[+] Kodlanmış token ID'leri: {token_ids}")

            return token_ids

        except Exception as e:
            logger.error(f"[X] Kodlama sırasında hata oluştu: {e}")
            raise BPETokenizerError(f"Encode Error: {e}")

    def decode(self, token_ids: List[int]) -> str:
        """
        Token ID'lerini çözerek metne dönüştürür.

        Args:
            token_ids (List[int]): Kodlanmış token ID listesi.

        Returns:
            str: Çözümlenmiş metin.
        """
        try:
            if not token_ids:
                raise ValueError("Çözülecek token listesi boş olamaz.")

            if not self.vocab:
                raise BPETokenizerError("[X] Vocab boş. Çözümleme yapılamaz.")  #  Hata yönetimi düzeltildi.

            logger.info(f"[+] Çözümleme işlemi başlatılıyor: {token_ids}")
            text = self.decoder.decode(token_ids)
            logger.debug(f"[+] Çözümlenmiş metin (post-process öncesi): {text}")

            #  Post-processing sırasındaki boşluk kaldırma işlemi kaldırıldı
            # Çıktı doğrudan çözülen token'lar üzerinden yapılacak
            processed_text = self._post_process(text)
            logger.debug(f"[+] Çözümlenmiş metin (post-process sonrası): {processed_text}")

            return processed_text

        except Exception as e:
            logger.error(f"[X] Çözümleme sırasında hata oluştu: {e}")
            raise BPETokenizerError(f"Decode Error: {e}")

    def get_token_ids(self, text: str) -> List[int]:
        """
        Metni kodlayarak token ID'lerini döner.

        Args:
            text (str): Kodlanacak metin.

        Returns:
            List[int]: Kodlanmış token ID listesi.
        """
        try:
            token_ids = self.encode(text)
            return token_ids

        except Exception as e:
            logger.error(f"[X] Token ID alma sırasında hata oluştu: {e}")
            raise BPETokenizerError(f"Get Token IDs Error: {e}")

    def get_text(self, token_ids: List[int]) -> str:
        """
        Token ID'lerinden metin oluşturur.

        Args:
            token_ids (List[int]): Kodlanmış token ID listesi.

        Returns:
            str: Çözümlenmiş metin.
        """
        try:
            text = self.decode(token_ids)
            return text

        except Exception as e:
            logger.error(f"[X] Token çözme sırasında hata oluştu: {e}")
            raise BPETokenizerError(f"Get Text Error: {e}")

    def _post_process(self, text: str) -> str:
        """
        Çözümlenmiş metin üzerinde son işlem.

        Args:
            text (str): İşlenecek metin.

        Returns:
            str: Düzeltilmiş metin.
        """
        #  Fazladan boşluk ekleyen işlem kaldırıldı
        # Metin olduğu gibi bırakılıyor
        return text.replace("  ", " ").strip()

