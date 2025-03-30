import logging
import unicodedata
import re
from typing import List, Union

logger = logging.getLogger(__name__)

class PretokenizationError(ValueError):
    """Tokenizasyon sırasında oluşan hatalar için özel istisna."""
    pass

class Pretokenizer:
    def __init__(self):
        """
        Pretokenizer sınıfı, metin üzerinde temel ön işleme işlemlerini yapar.
        İşlemler:
          - Unicode normalizasyonu (NFC)
          - Türkçe karakter desteği
          - Büyük harfleri küçültme
          - Özel karakter temizliği
          - Fazla boşlukların temizlenmesi ve tokenlere ayırma
        """
        logger.info("[+] Pretokenizer başlatılıyor...")

        # Unicode kategorilerini temel alan düzenli ifadeler (güçlü regex)
        self.cleaning_pattern = re.compile(r"[^\w\sçğıöşüÇĞİÖŞÜ]", re.UNICODE)
        self.spacing_pattern = re.compile(r"(\d)([^\d\s])|([^\d\s])(\d)")
        self.whitespace_pattern = re.compile(r"\s+")

        # Geçerli karakterler (Türkçe karakter desteği dahil)
        # Geçerli karakterler (Türkçe, Avrupa dilleri, matematik sembolleri ve özel işaretler dahil)
        self.valid_characters = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "çğıöşüÇĞİÖŞÜ"  # Türkçe karakterler
            "áéíóúñüÁÉÍÓÚÑÜÇ"  # İspanyolca ve Avrupa dilleri karakterleri
            "äöüßÄÖÜẞ"  # Almanca karakterler
            "àèìòùÀÈÌÒÙâêîôûÂÊÎÔÛ"  # Fransızca ve İtalyanca karakterler
            "øØåÅæÆ"  # İskandinav dilleri karakterleri
            "π∑∫∞√±≈≠≡≤≥∂∆∇"  # Matematik sembolleri
            "θλφψΩμτβγσϵζηξκ"  # Yunan harfleri
            "!@#$%^&*()-_=+[{]};:'\",<.>/?\\|`~"  # Özel karakterler
            " 0123456789\t\n\r"  # Boşluk ve kontrol karakterleri
            "ʃθðæø"  # Fonetik semboller
        )


        # State tutma (reset işlemi için)
        self._original_text = None
        self._cleaned_text = None
        self._tokens = None

        logger.info("[+] Pretokenizer başarıyla başlatıldı.")

    def tokenize(self, text: Union[str, List[str], dict]) -> List[str]:
        """
        Metni tokenlere ayırır ve temel ön işlemleri uygular.

        Args:
            text (Union[str, List[str], dict]): Girdi metni.

        Returns:
            List[str]: İşlenmiş token listesi.

        Raises:
            ValueError: Girdi metni boş veya yanlış tipte olursa.
            PretokenizationError: Tokenizasyon sonrası geçerli token bulunamazsa.
        """
        try:
            # 1. GİRİŞ TÜRÜ KONTROLÜ
            if text is None:
                raise ValueError("Girdi metni None olamaz.")
            if isinstance(text, list):
                text = " ".join(map(str, text))
            elif isinstance(text, dict):
                text = str(text.get("data", ""))

            if not isinstance(text, str):
                raise TypeError(f"Girdi metni string olmalıdır, {type(text)} verildi.")

            text = text.strip()
            if not text:
                logger.warning("[!] Girdi metni boş!")
                return ["<EMPTY>"]

            logger.debug(f"[+] Orijinal metin: '{text}'")

            # 2. STATE SAKLAMA
            self._original_text = text

            # 3. Unicode normalizasyonu ve lowercase dönüşümü
            text = unicodedata.normalize('NFC', text).lower()
            logger.debug(f"[+] Unicode normalizasyonu ve lowercase sonrası: '{text}'")

            # 4. Özel karakter temizliği (gelişmiş regex ile)
            text = self.cleaning_pattern.sub(' ', text)
            logger.debug(f"[+] Temizlenmiş metin: '{text}'")

            # 5. Sayısal ve harf ayrımı için boşluk ekleme
            text = self.spacing_pattern.sub(r"\1 \2 \3 \4", text)
            logger.debug(f"[+] Sayısal ve alfanümerik ayrımı sonrası: '{text}'")

            # 6. Fazla boşlukların kaldırılması
            text = self.whitespace_pattern.sub(' ', text).strip()
            logger.debug(f"[+] Fazla boşluklar kaldırıldıktan sonra: '{text}'")

            # 7. Tokenlere ayırma ve boş token kontrolü
            tokens = text.split()
            logger.debug(f"[+] Tokenler: {tokens}")

            # 8. Boş token kontrolü
            if not tokens:
                logger.warning("[!] Tokenizasyon sonrası geçerli token bulunamadı. '<EMPTY>' döndürüldü.")
                return ["<EMPTY>"]

            # 9. Geçersiz karakter kontrolü
            invalid_tokens = [token for token in tokens if not all(char in self.valid_characters for char in token)]
            if invalid_tokens:
                logger.warning(f"[!] Geçersiz tokenler bulundu: {invalid_tokens}")
                raise PretokenizationError(f"Geçersiz tokenler tespit edildi: {invalid_tokens}")

            # 10. State saklama
            self._cleaned_text = text
            self._tokens = tokens

            logger.info(f"[+] Tokenizasyon tamamlandı: {tokens}")
            return tokens

        except (ValueError, TypeError) as e:
            logger.error(f"[X] Pretokenizer hatası: {e}")
            raise e
        except Exception as e:
            logger.error(f"[X] Pretokenizer hatası: {e}")
            raise PretokenizationError(f"Pretokenizer Error: {e}")

    def reset(self):
        """
        Pretokenizer durumunu sıfırlar.
        """
        try:
            logger.warning("[!] Pretokenizer sıfırlanıyor...")

            # Pretokenizer'ı baştan başlatıyoruz
            self.__init__()

            logger.info("[+] Pretokenizer sıfırlandı.")

        except Exception as e:
            logger.error(f"[X] Pretokenizer reset hatası: {e}")
            raise PretokenizationError(f"Pretokenizer Reset Error: {e}")

    def validate_token(self, token: str) -> bool:
        """
        Token geçerliliğini kontrol eder.
        Args:
            token (str): Kontrol edilecek token.

        Returns:
            bool: Geçerli ise True, geçerli değilse False.
        """
        return all(char in self.valid_characters for char in token)

