import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ============================
#  Sabitler ve Kodlama Tanımları
# ============================
DEFAULT_ENCODING = 'utf-8'
FALLBACK_ENCODINGS = ['latin1', 'utf-16', 'ascii', 'cp1252']

# ============================
#  Hata Sınıfları
# ============================
class TXTLoaderError(Exception):
    """Genel TXTLoader hatası."""
    pass

class InvalidFileTypeError(TXTLoaderError):
    """Geçersiz dosya türü hatası."""
    pass

class FileTooLargeError(TXTLoaderError):
    """Dosya boyutu çok büyük olduğunda fırlatılır."""
    pass

class FileEmptyError(TXTLoaderError):
    """Boş dosya hatası."""
    pass

class EncodingError(TXTLoaderError):
    """Kodlama hatası."""
    pass


# ============================
#  TXTLoader Sınıfı
# ============================
class TXTLoader:
    """
    TXTLoader, metin dosyalarını yüklemek ve normalize etmek için kullanılan modüldür.
    
    Özellikler:
    - UTF-8 ile yükleme ve fallback encoding desteği.
    - ASCII olmayan karakterler temizlenir.
    - Maksimum uzunluk kontrolü.
    - Büyük dosyalarda performans optimizasyonu.
    """

    def __init__(self, max_length: int = 1_000_000):
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError(f"`max_length` pozitif bir tamsayı olmalıdır, alınan: {max_length}")

        self.max_length = max_length

    # ============================
    #  Dosya Yükleme
    # ============================
    def load_file(self, file_path: str) -> str:
        """
        TXT dosyasını yükler ve normalize eder.

        Args:
            file_path (str): Yüklenecek dosyanın yolu.

        Returns:
            str: Normalize edilmiş dosya içeriği.
        
        Raises:
            FileNotFoundError: Dosya bulunamazsa.
            InvalidFileTypeError: Geçersiz dosya türü ise.
            FileEmptyError: Dosya boşsa.
            EncodingError: Kodlama hatası varsa.
            TXTLoaderError: Diğer tüm hatalar.
        """
        if not isinstance(file_path, str):
            raise TypeError(f"`file_path` tipi 'str' olmalıdır, alınan: {type(file_path)}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f" Dosya bulunamadı: {file_path}")

        if not os.path.isfile(file_path):
            raise InvalidFileTypeError(f" Geçersiz dosya türü: {file_path}")

        if not os.access(file_path, os.R_OK):
            raise PermissionError(f" Dosya okunamıyor veya erişim izni yok: {file_path}")

        content = None

        try:
            #  İlk olarak UTF-8 ile yükleme yap
            try:
                with open(file_path, 'r', encoding=DEFAULT_ENCODING) as f:
                    content = f.read()
            except UnicodeDecodeError:
                logger.warning(f" UTF-8 ile dosya okunamadı, fallback encoding deneniyor: {file_path}")
                
                #  Fallback Encoding Denemesi
                for encoding in FALLBACK_ENCODINGS:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                            logger.info(f" {encoding} ile dosya yüklendi: {file_path}")
                            break
                    except UnicodeDecodeError:
                        logger.warning(f" {encoding} kodlaması ile dosya okunamadı.")
                        continue
            
            if content is None:
                raise EncodingError(f"Dosya kodlaması belirlenemedi: {file_path}")

            #  Boş dosya kontrolü
            if not content.strip():
                raise FileEmptyError(f"Dosya boş veya geçersiz içerik içeriyor: {file_path}")

            #  Normalize Etme
            normalized_content = self._normalize_text(content)

            #  Maksimum uzunluk kontrolü
            if len(normalized_content) > self.max_length:
                logger.warning(
                    f" Dosya içeriği maksimum uzunluğu ({self.max_length} karakter) aşıyor. Kesiliyor: {file_path}"
                )
                normalized_content = normalized_content[:self.max_length]

            logger.info(f" TXT dosyası başarıyla yüklendi -> Uzunluk: {len(normalized_content)} karakter")
            return normalized_content

        except FileNotFoundError:
            logger.error(f" Dosya bulunamadı: {file_path}", exc_info=True)
            raise

        except PermissionError:
            logger.error(f" Dosya erişim hatası: {file_path}", exc_info=True)
            raise

        except UnicodeDecodeError as e:
            logger.error(f" Kodlama hatası: {e}", exc_info=True)
            raise EncodingError(f"Kodlama hatası: {e}")

        except Exception as e:
            logger.error(f" Genel hata oluştu: {e}", exc_info=True)
            raise TXTLoaderError(f"Genel hata oluştu: {e}")


    # ============================
    #  Metin Normalizasyonu
    # ============================
    def _normalize_text(self, text: str) -> str:
        """
        Metin temizliği ve ASCII olmayan karakterlerin temizlenmesi.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        try:
            #  Satır sonları ve gereksiz boşlukları kaldır
            text = " ".join(text.split())

            #  ASCII olmayan karakterleri temizle (emoji ve semboller)
            text = re.sub(r"[^\x00-\x7F]+", " ", text)

            #  Fazla boşlukları kaldır
            text = re.sub(r"\s+", " ", text)

            logger.debug(f" Normalize edilmiş metin: {text[:100]}...")
            return text

        except Exception as e:
            logger.error(f" Metin normalize edilirken hata oluştu: {e}", exc_info=True)
            raise TXTLoaderError(f"Metin normalize edilirken hata oluştu: {e}")


