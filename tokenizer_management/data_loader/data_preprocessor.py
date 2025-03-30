import re
import logging
import unicodedata
from typing import Optional, Union

logger = logging.getLogger(__name__)

# ============================
#  Özel Token Tanımları
# ============================
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# ============================
#  Hata Sınıfları
# ============================
class PreprocessingError(Exception):
    pass

class InvalidTextError(PreprocessingError):
    pass

class TokenInsertionError(PreprocessingError):
    pass

# ============================
#  DataPreprocessor Sınıfı
# ============================
class DataPreprocessor:
    def __init__(self, 
                 bos_token: str = BOS_TOKEN, 
                 eos_token: str = EOS_TOKEN, 
                 pad_token: str = PAD_TOKEN,
                 unk_token: str = UNK_TOKEN):
        if not all(isinstance(t, str) for t in [bos_token, eos_token, pad_token, unk_token]):
            raise TypeError("Tüm özel tokenler string olmalıdır.")
        
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

    # ============================
    #  Metin Temizleme
    # ============================
    def clean_text(self, 
                   text: str, 
                   remove_special_chars: bool = True,
                   lowercase: bool = True,
                   remove_unicode: bool = True) -> str:
        if not isinstance(text, str):
            raise InvalidTextError(f"Giriş tipi geçersiz: Beklenen tür: `str`, alınan tür: {type(text)}")

        try:
            # Unicode karakterlerini normalize et
            if remove_unicode:
                text = unicodedata.normalize("NFKD", text)

            # Baş ve sondaki boşlukları kaldır
            text = text.strip()

            # Gereksiz boşlukları kaldır
            text = re.sub(r'\s+', ' ', text)

            # Özel karakterleri kaldır (isteğe bağlı)
            if remove_special_chars:
                text = re.sub(r'[^\w\s.,!?;:()\'"-]', '', text)

            # Küçük harfe çevir
            if lowercase:
                text = text.lower()

            logger.debug(f" Temizlenmiş metin: {text[:100]}...")
            return text
        
        except Exception as e:
            logger.error(f"Temizleme hatası: {e}", exc_info=True)
            raise PreprocessingError(f"Temizleme sırasında hata oluştu: {e}")

    # ============================
    #  Özel Token Ekleme
    # ============================
    def add_special_tokens(self, 
                           text: str, 
                           add_bos: bool = True, 
                           add_eos: bool = True) -> str:
        if not isinstance(text, str):
            raise InvalidTextError(f"Beklenen `str`, alınan `{type(text)}`")

        try:
            tokens = []
            if add_bos:
                tokens.append(self.bos_token)
            tokens.append(text)
            if add_eos:
                tokens.append(self.eos_token)
            
            output = " ".join(tokens)
            logger.debug(f" Özel tokenler eklenmiş metin: {output[:100]}...")
            return output
        
        except Exception as e:
            logger.error(f"Token ekleme hatası: {e}", exc_info=True)
            raise TokenInsertionError(f"Token ekleme sırasında hata oluştu: {e}")

    # ============================
    #  Genel İşlemci (Yeni Güncelleme)
    # ============================
    def preprocess_text(self, 
                        text: Union[str, list, dict], 
                        add_special: bool = True,
                        remove_special_chars: bool = True,
                        lowercase: bool = True,
                        remove_unicode: bool = True) -> str:
        """
        Farklı türdeki girişleri işleyebilen merkezi işleme metodu.
        
        - Liste veya sözlükler düz metne çevrilir.
        - Ardından temizleme ve token ekleme işlemi yapılır.
        """
        try:
            # Liste veya sözlük türlerini düz metne çeviriyoruz
            if isinstance(text, list):
                text = " ".join(map(str, text))
            elif isinstance(text, dict):
                text = " ".join([f"{k}: {v}" for k, v in text.items()])

            if not isinstance(text, str):
                raise InvalidTextError(f"Giriş tipi geçersiz: Beklenen tür `str`, alınan tür: {type(text)}")

            # Temizleme işlemi
            cleaned = self.clean_text(text, 
                                      remove_special_chars=remove_special_chars,
                                      lowercase=lowercase,
                                      remove_unicode=remove_unicode)
            # Özel token ekleme
            if add_special:
                cleaned = self.add_special_tokens(cleaned)

            return cleaned
        
        except Exception as e:
            logger.error(f"Ön işleme hatası: {e}", exc_info=True)
            raise PreprocessingError(f"Ön işleme sırasında hata oluştu: {e}")

