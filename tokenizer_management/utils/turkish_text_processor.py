import re
import unicodedata
import logging
from typing import List, Dict, Optional
from ..config import get_turkish_config

# === Logger Yapılandırması ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Yapılandırmayı Yükle ve Sınıf Düzeyinde Değişkenlere Ata ===
_turkish_config = get_turkish_config()
CHARACTERS = _turkish_config["characters"]
VOWELS = _turkish_config["vowels"]
CONSONANTS = _turkish_config["consonants"]
STOPWORDS = set(_turkish_config["stopwords"])  # Hızlı erişim için set
TEXT_PROCESSING = _turkish_config["text_processing"]
SYLLABIFICATION_RULES = _turkish_config["syllabification_rules"]
SPECIAL_CHARACTERS = _turkish_config["special_characters"]

class TurkishTextProcessor:
    def __init__(self, syllabifier=None):
        """
        Constructor: Yapılandırma bilgileri ve opsiyonel heceleyici (syllabifier) yüklenir.
        """
        try:
            self.syllabifier = syllabifier
            # Yapılandırma bilgilerini örnek düzeyde saklıyoruz:
            self.characters = CHARACTERS
            self.vowels = VOWELS
            self.consonants = CONSONANTS
            self.stopwords = STOPWORDS
            self.text_processing = TEXT_PROCESSING
            self.syllabification_rules = SYLLABIFICATION_RULES
            self.special_characters = SPECIAL_CHARACTERS
            logger.info("TurkishTextProcessor başarıyla başlatıldı.")
        except Exception as e:
            logger.error(f"Constructor başlatılırken hata oluştu: {e}")
            raise

    # === 1. Unicode Normalizasyonu ===
    def normalize_unicode(self, text: str) -> str:
        """
        Unicode normalizasyonu yapar ve özel karakter dönüşümlerini uygular.
        """
        try:
            norm_type = self.text_processing.get("normalize_unicode", "NFC")
            if norm_type:
                text = unicodedata.normalize(norm_type, text)

            # Özel karakterleri dönüşüm listesine göre uygula
            for old, new in self.special_characters.items():
                text = text.replace(old, new)

            logger.debug(f"Unicode normalizasyon sonrası: {text}")
            return text
        except Exception as e:
            logger.error(f"normalize_unicode sırasında hata oluştu: {e}")
            raise

    def apply_vowel_harmony(self, token: str) -> str:
        """
        Ünlü uyumu kurallarını uygular.
        """
        try:
            if self.text_processing.get("apply_vowel_harmony", False):
                back_vowels = set(self.vowel_harmony_rules["back_vowels"])
                front_vowels = set(self.vowel_harmony_rules["front_vowels"])

                last_vowel = next((c for c in reversed(token) if c in self.vowels), None)

                if last_vowel in back_vowels:
                    token = re.sub(r'[eiöü]', 'a', token)
                elif last_vowel in front_vowels:
                    token = re.sub(r'[aıou]', 'e', token)

            logger.debug(f"Ünlü uyumu sonrası: {token}")
            return token
        except Exception as e:
            logger.error(f"apply_vowel_harmony sırasında hata oluştu: {e}")
            raise

    def apply_consonant_harmony(self, token: str) -> str:
        """
        Ünsüz benzeşmesi kurallarını uygular.
        """
        try:
            if self.text_processing.get("apply_consonant_harmony", False):
                if token[-1] in self.consonant_harmony_rules["voiceless_consonants"]:
                    token = token[:-1] + "t"
                elif token[-1] in self.consonant_harmony_rules["voiced_consonants"]:
                    token = token[:-1] + "d"

            logger.debug(f"Ünsüz benzeşmesi sonrası: {token}")
            return token
        except Exception as e:
            logger.error(f"apply_consonant_harmony sırasında hata oluştu: {e}")
            raise

    def remove_suffixes(self, token: str) -> str:
        """
        Çekim ve yapım eklerini kaldırır.
        """
        try:
            for suffix in self.suffixes["noun_suffixes"]:
                if token.endswith(suffix):
                    token = token[:-len(suffix)]

            logger.debug(f"Ek kaldırma sonrası: {token}")
            return token
        except Exception as e:
            logger.error(f"remove_suffixes sırasında hata oluştu: {e}")
            raise



    # === 2. Küçük Harfe Çevirme ===
    def to_lowercase(self, text: str) -> str:
        """
        Metni, Türkçe karakter dönüşümleri yaparak tamamen küçük harfe çevirir.
        """
        try:
            if self.text_processing.get("lowercase", True):
                # Türkçe özel dönüşüm: 'İ' -> 'i' ve 'I' -> 'ı'
                text = text.replace("İ", "i").replace("I", "ı").lower()
            logger.debug(f"Küçük harfe çevrildi: {text}")
            return text
        except Exception as e:
            logger.error(f"to_lowercase sırasında hata oluştu: {e}")
            raise

    # === 3. Noktalama İşaretlerini Kaldırma ===
    def remove_punctuation(self, text: str) -> str:
        """
        Metindeki noktalama işaretlerini temizler.
        """
        try:
            if self.text_processing.get("remove_punctuation", True):
                text = re.sub(r'[^\w\s]', '', text)
            logger.debug(f"Noktalama kaldırıldı: {text}")
            return text
        except Exception as e:
            logger.error(f"remove_punctuation sırasında hata oluştu: {e}")
            raise

    # === 4. Stopword'leri Kaldırma ===
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Token listesinden stopword'leri çıkarır.
        """
        try:
            if self.text_processing.get("remove_stopwords", True):
                tokens = [token for token in tokens if token not in self.stopwords]
            logger.debug(f"Stopwords kaldırıldı: {tokens}")
            return tokens
        except Exception as e:
            logger.error(f"remove_stopwords sırasında hata oluştu: {e}")
            raise

    # === 5. Heceleme (Türkçe Kurallara Göre) ===
    def syllabify(self, tokens: List[str]) -> List[str]:
        """
        Heceleme işlemi yapar.
        """
        try:
            syllables = []
            for token in tokens:
                current = ""
                for char in token:
                    current += char
                    if len(current) >= self.syllabification_rules["max_syllable_length"]:
                        syllables.append(current)
                        current = ""
                if current:
                    syllables.append(current)
                    
            logger.debug(f"Heceleme sonucu: {syllables}")
            return syllables
        except Exception as e:
            logger.error(f"syllabify sırasında hata oluştu: {e}")
            raise



    # === 6. Kök Bulma (Stemming) ===
    def stem(self, word: str) -> str:
        try:
            suffixes = [
                r"(lar|ler)$", r"(lık|lik|luk|lük)$", r"(cı|ci|cu|cü)$", 
                r"(mak|mek)$", r"(da|de|den|dan)$", r"(mış|miş|muş|müş)$",
                r"(sınız|siniz)$", r"(casına|cesine)$"
            ]

            for suffix in suffixes:
                word = re.sub(suffix, '', word)
            
            logger.debug(f"Stemming sonucu: {word}")
            return word
        except Exception as e:
            logger.error(f"stem sırasında hata oluştu: {e}")
            raise


    # === 7. Lemmatization ===
    def lemmatize(self, word: str) -> str:
        try:
            lemma_map = {
                "kitaplar": "kitap",
                "evler": "ev",
                "koşuyor": "koş",
                "çalışıyor": "çalış",
                "ağaçlar": "ağaç",
                "yazıyorum": "yaz",
                "düşünüyor": "düşün"
            }
            result = lemma_map.get(word, word)
            logger.debug(f"Lemmatization sonucu: {result}")
            return result
        except Exception as e:
            logger.error(f"lemmatize sırasında hata oluştu: {e}")
            raise


    # === 8. Özel Karakter İşleme ===
    def process_special_characters(self, text: str) -> str:
        """
        Metindeki özel karakterleri, yapılandırmadan alınan eşleşmelere göre dönüştürür.
        """
        try:
            for old, new in SPECIAL_CHARACTERS.items():
                text = text.replace(old, new)
            logger.debug(f"Özel karakter işleme sonucu: {text}")
            return text
        except Exception as e:
            logger.error(f"process_special_characters sırasında hata oluştu: {e}")
            raise

    # === 9. Metni Temizleme (Clean Text) ===
    def clean_text(self, text: str) -> str:
        """
        Unicode normalizasyonu, küçük harfe çevirme ve noktalama temizleme adımlarını uygular.
        """
        try:
            text = self.normalize_unicode(text)
            text = self.to_lowercase(text)
            text = self.remove_punctuation(text)
            logger.debug(f"Clean text sonucu: {text}")
            return text
        except Exception as e:
            logger.error(f"clean_text sırasında hata oluştu: {e}")
            raise

    # === 10. Tokenizasyon ve İşleme ===
    def process(self, text: str) -> List[str]:
        """
        Metni tokenlere ayırır, stopword'leri kaldırır, stemming ve lemmatization uygular,
        ardından heceleme işlemi gerçekleştirir.
        """
        try:
            # 1. Metni temizle
            text = self.clean_text(text)
            
            # 2. Tokenizasyon: Boşluk, noktalama ve özel karakterlerle ayır
            tokens = re.split(r'\s+|[,.!?;:]', text)
            tokens = [token for token in tokens if token]  # Boş tokenleri çıkar
            
            # 3. Kök bulma ve lemmatization işlemi yap
            tokens = [self.stem(token) for token in tokens]
            tokens = [self.lemmatize(token) for token in tokens]
            
            # 4. Stopword'leri kaldır (stem ve lemma sonrası daha güvenli)
            tokens = self.remove_stopwords(tokens)
            
            # 5. Heceleme işlemi uygula
            tokens = self.syllabify(tokens)
            
            logger.debug(f"Process sonucu: {tokens}")
            return tokens
        
        except Exception as e:
            logger.error(f"process sırasında hata oluştu: {e}")
            raise

    def full_process(self, text: str) -> List[str]:
        """
        Metnin özel karakter işlenmesi, temizlenmesi, tokenizasyonu ve
        son olarak özel token eklenmesi işlemlerini gerçekleştirir.
        """
        try:
            # 1. Özel karakter işleme → Temizleme sırası düzeltildi
            text = self.process_special_characters(text)
            
            # 2. Temizleme ve tokenizasyon işlemi
            tokens = self.process(text)
            
            # 3. Boş tokenler oluşursa çıkar
            tokens = [token for token in tokens if token]
            
            # 4. Özel token ekle
            tokens = self.add_special_tokens(tokens)
            
            # 5. Log çıktısını sınırla (çok uzun çıktıları kırp)
            if len(tokens) > 20:
                display_tokens = tokens[:10] + ["..."] + tokens[-10:]
            else:
                display_tokens = tokens
            
            logger.debug(f"Full process sonucu: {display_tokens}")
            return tokens
        
        except Exception as e:
            logger.error(f"full_process sırasında hata oluştu: {e}")
            raise


        # === 11. Özel Token Ekleme ===
    def add_special_tokens(self, tokens: List[str]) -> List[str]:
        """
        Token listesine başlangıç (<BOS>) ve bitiş (<EOS>) token'larını ekler.
        """
        try:
            tokens.insert(0, "<BOS>")
            tokens.append("<EOS>")
            logger.debug(f"Özel token ekleme sonucu: {tokens}")
            return tokens
        except Exception as e:
            logger.error(f"add_special_tokens sırasında hata oluştu: {e}")
            raise

    # === 12. Genel İşleme Fonksiyonu (Full Process) ===
    def full_process(self, text: str) -> List[str]:
        """
        Metnin özel karakter işlenmesi, temizlenmesi, tokenizasyonu ve
        son olarak özel token eklenmesi işlemlerini gerçekleştirir.
        """
        try:
            text = self.process_special_characters(text)
            tokens = self.process(text)
            tokens = self.add_special_tokens(tokens)
            logger.debug(f"Full process sonucu: {tokens}")
            return tokens
        except Exception as e:
            logger.error(f"full_process sırasında hata oluştu: {e}")
            raise

    # === 13. Toplu İşleme (Batch Process) ===
    def batch_process(self, texts: List[str]) -> List[List[str]]:
        """
        Birden fazla metni toplu olarak işler.
        """
        try:
            results = []
            for text in texts:
                try:
                    if text.strip():  # Boş veya sadece boşluk içeren satırları atla
                        result = self.full_process(text)
                        results.append(result)
                    else:
                        results.append(["<BOS>", "<EOS>"])
                except Exception as e:
                    logger.warning(f"batch_process sırasında hata oluştu: {e}")
                    results.append(["<BOS>", "<EOS>"])  # Hata oluştuğunda boş liste döndür
            
            # Log çıktısını sınırla (max 3 örnek göster)
            if len(results) > 3:
                display_results = results[:2] + [["..."]] + results[-2:]
            else:
                display_results = results
            
            logger.debug(f"Batch process sonucu: {display_results}")
            return results
        
        except Exception as e:
            logger.error(f"batch_process sırasında hata oluştu: {e}")
            raise


    # === 14. Giriş Doğrulama (Input Validation) ===
    def validate_input(self, text: str) -> bool:
        """
        Girdi metninin doğru tipte olup olmadığını ve boş olmadığını kontrol eder.
        """
        if not isinstance(text, str):
            raise TypeError(f"Input text must be string. Got {type(text)}")
        return len(text) > 0
