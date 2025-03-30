import logging
import unicodedata
from typing import List
from tokenizer_management.config import get_turkish_config

logger = logging.getLogger(__name__)

# === Türkçe Yapılandırmayı Yükle ===
try:
    _turkish_config = get_turkish_config()
    VOWELS = set(_turkish_config.get("vowels", "aeıioöuü"))
    CONSONANTS = set(_turkish_config.get("consonants", "bcçdfgğhjklmnprsştvyz"))
    SYLLABIFICATION_RULES = _turkish_config.get("syllabification_rules", {})
except Exception as e:
    logger.error(f"Yapılandırma dosyası yüklenemedi: {e}")
    raise ImportError(f"Türkçe yapılandırma dosyası yüklenemedi: {e}")

# === Hata Sınıfı ===
class SyllabificationError(Exception):
    pass


class Syllabifier:
    def __init__(self, max_token_length=10000):
        """
        Türkçe heceleme işlemlerini gerçekleştiren sınıf.
        Yapılandırma dosyasından alınan kurallara göre heceleme yapar.
        """
        logger.info("[+] Syllabifier başlatıldı.")
        self.vowels = VOWELS
        self.consonants = CONSONANTS
        self.rules = SYLLABIFICATION_RULES
        self.max_token_length = max_token_length

    def split(self, tokens: List[str]) -> List[str]:
        """
        Token listesi üzerinde heceleme yapar.

        Args:
            tokens (List[str]): İşlenecek token listesi.

        Returns:
            List[str]: Hecelenmiş token listesi.

        Raises:
            ValueError: Token listesi boşsa veya geçersiz türdeyse.
            SyllabificationError: Heceleme hatası olursa.
        """
        if not tokens:
            raise ValueError("Token listesi boş olamaz.")
        if not isinstance(tokens, list):
            raise TypeError(f"Geçersiz token türü: {type(tokens)}")

        syllables = []
        for token in tokens:
            if not isinstance(token, str):
                raise TypeError(f"Geçersiz token türü: {type(token)}")

            token = unicodedata.normalize("NFC", token)

            # === Özel Etiket Kontrolü ===
            if token.startswith("__tag__"):
                logger.debug(f"[~] Özel etiket tespit edildi, doğrudan eklendi: {token}")
                syllables.append(token)
                continue

            # === Uzun Token Kontrolü ===
            if len(token) > self.max_token_length:
                logger.warning(f"[!] Uzun token ({len(token)} karakter) → Direkt döndürüldü.")
                syllables.append(token)
                continue

            # === Sesli Harf Kontrolü ===
            if not any(ch in self.vowels for ch in token if ch.isalpha()):
                logger.warning(f"[!] '{token}' sesli harf içermediği için döndürüldü.")
                syllables.append(token)
                continue

            # === Heceleme İşlemi ===
            word_syllables = self._syllabify_word(token)
            if word_syllables:
                syllables.extend(word_syllables)
            else:
                logger.warning(f"[!] Heceleme başarısız oldu → '{token}'")
                syllables.append(token)

        if not syllables:
            raise SyllabificationError("Heceleme sonucunda geçerli bir hece bulunamadı.")

        logger.debug(f"[+] Heceleme sonucu: {syllables}")
        return syllables


    def split_into_syllables(self, text: str) -> List[str]:
        if not isinstance(text, str):
            raise TypeError(f"Geçersiz metin türü: {type(text)}")

        text = unicodedata.normalize("NFC", text.lower())
        vowels = self.vowels
        consonants = self.consonants
        syllables = []
        current = ""

        i = 0
        while i < len(text):
            current += text[i]

            if text[i] in vowels:
                # === Sonraki karakter kontrolü ===
                if i + 1 < len(text):
                    next_char = text[i + 1]

                    #  1. Sesli harften sonra tek sessiz harf ve ardından sesli harf → Böl
                    if next_char in consonants:
                        if i + 2 < len(text) and text[i + 2] in vowels:
                            syllables.append(current)
                            current = ""

                        #  2. Arka arkaya iki sessiz harf gelirse → ilk sessizden sonra böl
                        elif i + 2 < len(text) and text[i + 2] in consonants:
                            syllables.append(current + text[i + 1])
                            current = ""
                            i += 1

                        #  3. Üçlü sessiz harf kümesi → ilk iki sessizden sonra böl
                        elif i + 3 < len(text) and all(ch in consonants for ch in text[i + 1:i + 4]):
                            syllables.append(current + text[i + 1] + text[i + 2])
                            current = ""
                            i += 2

                    #  4. Ünlü + sessiz + sessiz + ünlü yapısı → ilk sessizden sonra böl
                    elif (i + 2 < len(text) and 
                        text[i + 1] in consonants and 
                        text[i + 2] in vowels):
                        syllables.append(current)
                        current = ""

                else:
                    #  5. Metnin sonuna geldiysek kapat
                    syllables.append(current)
                    current = ""

            i += 1

        if current:
            syllables.append(current)

        # === Kural Tabanlı Son Düzeltmeler ===
        fixed_syllables = []
        for syllable in syllables:
            #  Eğer sessiz harf ile bitiyorsa → son sessizden sonra bölme yap
            if len(syllable) > 1 and syllable[-1] in consonants:
                fixed_syllables.append(syllable[:-1])
                fixed_syllables.append(syllable[-1])
            else:
                fixed_syllables.append(syllable)

        #  İkili sessiz harf + ünlü kombinasyonu için düzenleme
        final_syllables = []
        for i, syllable in enumerate(fixed_syllables):
            if i > 0 and len(syllable) == 1 and syllable in consonants:
                # Eğer önceki hece sessiz harf ile bittiyse → birleştir
                final_syllables[-1] += syllable
            else:
                final_syllables.append(syllable)

        logger.debug(f"[+] Heceleme sonucu: {final_syllables}")
        return final_syllables


    def _syllabify_word(self, word: str) -> List[str]:
        try:
            word = unicodedata.normalize("NFC", word.lower())
            syllables = []
            start = 0

            while start < len(word):
                first_vowel = next((i for i in range(start, len(word)) if word[i] in self.vowels), None)
                if first_vowel is None:
                    syllables.append(word[start:])
                    break

                next_vowel = next((i for i in range(first_vowel + 1, len(word)) if word[i] in self.vowels), None)
                if next_vowel is None:
                    syllables.append(word[start:])
                    break

                cluster = word[first_vowel + 1: next_vowel]
                boundary = next_vowel

                #  **Küme Kontrolü**
                if len(cluster) > 1:
                    # Tam sessiz kümeyse → İlk sessizden sonra böl
                    if all(ch in self.consonants for ch in cluster):
                        boundary = first_vowel + 1
                    elif len(cluster) == 2:
                        boundary = first_vowel + 1
                    elif len(cluster) == 3:
                        boundary = first_vowel + 2
                    elif cluster[0] in self.consonants and cluster[1] in self.vowels:
                        boundary = first_vowel + 1
                    else:
                        boundary = next_vowel
                else:
                    boundary = next_vowel

                #  Fiil kökü + ek zinciri kontrolü
                if boundary - start > 2 and word[boundary - 1] in self.consonants:
                    boundary -= 1

                syllable = word[start:boundary]
                syllables.append(syllable)

                start = boundary

            if not syllables:
                raise ValueError("Heceleme sonucunda geçerli bir hece bulunamadı.")

            return syllables

        except Exception as e:
            logger.error(f"Heceleme sırasında hata: {e}")
            raise ValueError(f"Heceleme hatası: {e}")



    def reset(self):
        """
        Heceleme yapısını sıfırlar ve yapılandırmayı yeniden yükler.
        """
        try:
            logger.warning("[!] Syllabifier sıfırlanıyor...")
            config = get_turkish_config()
            self.vowels = set(config.get("vowels", "aeıioöuü"))
            self.consonants = set(config.get("consonants", "bcçdfgğhjklmnprsştvyz"))
            self.rules = config.get("syllabification_rules", {})
            logger.info("[+] Syllabifier sıfırlandı.")
        except Exception as e:
            logger.error(f"Sıfırlama hatası: {e}")
            raise SyllabificationError(f"Sıfırlama hatası: {e}")

