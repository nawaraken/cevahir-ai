import logging
from typing import List

logger = logging.getLogger(__name__)

class MorphologyError(Exception):
    """Türkçe dilbilgisine özgü kök ve ek ayrımında oluşan hatalar için özel istisna."""
    pass


class Morphology:
    def __init__(self):
        logger.info("[+] Morphology başlatıldı.")

        self.vowels = set("aeıioöuü")
        self.consonants = set("bcçdfgğhjklmnprsştvyz")

        self.suffixes = {
            "ler", "lar", "den", "dan", "in", "ın", "e", "a", "de", "da", 
            "le", "la", "m", "n", "s", "y", "i", "u", "ü", "o", 
            "imiz", "iniz", "dir", "dır", "ken", "mek", "mak"
        }

        self.consonant_softening = {
            "p": "b",
            "ç": "c",
            "t": "d",
            "k": "ğ"
        }

        self.vowel_harmony_front = set("eiöü")
        self.vowel_harmony_back = set("aıou")

        self._original_syllables = None
        self._morphemes = None

    def analyze(self, syllables: List[str]) -> List[str]:
        if not syllables:
            raise ValueError("Hece listesi boş olamaz.")

        self._original_syllables = syllables
        morphemes = []

        for syllable in syllables:
            if syllable.startswith("__tag__"):
                # Özel etiket, işlem yapılmadan alınır
                logger.debug(f"[~] Özel etiket bulundu: {syllable}")
                morphemes.append(syllable)
            else:
                result = self._split_morpheme(syllable)
                morphemes.extend(result)

        if not morphemes:
            raise MorphologyError("Geçerli bir morfem bulunamadı.")

        self._morphemes = morphemes
        logger.debug(f"[+] Kök ve ek ayrımı sonucu: {morphemes}")
        return morphemes

    def _split_morpheme(self, word: str) -> List[str]:
        if len(word) <= 3 or word.startswith("__tag__"):
            return [word]

        root = ""
        suffix_chain = []
        original_word = word

        for i in range(len(word), 0, -1):
            root_candidate = word[:i]
            suffix_candidate = word[i:]

            if suffix_candidate in self.suffixes:
                root = root_candidate
                suffix_chain.insert(0, suffix_candidate)
                word = root_candidate
            else:
                softened_suffix = self._apply_consonant_softening(suffix_candidate)
                if softened_suffix in self.suffixes:
                    root = root_candidate
                    suffix_chain.insert(0, softened_suffix)
                    word = root_candidate

        if not root:
            root = original_word

        if len(root) > 1 and not self._check_vowel_harmony(root):
            logger.warning(f"[!] Ünlü uyumu ihlali tespit edildi: {root}")

        return [root] + suffix_chain

    def _apply_consonant_softening(self, suffix: str) -> str:
        if len(suffix) > 0 and suffix[0] in self.consonant_softening:
            softened = self.consonant_softening[suffix[0]] + suffix[1:]
            logger.debug(f"[+] Sert ünsüz dönüşümü: {suffix} → {softened}")
            return softened
        return suffix

    def _check_vowel_harmony(self, root: str) -> bool:
        front = any(ch in self.vowel_harmony_front for ch in root)
        back = any(ch in self.vowel_harmony_back for ch in root)
        return not (front and back)

    def split_into_syllables(self, text: str) -> List[str]:
        if text.startswith("__tag__"):
            return [text]

        vowels = self.vowels
        syllables = []
        current = ""

        for char in text:
            current += char
            if char in vowels:
                syllables.append(current)
                current = ""

        if current:
            syllables.append(current)

        return syllables

    def analyze_root_and_morphology(self, text: str) -> str:
        syllables = self.split_into_syllables(text)
        root_and_suffixes = self.analyze(syllables)
        root = root_and_suffixes[0]
        suffixes = root_and_suffixes[1:]
        return root + "".join(suffixes)

    def reset(self):
        logger.warning("[!] Morphology sıfırlanıyor...")
        self._original_syllables = None
        self._morphemes = None
        logger.info("[+] Morphology sıfırlandı.")
