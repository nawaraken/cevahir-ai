import os
import torch
from typing import Dict, List

# === Cihaz Yapılandırması ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Dosya Yolları ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Vocab dosyaları
VOCAB_DIR = os.path.join(BASE_DIR, "data", "vocab_lib")
VOCAB_PATH = os.path.join(VOCAB_DIR, "vocab.json")

# === Tokenizer Yapılandırması ===
TOKENIZER_CONFIG: Dict = {
    "max_seq_length": 128,
    "vocab_size": 75000,
    "padding_token": "<PAD>",
    "unknown_token": "<UNK>",
    "start_token": "<BOS>",
    "end_token": "<EOS>",
}

# === BPE Yapılandırması ===
BPE_CONFIG: Dict = {
    "merge_operations": 50000,
    "min_frequency": 2,
    "cache_dir": os.path.join(BASE_DIR, "cache", "bpe"),
}

# === SentencePiece Yapılandırması ===
SENTENCEPIECE_CONFIG: Dict = {
    "vocab_size": TOKENIZER_CONFIG["vocab_size"],
    "model_type": "bpe",  # 'bpe', 'unigram', 'char', 'word'
    "character_coverage": 0.9995,
    "cache_dir": os.path.join(BASE_DIR, "cache", "sentencepiece"),
}

# === Chat Yapılandırması ===
CHAT_CONFIG: Dict = {
    "context_window": 512,
    "max_length": 50,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.7,
}

# === Log Yapılandırması ===
LOGGING_CONFIG: Dict = {
    "log_level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "log_file": os.path.join(BASE_DIR, "logs", "app.log"),
}

# === Özel Token ID Eşleşmesi ===
TOKEN_MAPPING = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3
}


# === Yapılandırma Fonksiyonları ===
def get_vocab_config() -> Dict:
    return TOKENIZER_CONFIG

def get_bpe_config() -> Dict:
    return BPE_CONFIG

def get_sentencepiece_config() -> Dict:
    return SENTENCEPIECE_CONFIG

def get_chat_config() -> Dict:
    return CHAT_CONFIG

def get_token_mapping() -> Dict:
    return TOKEN_MAPPING


# ==============================
# 🚀 TÜRKÇE ÖZGÜ YAPILANDIRMALAR
# ==============================

# === Türkçe Karakterler ===
TURKISH_CHARACTERS = [
    "a", "b", "c", "ç", "d", "e", "f", "g", "ğ", "h",
    "ı", "i", "j", "k", "l", "m", "n", "o", "ö", "p",
    "r", "s", "ş", "t", "u", "ü", "v", "y", "z"
]

# === Sesli ve Sessiz Harfler ===
TURKISH_VOWELS = ["a", "e", "ı", "i", "o", "ö", "u", "ü"]
TURKISH_CONSONANTS = [
    "b", "c", "ç", "d", "f", "g", "ğ", "h", "j", "k",
    "l", "m", "n", "p", "r", "s", "ş", "t", "v", "y", "z"
]

# === Ünlü Uyumu Kuralları ===
VOWEL_HARMONY_RULES = {
    "front_vowels": ["e", "i", "ö", "ü"],  # İnce sesliler
    "back_vowels": ["a", "ı", "o", "u"],   # Kalın sesliler
    "rounded_vowels": ["o", "ö", "u", "ü"],
    "unrounded_vowels": ["a", "e", "ı", "i"]
}

# === Ünsüz Benzeşmesi Kuralları ===
CONSONANT_HARMONY_RULES = {
    "voiceless_consonants": ["p", "ç", "t", "k", "f", "h", "s", "ş"],
    "voiced_consonants": ["b", "c", "d", "g", "v", "z", "j", "l", "m", "n", "r"],
    "continuant_consonants": ["f", "v", "s", "ş", "z", "j", "h"],
    "non_continuant_consonants": ["b", "c", "d", "g", "k", "p", "t"]
}

# === Türkçe Stopwords (Genişletilmiş) ===
TURKISH_STOPWORDS = [
    "ve", "de", "mi", "ile", "gibi", "ama", "fakat", "ancak", "da", "ki",
    "çünkü", "ise", "için", "ama", "o", "bu", "şu", "şey", "neden", "nasıl",
    "bir", "bazı", "birçok", "sanki", "falan", "diğer", "herhangi", "çok", 
    "az", "hemen", "asla", "kadar", "sonra", "önce", "zaten", "yani", 
    "onun", "bunun", "şunun", "şimdi", "her", "hiç", "sadece", "artık"
]

# === Yapım ve Çekim Ekleri ===
TURKISH_SUFFIXES = {
    "noun_suffixes": [
        "lar", "ler", "lık", "lik", "luk", "lük", "cı", "ci", "cu", "cü",
        "sız", "siz", "suz", "süz", "den", "dan", "e", "a", "de", "da", 
        "la", "le", "ten", "tan", "ye", "ya", "ki", "im", "in", "iz", "iniz"
    ],
    "verb_suffixes": [
        "iyor", "mak", "mek", "mış", "miş", "du", "dü", "di", "dı",
        "sa", "se", "m", "n", "yız", "yızlar", "siniz", "sinizler",
        "ken", "ince", "erek", "arak", "ip", "meden", "meden önce"
    ],
    "tense_suffixes": [
        "miş", "mişti", "di", "dı", "du", "dü", "se", "sa", "acak", "ecek",
        "yordu", "mıştı", "iyordu", "miş olmalı", "ebilirdi", "abilir", "amaz"
    ],
    "possessive_suffixes": [
        "ım", "im", "um", "üm", "ınız", "iniz", "ları", "leri"
    ]
}

# === Heceleme Kuralları ===
SYLLABIFICATION_RULES = {
    "max_syllable_length": 3,
    "split_on_vowel": True,
    "split_on_consonant_cluster": True,
    "handle_diphtongs": True,
    "handle_double_consonants": True
}

# === Özel Karakterler ===
SPECIAL_CHARACTERS = {
    "ı": "i", "İ": "i", "ç": "c", "Ç": "C",
    "ş": "s", "Ş": "S", "ö": "o", "Ö": "O",
    "ü": "u", "Ü": "U", "ğ": "g", "Ğ": "G",
    # Fars Dili Desteği
    "ی": "i", "ک": "k", "آ": "a", "و": "v",
    # Arap Dili Desteği
    "ء": "'", "أ": "a", "ب": "b", "ت": "t", "ث": "th",
    "ج": "j", "ح": "h", "خ": "kh", "د": "d", "ذ": "dh",
    "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "s",
    "ض": "d", "ط": "t", "ظ": "z", "ع": "a", "غ": "gh",
    "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m",
    "ن": "n", "ه": "h", "و": "w", "ي": "y"
}

# === Türkçe Dil İşleme Kuralları ===
TURKISH_TEXT_PROCESSING = {
    "lowercase": True,
    "remove_stopwords": True,
    "remove_punctuation": True,
    "normalize_unicode": "NFKD",  # 'NFC', 'NFD', 'NFKD' seçenekleri var
    "stemming": True,
    "lemmatization": True,
    "apply_vowel_harmony": True,
    "apply_consonant_harmony": True,
    "apply_suffix_rules": True
}

# === Bağımlılık Yapılandırması ===
DEPENDENCY_CONFIG: Dict = {
    "turkish_characters": TURKISH_CHARACTERS,
    "turkish_vowels": TURKISH_VOWELS,
    "turkish_consonants": TURKISH_CONSONANTS,
    "turkish_stopwords": TURKISH_STOPWORDS,
    "syllabification_rules": SYLLABIFICATION_RULES,
    "special_characters": SPECIAL_CHARACTERS,
    "text_processing": TURKISH_TEXT_PROCESSING,
    "suffixes": TURKISH_SUFFIXES,
    "vowel_harmony_rules": VOWEL_HARMONY_RULES,
    "consonant_harmony_rules": CONSONANT_HARMONY_RULES,
}

# === Yapılandırma Fonksiyonları ===
def get_turkish_config() -> Dict:
    return {
        "characters": TURKISH_CHARACTERS,
        "vowels": TURKISH_VOWELS,
        "consonants": TURKISH_CONSONANTS,
        "stopwords": TURKISH_STOPWORDS,
        "syllabification_rules": SYLLABIFICATION_RULES,
        "special_characters": SPECIAL_CHARACTERS,
        "text_processing": TURKISH_TEXT_PROCESSING,
        "suffixes": TURKISH_SUFFIXES,
        "vowel_harmony_rules": VOWEL_HARMONY_RULES,
        "consonant_harmony_rules": CONSONANT_HARMONY_RULES
    }
