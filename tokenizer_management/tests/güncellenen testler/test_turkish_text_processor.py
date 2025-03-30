import pytest
from tokenizer_management.utils.turkish_text_processor import TurkishTextProcessor

# === Gelişmiş Mock Syllabifier ===
class MockSyllabifier:
    def split(self, tokens):
        result = []
        for token in tokens:
            current = ""
            for i, char in enumerate(token):
                current += char
                if i < len(token) - 1:
                    next_char = token[i + 1]
                    if (char in "aeıioöuü" and next_char not in "aeıioöuü") or \
                       (char not in "aeıioöuü" and next_char in "aeıioöuü"):
                        result.append(current)
                        current = ""
            if current:
                result.append(current)
        return result

@pytest.fixture
def processor():
    syllabifier = MockSyllabifier()
    return TurkishTextProcessor(syllabifier)

# === 1. Unicode Normalizasyonu Testi (Gelişmiş) ===
@pytest.mark.parametrize("text, expected", [
    ("İstanbul", "İstanbul"),
    ("ĞÜçlü", "ĞÜçlü"),
    ("Çalışıyor", "Çalışıyor"),
    ("Şeker", "Şeker"),
    ("İzmir’de", "İzmir’de")
])
def test_normalize_unicode(processor, text, expected):
    result = processor.normalize_unicode(text)
    assert result == expected

# === 2. Küçük Harfe Çevirme Testi ===
@pytest.mark.parametrize("text, expected", [
    ("İstanbul", "istanbul"),
    ("ĞÜçlü", "ğüçlü"),
    ("Şeker", "şeker")
])
def test_to_lowercase(processor, text, expected):
    result = processor.to_lowercase(text)
    assert result == expected

# === 3. Noktalama İşaretlerini Kaldırma Testi ===
@pytest.mark.parametrize("text, expected", [
    ("Merhaba, dünya!", "Merhaba dünya"),
    ("Bu bir test.", "Bu bir test"),
    ("Yaprak, düşüyor.", "Yaprak düşüyor")
])
def test_remove_punctuation(processor, text, expected):
    result = processor.remove_punctuation(text)
    assert result == expected

# === 4. Stopword Kaldırma Testi (Gelişmiş) ===
@pytest.mark.parametrize("tokens, expected", [
    (["bu", "kitap", "çok", "iyi"], ["kitap", "çok", "iyi"]),
    (["ve", "ama", "ile", "çalışıyor"], ["çalışıyor"])
])
def test_remove_stopwords(processor, tokens, expected):
    result = processor.remove_stopwords(tokens)
    assert result == expected

# === 5. Heceleme Testi (Gelişmiş) ===
@pytest.mark.parametrize("tokens, expected", [
    (["merhaba", "dünya", "kaide", "serzeniş"], 
     ["mer", "ha", "ba", "dün", "ya", "kai", "de", "ser", "ze", "niş"]),
    (["kitaplar", "ağaç", "şehir"], 
     ["ki", "tap", "lar", "a", "ğaç", "şe", "hir"])
])
def test_syllabify(processor, tokens, expected):
    result = processor.syllabify(tokens)
    assert result == expected

# === 6. Kök Bulma (Stemming) Testi ===
@pytest.mark.parametrize("word, expected", [
    ("kitaplar", "kitap"),
    ("evler", "ev"),
    ("koşuyor", "koş")
])
def test_stem(processor, word, expected):
    result = processor.stem(word)
    assert result == expected

# === 7. Lemmatization Testi ===
@pytest.mark.parametrize("word, expected", [
    ("kitaplar", "kitap"),
    ("çalışıyor", "çalış"),
    ("ağaçlar", "ağaç")
])
def test_lemmatize(processor, word, expected):
    result = processor.lemmatize(word)
    assert result == expected

# === 8. Özel Karakter Dönüşümü Testi ===
@pytest.mark.parametrize("text, expected", [
    ("Çalışıyor, İstanbul’da!", "Calisiyor, Istanbul’da!")
])
def test_process_special_characters(processor, text, expected):
    result = processor.process_special_characters(text)
    assert result == expected

# === 9. Özel Token Ekleme Testi ===
def test_add_special_tokens(processor):
    tokens = ["kitap", "okumak"]
    result = processor.add_special_tokens(tokens)
    assert result == ["<BOS>", "kitap", "okumak", "<EOS>"]

# === 10. Boş Girdi Testi ===
def test_empty_input(processor):
    text = ""
    result = processor.full_process(text)
    assert result == ["<BOS>", "<EOS>"]

# === 11. Büyük Metin İşleme Testi ===
def test_large_input(processor):
    text = " ".join(["kitap"] * 1000)
    result = processor.full_process(text)
    assert len(result) == 1002

# === 12. Batch İşleme Testi ===
def test_batch_process(processor):
    texts = ["Merhaba dünya", "Kitap okumak güzel"]
    result = processor.batch_process(texts)
    assert result == [
        ["<BOS>", "mer", "ha", "ba", "dün", "ya", "<EOS>"],
        ["<BOS>", "ki", "tap", "oku", "mak", "gu", "zel", "<EOS>"]
    ]

# === 13. Uzun Kelime Testi ===
def test_long_words(processor):
    text = "anlamlandıramadıklarımızdanmışsınızcasına"
    result = processor.full_process(text)
    assert result == [
        "<BOS>", "an", "lam", "lan", "dı", "ra", "ma", "dık", "la", "rı", "mız", 
        "dan", "mış", "sı", "nız", "ca", "sı", "na", "<EOS>"
    ]

