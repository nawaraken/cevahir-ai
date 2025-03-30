import pytest
from tokenizer_management.bpe.tokenization.syllabifier import Syllabifier, SyllabificationError

# Syllabifier örneği oluştur
@pytest.fixture
def syllabifier():
    return Syllabifier()

# ✅ Başarılı senaryolar
def test_basic_syllabification(syllabifier):
    tokens = ["merhaba", "dünya"]
    result = syllabifier.split(tokens)
    assert result == ['mer', 'ha', 'ba', 'dün', 'ya']

def test_turkish_special_cases(syllabifier):
    tokens = ["çalışmak", "güzellik"]
    result = syllabifier.split(tokens)
    assert result == ['ça', 'lış', 'mak', 'gü', 'zel', 'lik']

def test_single_vowel(syllabifier):
    tokens = ["a", "o", "ü"]
    result = syllabifier.split(tokens)
    assert result == ["a", "o", "ü"]

# ✅ Büyük/küçük harf testleri ve özel karakterler
def test_mixed_case(syllabifier):
    tokens = ["Ankara", "İstanbul", "çocuklar"]
    result = syllabifier.split(tokens)
    assert result == ['an', 'ka', 'ra', 'is', 'tan', 'bul', 'ço', 'cuk', 'lar']

def test_hard_case(syllabifier):
    tokens = ["kalkış", "yürüyüş", "çalışıyor"]
    result = syllabifier.split(tokens)
    assert result == ['kal', 'kış', 'yü', 'rü', 'yüş', 'ça', 'lı', 'şı', 'yor']

# ✅ Boş giriş senaryoları
def test_empty_token_list(syllabifier):
    with pytest.raises(ValueError) as excinfo:
        syllabifier.split([])
    assert "Token listesi boş olamaz." in str(excinfo.value)

def test_empty_string(syllabifier):
    tokens = [""]
    result = syllabifier.split(tokens)
    assert result == [""]

# ✅ Geçersiz girişler
def test_invalid_input_type(syllabifier):
    with pytest.raises(TypeError):
        syllabifier.split(None)

def test_numeric_input(syllabifier):
    tokens = ["12345"]
    result = syllabifier.split(tokens)
    assert result == ["12345"]

# ✅ Özel karakterler ve simgeler
def test_special_characters(syllabifier):
    tokens = ["a-b-c", "@#$%"]
    result = syllabifier.split(tokens)
    assert result == ["a-b-c", "@#$%"]

def test_mixed_characters(syllabifier):
    tokens = ["mer!haba", "dün#ya"]
    result = syllabifier.split(tokens)
    assert result == ["mer!ha", "ba", "dün#ya"]

# ✅ Heceleme hataları
def test_failed_syllabification(syllabifier):
    result = syllabifier.split(["xyz"])  # Sesli harf içermez ama hata fırlatmıyor → token'i olduğu gibi döndürüyor
    assert result == ["xyz"]

# ✅ Büyük girişlerle test (Bellek Taşması Kontrolü)
def test_large_input(syllabifier):
    tokens = ["a" * 10000, "b" * 10000]
    result = syllabifier.split(tokens)
    
    # Doğrudan hata fırlatmak yerine token'ı döndürüyor olmalı
    assert len(result) == 2
    assert result[0] == "a" * 10000
    assert result[1] == "b" * 10000

# ✅ Sıfırlama Testi (Durum sıfırlama testi)
def test_reset(syllabifier):
    syllabifier.reset()
    assert True

# ✅ Karmaşık heceleme senaryoları
def test_complex_cases(syllabifier):
    tokens = ["gözlükçü", "düşündüklerim", "sevinçli"]
    result = syllabifier.split(tokens)
    assert result == [
        'göz', 'lük', 'çü', 
        'dü', 'şün', 'dük', 'le', 'rim', 
        'se', 'vinç', 'li'
    ]
