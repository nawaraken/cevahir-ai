import pytest
from tokenizer_management.bpe.tokenization.pretokenizer import Pretokenizer, PretokenizationError

# ✅ Pretokenizer örneğini oluşturmak için fixture
@pytest.fixture
def pretokenizer():
    return Pretokenizer()

# Başarılı senaryolar

def test_basic_tokenization(pretokenizer):
    text = "Merhaba Dünya!"
    # Beklenen: büyük harfler küçültüldükten, özel karakterler kaldırıldıktan sonra ["merhaba", "dünya"]
    result = pretokenizer.tokenize(text)
    assert result == ["merhaba", "dünya"]

def test_multiple_spaces(pretokenizer):
    text = "   Merhaba     Dünya   "
    result = pretokenizer.tokenize(text)
    assert result == ["merhaba", "dünya"]

def test_numeric_tokens(pretokenizer):
    text = "123 456"
    result = pretokenizer.tokenize(text)
    assert result == ["123", "456"]

def test_mixed_alphanumeric(pretokenizer):
    text = "Hello, world! 42"
    # Beklenen: "hello", "world", "42"
    result = pretokenizer.tokenize(text)
    assert result == ["hello", "world", "42"]

def test_hyphenated_input(pretokenizer):
    text = "merhaba-dünya"
    # '-' karakteri özel karakter olarak kabul edilip boşlukla değiştirilir.
    result = pretokenizer.tokenize(text)
    assert result == ["merhaba", "dünya"]

# Hata senaryoları

def test_empty_input(pretokenizer):
    with pytest.raises(ValueError) as excinfo:
        pretokenizer.tokenize("")
    assert "Girdi metni boş olamaz" in str(excinfo.value)

def test_only_special_characters(pretokenizer):
    # Hiç alfabetik karakter içermediği için, tokenizasyon sonrası token listesi boş olacağı
    # ve bu durumda PretokenizerError fırlatılacaktır.
    with pytest.raises(PretokenizationError) as excinfo:
        pretokenizer.tokenize("!@#$")
    assert "Tokenizasyon sonrası geçerli bir token bulunamadı" in str(excinfo.value)

def test_invalid_input_type(pretokenizer):
    with pytest.raises(ValueError) as excinfo:
        pretokenizer.tokenize(None)  # Bu durumda boş/None girildiğinden hata beklenir.
    # Mesajı doğrulayabilir veya TypeError beklediğimizi belirtebiliriz. (Kodunuzda boşluk kontrolü önce yapılıyor.)

# (Eğer None için özel bir hata verilmek isteniyorsa, kodunuzda None kontrolü eklenmelidir.)

# Diğer senaryolar

def test_reset(pretokenizer):
    # reset fonksiyonu hata fırlatmamalı.
    pretokenizer.reset()
    assert True
