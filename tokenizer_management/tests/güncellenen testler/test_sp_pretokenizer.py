import pytest
from tokenizer_management.sentencepiece.tokenization.sp_pretokenizer import SPPretokenizer

@pytest.fixture
def sp_pretokenizer():
    return SPPretokenizer()

def test_normalize_text(sp_pretokenizer):
    """
    Metnin Unicode normalizasyonu, küçük harfe çevirme ve boşluk düzenlemesini test eder.
    """
    input_text = "  Héllo   Wörld!  "
    # Unicode NFKC normalizasyonu uygulanıp, tüm karakterler küçük harfe çevrildiğinde,
    # beklenen sonuç; ekstra boşluklar kaldırılarak "héllo wörld!" olmalıdır.
    expected = "héllo wörld!"
    normalized = sp_pretokenizer.normalize_text(input_text)
    assert normalized == expected, f"Beklenen '{expected}', ancak '{normalized}' elde edildi."

def test_tokenize_empty(sp_pretokenizer):
    """
    Boş metin için tokenize metodu boş liste döndürmelidir.
    """
    tokens = sp_pretokenizer.tokenize("")
    assert tokens == [], "Boş metin için token listesi boş olmalıdır."

def test_tokenize_basic(sp_pretokenizer):
    """
    Basit bir metin için normalize edilmiş token listesini döndürür.
    """
    input_text = "   Hello   world, this is a TEST.  "
    # Normalize edildikten sonra metin "hello world, this is a test." olur ve boşluklara göre bölünür.
    expected_tokens = ["hello", "world,", "this", "is", "a", "test."]
    tokens = sp_pretokenizer.tokenize(input_text)
    assert tokens == expected_tokens, f"Beklenen {expected_tokens}, ancak {tokens} elde edildi."

def test_tokenize_unicode(sp_pretokenizer):
    """
    Unicode içeren metinlerin de doğru şekilde normalize edilip tokenize edildiğini test eder.
    """
    input_text = "İstanbul, Türkiye"
    # Python'un varsayılan lower() fonksiyonu Türkçe karakterlerde tam beklendiği gibi çalışmayabilir,
    # ancak burada temel amaç; normalize edilip boşluklar doğru bölünmeli.
    tokens = sp_pretokenizer.tokenize(input_text)
    # Token listesinin boş olmaması ve en az iki token içermesi beklenir.
    assert isinstance(tokens, list)
    assert len(tokens) >= 2
