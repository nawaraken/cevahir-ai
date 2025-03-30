import pytest
from tokenizer_management.bpe.tokenization.postprocessor import Postprocessor, PostProcessingError

# Basit bir token listesinin işlenmesi: özel token dönüşümü, noktalama düzeltmesi ve cümle başı büyük harf
def test_basic_postprocessing():
    pp = Postprocessor()
    tokens = ["hello", "world", "<EOS>"]
    # Beklenen: "<EOS>" -> ".", join: "hello world ." → noktalama düzeltmesi: "hello world." 
    # → cümle başı büyük: "Hello world."
    result = pp.process(tokens)
    assert result == "Hello world."

# Özel tokenlerin doğru dönüşümünü test et: <BOS> silinir, <EOS> nokta ile değiştirilir.
def test_special_tokens():
    pp = Postprocessor()
    tokens = ["<BOS>", "this", "is", "a", "test", "<EOS>"]
    result = pp.process(tokens)
    # Join: " this is a test ." → After filtering, becomes "this is a test." 
    # → cümle başı büyük: "This is a test."
    assert result == "This is a test."

# Noktalama düzeltme: Tokenler join edildikten sonra fazladan boşluk bırakılan noktalama işaretleri düzeltilmeli.
def test_punctuation_fix():
    pp = Postprocessor()
    tokens = ["hello", ",", "world", "!"]
    result = pp.process(tokens)
    # Join: "hello , world !" → Punctuation fix: "hello, world!" → cümle başı büyük: "Hello, world!"
    assert result == "Hello, world!"

# Token listesi boş ise hata fırlatmalı.
def test_empty_tokens():
    pp = Postprocessor()
    with pytest.raises(PostProcessingError):
        pp.process([])

# Sadece boş özel token'lar (örneğin, <PAD>) kullanılırsa; özel token dönüşümü sonrasında metin boş kalır,
# bu durumda hata fırlatılmalı.
def test_only_special_token_result():
    pp = Postprocessor()
    tokens = ["<PAD>"]
    with pytest.raises(PostProcessingError):
        pp.process(tokens)

# Karmaşık token listesi: özel tokenler, noktalama ve cümle başı büyük harf dönüşümü birlikte test ediliyor.
def test_complex_postprocessing():
    pp = Postprocessor()
    tokens = ["<BOS>", "hello", ",", "world", "this", "is", "an", "example", "<EOS>"]
    result = pp.process(tokens)
    # İşlem sırası:
    # 1. <BOS> -> "" ve <EOS> -> "."
    # 2. Boş tokenler çıkarılır, join ile "hello , world this is an example ." elde edilir.
    # 3. Punctuation fix: "hello, world this is an example." (yanlış boşluklar kaldırılır)
    # 4. _capitalize ile: "Hello, world this is an example."
    assert result == "Hello, world this is an example."

# Reset metodunun çağrılması (reset sadece loglama yapıyor, bu yüzden hata fırlatmaz)
def test_reset_postprocessor():
    pp = Postprocessor()
    pp.reset()
    assert True
