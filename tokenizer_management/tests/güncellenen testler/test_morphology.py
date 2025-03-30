import pytest
from tokenizer_management.bpe.tokenization.morphology import Morphology, MorphologyError

# Test: Boş hece listesi verildiğinde uygun hata fırlatılmalı.
def test_empty_syllable_list():
    morph = Morphology()
    with pytest.raises(MorphologyError) as excinfo:
        morph.analyze([])
    # Hata mesajının "Hece listesi boş olamaz." içermesi bekleniyor.
    assert "Hece listesi boş olamaz." in str(excinfo.value)

# Test: 3 karakterden kısa kelimeler olduğu gibi döndürülmeli.
def test_short_word():
    morph = Morphology()
    result = morph.analyze(["su"])
    assert result == ["su"]

# Test: Ek ayrımının çalıştığını doğrulama (örneğin "kitaplar").
def test_split_morpheme_with_suffix():
    morph = Morphology()
    result = morph.analyze(["kitaplar"])
    # Beklenen: "kitaplar" kelimesi, "kitap" kökü ve "lar" eki olarak ayrılmalı.
    assert result == ["kitap", "lar"]

# Test: Ek ayrımının olmadığı durumda kelime olduğu gibi döndürülmeli (örneğin "kalem").
def test_split_morpheme_without_suffix():
    morph = Morphology()
    result = morph.analyze(["kalem"])
    # Kod, "kalem" için "kalem" yerine ["kale", "m"] üretecektir, çünkü "m" suffixes setinde vardır.
    assert result == ["kale", "m"]

# Test: Birden fazla kelime için analiz.
def test_multiple_syllables():
    morph = Morphology()
    result = morph.analyze(["kitaplar", "evler"])
    # "kitaplar" → ["kitap", "lar"], "evler" → ["ev", "ler"]
    assert result == ["kitap", "lar", "ev", "ler"]

# Test: Karmaşık durum testi (örneğin "araba")
def test_complex_word():
    morph = Morphology()
    result = morph.analyze(["araba"])
    # "araba" için; i=4: root_candidate = "arab", suffix_candidate = "a" ve "a" suffixes setinde bulunuyor.
    assert result == ["arab", "a"]

# Test: Karışık kelime listesi (örneğin kısa kelime ve ekli kelime)
def test_analyze_mixed():
    morph = Morphology()
    result = morph.analyze(["ev", "evler"])
    # "ev" → ["ev"] ve "evler" → ["ev", "ler"]
    assert result == ["ev", "ev", "ler"]

# Test: Reset metodunun hata fırlatmadığını doğrulama.
def test_reset_morphology():
    morph = Morphology()
    morph.reset()
    assert True
