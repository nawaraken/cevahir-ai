import pytest
from tokenizer_management.sentencepiece.tokenization.language_processor import LanguageProcessor

def test_process_normal():
    """
    Basit bir token listesi verildiğinde, 
    tüm tokenların küçük harfe çevrildiğini, noktalama işaretlerinin kaldırıldığını
    ve gereksiz boşlukların temizlendiğini test eder.
    """
    processor = LanguageProcessor()
    tokens = ["Hello,", "World!", "Test.", "Python?"]
    processed = processor.process(tokens)
    expected = ["hello", "world", "test", "python"]
    assert processed == expected

def test_process_empty():
    """
    Boş bir token listesi verildiğinde boş liste döndürüldüğünü test eder.
    """
    processor = LanguageProcessor()
    tokens = []
    processed = processor.process(tokens)
    assert processed == []

def test_process_tr_characters():
    """
    Türkçe karakterlerin doğru şekilde normalize edildiğini test eder.
    Büyük harflerin doğru şekilde küçüğe dönüştürüldüğünü ve noktalama işaretlerinin kaldırıldığını kontrol eder.
    """
    processor = LanguageProcessor()
    tokens = ["İSTANBUL,", "ÇOK!", "GÜZEL."]
    processed = processor.process(tokens)
    # "İSTANBUL".casefold() ü "i̇stanbul" (küçük i, noktalı) döndürür.
    expected = ["i̇stanbul", "çok", "güzel"]
    assert processed == expected

def test_process_strip_spaces():
    """
    Tokenlardaki kenardaki boşlukların temizlendiğini test eder.
    """
    processor = LanguageProcessor()
    tokens = ["  Merhaba  ", "  Dünya!!!  "]
    processed = processor.process(tokens)
    expected = ["merhaba", "dünya"]
    assert processed == expected
