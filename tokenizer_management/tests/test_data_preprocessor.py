import pytest
from tokenizer_management.data_loader.data_preprocessor import DataPreprocessor, BOS_TOKEN, EOS_TOKEN

def test_clean_text():
    raw = "  This    is a \n Test.  "
    cleaned = DataPreprocessor.clean_text(raw)
    # clean_text() metodu .lower() uyguladığı için:
    assert cleaned == "this is a test."

def test_add_special_tokens():
    text = "sample text"
    with_tokens = DataPreprocessor.add_special_tokens(text)
    expected = f"{BOS_TOKEN} sample text {EOS_TOKEN}"
    assert with_tokens == expected

def test_preprocess_text():
    raw = "  Sample   Text!  "
    processed = DataPreprocessor.preprocess_text(raw)
    # Önce temizleme (lowercase ve normalize) sonra special token ekleme uygulanır.
    expected = f"{BOS_TOKEN} sample text! {EOS_TOKEN}"
    assert processed == expected
