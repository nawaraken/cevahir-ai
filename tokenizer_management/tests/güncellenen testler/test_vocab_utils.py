import pytest
import os
import json
from tokenizer_management.vocab.vocab_utils import (
    load_json_file,
    save_json_file,
    id_to_token,
    token_to_id,
    validate_token_mapping,
    calculate_frequency,
    ids_to_tokens,
    tokens_to_ids,
    VocabLoadError,
    VocabFormatError,
)
from tokenizer_management.config import VOCAB_PATH, TOKEN_MAPPING

@pytest.fixture
def sample_vocab():
    return {
        "<PAD>": 0,
        "<UNK>": 1,
        "<EOS>": 2,
        "<BOS>": 3,
        "merhaba": 4,
        "dünya": 5
    }

@pytest.fixture
def setup_json_file(tmpdir):
    file = tmpdir.join("test_vocab.json")
    data = {"merhaba": 1, "dünya": 2}
    file.write(json.dumps(data, ensure_ascii=False))
    return str(file)

@pytest.fixture
def invalid_json_file(tmpdir):
    file = tmpdir.join("invalid.json")
    file.write("invalid json content")
    return str(file)

@pytest.fixture
def non_dict_json_file(tmpdir):
    file = tmpdir.join("non_dict.json")
    file.write(json.dumps(["merhaba", "dünya"]))
    return str(file)


# === TEST: load_json_file ===

def test_load_json_file_success(setup_json_file):
    data = load_json_file(setup_json_file)
    assert data["merhaba"] == 1
    assert data["dünya"] == 2

def test_load_json_file_not_found():
    with pytest.raises(VocabLoadError, match="Dosya bulunamadı"):
        load_json_file("non_existing_file.json")

def test_load_json_file_invalid_json(invalid_json_file):
    with pytest.raises(VocabFormatError, match="Tüm denenen kodlamalar başarısız"):
        load_json_file(invalid_json_file)


def test_load_json_file_invalid_format(non_dict_json_file):
    with pytest.raises(VocabFormatError, match="JSON formatı hatalı"):
        load_json_file(non_dict_json_file)


# === TEST: save_json_file ===

def test_save_json_file_success(tmpdir):
    file_path = os.path.join(tmpdir, "saved_vocab.json")
    data = {"test": 123}
    save_json_file(file_path, data)
    assert os.path.exists(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
        assert loaded_data == data


# === TEST: id_to_token ===

def test_id_to_token_found(sample_vocab):
    assert id_to_token(4, sample_vocab) == "merhaba"

def test_id_to_token_not_found(sample_vocab):
    assert id_to_token(10, sample_vocab) is None


# === TEST: token_to_id ===

def test_token_to_id_found(sample_vocab):
    assert token_to_id("dünya", sample_vocab) == 5

def test_token_to_id_not_found(sample_vocab):
    assert token_to_id("yok", sample_vocab) is None


# === TEST: validate_token_mapping ===

def test_validate_token_mapping_success(sample_vocab):
    # TOKEN_MAPPING vocab ile aynı yapıya sahip olmalı
    original_mapping = TOKEN_MAPPING.copy()
    TOKEN_MAPPING.update({"<PAD>": 0, "<UNK>": 1, "<EOS>": 2, "<BOS>": 3})
    
    assert validate_token_mapping(sample_vocab) is True

    TOKEN_MAPPING.clear()
    TOKEN_MAPPING.update(original_mapping)

def test_validate_token_mapping_failure(sample_vocab, caplog):
    original_mapping = TOKEN_MAPPING.copy()
    TOKEN_MAPPING.update({"<PAD>": 100})  # Yanlış değer vererek hata oluştur
    
    assert validate_token_mapping(sample_vocab) is False
    assert "Özel token eşleşmesi hatalı" in caplog.text

    TOKEN_MAPPING.clear()
    TOKEN_MAPPING.update(original_mapping)


# === TEST: calculate_frequency ===

def test_calculate_frequency():
    tokens = ["a", "b", "a", "c", "b", "a"]
    frequency = calculate_frequency(tokens)
    assert frequency == {"a": 3, "b": 2, "c": 1}

def test_calculate_frequency_empty():
    assert calculate_frequency([]) == {}


# === TEST: ids_to_tokens ===

def test_ids_to_tokens(sample_vocab):
    ids = [4, 5, 0, 3]
    tokens = ids_to_tokens(ids, sample_vocab)
    assert tokens == ["merhaba", "dünya", "<PAD>", "<BOS>"]

def test_ids_to_tokens_with_invalid(sample_vocab):
    ids = [4, 99, 5]
    tokens = ids_to_tokens(ids, sample_vocab)
    assert tokens == ["merhaba", None, "dünya"]


# === TEST: tokens_to_ids ===

def test_tokens_to_ids(sample_vocab):
    tokens = ["merhaba", "dünya", "<UNK>"]
    ids = tokens_to_ids(tokens, sample_vocab)
    assert ids == [4, 5, 1]

def test_tokens_to_ids_with_invalid(sample_vocab):
    tokens = ["merhaba", "olmayan_token"]
    ids = tokens_to_ids(tokens, sample_vocab)
    assert ids == [4, None]
