import json
import os
from typing import Dict, List, Optional
import logging

from tokenizer_management.config import VOCAB_PATH, TOKEN_MAPPING

# === Logger Yapılandırması ===
logger = logging.getLogger(__name__)

# === Özel Hata Sınıfları ===
class VocabLoadError(Exception):
    pass

class VocabFormatError(Exception):
    pass

class TokenMappingError(Exception):
    pass


# === JSON Dosya Yükleyici ===
def load_json_file(file_path: str, encodings: List[str] = ["utf-8", "utf-8-sig", "latin-1"]) -> Dict:
    if not os.path.exists(file_path):
        raise VocabLoadError(f"Dosya bulunamadı: {file_path}")

    last_exception = None
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise VocabFormatError(f"JSON formatı hatalı (dict bekleniyor): {file_path}")
                return data

        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            last_exception = e
            logger.warning(f"{encoding} kodlamasıyla hata: {e}")

    raise VocabFormatError(f"Tüm denenen kodlamalar başarısız: {last_exception}")


# === JSON Kaydedici ===
def save_json_file(file_path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Dosya başarıyla kaydedildi: {file_path}")
    except IOError as e:
        raise IOError(f"Dosya kaydedilemedi: {e}")


# === Token ID - Token Eşleşmesi ===
def id_to_token(token_id: int, vocab: Dict[str, int]) -> Optional[str]:
    token = {v: k for k, v in vocab.items()}.get(token_id)
    if token is None:
        logger.warning(f"ID bulunamadı: {token_id}")
    return token


def token_to_id(token: str, vocab: Dict[str, int]) -> Optional[int]:
    token_id = vocab.get(token)
    if token_id is None:
        logger.warning(f"Token bulunamadı: {token}")
    return token_id


# === Token Mapping Doğrulayıcı ===
def validate_token_mapping(vocab: Dict[str, int]) -> bool:
    valid = True
    for token, id_ in TOKEN_MAPPING.items():
        if vocab.get(token) != id_:
            logger.warning(f"Özel token eşleşmesi hatalı: {token} -> {id_}")
            valid = False
    return valid


# === Token Frekans Hesaplayıcı ===
def calculate_frequency(tokens: List[str]) -> Dict[str, int]:
    frequency = {}
    for token in tokens:
        frequency[token] = frequency.get(token, 0) + 1
    logger.info(f"Token frekansları hesaplandı. Toplam benzersiz token: {len(frequency)}")
    return frequency


# === ID'lerden Token Oluşturucu ===
def ids_to_tokens(ids: List[int], vocab: Dict[str, int]) -> List[Optional[str]]:
    id_map = {v: k for k, v in vocab.items()}
    tokens = []
    for id_ in ids:
        token = id_map.get(id_)
        if token is None:
            logger.warning(f"Geçersiz ID: {id_}")
        tokens.append(token)
    return tokens


# === Tokenlerden ID Oluşturucu ===
def tokens_to_ids(tokens: List[str], vocab: Dict[str, int]) -> List[Optional[int]]:
    ids = []
    for token in tokens:
        token_id = vocab.get(token)
        if token_id is None:
            logger.warning(f"Geçersiz token: {token}")
        ids.append(token_id)
    return ids
