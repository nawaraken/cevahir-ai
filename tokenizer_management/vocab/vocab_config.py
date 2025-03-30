from typing import Dict
from tokenizer_management.config import TOKENIZER_CONFIG, TOKEN_MAPPING

class VocabConfig:
    """
    Özel token yapılandırmasını tutan yapı.
    """

    @classmethod
    def get_special_tokens(cls) -> Dict[str, str]:
        """
        Özel token tanımlarını döner.

        Returns:
            Dict[str, str]: Özel token eşleşmelerini döner.
        """
        return {
            "padding_token": TOKENIZER_CONFIG["padding_token"],
            "unknown_token": TOKENIZER_CONFIG["unknown_token"],
            "start_token": TOKENIZER_CONFIG["start_token"],
            "end_token": TOKENIZER_CONFIG["end_token"],
        }
    
    @classmethod
    def get_special_token_ids(cls) -> Dict[str, int]:
        """
        Özel token ID eşleşmelerini döner.

        Returns:
            Dict[str, int]: Özel token ID eşleşmelerini döner.
        """
        special_tokens = cls.get_special_tokens()
        token_ids = {}
        for token_name, token in special_tokens.items():
            if token in TOKEN_MAPPING:
                token_ids[token] = TOKEN_MAPPING[token]
            else:
                raise KeyError(f"Token mapping bulunamadı: {token_name} -> {token}")
        return token_ids
    
    @classmethod
    def get_vocab_settings(cls) -> Dict[str, int]:
        """
        Vocab yapılandırmasını döner.

        Returns:
            Dict[str, int]: Vocab yapılandırma detaylarını döner.
        """
        return {
            "vocab_size": TOKENIZER_CONFIG["vocab_size"],
            "max_seq_length": TOKENIZER_CONFIG["max_seq_length"],
        }

    @classmethod
    def validate_special_tokens(cls) -> bool:
        """
        Özel tokenların token mapping ile uyumunu doğrular.

        Returns:
            bool: Tüm eşleşmeler doğruysa True, değilse False.
        """
        try:
            special_tokens = cls.get_special_tokens()
            token_ids = cls.get_special_token_ids()
            for token in special_tokens.values():
                if token not in TOKEN_MAPPING:
                    raise KeyError(f"Eşleşmeyen token: {token}")
            return True
        except KeyError as e:
            print(f"[!] Özel token hatası: {e}")
            return False


# === Test Amaçlı Çalıştırma ===
if __name__ == "__main__":
    print("\n🔎 Özel Tokenlar:")
    print(VocabConfig.get_special_tokens())

    print("\n🔎 Özel Token ID'leri:")
    print(VocabConfig.get_special_token_ids())

    print("\n🔎 Vocab Yapılandırması:")
    print(VocabConfig.get_vocab_settings())

    print("\n🔎 Eşleşme Doğrulaması:")
    if VocabConfig.validate_special_tokens():
        print("[✔️] Tüm token eşleşmeleri doğru!")
    else:
        print("[❌] Token eşleşmelerinde hata var!")
