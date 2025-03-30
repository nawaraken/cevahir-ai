import os
from typing import Dict, List, Optional
import logging
from collections import Counter

from .vocab_loader import VocabLoader
from .vocab_builder import VocabBuilder
from .vocab_updater import VocabUpdater
from .vocab_utils import (
    load_json_file,
    save_json_file,
    calculate_frequency,
    validate_token_mapping,
    token_to_id,
    id_to_token,
)

logger = logging.getLogger(__name__)


class VocabManager:
    _instance = None

    def __new__(cls, vocab_path: str, special_tokens: Optional[Dict[str, int]] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(vocab_path, special_tokens)
        return cls._instance

    def _initialize(self, vocab_path: str, special_tokens: Optional[Dict[str, int]] = None):
        """
        VocabManager yapılandırmasını başlatır.
        """
        self.vocab_path = vocab_path
        self.special_tokens = special_tokens or {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }

        # Yükleyici ve Yapılandırıcı Başlatılıyor
        self.vocab_loader = VocabLoader(self.vocab_path)
        self.vocab_builder = VocabBuilder(self.vocab_path, self.special_tokens)

        # Vocab dosyası mevcutsa yükle, değilse oluştur
        if os.path.exists(self.vocab_path):
            self.vocab = self.vocab_loader.load_vocab()
            logger.info(f"[+] Vocab dosyası yüklendi: {self.vocab_path}")
        else:
            logger.warning(f"[!] Vocab dosyası bulunamadı. Varsayılan vocab oluşturuluyor...")
            self.vocab = self.vocab_builder.build_vocab([])

        # Güncelleyici Başlat
        self.vocab_updater = VocabUpdater(self.vocab, self.vocab_path, self.special_tokens)

        # Özel tokenları doğrula
        validate_token_mapping(self.special_tokens)
        self.update_reverse_vocab()

    def update_reverse_vocab(self) -> None:
        self.reverse_vocab = {value["id"]: key for key, value in self.vocab.items()}
        special_ids = set(self.special_tokens.values())

        for token, value in self.vocab.items():
            if value["id"] in special_ids and token not in self.special_tokens:
                logger.warning(f"[!] Özel token ID çakışması: {token} -> {value['id']}")
                new_id = self.get_next_token_id()
                value["id"] = new_id



    def load_vocab(self) -> Dict[str, dict]:
        """
        Vocab dosyasını yükler.
        """
        try:
            self.vocab = load_json_file(self.vocab_path)
            self.update_reverse_vocab()
            logger.info(f"[+] Vocab dosyası yüklendi: {self.vocab_path}")
            return self.vocab
        except Exception as e:
            logger.error(f"[X] Vocab dosyası yüklenemedi: {e}")
            raise

    def save_vocab(self) -> None:
        """
        Vocab dosyasını kayıt eder.
        """
        if not self.vocab:
            logger.warning("[!] Kayıt için boş vocab tespit edildi!")
            return

        save_json_file(self.vocab_path, self.vocab)
        logger.info(f"[+] Vocab dosyası başarıyla kaydedildi: {self.vocab_path}")

    def update_vocab(self, new_tokens: List[str], positions: Optional[List[int]] = None, method: str = "bpe") -> None:
        """
        Yeni bir vocab yükleyip mevcut vocab'ı günceller.
        Token frekansı ve pozisyon bilgileri ile birlikte güncelleme yapılır.

        Args:
            new_tokens (List[str]): Güncellenecek yeni token listesi.
            positions (Optional[List[int]]): Tokenlerin pozisyon bilgileri. Her token için global pozisyon belirtilmelidir.
            method (str): Kullanılan tokenizasyon yöntemi (örn. "bpe").
        """
        if not new_tokens or not isinstance(new_tokens, list):
            raise ValueError("Yeni token listesi geçersiz veya boş.")

        # Tokenleri temizle
        cleaned_tokens = [token.strip() for token in new_tokens if token.strip()]
        if not cleaned_tokens:
            logger.warning("[!] Geçerli token bulunamadı. Temizleme sonrası liste boş.")
            return

        # Pozisyonlar sağlanmamışsa ya da uzunluk uyuşmuyorsa uyarı ver ve sıfırla
        if positions is None or len(positions) != len(cleaned_tokens):
            logger.warning("[!] Token ve pozisyon uzunlukları uyuşmuyor. Tüm pozisyonlar sıfırlandı.")
            positions = [0] * len(cleaned_tokens)

        # Pozisyon bilgilerini birden fazla tekrar için grupla
        token_positions_map = {}
        for token, pos in zip(cleaned_tokens, positions):
            if token not in token_positions_map:
                token_positions_map[token] = []
            token_positions_map[token].append(pos)

        # Kullanılan ID'ler
        used_ids = set(self.reverse_vocab.keys())
        special_ids = set(self.special_tokens.values())

        # Her benzersiz token için güncelleme yap
        for token, pos_list in token_positions_map.items():
            freq = len(pos_list)

            if token in self.vocab:
                # Mevcut token için frekansı artır, pozisyonları ekle
                self.vocab[token]["total_freq"] += freq
                for pos in pos_list:
                    if pos not in self.vocab[token]["positions"]:
                        self.vocab[token]["positions"].append(pos)

                logger.debug(f"[+] Token güncellendi: {token} → freq: {self.vocab[token]['total_freq']}, pozisyonlar: {self.vocab[token]['positions']}")
            else:
                # Yeni token için ID ata
                new_id = self.get_next_token_id()
                while new_id in used_ids or new_id in special_ids:
                    new_id += 1

                self.vocab[token] = {
                    "id": new_id,
                    "total_freq": freq,
                    "positions": pos_list
                }
                self.reverse_vocab[new_id] = token
                used_ids.add(new_id)

                logger.debug(f"[+] Yeni token eklendi: {token} → ID: {new_id}, freq: {freq}, pozisyonlar: {pos_list}")

        self.update_reverse_vocab()
        self.save_vocab()
        logger.info(f"[✓] Vocab güncelleme tamamlandı → Yöntem: {method}, Token sayısı: {len(token_positions_map)}")




    def update_reverse_vocab(self) -> None:
        """
        Reverse vocab günceller ve ID çakışmalarını önler.
        """
        self.reverse_vocab = {value["id"]: key for key, value in self.vocab.items()}
        special_ids = set(self.special_tokens.values())
        used_ids = set(self.reverse_vocab.keys())

        for token, value in self.vocab.items():
            token_id = value["id"]

            # ID çakışması kontrolü ve çözümü
            if token_id in special_ids and token not in self.special_tokens:
                logger.warning(f"[!] Özel token ID çakışması tespit edildi: {token} → ID: {token_id}")

                # Yeni ID belirle ve çakışmayı çöz
                new_id = self.get_next_token_id()
                while new_id in used_ids or new_id in special_ids:
                    logger.warning(f"[!] ID çakışması devam ediyor, yeni ID atanıyor: {new_id}")
                    new_id += 1
                
                # Yeni ID ataması yap
                value["id"] = new_id
                used_ids.add(new_id)
                self.reverse_vocab[new_id] = token

                logger.info(f"[+] ID çakışması çözüldü → '{token}' için yeni ID: {new_id}")

        logger.info(f"[✓] Reverse vocab başarıyla güncellendi. Toplam token sayısı: {len(self.reverse_vocab)}")




    def build_vocab(self, token_list: List[str]) -> Dict[str, dict]:
        """
        Vocab'i sıfırdan oluşturur.
        """
        vocab = self.vocab_builder.build_vocab(token_list)
        self.vocab.update(vocab)
        self.update_reverse_vocab()
        self.save_vocab()
        return self.vocab

    def reset_vocab(self) -> None:
        """
        Vocab'i sıfırlar ve varsayılan tokenları yükler.
        """
        try:
            if os.path.exists(self.vocab_path):
                os.remove(self.vocab_path)

            # Varsayılan özel tokenlar ile sıfırla
            self.vocab = {
                "<PAD>": {"id": 0, "positions": [], "total_freq": 0},
                "<UNK>": {"id": 1, "positions": [], "total_freq": 0},
                "<BOS>": {"id": 2, "positions": [], "total_freq": 0},
                "<EOS>": {"id": 3, "positions": [], "total_freq": 0},
            }

            self.update_reverse_vocab()
            self.save_vocab()

            logger.info("[+] Vocab sıfırlandı ve varsayılan tokenlar yüklendi.")
        except Exception as e:
            logger.error(f"[X] Vocab sıfırlama hatası: {e}")
            raise

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def get_token_info(self, token: str) -> Optional[dict]:
        return self.vocab.get(token)

    def get_token_id(self, token: str) -> Optional[int]:
        return token_to_id(token, self.vocab)

    def get_token_from_id(self, token_id: int) -> Optional[str]:
        return id_to_token(token_id, self.vocab)

    def bulk_update_vocab(self, token_batches: List[List[str]]) -> None:
        for batch in token_batches:
            self.update_vocab(batch)  # ID çakışması kontrolü güncellendi

        self.save_vocab()


    def calculate_frequency(self, tokens: List[str]) -> Dict[str, int]:
        """
        Token frekansını hesaplar.
        """
        return calculate_frequency(tokens)

    def get_next_token_id(self) -> int:
        # Hem reverse_vocab hem de vocab üzerinden kullanılmayan ID'yi al
        used_ids = set(self.reverse_vocab.keys()).union(
            value["id"] for value in self.vocab.values()
        )
        special_ids = set(self.special_tokens.values())

        # Yeni ID belirle
        next_id = max(used_ids) + 1
        while next_id in used_ids or next_id in special_ids:
            next_id += 1
        
        return next_id



    def set_token_id(self, token: str, token_id: int) -> None:
        if token in self.vocab:
            self.vocab[token]["id"] = token_id
            self.update_reverse_vocab()
        else:
            raise ValueError(f"Token '{token}' vocab içinde bulunamadı.")

    def set_vocab(self, new_vocab: Dict[str, dict]) -> None:
        if not isinstance(new_vocab, dict):
            raise TypeError("Yeni vocab geçerli bir sözlük formatında olmalıdır.")
            
        self.vocab = new_vocab
        self.update_reverse_vocab()
        self.save_vocab()

        logger.info(f"[+] Yeni vocab doğrudan yüklendi. Toplam token sayısı: {len(self.vocab)}")



# ÖRNEK ÇAĞRI:
# vocab_manager = VocabManager("data/vocab.json")
# vocab_manager.reset_vocab()
# vocab_manager.update_vocab(["test", "deneme", "token"])
