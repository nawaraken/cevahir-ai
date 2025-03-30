import logging
import os
import json
import re
from collections import Counter
from typing import List, Dict, Optional, Set, Tuple

from .bpe_encoder import BPEEncoder
from .bpe_decoder import BPEDecoder,BPEDecodingError
from .bpe_trainer import BPETrainer
from .tokenization.pretokenizer import Pretokenizer
from .tokenization.syllabifier import Syllabifier
from .tokenization.morphology import Morphology
from .tokenization.postprocessor import Postprocessor

from tokenizer_management.base_tokenizer_manager import BaseTokenizerManager

logger = logging.getLogger(__name__)


class BPETokenError(Exception):
    pass

class BPEDecodingError(Exception):
    """BPEDecoder ile ilgili hataları tanımlamak için özel exception."""
    
class BPEManager(BaseTokenizerManager):
    _instance = None

    def __new__(cls, vocab: Optional[Dict[str, dict]] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._vocab = vocab if vocab else cls._instance._default_vocab()
            cls._instance._initialize()
            logger.info("[+] Yeni BPEManager örneği oluşturuldu.")
        elif vocab is not None and cls._instance._vocab != vocab:
            logger.info("[+] BPEManager mevcut vocab ile güncelleniyor...")
            cls._instance.set_vocab(vocab)
        else:
            logger.info("[+] Mevcut BPEManager örneği kullanılıyor.")
        return cls._instance

    def _initialize(self):
        if not self._vocab:
            raise BPETokenError("[X] Vocab yüklenemedi veya boş.")
        logger.info("[+] BPEManager başlatılıyor...")
        self.encoder = BPEEncoder(self._vocab)
        self.decoder = BPEDecoder(self._vocab)
        self.trainer = BPETrainer(self._vocab)
        self.pretokenizer = Pretokenizer()
        self.syllabifier = Syllabifier()
        self.morphology = Morphology()
        self.postprocessor = Postprocessor()
        self.update_reverse_vocab()
        logger.info("[+] BPEManager başarıyla başlatıldı.")

    def encode(self, text: str, mode: str = "train") -> Tuple[List[str], List[int]]:
        if not self._vocab:
            raise BPETokenError("[X] Vocab boş. Kodlama yapılamaz.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("[X] Kodlama için geçerli bir metin girilmelidir.")

        # Metni tokenize et
        tokens = self.pretokenizer.tokenize(text)
        if not tokens:
            raise ValueError("[X] Tokenizasyon sonucu boş çıktı.")

        # Tüm tokenleri lowercase yap
        tokens = [token.lower() for token in tokens]

        # Özel token’ları (BOS, EOS) ekle
        tokens = ["<BOS>"] + tokens + ["<EOS>"]

        token_ids = []

        for idx, token in enumerate(tokens):
            is_special = token in {"<PAD>", "<UNK>", "<BOS>", "<EOS>"} or token.startswith("__tag__")
            entry = self._vocab.get(token)
            if entry is not None:
                entry["total_freq"] += 1
                entry["positions"].append(idx)
                token_id = entry["id"]
            else:
                if mode == "inference":
                    if not is_special:
                        root = self.morphology.find_root(token)
                        if root and root in self._vocab:
                            logger.info(f"[~] Bilinmeyen token: '{token}', kök bulundu: '{root}'")
                            token_id = self._vocab[root]["id"]
                        else:
                            logger.info(f"[~] Bilinmeyen token: '{token}', kök bulunamadı. '<UNK>' atanıyor.")
                            unk_entry = self._vocab.get("<UNK>")
                            if unk_entry is None:
                                raise BPETokenError("[X] '<UNK>' token'ı vocab içinde tanımlı değil.")
                            token_id = unk_entry["id"]
                    else:
                        special_entry = self._vocab.get(token)
                        if special_entry is None:
                            raise BPETokenError(f"[X] Özel token '{token}' vocab içinde tanımlı değil.")
                        token_id = special_entry["id"]
                else:
                    # Eğitim modunda token vocab'a ekleniyor.
                    next_id = max((v["id"] for v in self._vocab.values()), default=0) + 1
                    self._vocab[token] = {
                        "id": next_id,
                        "total_freq": 1,
                        "positions": [idx]
                    }
                    token_id = next_id
                    self.reverse_vocab[token_id] = token
                    logger.info(f"[+] Yeni token eklendi: '{token}' → ID: {token_id}")

            if token_id is not None:
                token_ids.append(token_id)

        if not self.reverse_vocab:
            logger.warning("[!] reverse_vocab eksik. Yeniden oluşturuluyor...")
            self.update_reverse_vocab()

        missing_ids = [tid for tid in token_ids if tid not in self.reverse_vocab]
        if missing_ids:
            raise ValueError(f"[X] Hatalı ID'ler tespit edildi: {missing_ids}")

        logger.info(f"[✓] Kodlama tamamlandı. Tokenler: {tokens} | ID'ler: {token_ids}")
        return tokens, token_ids

    def decode(self, token_ids: List[int]) -> str:
        if not self._vocab:
            raise BPEDecodingError("[X] Vocab boş. Çözümleme yapılamaz.")
        
        if not isinstance(token_ids, list) or not all(isinstance(tid, int) for tid in token_ids):
            raise TypeError("[X] Çözümleme için geçerli bir ID listesi sağlanmalıdır.")
        
        if not token_ids:
            raise ValueError("[X] Token ID listesi boş olamaz.")
        
        if not self.reverse_vocab:
            logger.warning("[!] reverse_vocab eksik veya boş. Yeniden oluşturuluyor...")
            self.reverse_vocab = self._build_reverse_vocab()
        
        tokens = [self.reverse_vocab.get(tid, "<UNK>") for tid in token_ids]
        
        filtered_tokens = [
            token for token in tokens
            if token not in {"<BOS>", "<EOS>", "<PAD>"} and not token.startswith("__tag__")
        ]
        
        try:
            decoded_text = self.postprocessor.process(filtered_tokens)
            decoded_text = decoded_text.strip()
            if not decoded_text.endswith("..."):
                decoded_text = decoded_text.rstrip(".")
            if not decoded_text:
                logger.warning("[!] Çözümleme sonucu boş çıktı. '<EMPTY>' döndürüldü.")
                decoded_text = "<EMPTY>"
        except Exception as e:
            logger.error(f"[X] Postprocessor hatası: {e}")
            raise BPEDecodingError(f"[X] Postprocessor hatası: {e}")
        
        logger.info(f"[+] Çözümleme başarıyla tamamlandı: {decoded_text}")
        return decoded_text

    def train(self, corpus: List[str], vocab_size: Optional[int] = None) -> None:
        if not corpus:
            raise ValueError("[X] Eğitim verisi boş olamaz.")
        self.trainer.train(corpus, vocab_size)
        self._vocab = self.trainer.get_vocab()
        self.update_reverse_vocab()
        logger.info("[+] BPE eğitimi tamamlandı ve vocab güncellendi.")

    def get_vocab(self) -> Dict[str, dict]:
        return self._vocab

    def set_vocab(self, new_vocab: Dict[str, dict]) -> None:
        if not new_vocab:
            raise ValueError("[X] Yeni vocab boş olamaz.")
        self._vocab = new_vocab
        self.encoder.set_vocab(new_vocab)
        self.decoder.set_vocab(new_vocab)
        # BPETrainer’da set_vocab metodu olmadığı için bu satır kaldırıldı.
        # self.trainer.set_vocab(new_vocab)
        self.update_reverse_vocab()
        logger.info("[+] Dahili vocab başarıyla güncellendi.")

    def update_reverse_vocab(self) -> None:
        if not self._vocab:
            raise BPETokenError("[X] Reverse vocab güncellenemedi çünkü vocab boş.")
        
        seen = set()
        for token, value in list(self._vocab.items()):
            token_id = value["id"]
            if token_id in seen:
                logger.warning(f"[!] ID çakışması: {token} -> {token_id}")
                new_id = self.get_next_token_id(seen)
                value["id"] = new_id
                token_id = new_id
            seen.add(token_id)
        
        self.decoder.reverse_vocab = {
            value["id"]: token for token, value in self._vocab.items()
        }
        logger.info(f"[+] reverse_vocab başarıyla güncellendi. Toplam token sayısı: {len(self.decoder.reverse_vocab)}")

    def get_next_token_id(self, excluded_ids: Set[int]) -> int:
        used_ids = set(value["id"] for value in self._vocab.values())
        next_id = max(used_ids.union(excluded_ids), default=3) + 1
        while next_id in excluded_ids:
            next_id += 1
        return next_id

    @property
    def reverse_vocab(self) -> Dict[int, str]:
        return self.decoder.reverse_vocab

    def reset(self) -> None:
        logger.warning("[!] Vocab sıfırlanıyor...")
        self._vocab = self._default_vocab()
        self.encoder = BPEEncoder(self._vocab)
        self.decoder = BPEDecoder(self._vocab)
        self.trainer = BPETrainer(self._vocab)
        self.update_reverse_vocab()
        logger.info("[+] Vocab sıfırlandı ve bileşenler yeniden başlatıldı.")

    def _default_vocab(self) -> Dict[str, dict]:
        return {
            "<PAD>": {"id": 0, "total_freq": 0, "positions": []},
            "<UNK>": {"id": 1, "total_freq": 0, "positions": []},
            "<BOS>": {"id": 2, "total_freq": 0, "positions": []},
            "<EOS>": {"id": 3, "total_freq": 0, "positions": []},
        }

    def auto_update_vocab(self, tokens: List[str]) -> None:
        if not tokens or not isinstance(tokens, list):
            raise ValueError("[X] Yeni token listesi boş veya geçersiz formatta.")
        
        used_ids = set(self._vocab[token]["id"] for token in self._vocab)
        added_count = 0

        for token in tokens:
            if token not in self._vocab:
                new_id = self.get_next_token_id(used_ids)
                self._vocab[token] = {
                    "id": new_id,
                    "total_freq": 1,
                    "positions": []
                }
                used_ids.add(new_id)
                added_count += 1

        if added_count > 0:
            self.update_reverse_vocab()
            logger.info(f"[+] {added_count} yeni token eklendi.")
