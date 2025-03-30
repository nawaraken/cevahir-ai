"""
training_manager.py

Bu modül, eğitim sürecini yöneten TrainingManager sınıfını içerir.
"""

import logging
import json
import os
from typing import List, Dict, Any

from .training_preprocessor import TrainingPreprocessor
from .training_tokenizer import TrainingTokenizer
from .training_postprocessor import TrainingPostprocessor
from .training_tensorizer import TrainingTensorizer

from tokenizer_management.base_tokenizer_manager import BaseTokenizerManager

logger = logging.getLogger(__name__)


class TrainingManagerError(Exception):
    pass


class TrainingManager(BaseTokenizerManager):
    """
    Eğitim sürecini yöneten ana sınıf.
    """

    def __init__(self, config: Dict[str, Any] = None):
        if config is not None and not isinstance(config, dict):
            raise TypeError("Config bir sözlük olmalıdır.")
            
        self.config = config if config is not None else {}
        self.preprocessor = TrainingPreprocessor()
        self.tokenizer = TrainingTokenizer()
        self.postprocessor = TrainingPostprocessor()
        self.tensorizer = TrainingTensorizer()
        self.vocab = {}  # Eğitim öncesi boş vocab
        logger.info("TrainingManager başarıyla başlatıldı.")

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError("TrainingManager encode işlemi desteklemez.")

    def decode(self, token_ids: List[int]) -> str:
        raise NotImplementedError("TrainingManager decode işlemi desteklemez.")

    def train(self, corpus: List[str], target_vocab_size: int) -> None:
        if not corpus:
            logger.warning("Eğitim için verilen corpus boş.")
            return

        try:
            preprocessed = [self.preprocessor.preprocess(text) for text in corpus]
            tokenized = [self.tokenizer.tokenize(text) for text in preprocessed]
            new_vocab = self._build_vocab(tokenized, target_vocab_size)
            self.update_vocab(new_vocab)
            logger.info(f"Eğitim tamamlandı. Güncellenmiş vocab boyutu: {len(self.vocab)}")
        except Exception as e:
            logger.error(f"[X] Eğitim süreci sırasında hata: {e}")
            raise TrainingManagerError(f"Eğitim süreci sırasında hata: {e}")

    def _build_vocab(self, tokenized_corpus: List[List[str]], target_vocab_size: int) -> Dict[str, dict]:
        special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }

        vocab = {
            token: {
                "id": idx,
                "total_freq": 0,
                "positions": []
            } for token, idx in special_tokens.items()
        }

        word_freq = {}
        for tokens in tokenized_corpus:
            for token in tokens:
                token = token.strip()
                if token and token not in special_tokens:
                    word_freq[token] = word_freq.get(token, 0) + 1

        sorted_tokens = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        next_id = max(special_tokens.values()) + 1
        for token, freq in sorted_tokens[:target_vocab_size]:
            if token not in vocab:
                vocab[token] = {
                    "id": next_id,
                    "total_freq": freq,
                    "positions": []
                }
                next_id += 1
            else:
                vocab[token]["total_freq"] += freq

        logger.info(f"[+] Vocab başarıyla oluşturuldu. Toplam {len(vocab)} token mevcut.")
        return vocab

    def get_vocab(self) -> Dict[str, dict]:
        return self.vocab

    def set_vocab(self, new_vocab: Dict[str, dict]) -> None:
        if not isinstance(new_vocab, dict):
            raise TypeError("Vocab bir sözlük olmalıdır.")
        self.vocab = new_vocab
        logger.info("[+] Eğitim vocab dışarıdan set edildi.")

    def update_vocab(self, new_vocab: Dict[str, dict]) -> None:
        if not isinstance(new_vocab, dict):
            raise TypeError("Vocab bir sözlük olmalıdır.")

        logger.info("[+] Vocab güncelleniyor...")
        existing_ids = set(info["id"] for info in self.vocab.values())
        next_id = max(existing_ids, default=-1) + 1

        special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }

        for token, idx in special_tokens.items():
            if token not in self.vocab:
                self.vocab[token] = {
                    "id": idx,
                    "total_freq": 0,
                    "positions": []
                }

        for token, info in new_vocab.items():
            if token not in self.vocab:
                self.vocab[token] = {
                    "id": next_id,
                    "total_freq": info.get("total_freq", 0),
                    "positions": info.get("positions", [])
                }
                next_id += 1
            else:
                self.vocab[token]["total_freq"] += info.get("total_freq", 0)

        logger.info(f"[+] Vocab başarıyla güncellendi. Toplam {len(self.vocab)} token mevcut.")

    def update_reverse_vocab(self) -> None:
        logger.debug("[~] TrainingManager reverse_vocab gerektirmez, işlem atlandı.")

    def tensorize(self, corpus: List[str]) -> Any:
        if not self.vocab:
            raise TrainingManagerError("Vocab boş olduğu için tensorize yapılamaz.")

        try:
            preprocessed = [self.preprocessor.preprocess(text) for text in corpus]
            tokenized = [self.tokenizer.tokenize(text) for text in preprocessed]
            postprocessed = [self.postprocessor.process(tokens) for tokens in tokenized]
            tensor_data = self.tensorizer.tensorize(postprocessed)
            return tensor_data
        except Exception as e:
            logger.error("Tensorize işlemi sırasında hata: %s", e)
            raise TrainingManagerError(f"Tensorize işlemi sırasında hata: {e}")

    def save_vocab(self, path: str) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            special_tokens = {
                "<PAD>": 0,
                "<UNK>": 1,
                "<BOS>": 2,
                "<EOS>": 3
            }

            for token, idx in special_tokens.items():
                if token not in self.vocab:
                    self.vocab[token] = {
                        "id": idx,
                        "total_freq": 0,
                        "positions": []
                    }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.vocab, f, indent=4, ensure_ascii=False)

            logger.info(f"[+] Vocab dosyası başarıyla kaydedildi: {path}")

        except Exception as e:
            logger.error(f"[X] Vocab dosyası kaydedilemedi: {e}")
            raise OSError(f"Vocab dosyası kaydedilemedi: {e}")
