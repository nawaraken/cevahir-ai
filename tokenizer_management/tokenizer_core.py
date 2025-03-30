import os
import json
import logging
from typing import Any, Dict, List, Union, Optional,Tuple
import torch

# Yapılandırma dosyasını içe aktar
from .config import VOCAB_PATH, TOKENIZER_CONFIG

# Interface yapısını içe aktar
from .base_tokenizer_manager import BaseTokenizerManager

# Modül yöneticilerini içe aktaralım
from tokenizer_management.vocab.vocab_manager import VocabManager
from tokenizer_management.training.training_manager import TrainingManager
from tokenizer_management.bpe.bpe_manager import BPEManager
from tokenizer_management.sentencepiece.sentencepiece_manager import SentencePieceManager
from tokenizer_management.chatting.chatting_manager import ChattingManager

# DataLoaderManager'ı içe aktaralım
from tokenizer_management.data_loader.data_loader_manager import DataLoaderManager

logger = logging.getLogger(__name__)


class TokenizerCore:
    """
    TokenizerCore, tüm tokenizasyon işlemlerini merkezi olarak yöneten ana sınıftır.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config if config else {}

        try:
            # -------------------------------
            # VocabManager Başlatma
            # -------------------------------
            vocab_path = self.config.get("vocab_path", VOCAB_PATH)
            if not vocab_path or not os.path.exists(vocab_path):
                logger.warning(f"[!] Vocab dosyası bulunamadı: {vocab_path}")
                logger.warning("[!] Varsayılan vocab dosyası oluşturuluyor...")
                os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
                self._create_default_vocab(vocab_path)

            logger.info(f"[+] Vocab dosyası yolu: {vocab_path}")

            self.vocab_manager = VocabManager(vocab_path=vocab_path)
            vocab = self.vocab_manager.vocab
            if not vocab:
                raise ValueError("Vocab dosyası yüklenemedi veya boş.")
            logger.info(f"[+] Vocab dosyası başarıyla yüklendi. Toplam token sayısı: {len(vocab)}")

            # -------------------------------
            # Tokenizer Modüllerini Başlat
            # -------------------------------
            self.bpe_manager = BPEManager(vocab=self.vocab_manager.vocab)
            self.sentencepiece_manager = SentencePieceManager(vocab=self.vocab_manager.vocab)
            self.training_manager = TrainingManager(config=self.config.get("training", {}))
            self.chatting_manager = ChattingManager(vocab_path=vocab_path)


            logger.info("[+] Tüm tokenizer yöneticileri başarıyla başlatıldı.")

            # -------------------------------
            # DataLoaderManager Başlat
            # -------------------------------
            data_directory = self.config.get("data_directory", "data/")
            batch_size = self.config.get("batch_size", TOKENIZER_CONFIG.get("batch_size", 8))

            if not os.path.exists(data_directory):
                os.makedirs(data_directory)

            self.data_loader = DataLoaderManager(
                data_directory=data_directory,
                batch_size=batch_size
            )
            logger.info("[+] DataLoaderManager başarıyla başlatıldı.")

            logger.info("[✓] TokenizerCore başarıyla başlatıldı.")

        except Exception as e:
            logger.error(f"[X] TokenizerCore başlatılırken hata: {e}")
            raise

    def _create_default_vocab(self, vocab_path: str) -> None:
        default_vocab = {
            "<PAD>": {"id": 0, "total_freq": 1, "positions": []},
            "<UNK>": {"id": 1, "total_freq": 1, "positions": []},
            "<BOS>": {"id": 2, "total_freq": 1, "positions": []},
            "<EOS>": {"id": 3, "total_freq": 1, "positions": []},
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(default_vocab, f, indent=4, ensure_ascii=False)
        logger.info("[+] Varsayılan vocab dosyası oluşturuldu.")

    def _normalize_input(self, text: Union[str, List[str], Dict]) -> str:
        if isinstance(text, list):
            text = " ".join(map(str, text))
        elif isinstance(text, dict):
            if "data" not in text:
                raise ValueError("Geçersiz dict formatı: 'data' anahtarı eksik.")
            text = str(text["data"])
        elif not isinstance(text, str):
            raise TypeError(f"Geçersiz giriş formatı: {type(text)}")
        return text.strip()

    def encode_text(
        self,
        text: Union[str, List[str], Dict],
        method: str = "bpe",
        mode: str = "train"
    ) -> List[int]:
        if text is None:
            raise ValueError("Girdi metni boş veya geçersiz formatta.")

        text = self._normalize_input(text)

        manager = self._get_manager(method)
        manager.update_reverse_vocab()

        if method == "bpe":
            tokens, token_ids = manager.encode(text, mode=mode)

            #  Vocab güncellemesi bu aşamada yapılmıyor
            # finalize_vocab() çağrısıyla işlem tamamlanacak
        else:
            token_ids = manager.encode(text)

        # Geçersiz token ID kontrolü
        reverse_vocab = getattr(manager, "reverse_vocab", None)
        if reverse_vocab and method == "bpe":
            missing_ids = [token_id for token_id in token_ids if token_id not in reverse_vocab]
            if missing_ids:
                raise ValueError(f"Geçersiz token ID'leri: {missing_ids}")

        return token_ids

    def generate_response(self, model: torch.nn.Module, tensor_data: Dict[str, torch.Tensor], max_length: int = 64) -> str:
        """
        ChattingManager üzerinden model çıktısını çözümleyip cevap üretir.
        """
        return self.chatting_manager.generate_response(model, tensor_data, max_length)

    def finalize_vocab(self) -> None:
        """
        Tokenizasyon işlemleri tamamlandıktan sonra vocab dosyasını toplu şekilde günceller ve kaydeder.
        """
        try:
            updated_vocab = self.bpe_manager.get_vocab()
            self.vocab_manager.set_vocab(updated_vocab)
            self.vocab_manager.save_vocab()
            logger.info("[✓] Vocab toplu olarak güncellendi ve kaydedildi.")
        except Exception as e:
            logger.error(f"[X] finalize_vocab() hatası: {e}")
            raise


    def decode_text(self, token_ids: List[int], method: str = "bpe") -> str:
        if not token_ids or not isinstance(token_ids, list):
            raise ValueError("Geçersiz token ID listesi.")

        manager = self._get_manager(method)
        return manager.decode(token_ids).strip()

    def train_model(self, corpus: List[str], method: str = "training", vocab_size: int = None) -> None:
        if not corpus or not isinstance(corpus, list):
            raise ValueError("Eğitim verisi boş veya hatalı formatta.")

        manager = self._get_manager(method)
        manager.train(corpus, vocab_size)

        # Vocab güncellemesi
        new_vocab = manager.get_vocab()
        self.vocab_manager.set_vocab(new_vocab)
        self._update_modules_vocab()
        logger.info("[✓] Eğitim işlemi başarıyla tamamlandı.")

    def _get_manager(self, method: str) -> BaseTokenizerManager:
        method = method.lower()
        manager_map: Dict[str, BaseTokenizerManager] = {
            "bpe": self.bpe_manager,
            "sentencepiece": self.sentencepiece_manager,
            "chat": self.chatting_manager,
            "training": self.training_manager
        }
        if method not in manager_map:
            raise ValueError(f"Geçersiz yöntem: {method}")
        return manager_map[method]

    def _update_modules_vocab(self) -> None:
        """
        Vocab güncellendiğinde tüm modüllerin hem vocab hem reverse vocab yapıları eşitlenir.
        """
        updated_vocab = self.vocab_manager.vocab

        self.bpe_manager.set_vocab(updated_vocab)
        self.sentencepiece_manager.set_vocab(updated_vocab)
        self.chatting_manager.set_vocab(updated_vocab)

        # Eğer modül reverse_vocab kullanıyorsa, onun da güncellenmesi gerekir
        if hasattr(self.bpe_manager, "update_reverse_vocab"):
            self.bpe_manager.update_reverse_vocab()

        if hasattr(self.sentencepiece_manager, "update_reverse_vocab"):
            self.sentencepiece_manager.update_reverse_vocab()

        if hasattr(self.chatting_manager, "update_reverse_vocab"):
            self.chatting_manager.update_reverse_vocab()

        logger.info("[✓] Tüm modüllerin vocab ve reverse vocab yapıları güncellendi.")


    def update_vocab(self, new_tokens: List[str], method: str = "bpe") -> None:
        self.vocab_manager.update_vocab(new_tokens, method)
        self._update_modules_vocab()

    def _split_sentences(self, text: str) -> List[str]:
        """
        Noktalama işaretlerine göre cümlelere ayırır.
        Nokta, soru işareti, ünlem gibi ayırıcıları kullanır.
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def load_training_data(self) -> List[Tuple[List[int], List[int]]]:
        """
        Eğitim verisini yükler ve token ID çiftleri olarak döndürür.
        - JSON: (input=soru, target=cevap)
        - DOCX/text: (__tag__text) etiketli çiftler (input=target)
        """
        data = self.data_loader.load_data()
        tokenized_data = []
        max_seq_len = self.config.get("max_seq_length", 64)

        for entry_index, entry in enumerate(data):
            try:
                normalized_entry = self._normalize_input(entry)
                source_file = entry.get("source_file", "bilinmiyor")
                modality = entry.get("modality", "text")

                logger.debug(f"[{entry_index}] Veri işleniyor | Modality: {modality} | Kaynak: {source_file}")

                if "__tag__soru" in normalized_entry and "__tag__cevap" in normalized_entry:
                    qa_pairs = self._extract_json_qa_pairs(normalized_entry)
                    for s_text, c_text in qa_pairs:
                        input_ids = self.encode_text(s_text, method="bpe", mode="train")[:max_seq_len]
                        target_ids = self.encode_text(c_text, method="bpe", mode="train")[:max_seq_len]
                        if input_ids and target_ids:
                            tokenized_data.append((input_ids, target_ids))
                            logger.debug(f"[{entry_index}] QA eklendi | Len: {len(input_ids)} → {len(target_ids)}")
                else:
                    sentences = self._split_sentences(normalized_entry)
                    chunk = []

                    for idx, sentence in enumerate(sentences):
                        chunk.append(sentence.strip())
                        if len(chunk) == 3 or idx == len(sentences) - 1:
                            text_block = " ".join(chunk)
                            tagged = f"__tag__text {text_block}"
                            ids = self.encode_text(tagged, method="bpe", mode="train")[:max_seq_len]
                            if ids:
                                tokenized_data.append((ids, ids.copy()))
                                logger.debug(f"[{entry_index}] DOCX CHUNK eklendi | Len: {len(ids)}")
                            chunk = []

            except Exception as e:
                logger.warning(f"[!] Satır {entry_index} tokenize edilirken hata: {e}")
                continue

        if not tokenized_data:
            logger.warning("[!] Hiçbir geçerli eğitim verisi tokenize edilemedi.")
            return []

        logger.info(f"[✓] Tokenize edilen toplam örnek sayısı: {len(tokenized_data)}")

        if not self.verify_training_data(tokenized_data):
            logger.error("[X] Eğitim verisi doğrulama başarısız! Veri yapısında bozulma olabilir.")
            raise ValueError("Tokenized training data is invalid!")

        return tokenized_data

    def _extract_json_qa_pairs(self, text: str) -> List[Tuple[str, str]]:
        """
        __tag__soru ve __tag__cevap etiketlerine göre soru-cevap çiftlerini çıkarır.
        """
        pairs = []
        segments = text.split("__tag__soru")
        for segment in segments:
            if not segment.strip():
                continue
            parts = segment.split("__tag__cevap")
            if len(parts) == 2:
                soru = parts[0].strip()
                cevap = parts[1].strip()
                tagged_soru = f"__tag__soru {soru}"
                tagged_cevap = f"__tag__cevap {cevap}"
                pairs.append((tagged_soru, tagged_cevap))
            else:
                logger.warning(f"[!] Eksik __tag__cevap etiketi tespit edildi: {segment[:50]}...")
        return pairs

    def verify_training_data(self, data: List[Tuple[List[int], List[int]]]) -> bool:
        for i, pair in enumerate(data):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                logger.warning(f"[!] Eğitim verisi format hatası: Satır {i} -> {type(pair)}")
                return False
            input_ids, target_ids = pair
            if not all(isinstance(t, int) for t in input_ids + target_ids):
                logger.warning(f"[!] Eğitim verisi tipi hatalı: Satır {i}")
                return False
        return True






# Global instance
tokenizer_core = TokenizerCore()
