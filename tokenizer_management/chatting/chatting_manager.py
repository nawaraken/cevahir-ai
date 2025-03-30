import logging
from typing import List, Dict, Any,Optional
import os
import json
import torch
from .chat_preprocessor import ChatPreprocessor
from .chat_tokenizer import ChatTokenizer
from .chat_encoder import ChatEncoder
from .chat_decoder import ChatDecoder
from .chat_postprocessor import ChatPostprocessor

from tokenizer_management.base_tokenizer_manager import BaseTokenizerManager

logger = logging.getLogger(__name__)


class ChattingManagerError(Exception):
    """ChattingManager ile ilgili hatalar için özel exception sınıfı."""
    pass


class ChattingManager(BaseTokenizerManager):
    """
    Chat tabanlı tokenizasyon işlemlerini yöneten ana sınıf.
    """

    def __init__(self, vocab_path: Optional[str] = None):
        self.preprocessor = ChatPreprocessor()
        self.tokenizer = ChatTokenizer()
        self.encoder = ChatEncoder()
        self.decoder = ChatDecoder()
        self.postprocessor = ChatPostprocessor()

        # Vocab: basit token -> id mapping (str -> int)
        self.vocab: Dict[str, int] = {}
        self._reverse_vocab: Dict[int, str] = {}   # ID -> Token

        logger.info("ChattingManager başarıyla başlatıldı.")

        # Eğer yol verildiyse vocab otomatik yüklenir
        if vocab_path:
            self.load_vocab(vocab_path)


    def encode(self, text: str) -> List[int]:
        try:
            preprocessed = self.preprocessor.preprocess(text)
            tokens = self.tokenizer.tokenize(preprocessed)
            token_ids = self.encoder.encode(tokens)
            return token_ids
        except Exception as e:
            logger.error("Chat mesajını kodlama sırasında hata: %s", e)
            raise ChattingManagerError(f"encode() hatası: {e}")

    def decode(self, token_ids: List[int]) -> str:
        try:
            # Decoder'dan raw string al
            decoded_raw = self.decoder.decode(token_ids)

            # Eğer <UNK> içeriyorsa logla
            if "<UNK>" in decoded_raw:
                unk_count = decoded_raw.count("<UNK>")
                logger.warning(f"[!] {unk_count} adet bilinmeyen token <UNK> tespit edildi.")

            # Post-process
            final_text = self.postprocessor.process(decoded_raw)

            # Ek güvenlik: boş string dönerse uyar
            if not final_text.strip():
                logger.warning("[!] decode() sonucu boş metin.")
                return "<boş>"

            return final_text

        except Exception as e:
            logger.error("Chat mesajını çözme sırasında hata: %s", e)
            raise ChattingManagerError(f"decode() hatası: {e}")

    def train(self, corpus: List[str], vocab_size: int = None) -> None:
        """
        Eğitim metodu: Tüm corpus verilerinden vocab inşa eder.
        
        Args:
            corpus (List[str]): Eğitim verisi
            vocab_size (int, optional): Vocab sınırı (şimdilik kullanılmıyor)
        """
        try:
            token_counter = {}

            for text in corpus:
                preprocessed = self.preprocessor.preprocess(text)
                tokens = self.tokenizer.tokenize(preprocessed)
                for token in tokens:
                    token_counter[token] = token_counter.get(token, 0) + 1

            sorted_tokens = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)

            # Basit bir token → id eşlemesi oluşturuyoruz.
            self.vocab = {token: idx for idx, (token, _) in enumerate(sorted_tokens)}

            self.update_reverse_vocab()
            self.encoder.set_vocab(self.vocab)
            self.decoder.set_reverse_vocab(self.reverse_vocab)

            logger.info("[+] Chatting vocab başarıyla eğitildi. Token sayısı: %d", len(self.vocab))

        except Exception as e:
            logger.error("[X] Chatting train sırasında hata: %s", e)
            raise ChattingManagerError(f"train() hatası: {e}")

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def set_vocab(self, vocab: Dict[str, Any]) -> None:
        """
        Gelen vocab sözlüğü (değerler dict ya da int olabilir) 
        token → id eşlemesine dönüştürülerek kaydedilir.
        """
        if not isinstance(vocab, dict):
            raise ValueError("Vocab dict formatında olmalıdır.")

        converted_vocab = {}
        for token, value in vocab.items():
            if isinstance(value, dict):
                if "id" in value:
                    converted_vocab[token] = value["id"]
                else:
                    raise ValueError(f"Token '{token}' için 'id' bulunamadı.")
            elif isinstance(value, int):
                converted_vocab[token] = value
            else:
                raise ValueError(f"Token '{token}' için geçersiz değer tipi: {type(value)}")

        self.vocab = converted_vocab
        self.update_reverse_vocab()

        self.encoder.set_vocab(self.vocab)
        self.decoder.set_reverse_vocab(self.reverse_vocab)

        logger.info("[+] Chatting vocab başarıyla yüklendi.")

    def update_reverse_vocab(self) -> None:
        try:
            self._reverse_vocab = {idx: token for token, idx in self.vocab.items()}
            logger.info("[+] Reverse vocab güncellendi. Toplam token sayısı: %d", len(self._reverse_vocab))
        except Exception as e:
            logger.error("[X] Reverse vocab güncelleme sırasında hata: %s", e)
            raise ChattingManagerError(f"Reverse vocab güncelleme hatası: {e}")

    @property
    def reverse_vocab(self) -> Dict[int, str]:
        return self._reverse_vocab

    @reverse_vocab.setter
    def reverse_vocab(self, value: Dict[int, str]):
        self._reverse_vocab = value

    def save_vocab(self, path: str) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.vocab, f, indent=4, ensure_ascii=False)
            logger.info("[+] Chatting vocab dosyası kaydedildi: %s", path)
        except Exception as e:
            logger.error("[X] Chatting vocab dosyası kaydedilemedi: %s", e)
            raise IOError(f"Chatting vocab save error: {e}")

    def load_vocab(self, path: str) -> None:
        """
        Belirtilen JSON dosyasından vocab yükler ve encoder/decoder ile eşler.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded_vocab = json.load(f)

            # Eğer vocab dosyası dict değerler içeriyorsa, bunları token → id eşlemesine çeviriyoruz.
            converted_vocab = {}
            for token, value in loaded_vocab.items():
                if isinstance(value, dict):
                    if "id" in value:
                        converted_vocab[token] = value["id"]
                    else:
                        raise ValueError(f"Token '{token}' için 'id' bulunamadı.")
                elif isinstance(value, int):
                    converted_vocab[token] = value
                else:
                    raise ValueError(f"Token '{token}' için geçersiz değer tipi: {type(value)}")

            # Ana vocab güncellemesi
            self.vocab = converted_vocab
            self.update_reverse_vocab()

            # Encoder ve decoder da bu yeni vocab ile güncellenmeli
            self.encoder.set_vocab(self.vocab)
            self.decoder.set_reverse_vocab(self.reverse_vocab)

            logger.info("[+] Chatting vocab başarıyla yüklendi: %s", path)

        except Exception as e:
            logger.error("[X] Chatting vocab dosyası yüklenemedi: %s", e)
            raise IOError(f"Chatting vocab load error: {e}")


    def auto_update_vocab(self, tokens: List[str]) -> None:
        if not tokens or not isinstance(tokens, list):
            raise ChattingManagerError("[X] Geçerli bir token listesi sağlanmalıdır.")

        new_tokens = [token.lower() for token in tokens if token.lower() not in self.vocab]

        if new_tokens:
            logger.info(f"[+] Yeni tokenler tespit edildi: {new_tokens}")

            existing_ids = {id for id in self.vocab.values()}
            next_available_id = max(existing_ids, default=3) + 1

            for token in new_tokens:
                if token not in self.vocab:
                    self.vocab[token] = next_available_id
                    logger.info(f"[+] Yeni token eklendi: '{token}' -> ID: {next_available_id}")
                    next_available_id += 1
                else:
                    logger.debug(f"[~] Token zaten mevcut: '{token}'")

            self.update_reverse_vocab()
            logger.info("[✓] Chatting vocab güncellemesi tamamlandı.")

    def generate_response(self, model: torch.nn.Module, tensor_data: Dict[str, torch.Tensor], max_length: int = 64) -> str:
        """
        Oto-regresif şekilde token token cevap üretir. <EOS> tokenı üretildiğinde durur.
        """
        try:
            model.eval()
            input_ids = tensor_data["input_ids"].clone().detach()
            generated_ids = input_ids.clone()

            with torch.no_grad():
                for _ in range(max_length):
                    outputs, _ = model(generated_ids)
                    next_token_logits = outputs[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

                    # <EOS> üretildiyse döngüyü durdur
                    if next_token_id.item() == self.encoder.eos_token_id:
                        break

                    generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

            predicted_ids = generated_ids[0].tolist()
            decoded = self.decode(predicted_ids)
            return decoded

        except Exception as e:
            logger.error(f"[X] Cevap üretimi sırasında hata: {e}")
            raise ChattingManagerError(f"generate_response hatası: {e}")

