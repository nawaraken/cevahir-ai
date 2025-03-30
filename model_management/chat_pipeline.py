import logging
import sys
import os
import torch
from typing import Dict,Any

# Proje kök dizinini modül yoluna ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# src dizinini ekle
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

print(f"PYTHONPATH Güncellendi: {sys.path}")

# --- Gerekli modülleri içe aktar ---
from tokenizer_management.tokenizer_core import TokenizerCore
from model_management.model_manager import ModelManager

# Log yapılandırması
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ChatPipeline")


def main() -> None:
    """
    Terminal üzerinden kullanıcı ile sohbet etmek için chat pipeline.
    """
    logger.info("ChatPipeline başlatılıyor...")

    # --- Konfigürasyon Ayarları ---
    config: Dict[str, Any] = {
        "vocab_path": os.path.join("data", "vocab_lib", "vocab.json"),
        "max_seq_length": 64,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # --- TokenizerCore Başlat ---
    try:
        tokenizer_core = TokenizerCore(config=config)

        # ChattingManager'ın vocab'ini senkronize et
        tokenizer_core.chatting_manager.set_vocab(tokenizer_core.vocab_manager.vocab)

        logger.info("TokenizerCore başarıyla başlatıldı.")
    except Exception as e:
        logger.error(f"TokenizerCore başlatılamadı: {e}", exc_info=True)
        return

    # --- Modeli Başlat ve Ağırlıkları Yükle ---
    try:
        model_manager = ModelManager(config=config)
        setattr(model_manager, "device", config["device"])

        model_path = os.path.join("saved_models", "cevahir_model.pth")
        model_manager.load(model_path)

        logger.info(f"Model başarıyla yüklendi: {model_path}")
    except Exception as e:
        logger.error(f"Model yüklenirken hata oluştu: {e}", exc_info=True)
        return

    print("Cevahir AI Modeli Sohbet Arayüzüne hoş geldin! Çıkmak için 'kapat' yazabilirsin.")

    # --- Sohbet Döngüsü ---
    while True:
        user_input = input("Muhammed: ").strip()
        if user_input.lower() == "kapat":
            print("Sohbet sonlandırıldı.")
            break

        try:
            # Kullanıcı girdisini encode et
            token_ids = tokenizer_core.encode_text(user_input, method="chat")
            input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(model_manager.device)

            # Attention mask (zorunlu değil ama modellerde kullanışlı)
            attention_mask = (input_tensor != 0).long()

            # Yanıt üretimi
            response = tokenizer_core.chatting_manager.generate_response(
                model=model_manager.model,
                tensor_data={"input_ids": input_tensor, "attention_mask": attention_mask},
                max_length=50
            )

            print(f"Cevahir: {response}")

        except Exception as e:
            logger.error(f"Sohbet sırasında hata oluştu: {e}", exc_info=True)
            print("Cevahir: Üzgünüm, mesajını işleyemedim. Lütfen tekrar dene.")


if __name__ == "__main__":
    main()
