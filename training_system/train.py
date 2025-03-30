import os
import sys
import torch
import logging

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from training_service import TrainingService

# Log yapılandırması
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
train_logger = logging.getLogger("TrainScript")

# Model ve Eğitim Konfigürasyonu
TRAIN_CONFIG = {
    # === Yol Ayarları ===
    "vocab_path": "data/vocab_lib/vocab.json",
    "data_directory": "education",
    "checkpoint_dir": "saved_models/checkpoints/",
    "training_history_path": "saved_models/training_history.json",
    "model_save_path": "saved_models/cevahir_model.pth",  # Service içinde kullanılacak

    # === Eğitim Parametreleri ===
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.00005,
    "early_stopping_patience": 3,
    "checkpoint_interval": 2,   # Her epoch'ta checkpoint alınsın
    "tb_log_dir": "runs/cevahir_training",

    # === Model Yapısı ===
    "vocab_size": 75000,
    "max_seq_length": 256,
    "embed_dim": 1024,
    "seq_proj_dim": 1024,
    "num_heads": 16,
    "dropout": 0.2,
    "attention_type": "multi_head",
    "normalization_type": "layer_norm",

    # === Cihaz ve Diğer ===
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_files": 45,
    "log_level": "INFO",
    "gradient_clip": 1.0
}

def main():
    try:
        train_logger.info("Eğitim süreci başlatılıyor...")

        # Eğitim servisini başlat
        training_service = TrainingService(TRAIN_CONFIG)

        # Modelin eğitimi
        training_service.train()

        train_logger.info("Eğitim tamamlandı, model kaydedildi.")

    except Exception as e:
        train_logger.error(f"Eğitim sırasında hata oluştu: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
