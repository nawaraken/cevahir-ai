import os
import sys
import torch
import logging
from typing import Dict, List, Tuple, Union, Any
from torch.utils.data import DataLoader as TorchDataLoader

from torch.utils.data import TensorDataset, random_split, DataLoader as TorchDataLoader
from torch.utils.data._utils.collate import default_collate

from torch.utils.data import TensorDataset, random_split, DataLoader as TorchDataLoader

# Proje dizinini sys.path'e ekleyelim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Gerekli modüller
from model_management.model_manager import ModelManager
from training_management.training_manager import TrainingManager
from tokenizer_management.tokenizer_core import TokenizerCore

# Log yapılandırması
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
service_logger = logging.getLogger("TrainingService")

MODEL_SAVE_PATH = os.path.join("saved_models", "cevahir_model.pth")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


class TrainingService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        service_logger.info(f"Training device: {self.device}")

        self.model_manager = ModelManager(config)
        self.tokenizer_core = TokenizerCore(config)
        self.training_manager = None

        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            if os.path.exists(MODEL_SAVE_PATH):
                service_logger.info(f"Saved model found. Loading: {MODEL_SAVE_PATH}")
                self.model_manager.load(MODEL_SAVE_PATH)
                self.model_manager.model.train()  # <== MODELİ EĞİTİM MODUNA AL
            else:
                service_logger.info("No saved model found. Initializing new model...")
                self.model_manager.initialize()
                self.model_manager.model.train()  # <== YENİ MODELİ DE EĞİTİM MODUNA AL
        except Exception as e:
            service_logger.error(f"Model initialization error: {e}", exc_info=True)
            self.model_manager.initialize()
            self.model_manager.model.train()  # <== HATA SONRASI BAŞLATILAN MODELİ DE TRAIN MODUNA AL


    @staticmethod
    def custom_collate(batch):
        """
        Batch içindeki her örneğin (input_tensor, target_tensor) tuple'ı olduğunu varsayarak,
        bunları ayrı ayrı stackleyip, (inputs, targets) şeklinde tek bir tuple'a dönüştürür.
        """
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        return inputs, targets

    def _prepare_data(self) -> Tuple[TorchDataLoader, TorchDataLoader, int]:
        try:
            service_logger.info(" [1] TokenizerCore üzerinden eğitim verisi yükleniyor...")
            raw_data = self.tokenizer_core.load_training_data()
            self.tokenizer_core.finalize_vocab()
            if not raw_data:
                raise ValueError(" Hiç veri yüklenmedi! TokenizerCore'dan dönen veri boş.")

            service_logger.info(f" [2] Veri başarıyla yüklendi. Toplam örnek sayısı: {len(raw_data)}")

            max_seq_len = self.config.get("max_seq_length", 64)
            PAD_ID = self.tokenizer_core.vocab_manager.vocab.get("<PAD>", {}).get("id", 0)
            BOS_ID = self.tokenizer_core.vocab_manager.vocab.get("<BOS>", {}).get("id", 2)
            EOS_ID = self.tokenizer_core.vocab_manager.vocab.get("<EOS>", {}).get("id", 3)

            dataset = []

            for idx, pair in enumerate(raw_data):
                try:
                    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                        raise TypeError(f"[!] Satır {idx} format hatası: Beklenen (input_ids, target_ids), gelen: {type(pair)}")

                    input_ids, target_ids = pair

                    if not all(isinstance(t, int) for t in input_ids + target_ids):
                        raise TypeError(f"[!] Satır {idx} token tipi hatalı: Tüm token'lar int olmalı.")

                    # BOS/EOS kontrolü
                    if input_ids[0] != BOS_ID:
                        input_ids = [BOS_ID] + input_ids
                    if input_ids[-1] != EOS_ID:
                        input_ids.append(EOS_ID)
                    if target_ids[0] != BOS_ID:
                        target_ids = [BOS_ID] + target_ids
                    if target_ids[-1] != EOS_ID:
                        target_ids.append(EOS_ID)

                    # Padding
                    input_ids = (
                        input_ids[:max_seq_len]
                        if len(input_ids) > max_seq_len
                        else input_ids + [PAD_ID] * (max_seq_len - len(input_ids))
                    )
                    target_ids = (
                        target_ids[:max_seq_len]
                        if len(target_ids) > max_seq_len
                        else target_ids + [PAD_ID] * (max_seq_len - len(target_ids))
                    )

                    input_tensor = torch.tensor(input_ids, dtype=torch.long)
                    target_tensor = torch.tensor(target_ids, dtype=torch.long)

                    assert input_tensor.shape[0] == max_seq_len, f"Input uzunluğu {input_tensor.shape[0]} != {max_seq_len}"
                    assert target_tensor.shape[0] == max_seq_len, f"Target uzunluğu {target_tensor.shape[0]} != {max_seq_len}"

                    dataset.append((input_tensor, target_tensor))
                    service_logger.debug(f"[✓] Satır {idx}: input ve target tensorları başarıyla oluşturuldu.")

                except Exception as tokenization_error:
                    service_logger.warning(f"[!] Satır {idx} token işleme hatası: {tokenization_error}")

            if not dataset:
                raise ValueError(" Tokenizasyon sonrası geçerli veri bulunamadı!")

            full_dataset = dataset

            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

            batch_size = self.config.get("batch_size", 8)
            train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=TrainingService.custom_collate)
            val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=TrainingService.custom_collate)

            service_logger.info(f" [3] Eğitim veri seti hazır. Toplam: {len(full_dataset)} | Train: {train_size}, Val: {val_size}")
            service_logger.debug(f"[✓] Batch size: {batch_size}, Seq len: {max_seq_len}, PAD ID: {PAD_ID}, BOS ID: {BOS_ID}, EOS ID: {EOS_ID}")

            return train_loader, val_loader, max_seq_len

        except Exception as e:
            service_logger.error(f" [X] prepare_data aşamasında hata oluştu: {e}", exc_info=True)
            raise



    def verify_training_data(self, data: List[Tuple[torch.Tensor, torch.Tensor]]) -> bool:
        """Her örneğin (input_tensor, target_tensor) tuple'ı olduğunu ve her iki tensörün de doğru tipte olduğunu doğrular."""
        for i, pair in enumerate(data):
            if not (isinstance(pair, tuple) and len(pair) == 2):
                service_logger.warning(f"[!] Eğitim verisi format hatası: Satır {i} -> {type(pair)}")
                return False
            input_tensor, target_tensor = pair
            if not (isinstance(input_tensor, torch.Tensor) and isinstance(target_tensor, torch.Tensor)):
                service_logger.warning(f"[!] Eğitim verisi tipi hatalı: Satır {i}")
                return False
        return True




    def train(self) -> None:
        service_logger.info("Training pipeline is starting...")
        try:
            self.model_manager.model.train()  # <== GÜVENLİK AMAÇLI BURAYA DA KOY
            train_loader, val_loader, seq_len = self._prepare_data()
            self.training_manager = TrainingManager(
                model=self.model_manager.model,
                optimizer=self.model_manager.optimizer,
                criterion=self.model_manager.criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                config={**self.config, "seq_len": seq_len}
            )

            train_loss, val_loss = self.training_manager.train()
            if torch.isnan(torch.tensor(train_loss)) or torch.isnan(torch.tensor(val_loss)):
                raise ValueError("NaN loss detected.")

            service_logger.info(f"Training complete. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self.save_model()

        except Exception as e:
            service_logger.error(f"Training failed: {e}", exc_info=True)


    def save_model(self) -> None:
        try:
            service_logger.info(f"Saving model to: {MODEL_SAVE_PATH}")
            self.model_manager.save(MODEL_SAVE_PATH)
        except Exception as e:
            service_logger.error(f"Model saving error: {e}", exc_info=True)

    def predict(self, input_tensor: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Any], None]:
        if not isinstance(input_tensor, torch.Tensor):
            service_logger.error("Invalid input type. Expected torch.Tensor.")
            return None

        input_tensor = input_tensor.to(self.device)
        try:
            with torch.no_grad():
                output = self.model_manager.forward(input_tensor, inference=True)
                return output
        except Exception as e:
            service_logger.error(f"Prediction error: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    CONFIG = {
        "vocab_path": os.path.join("data", "vocab_lib", "vocab.json"),
        "data_directory": "education",
        "batch_size": 8,
        "training": {"epochs": 5, "learning_rate": 0.001},
        "max_seq_length": 64,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    trainer = TrainingService(CONFIG)
    trainer.train()
