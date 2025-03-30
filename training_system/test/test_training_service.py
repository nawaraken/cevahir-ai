import os
import sys
import torch
import pytest

# Yol ayarı (training_service import edilebilsin)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from training_system.training_service import TrainingService

# ================================
# Test yapılandırması
# ================================
TEST_CONFIG = {
    "vocab_path": os.path.join("data", "vocab_lib", "vocab.json"),
    "data_directory": "education",
    "batch_size": 2,
    "training": {"epochs": 2, "learning_rate": 0.00005},
    "max_seq_length": 128,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ================================
# Test 1: TrainingService başlatılabiliyor mu?
# ================================
def test_training_service_initialization():
    service = TrainingService(TEST_CONFIG)
    assert service.model_manager.model is not None
    assert hasattr(service.tokenizer_core, "data_loader")

# ================================
# Test 2: Veri hazırlama işlemi düzgün çalışıyor mu?
# ================================
def test_training_service_prepare_data():
    service = TrainingService(TEST_CONFIG)
    train_loader, val_loader, seq_len = service._prepare_data()
    assert hasattr(train_loader, "__iter__")
    assert hasattr(val_loader, "__iter__")
    assert isinstance(seq_len, int)
    for batch in train_loader:
        assert isinstance(batch, tuple)
        assert len(batch) == 2
        assert isinstance(batch[0], torch.Tensor)
        assert isinstance(batch[1], torch.Tensor)
        break

# ================================
# Test 3: Eğitim işlemi sorunsuz çalışıyor mu?
# ================================
def test_training_service_train_process():
    service = TrainingService(TEST_CONFIG)
    service.train()
    assert service.training_manager is not None

# ================================
# Test 4: Dummy input ile tahmin alınabiliyor mu?
# ================================
def test_training_service_prediction():
    service = TrainingService(TEST_CONFIG)
    dummy_input = torch.randint(0, 100, (1, TEST_CONFIG["max_seq_length"]), dtype=torch.long)
    output = service.predict(dummy_input)
    assert output is not None
