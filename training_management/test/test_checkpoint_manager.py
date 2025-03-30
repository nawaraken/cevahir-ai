"""
pytest_checkpoint_manager.py
=============================

Bu dosya, CheckpointManager sınıfını test etmek için pytest çerçevesini kullanır.
Tüm temel ve hata senaryolarını test ederek, checkpoint yönetiminin eksiksiz çalışmasını sağlar.

Testler:
--------
1. **Checkpoint kaydetme (save_checkpoint)**
    - Model ve optimizer durumlarını başarıyla kaydediyor mu?
    - Kaydedilen dosya var mı?
    - Maksimum checkpoint sınırını aştığında en eski dosyayı siliyor mu?

2. **Checkpoint yükleme (load_checkpoint)**
    - Kaydedilen dosya başarıyla yükleniyor mu?
    - Epoch ve eğitim geçmişi doğru mu?
    - Eksik veya bozuk dosya durumunda hata veriyor mu?

3. **Checkpoint rotasyonu (_manage_checkpoint_rotation)**
    - Maksimum dosya sınırını aşan eski checkpoint dosyalarını temizliyor mu?

4. **Checkpoint listeleme (list_checkpoints)**
    - Var olan checkpoint dosyaları eksiksiz listeleniyor mu?
    - Geçersiz dizin senaryosunu ele alıyor mu?
"""

import os
import pytest
import torch
from torch import nn, optim

from training_management.checkpoint_manager import CheckpointManager
from config.parameters import CHECKPOINT_MODEL, DEVICE

# Sahte Model Tanımla (Test İçin)
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def checkpoint_manager(tmp_path):
    """
    Geçici dizinde çalışan bir CheckpointManager örneği oluşturur.
    """
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_manager = CheckpointManager(
        checkpoint_model_dir=str(checkpoint_dir), max_checkpoints=3, device=DEVICE
    )
    return checkpoint_manager


@pytest.fixture
def dummy_model_and_optimizer():
    """
    Testler için basit bir model ve optimizer döndürür.
    """
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    return model, optimizer


def test_checkpoint_saving(checkpoint_manager, dummy_model_and_optimizer):
    """
    Checkpoint kaydetme fonksiyonunu test eder.
    """
    model, optimizer = dummy_model_and_optimizer
    epoch = 1
    training_history = {"train_loss": [0.1], "val_loss": [0.2]}

    # Checkpoint kaydet
    checkpoint_path = checkpoint_manager.save_checkpoint(model, optimizer, epoch, training_history)

    # Checkpoint'in gerçekten kaydedildiğini doğrula
    assert os.path.exists(checkpoint_path), "Checkpoint dosyası kaydedilmedi."
    assert len(os.listdir(checkpoint_manager.checkpoint_model_dir)) == 1, "Checkpoint dosya sayısı yanlış."


def test_checkpoint_loading(checkpoint_manager, dummy_model_and_optimizer):
    """
    Kaydedilen checkpoint'in yüklenmesini test eder.
    """
    model, optimizer = dummy_model_and_optimizer
    epoch = 2
    training_history = {"train_loss": [0.15], "val_loss": [0.25]}

    # Checkpoint kaydet
    checkpoint_path = checkpoint_manager.save_checkpoint(model, optimizer, epoch, training_history)

    # Modeli sıfırla ve tekrar yükle
    new_model, new_optimizer = DummyModel(), optim.Adam(model.parameters(), lr=0.01)
    loaded_data = checkpoint_manager.load_checkpoint(new_model, new_optimizer, checkpoint_path)

    # Yüklenen epoch ve eğitim geçmişini kontrol et
    assert loaded_data["epoch"] == epoch, "Checkpoint'ten yüklenen epoch yanlış."
    assert loaded_data["training_history"] == training_history, "Checkpoint'ten yüklenen eğitim geçmişi yanlış."


def test_checkpoint_file_not_found(checkpoint_manager, dummy_model_and_optimizer):
    """
    Geçersiz bir dosya yolu ile yükleme yapıldığında FileNotFoundError fırlatılmalı.
    """
    model, optimizer = dummy_model_and_optimizer
    fake_path = os.path.join(checkpoint_manager.checkpoint_model_dir, "non_existent_checkpoint.pth")

    with pytest.raises(FileNotFoundError):
        checkpoint_manager.load_checkpoint(model, optimizer, fake_path)


def test_checkpoint_rotation(checkpoint_manager, dummy_model_and_optimizer):
    """
    Maksimum checkpoint sınırı aşıldığında en eski checkpoint'in silindiğini test eder.
    """
    model, optimizer = dummy_model_and_optimizer

    # 4 farklı checkpoint kaydet (max_checkpoints = 3 olduğu için biri silinmeli)
    for epoch in range(1, 5):
        checkpoint_manager.save_checkpoint(model, optimizer, epoch)

    # Kaydedilen checkpoint dosyalarını al
    checkpoint_files = os.listdir(checkpoint_manager.checkpoint_model_dir)

    # Maksimum checkpoint sınırına göre eski dosyaların silindiğini kontrol et
    assert len(checkpoint_files) == 3, "Checkpoint dosya rotasyonu doğru çalışmıyor."
    assert "checkpoint_epoch_1.pth" not in checkpoint_files, "En eski checkpoint dosyası silinmedi."


def test_checkpoint_listing(checkpoint_manager, dummy_model_and_optimizer):
    """
    Checkpoint dosyalarının eksiksiz listelendiğini test eder.
    """
    model, optimizer = dummy_model_and_optimizer

    # 3 farklı checkpoint kaydet
    for epoch in range(1, 4):
        checkpoint_manager.save_checkpoint(model, optimizer, epoch)

    # Checkpoint dosyalarının listelenmesini kontrol et
    checkpoint_list = checkpoint_manager.list_checkpoints()

    assert len(checkpoint_list) == 3, "Checkpoint dosya sayısı eksik listelendi."
    assert all(os.path.exists(f) for f in checkpoint_list), "Listeleme sırasında olmayan dosya döndürüldü."


def test_checkpoint_listing_with_empty_directory(tmp_path):
    """
    Boş bir dizinde checkpoint listesi döndürmeye çalıştığında hata vermemeli, boş liste döndürmeli.
    """
    empty_checkpoint_manager = CheckpointManager(checkpoint_model_dir=str(tmp_path / "empty_checkpoints"))

    checkpoint_list = empty_checkpoint_manager.list_checkpoints()
    assert checkpoint_list == [], "Boş dizin için list_checkpoints boş liste döndürmelidir."
