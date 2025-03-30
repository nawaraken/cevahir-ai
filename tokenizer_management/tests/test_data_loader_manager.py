import os
import pytest
import torch
import numpy as np
from pathlib import Path
from tokenizer_management.data_loader.data_loader_manager import DataLoaderManager
from tokenizer_management.data_loader.data_loader_manager import FileNotFoundError, DataLoadError, DataPreprocessError

# ================================================
# Gerçek Dosya Dizini İçin Fixture
# ================================================
@pytest.fixture(scope="module")
def real_data_dir():
    data_dir = Path(__file__).resolve().parents[2] / "education"
    if not data_dir.exists():
        pytest.skip(f"Gerçek 'education' dizini bulunamadı: {data_dir}")
    return data_dir

# ================================================
# TEMEL TESTLER
# ================================================
def test_directory_exists(real_data_dir):
    assert real_data_dir.exists(), "Veri dizini mevcut olmalı."

def test_load_data_returns_list(real_data_dir):
    dlm = DataLoaderManager(str(real_data_dir))
    data = dlm.load_data()
    assert isinstance(data, list), "load_data() bir liste döndürmeli."
    assert len(data) >= 1, "En az bir dosya başarıyla işlenmeli."

def test_no_non_file_entries(real_data_dir):
    dlm = DataLoaderManager(str(real_data_dir))
    data = dlm.load_data()
    for entry in data:
        assert entry["data"] is not None, "Yüklenen veri boş olmamalı."

# ================================================
# JSON ve TXT Dosyaları Testleri
# ================================================
def test_text_processing(real_data_dir):
    dlm = DataLoaderManager(str(real_data_dir))
    data = dlm.load_data()
    text_entries = [entry for entry in data if entry["modality"] == "text"]
    
    if text_entries:
        for entry in text_entries:
            text = entry["data"]
            assert isinstance(text, str), "Metin formatı string olmalı."
            assert len(text) > 0, "Metin dosyası boş olmamalı."
    else:
        pytest.skip("Metin modality'si bulunamadı.")

# ================================================
# Ses (MP3) Dosyası Testleri
# ================================================
def test_audio_processing(real_data_dir):
    dlm = DataLoaderManager(str(real_data_dir))
    data = dlm.load_data()
    audio_entries = [entry for entry in data if entry["modality"] == "audio"]

    if audio_entries:
        for entry in audio_entries:
            assert isinstance(entry["data"], torch.Tensor), "Ses verisi uygun formatta değil."
            assert entry["data"].shape[0] > 0, "Ses dosyası boş olmamalı."
    else:
        pytest.skip("Ses modality'si bulunamadı.")

# ================================================
# Görsel (IMAGE) Dosyası Testleri
# ================================================
def test_image_processing(real_data_dir):
    dlm = DataLoaderManager(str(real_data_dir))
    data = dlm.load_data()
    image_entries = [entry for entry in data if entry["modality"] == "image"]

    if image_entries:
        for entry in image_entries:
            assert isinstance(entry["data"], torch.Tensor), "Görsel verisi torch.Tensor formatında olmalı."
            assert entry["data"].shape[0] > 0, "Görsel dosyası boş olmamalı."
    else:
        pytest.skip("Görsel modality'si bulunamadı.")

# ================================================
# Video Dosyası Testleri
# ================================================
def test_video_processing(real_data_dir):
    dlm = DataLoaderManager(str(real_data_dir))
    data = dlm.load_data()
    video_entries = [entry for entry in data if entry["modality"] == "video"]

    if video_entries:
        for entry in video_entries:
            assert isinstance(entry["data"], torch.Tensor), "Video verisi torch.Tensor formatında olmalı."
            assert entry["data"].shape[0] > 0, "Video dosyası boş olmamalı."
    else:
        pytest.skip("Video modality'si bulunamadı.")

# ================================================
# Tensor Dönüşümü Testleri
# ================================================
def test_convert_to_tensor():
    token_ids = [1, 2, 3]
    dlm = DataLoaderManager(data_directory=".")
    tensor = dlm.convert_to_tensor(token_ids, max_length=5)
    expected_tensor = torch.tensor([1, 2, 3, 0, 0], dtype=torch.long)

    assert torch.equal(tensor, expected_tensor), "convert_to_tensor() yanlış çalışıyor."

def test_tensorize_batch():
    batch = [[1, 2], [3, 4, 5]]
    dlm = DataLoaderManager(data_directory=".")
    tensor = dlm.tensorize_batch(batch, max_length=4)
    expected_tensor = torch.tensor([
        [1, 2, 0, 0],
        [3, 4, 5, 0]
    ], dtype=torch.long)

    assert torch.equal(tensor, expected_tensor), "tensorize_batch() yanlış çalışıyor."

# ================================================
# PyTorch DataLoader Testleri
# ================================================

