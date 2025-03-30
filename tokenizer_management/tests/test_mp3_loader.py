import numpy as np
import pytest
from tokenizer_management.data_loader.mp3_loader import MP3Loader

# Dummy fonksiyonlar oluşturup, librosa fonksiyonlarını monkeypatch ile değiştirelim.
def dummy_librosa_load(file_path, sr):
    # Örnek: 1 saniyelik sinüs dalgası üreteceğiz.
    t = np.linspace(0, 1, sr)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    return y, sr

def dummy_librosa_feature_mfcc(y, sr, n_mfcc):
    # Dummy: (n_mfcc, 10) boyutunda sabit değerlerden oluşan bir matris döndürsün.
    return np.ones((n_mfcc, 10))

def test_mp3_loader(monkeypatch):
    loader = MP3Loader(sr=22050, n_mfcc=13)
    monkeypatch.setattr("tokenizer_management.data_loader.mp3_loader.librosa.load", dummy_librosa_load)
    monkeypatch.setattr("tokenizer_management.data_loader.mp3_loader.librosa.feature.mfcc", dummy_librosa_feature_mfcc)
    
    output = loader.load_file("dummy.mp3")
    # Çıktı, n_mfcc sayıda eleman içeren numpy array olmalı.
    assert isinstance(output, np.ndarray)
    assert output.shape == (13,)
