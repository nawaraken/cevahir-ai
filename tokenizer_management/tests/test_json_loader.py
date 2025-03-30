import os
from pathlib import Path
import pytest
from tokenizer_management.data_loader.json_loader import JSONLoader, JSONFormatError, JSONFileNotFoundError

@pytest.fixture(scope="module")
def education_dir():
    # 🔥 education, tokenizer_management ile aynı seviyedeyse bir üst klasörden erişelim
    path = Path(__file__).resolve().parent.parent.parent / "education"
    print(f"Eğitim klasörü yolu: {path}")
    assert path.exists(), f"Education klasörü bulunamadı: {path}"
    return path

def test_valid_json(education_dir):
    loader = JSONLoader(max_depth=1000)
    valid_file = education_dir / "ksakjdka.json"
    print(f"Test edilen JSON dosyası: {valid_file}")
    assert valid_file.exists(), f"Dosya mevcut değil: {valid_file}"
    
    text = loader.load_file(str(valid_file))
    assert isinstance(text, str), "Çıktı metin formatında olmalıdır."
    assert text.strip() != "", "Çıktı boş olmamalıdır."

def test_invalid_json(education_dir):
    loader = JSONLoader(max_depth=1000)
    invalid_file = education_dir / "dyhjkgashjkdqa.json"
    print(f"Test edilen JSON dosyası: {invalid_file}")
    assert invalid_file.exists(), f"Dosya mevcut değil: {invalid_file}"
    
    with pytest.raises(JSONFormatError):
        loader.load_file(str(invalid_file))

def test_missing_json(education_dir):
    loader = JSONLoader(max_depth=1000)
    missing_file = education_dir / "non_existent.json"
    
    with pytest.raises(JSONFileNotFoundError):
        loader.load_file(str(missing_file))
