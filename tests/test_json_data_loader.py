import os
import pytest
from pathlib import Path
from tokenizer_management.data_loader.json_loader import JSONLoader, JSONFormatError, JSONFileNotFoundError, JSONLoaderError

# Eğitim verilerinin bulunduğu klasörü belirten fixture
@pytest.fixture(scope="module")
def education_dir():
    # Örneğin proje kök dizininde "education" adlı bir klasör olduğunu varsayıyoruz.
    # Kendi proje yapınıza göre yolu ayarlayabilirsiniz.
    return Path(__file__).resolve().parent.parent / "education"

def test_valid_json(education_dir):
    """
    Geçerli JSON dosyasının başarılı şekilde yüklendiğini ve metin çıktısı üretildiğini test eder.
    """
    loader = JSONLoader(max_depth=1000)
    valid_file = education_dir / "ksakjdka.json"  # Bu dosya geçerli JSON içeriğine sahip olmalıdır.
    text = loader.load_file(str(valid_file))
    assert isinstance(text, str), "Çıktı metin formatında olmalıdır."
    assert text.strip() != "", "Çıktı boş olmamalıdır."

def test_invalid_json(education_dir):
    """
    Geçersiz JSON dosyasında (örneğin, yapı hatalı veya eksik anahtarlar) JSONFormatError fırlatıldığını doğrular.
    """
    loader = JSONLoader(max_depth=1000)
    invalid_file = education_dir / "dyhjkgashjkdqa.json"  # Bu dosyada JSON format hatası olmalıdır.
    with pytest.raises(JSONFormatError) as excinfo:
        loader.load_file(str(invalid_file))
    assert "JSON format hatası" in str(excinfo.value)

def test_missing_json(education_dir):
    """
    Var olmayan bir JSON dosyası için JSONFileNotFoundError fırlatıldığını doğrular.
    """
    loader = JSONLoader(max_depth=1000)
    missing_file = education_dir / "non_existent.json"
    with pytest.raises(JSONFileNotFoundError):
        loader.load_file(str(missing_file))
