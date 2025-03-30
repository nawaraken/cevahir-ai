from pathlib import Path
import pytest
from tokenizer_management.data_loader.txt_loader import TXTLoader

def test_txt_loader_normalization(tmp_path: Path):
    # Fazla boşluk, satır sonu ve sekme içeren örnek içerik.
    content = "  This is    a  test.\nNew Line\tand spaces.  "
    file = tmp_path / "test.txt"
    file.write_text(content, encoding="utf-8")

    loader = TXTLoader()
    output = loader.load_file(str(file))
    # ' '.join(content.split()) ile normalizasyon yapılacağından:
    expected = "This is a test. New Line and spaces."
    assert output == expected
