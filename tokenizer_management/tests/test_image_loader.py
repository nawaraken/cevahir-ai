from pathlib import Path
import pytest
from PIL import Image
import torch
from tokenizer_management.data_loader.image_loader import ImageLoader

def test_image_loader(tmp_path: Path):
    # Kırmızı renkli 100x100 boyutunda örnek bir resim oluşturup kaydediyoruz.
    img = Image.new("RGB", (100, 100), color="red")
    file = tmp_path / "test.jpg"
    img.save(str(file))
    
    loader = ImageLoader()
    output = loader.load_file(str(file))
    # Çıktı, torch.Tensor formatında olmalı.
    assert isinstance(output, torch.Tensor)
    # Özellik vektörü modelden gelen boyuta bağlı, ancak en az 1 boyutlu olmalı.
    assert len(output.shape) == 1
