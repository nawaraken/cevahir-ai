import pytest
import torch
import logging
from src.neural_network import CevahirNeuralNetwork

# **Örnek Yapılandırma**
CONFIG = {
    "vocab_size": 150000,
    "embed_dim": 64,
    "seq_proj_dim": 2048,
    "num_heads": 16,
    "dropout": 0.1,
    "learning_rate": 0.00001,
    "attention_type": "multi_head",
    "normalization_type": "layer_norm",
}

# **Logger Ayarı**
logger = logging.getLogger("test_neural_network")
logger.setLevel(logging.DEBUG)

@pytest.fixture
def neural_network():
    """
    Varsayılan bir Cevahir Neural Network örneği oluşturur.
    """
    return CevahirNeuralNetwork(
        vocab_size=CONFIG["vocab_size"],
        embed_dim=CONFIG["embed_dim"],
        seq_proj_dim=CONFIG["seq_proj_dim"],
        num_heads=CONFIG["num_heads"],
        attention_type=CONFIG["attention_type"],
        normalization_type=CONFIG["normalization_type"],
        dropout=CONFIG["dropout"],
        learning_rate=CONFIG["learning_rate"],
    )

def test_initialization(neural_network):
    """
    Modelin bileşenlerinin düzgün şekilde başlatıldığını doğrular.
    """
    logger.info("[TEST] Model bileşen başlatma testi başladı.")
    
    assert neural_network.dil_katmani is not None, "DilKatmani başlatılamadı!"
    assert neural_network.layer_processor is not None, "LayerProcessor başlatılamadı!"
    assert neural_network.memory_manager is not None, "MemoryManager başlatılamadı!"
    assert neural_network.tensor_processing_manager is not None, "TensorProcessingManager başlatılamadı!"
    
    logger.info("[TEST] Model bileşen başlatma testi başarılı.")

def test_forward_pass(neural_network):
    """
    Modelin ileri yayılım işlemi başarıyla çalışıyor mu test eder.
    """
    logger.info("[TEST] İleri yayılım testi başladı.")

    batch_size, seq_len = 2, 10
    input_tensor = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    output, attn_weights = neural_network(input_tensor)

    # ** Çıktının Tensor Olduğunu Doğrula**
    assert isinstance(output, torch.Tensor), "Çıktı bir tensör olmalı!"
    assert isinstance(attn_weights, torch.Tensor) or attn_weights is None, "Attention weights yanlış türde!"

    # ** Çıktı Boyutunu Test Et**
    expected_output_shape = (batch_size, seq_len, CONFIG["vocab_size"])
    assert output.shape == torch.Size(expected_output_shape), \
        f"Çıktı boyutu hatalı! Beklenen: {expected_output_shape}, Gerçek: {output.shape}"

    # ** Çıktı Veri Türünü Test Et**
    assert output.dtype == torch.float32, f"Çıktı veri türü hatalı! Beklenen: float32, Gerçek: {output.dtype}"

    logger.info("[TEST] İleri yayılım testi başarıyla tamamlandı.")

def test_attention_mechanism(neural_network):
    """
    Dikkat mekanizmasının çıktısının doğru şekle sahip olup olmadığını test eder.
    """
    logger.info("[TEST] Dikkat mekanizması testi başladı.")

    batch_size, seq_len = 2, 10
    input_tensor = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    _, attn_weights = neural_network(input_tensor)

    if attn_weights is not None:
        expected_attn_shape = (batch_size, CONFIG["num_heads"], seq_len, seq_len)
        assert attn_weights.shape == torch.Size(expected_attn_shape), \
            f"Dikkat ağırlıklarının boyutu hatalı! Beklenen: {expected_attn_shape}, Gerçek: {attn_weights.shape}"

        # ** Dikkat Ağırlıkları 0-1 Aralığında mı?**
        assert torch.all((attn_weights >= 0) & (attn_weights <= 1)), "Dikkat ağırlıkları 0-1 aralığında değil!"

    logger.info("[TEST] Dikkat mekanizması testi başarıyla tamamlandı.")

def test_memory_manager(neural_network):
    """
    Bellek yöneticisinin çıktıyı başarıyla sakladığını test eder.
    """
    logger.info("[TEST] Bellek yönetimi testi başladı.")

    batch_size, seq_len = 2, 10
    input_tensor = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    output, _ = neural_network(input_tensor)

    stored_output = neural_network.memory_manager.retrieve("final_output")

    assert stored_output is not None, "Bellek yöneticisi çıktıyı kaydetmedi!"
    assert torch.equal(output, stored_output), "MemoryManager çıktısı ile model çıktısı eşleşmiyor!"

    logger.info("[TEST] Bellek yönetimi testi başarıyla tamamlandı.")
