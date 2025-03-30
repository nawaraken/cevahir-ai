import pytest
import torch
import time
import logging
from neural_network_module.dil_katmani import DilKatmani

# Fixture: DilKatmani örneği oluşturuyoruz.
@pytest.fixture
def dil_katmani():
    return DilKatmani(
        vocab_size=5000,
        embed_dim=128,
        seq_proj_dim=64,
        embed_init_method="xavier",
        seq_init_method="xavier",
        log_level=logging.DEBUG,
        dropout=0.1
    )

# Test 1: Forward pass sonrası çıkışın beklenen boyutta olduğunu doğrular.
def test_forward_output_shape(dil_katmani):
    batch_size, seq_len = 32, 50
    input_tensor = torch.randint(0, 5000, (batch_size, seq_len), dtype=torch.long)
    output = dil_katmani(input_tensor)
    assert output.shape == (batch_size, seq_len, 64), f"Expected shape {(batch_size, seq_len, 64)}, got {output.shape}"

# Test 2: Positional encoding'in, embedding üzerine eklenip farklılık yarattığını doğrular.
def test_positional_encoding_effect(dil_katmani):
    batch_size, seq_len = 4, 10
    input_tensor = torch.randint(0, 5000, (batch_size, seq_len), dtype=torch.long)
    embedded = dil_katmani.language_embedding(input_tensor)
    encoded = dil_katmani.positional_encoding(embedded)
    # Encoded tensor'ın, orijinal embedding'den farklı olması beklenir.
    assert not torch.equal(embedded, encoded), "Positional encoding did not modify the embeddings."

# Test 3: LayerNorm sonrası tensorun ortalamasının 0 ve std'sinin 1'e yakın olduğunu kontrol eder.
def test_layer_norm_statistics(dil_katmani):
    batch_size, seq_len = 4, 10
    input_tensor = torch.randint(0, 5000, (batch_size, seq_len), dtype=torch.long)
    embedded = dil_katmani.language_embedding(input_tensor)
    encoded = dil_katmani.positional_encoding(embedded)
    normalized = dil_katmani.layer_norm(encoded)
    mean_val = normalized.mean().item()
    std_val = normalized.std().item()
    assert abs(mean_val) < 1e-3, f"LayerNorm mean is not near 0: {mean_val}"
    assert abs(std_val - 1) < 1e-2, f"LayerNorm std is not near 1: {std_val}"

# Test 4: Eğitim modunda dropout'un farklı forward pass'lerde farklı sonuçlar ürettiğini kontrol eder.
def test_dropout_randomness_training(dil_katmani):
    dil_katmani.train()  # Eğitim modunda dropout aktif
    batch_size, seq_len = 4, 10
    input_tensor = torch.randint(0, 5000, (batch_size, seq_len), dtype=torch.long)
    output1 = dil_katmani(input_tensor)
    output2 = dil_katmani(input_tensor)
    # Eğitim modunda dropout aktif olduğundan, çıktılar arasında fark olması gerekir.
    assert not torch.allclose(output1, output2, atol=1e-5), "Dropout in training mode is not introducing randomness."

# Test 5: Eval modunda dropout kapalıyken deterministik sonuçlar üretilmeli.
def test_eval_mode_determinism(dil_katmani):
    dil_katmani.eval()  # Dropout devre dışı
    batch_size, seq_len = 4, 10
    input_tensor = torch.randint(0, 5000, (batch_size, seq_len), dtype=torch.long)
    output1 = dil_katmani(input_tensor)
    output2 = dil_katmani(input_tensor)
    assert torch.allclose(output1, output2, atol=1e-6), "Outputs are not deterministic in eval mode."

# Test 6: SeqProjection katmanının çıkış değerlerinin makul bir aralıkta olduğunu doğrular.
def test_seq_projection_output_range(dil_katmani):
    dil_katmani.eval()
    batch_size, seq_len = 4, 15
    input_tensor = torch.randint(0, 5000, (batch_size, seq_len), dtype=torch.long)
    output = dil_katmani(input_tensor)
    # Beklenen: SeqProjection katmanı çıkışı makul bir aralıkta (örneğin -10 ile 10)
    assert output.min() > -10 and output.max() < 10, "SeqProjection output is out of expected range (-10, 10)."

# Test 7: Yanlış giriş tipi verildiğinde (örneğin float tensor) hata alınmalı.
def test_invalid_input_type_dilkatmani():
    model = DilKatmani(
        vocab_size=5000,
        embed_dim=128,
        seq_proj_dim=64,
        embed_init_method="xavier",
        seq_init_method="xavier",
        log_level=logging.DEBUG,
        dropout=0.1
    )
    with pytest.raises(Exception):
        model(torch.rand(32, 50))  # Float tensor, beklenen long tensor

# Test 8: Yanlış giriş boyutunda tensor (örn. 3D tensor) verildiğinde hata alınmalı.
def test_invalid_input_dimension_dilkatmani():
    model = DilKatmani(
        vocab_size=5000,
        embed_dim=128,
        seq_proj_dim=64,
        embed_init_method="xavier",
        seq_init_method="xavier",
        log_level=logging.DEBUG,
        dropout=0.1
    )
    with pytest.raises(Exception):
        model(torch.randint(0, 5000, (32, 50, 10), dtype=torch.long))

# Test 9: Loglama çıktılarının beklendiği gibi oluşturulup oluşturulmadığını (caplog) kontrol eder.
def test_logging_output_dilkatmani(caplog):
    caplog.set_level(logging.DEBUG)
    model = DilKatmani(
        vocab_size=5000,
        embed_dim=128,
        seq_proj_dim=64,
        embed_init_method="xavier",
        seq_init_method="xavier",
        log_level=logging.DEBUG,
        dropout=0.1
    )
    batch_size, seq_len = 4, 10
    input_tensor = torch.randint(0, 5000, (batch_size, seq_len), dtype=torch.long)
    _ = model(input_tensor)
    # Kontrol: Loglarda "After LanguageEmbedding", "After PositionalEncoding", "After LayerNorm", "After Dropout", "After SeqProjection" gibi ifadelerin geçmesi beklenir.
    assert "After LanguageEmbedding" in caplog.text or "LanguageEmbedding" in caplog.text
    assert "After PositionalEncoding" in caplog.text or "PositionalEncoding" in caplog.text
    assert "After LayerNorm" in caplog.text or "LayerNorm" in caplog.text
    assert "After Dropout" in caplog.text or "Dropout" in caplog.text
    assert "After SeqProjection" in caplog.text or "SeqProjection" in caplog.text

# Test 10: Sabit girişler için, dropout kapalıyken çıktının deterministik olduğunu doğrular.
def test_determinism_fixed_input():
    model = DilKatmani(
        vocab_size=5000,
        embed_dim=128,
        seq_proj_dim=64,
        embed_init_method="xavier",
        seq_init_method="xavier",
        log_level=logging.DEBUG,
        dropout=0.0  # Dropout kapalı
    )
    model.eval()
    torch.manual_seed(42)
    batch_size, seq_len = 4, 10
    input_tensor = torch.randint(0, 5000, (batch_size, seq_len), dtype=torch.long)
    output1 = model(input_tensor)
    output2 = model(input_tensor)
    assert torch.allclose(output1, output2, atol=1e-6), "Determinism failed: outputs differ with fixed input and dropout disabled."