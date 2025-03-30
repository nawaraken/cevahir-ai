import pytest
import time
import torch
import logging
from neural_network_module.ortak_katman_module.neural_layer_processor import NeuralLayerProcessor

# Fixture'lar ile farklı attention türlerine sahip NeuralLayerProcessor örnekleri oluşturuyoruz.
@pytest.fixture
def nlp_multi_head():
    return NeuralLayerProcessor(
        embed_dim=512,
        num_heads=32,
        attention_type="multi_head",
        dropout=0.2,
        debug=False,
        normalization_type="layer_norm",
        scaling_strategy="sqrt",
        scaling_method="softmax",
        verbose=True
    )

@pytest.fixture
def nlp_self():
    return NeuralLayerProcessor(
        embed_dim=512,
        num_heads=32,
        attention_type="self",
        dropout=0.2,
        debug=False,
        normalization_type="layer_norm",
        scaling_strategy="sqrt",
        scaling_method="softmax",
        verbose=True
    )

@pytest.fixture
def nlp_cross():
    return NeuralLayerProcessor(
        embed_dim=512,
        num_heads=32,
        attention_type="cross",
        dropout=0.2,
        debug=False,
        normalization_type="layer_norm",
        scaling_strategy="sqrt",
        scaling_method="softmax",
        verbose=True
    )

# Test 1: Geçerli multi_head initialization
def test_initialization_multi_head(nlp_multi_head):
    model = nlp_multi_head
    assert model.embed_dim == 512
    assert model.num_heads == 32
    assert hasattr(model, 'multi_head_attention')



# Test 3: Multi-head attention için forward pass çıktısı ve dikkat ağırlıkları
def test_forward_multi_head(nlp_multi_head):
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    output, attn_weights = nlp_multi_head(query, key, value)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights is not None



# Test 5: Cross-attention için forward pass çıktısı
def test_forward_cross(nlp_cross):
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    output, attn_weights = nlp_cross(query, key, value)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights is not None

# Test 6: Forward pass sırasında NaN veya Inf değerin oluşmadığını kontrol et
def test_forward_no_nan_inf(nlp_multi_head):
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    output, _ = nlp_multi_head(query, key, value)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

# Test 7: Eğitim modunda dropout etkisinin olduğunu doğrula (aynı girdiler farklı çıktılar üretmeli)
def test_dropout_training(nlp_multi_head):
    nlp_multi_head.train()
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    output1, _ = nlp_multi_head(query, key, value)
    output2, _ = nlp_multi_head(query, key, value)
    with pytest.raises(AssertionError):
        # Beklenen: farklı forward pass'ler aynı olmamalı
        assert torch.allclose(output1, output2, atol=1e-5)

# Test 8: Eval modunda dropout kapalıyken deterministik çıktılar üretilmeli
def test_eval_mode_determinism(nlp_multi_head):
    nlp_multi_head.eval()
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    output1, _ = nlp_multi_head(query, key, value)
    output2, _ = nlp_multi_head(query, key, value)
    assert torch.allclose(output1, output2, atol=1e-5)

# Test 9: initialize_attention metodunun giriş tensörü ile aynı boyutta başlatılmış tensör döndürdüğünü kontrol et
def test_initialize_attention(nlp_multi_head):
    batch_size, seq_len, embed_dim = 4, 10, 512512
    inputs = torch.rand(batch_size, seq_len, embed_dim)
    initialized = nlp_multi_head.initialize_attention(inputs)
    assert initialized.shape == inputs.shape

# Test 10: normalize_attention metodunun aynı boyutta tensor döndürdüğünü kontrol et
def test_normalize_attention(nlp_multi_head):
    batch_size, seq_len, embed_dim = 4, 10, 512
    inputs = torch.rand(batch_size, seq_len, embed_dim)
    normalized = nlp_multi_head.normalize_attention(inputs)
    assert normalized.shape == inputs.shape


# Test 12: _validate_tensor metodunun yanlış türde veri aldığında hata fırlattığını doğrula
def test_validate_tensor_error(nlp_multi_head):
    with pytest.raises(TypeError):
        nlp_multi_head._validate_tensor("not a tensor", name="test")

# Test 13: Yanlış boyutta input verildiğinde forward metodunun hata fırlattığını doğrula
def test_forward_wrong_dimension(nlp_multi_head):
    wrong_input = torch.rand(4, 10)  # 2D tensor
    with pytest.raises(ValueError):
        nlp_multi_head(wrong_input)

# Test 14: optimize_attention metodunun giriş tensörü ile aynı boyutta çıktı ürettiğini kontrol et
def test_optimize_attention(nlp_multi_head):
    batch_size, seq_len, embed_dim = 4, 10, 512
    outputs = torch.rand(batch_size, seq_len, embed_dim)
    optimized = nlp_multi_head.attention_optimizer.forward(outputs)
    assert optimized.shape == outputs.shape

# Test 15: Forward pass performansının küçük girdilerde makul sürede tamamlandığını doğrula (< 0.1 saniye)
def test_forward_performance(nlp_multi_head):
    nlp_multi_head.eval()
    batch_size, seq_len, embed_dim = 16, 50, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    start = time.time()
    output, _ = nlp_multi_head(query, key, value)
    duration = time.time() - start
    assert duration < 0.1

# Test 16: Cross-attention'da 4D mask'in doğru sıkıştırıldığını doğrula
def test_cross_attention_mask_squeeze(nlp_cross):
    batch_size, seq_len, embed_dim = 4, 10, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    mask = torch.randint(0, 2, (batch_size, 1, 1, seq_len)).float()
    output, attn_weights = nlp_cross(query, key, value, mask=mask)
    assert output.shape == (batch_size, seq_len, embed_dim)

# Test 17: Eval modunda birden fazla forward pass'te çıkışların stabil olduğunu doğrula
def test_forward_stability(nlp_multi_head):
    nlp_multi_head.eval()
    batch_size, seq_len, embed_dim = 8, 20, 512
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    outputs = [nlp_multi_head(query, key, value)[0] for _ in range(5)]
    stacked = torch.stack(outputs)
    variance = torch.var(stacked, dim=0)
    assert torch.max(variance) < 1e-5

# Test 18: Forward pass çıktılarının mantıklı bir numerik aralıkta olduğunu kontrol et (örneğin -10 ile 10 arasında)
def test_forward_output_range(nlp_multi_head):
    nlp_multi_head.eval()
    batch_size, seq_len, embed_dim = 4, 15, 512
    query = torch.rand(batch_size, seq_len, embed_dim) * 2 - 1
    key = torch.rand(batch_size, seq_len, embed_dim) * 2 - 1
    value = torch.rand(batch_size, seq_len, embed_dim) * 2 - 1
    output, _ = nlp_multi_head(query, key, value)
    assert output.min() > -10 and output.max() < 10

# Test 19: Farklı sequence uzunluklarına sahip batch'ler işlenirken çıktı boyutlarının doğru olduğunu kontrol et
def test_varying_sequence_lengths(nlp_multi_head):
    nlp_multi_head.eval()
    batch_size = 4
    embed_dim = 512
    seq_lens = [10, 15, 20, 12]
    outputs = []
    for seq_len in seq_lens:
        query = torch.rand(1, seq_len, embed_dim)
        key = torch.rand(1, seq_len, embed_dim)
        value = torch.rand(1, seq_len, embed_dim)
        output, _ = nlp_multi_head(query, key, value)
        outputs.append(output)
        assert output.shape == (1, seq_len, embed_dim)
    concatenated = torch.cat(outputs, dim=1)
    assert concatenated.shape[1] == sum(seq_lens)
