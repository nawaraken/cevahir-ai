import pytest
import torch
import logging
import time

# Test edilecek modülün yolunu doğru şekilde ayarladığınızdan emin olun.
# Örneğin, projenizin kök dizininden çalışıyorsanız:
from neural_network_module.ortak_katman_module.tensor_processing_manager import TensorProcessingManager

# Fixture: Geçerli parametrelerle TensorProcessingManager örneği oluşturur.
@pytest.fixture
def tensor_processing_manager():
    # Örneğin; input_dim=128, output_dim=64, num_tasks varsayılan (tek görev) ve lr=0.0001
    return TensorProcessingManager(input_dim=128, output_dim=64, num_tasks=1, learning_rate=0.0001, log_level=logging.DEBUG)

def test_validate_dimension():
    # Geçerli ve geçersiz boyutların doğrulanması
    tpm = TensorProcessingManager(input_dim=128, output_dim=64, num_tasks=1, learning_rate=0.0001, log_level=logging.DEBUG)
    # Geçerli değerler döndürülmeli
    assert tpm._validate_dimension(128, "input_dim") == 128
    # Geçersiz değerlerde hata fırlatmalı
    with pytest.raises(ValueError):
        tpm._validate_dimension(-5, "input_dim")
    with pytest.raises(ValueError):
        tpm._validate_dimension("128", "input_dim")

def test_initialize_zeros(tensor_processing_manager):
    # initialize metodu ile sıfırlardan bir tensör oluşturulmalı.
    shape = (32, 128)
    tensor = tensor_processing_manager.initialize(shape, method="zeros")
    assert tensor.shape == torch.Size(shape)
    assert torch.all(tensor == 0)

def test_project(tensor_processing_manager):
    # Projeksiyon katmanını test edelim: Giriş boyutu input_dim, çıkış boyutu output_dim olmalı.
    input_tensor = torch.randn(32, tensor_processing_manager.input_dim)
    projected = tensor_processing_manager.project(input_tensor)
    assert projected.shape == (32, tensor_processing_manager.output_dim)
    assert projected.dtype == input_tensor.dtype

def test_validate_tensor_error(tensor_processing_manager):
    # _validate_tensor metodu, geçersiz giriş tipinde hata fırlatmalı.
    with pytest.raises(TypeError):
        tensor_processing_manager._validate_tensor("not a tensor", "test_tensor")
    with pytest.raises(ValueError):
        tensor_processing_manager._validate_tensor(torch.tensor([]), "empty_tensor")

def test_log_tensor_stats(caplog, tensor_processing_manager):
    # _log_tensor_stats metodunun doğru log çıktısı verdiğini kontrol edelim.
    caplog.set_level(logging.DEBUG)
    dummy_tensor = torch.randn(16, 128)
    tensor_processing_manager._log_tensor_stats(dummy_tensor, "Dummy Tensor")
    # Loglarda tensor shape, min, max, mean ve std bilgileri yer almalı.
    log_text = caplog.text
    assert "Dummy Tensor stats:" in log_text
    assert "shape=" in log_text
    assert "min=" in log_text
    assert "max=" in log_text
    assert "mean=" in log_text
    assert "std=" in log_text

def test_log_execution(caplog, tensor_processing_manager):
    # _log_execution metodunun doğru log çıktısını verdiğini kontrol edelim.
    caplog.set_level(logging.DEBUG)
    dummy_tensor = torch.randn(8, 64)
    tensor_processing_manager._log_execution("TestOperation", dummy_tensor)
    log_text = caplog.text
    assert "TestOperation completed successfully." in log_text
    assert "Tensor shape:" in log_text

def test_invalid_initialize_method(tensor_processing_manager):
    # initialize metoduna geçersiz method adı verildiğinde hata fırlatmalı.
    with pytest.raises(Exception):
        tensor_processing_manager.initialize((32, 128), method="invalid_method")

def test_project_invalid_input(tensor_processing_manager):
    # project metoduna geçersiz input tipi verildiğinde hata fırlatmalı.
    with pytest.raises(TypeError):
        tensor_processing_manager.project("invalid_input")

def test_performance_of_optimize(tensor_processing_manager):
    # Optimize metodunun makul sürede çalıştığını test edelim.
    tensor = torch.randn(64, 128)
    gradients = torch.randn(64, 128)
    start_time = time.time()
    _ = tensor_processing_manager.optimize(tensor, gradients, method="sgd")
    duration = time.time() - start_time
    # Örneğin, optimize 0.1 saniyeden kısa sürmeli (bu değeri ihtiyaca göre ayarlayabilirsiniz).
    assert duration < 0.1, f"Optimize işlemi çok yavaş: {duration:.6f} saniye"

def test_optimize_sgd(tensor_processing_manager):
    # Basit SGD optimizasyonunu test edelim: 
    # Çıktı, T - lr*G olmalı.
    T = torch.ones(10, 128)
    G = torch.full((10, 128), 0.5)
    lr = tensor_processing_manager.learning_rate  # 0.0001
    optimized = tensor_processing_manager.optimize(T, G, method="sgd")
    expected = T - lr * G
    assert torch.allclose(optimized, expected, atol=1e-6)

def test_optimize_adam(tensor_processing_manager):
    # Adam optimizasyonu test edilebilir: burada daha çok şekil ve çıkış tipi kontrolü yapalım.
    T = torch.ones(10, 128)
    G = torch.full((10, 128), 0.5)
    optimized = tensor_processing_manager.optimize(T, G, method="adam", beta1=0.9, beta2=0.999, epsilon=1e-8, t=1)
    # Adam güncellemesi karmaşık olacağından, sadece şekil ve tip kontrolü yapalım.
    assert optimized.shape == T.shape
    assert optimized.dtype == T.dtype