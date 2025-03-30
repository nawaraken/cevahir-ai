import pytest
import torch
import logging
from neural_network_module.ortak_katman_module.memory_manager import MemoryManager

@pytest.fixture
def memory_manager():
    return MemoryManager(init_type="xavier", normalization_type="layer_norm", scaling_type="min_max", log_level=logging.DEBUG)

def test_allocate_memory(memory_manager):
    tensor = memory_manager.allocate_memory((10, 20, 64))
    assert tensor.shape == (10, 20, 64)
    assert tensor.dtype == torch.float32
    assert tensor.device == torch.device('cpu')

def test_initialize_memory(memory_manager):
    tensor = memory_manager.allocate_memory((10, 20, 64))
    initialized_tensor = memory_manager.initialize_memory(tensor)
    assert initialized_tensor.shape == tensor.shape
    assert initialized_tensor.dtype == tensor.dtype
    assert initialized_tensor.device == tensor.device

def test_normalize_memory(memory_manager):
    tensor = memory_manager.allocate_memory((10, 20, 64))
    initialized_tensor = memory_manager.initialize_memory(tensor)
    normalized_tensor = memory_manager.normalize_memory(initialized_tensor)
    assert normalized_tensor.shape == tensor.shape
    assert normalized_tensor.dtype == tensor.dtype
    assert normalized_tensor.device == tensor.device

def test_scale_memory(memory_manager):
    tensor = memory_manager.allocate_memory((10, 20, 64))
    initialized_tensor = memory_manager.initialize_memory(tensor)
    normalized_tensor = memory_manager.normalize_memory(initialized_tensor)
    scaled_tensor = memory_manager.scale_memory(normalized_tensor)
    assert scaled_tensor.shape == tensor.shape
    assert scaled_tensor.dtype == tensor.dtype
    assert scaled_tensor.device == tensor.device

def test_optimize_memory(memory_manager):
    tensor = memory_manager.allocate_memory((10, 20, 64))
    initialized_tensor = memory_manager.initialize_memory(tensor)
    normalized_tensor = memory_manager.normalize_memory(initialized_tensor)
    scaled_tensor = memory_manager.scale_memory(normalized_tensor)
    optimized_tensor = memory_manager.optimize_memory(scaled_tensor)
    assert optimized_tensor.shape == tensor.shape
    assert optimized_tensor.dtype == tensor.dtype
    assert optimized_tensor.device == tensor.device

def test_bridge_attention(memory_manager):
    memory_tensor = memory_manager.allocate_memory((10, 20, 64))
    attention_tensor = torch.randn(10, 20, 64)
    bridged_tensor = memory_manager.bridge_attention(memory_tensor, attention_tensor)
    assert bridged_tensor.shape == (10, 20, 128)
    assert bridged_tensor.dtype == memory_tensor.dtype
    assert bridged_tensor.device == memory_tensor.device

def test_memory_manager_with_different_initialization():
    manager = MemoryManager(init_type="he", normalization_type="batch_norm", scaling_type="standard", log_level=logging.DEBUG)
    tensor = manager.allocate_memory((10, 20, 64))
    tensor = manager.initialize_memory(tensor)
    tensor = manager.normalize_memory(tensor)
    tensor = manager.scale_memory(tensor)
    tensor = manager.optimize_memory(tensor)
    attention_tensor = torch.randn(10, 20, 64)
    bridged_tensor = manager.bridge_attention(tensor, attention_tensor)
    assert bridged_tensor.shape == (10, 20, 128)

def test_memory_manager_with_different_normalization():
    manager = MemoryManager(init_type="normal", normalization_type="instance_norm", scaling_type="robust", log_level=logging.DEBUG)
    tensor = manager.allocate_memory((10, 20, 64))
    tensor = manager.initialize_memory(tensor)
    tensor = manager.normalize_memory(tensor)
    tensor = manager.scale_memory(tensor)
    tensor = manager.optimize_memory(tensor)
    attention_tensor = torch.randn(10, 20, 64)
    bridged_tensor = manager.bridge_attention(tensor, attention_tensor)
    assert bridged_tensor.shape == (10, 20, 128)

def test_memory_manager_with_different_scaling():
    manager = MemoryManager(init_type="xavier", normalization_type="group_norm", scaling_type="min_max", scale_range=(-1, 1), log_level=logging.DEBUG)
    tensor = manager.allocate_memory((10, 20, 64))
    tensor = manager.initialize_memory(tensor)
    tensor = manager.normalize_memory(tensor)
    tensor = manager.scale_memory(tensor)
    tensor = manager.optimize_memory(tensor)
    attention_tensor = torch.randn(10, 20, 64)
    bridged_tensor = manager.bridge_attention(tensor, attention_tensor)
    assert bridged_tensor.shape == (10, 20, 128)

# Yeni Test Metodları

def test_memory_initialization_with_different_dtypes():
    manager = MemoryManager(init_type="xavier", log_level=logging.DEBUG)
    tensor = manager.allocate_memory((10, 20, 64), dtype=torch.float64)
    initialized_tensor = manager.initialize_memory(tensor)
    assert initialized_tensor.dtype == torch.float64

def test_memory_normalization_with_different_shapes():
    manager = MemoryManager(normalization_type="batch_norm", log_level=logging.DEBUG)
    tensor = manager.allocate_memory((32, 10, 128))
    initialized_tensor = manager.initialize_memory(tensor)
    normalized_tensor = manager.normalize_memory(initialized_tensor)
    assert normalized_tensor.shape == (32, 10, 128)

def test_memory_scaling_with_different_scale_ranges():
    manager = MemoryManager(scaling_type="min_max", scale_range=(0, 1), log_level=logging.DEBUG)
    tensor = manager.allocate_memory((10, 20, 64))
    initialized_tensor = manager.initialize_memory(tensor)
    normalized_tensor = manager.normalize_memory(initialized_tensor)
    scaled_tensor = manager.scale_memory(normalized_tensor)
    assert scaled_tensor.min() >= 0
    assert scaled_tensor.max() <= 1

def test_memory_optimizer_with_edge_cases():
    manager = MemoryManager(log_level=logging.DEBUG)
    tensor = manager.allocate_memory((10, 20, 64))
    tensor.fill_(0)  # Edge case: all elements are zero
    optimized_tensor = manager.optimize_memory(tensor)
    assert torch.all(optimized_tensor == 0)

def test_bridge_attention_with_masks():
    manager = MemoryManager(log_level=logging.DEBUG)
    memory_tensor = manager.allocate_memory((10, 20, 64))
    attention_tensor = torch.randn(10, 20, 64)
    memory_mask = torch.ones(10, 20, dtype=torch.bool)
    attention_mask = torch.ones(10, 20, dtype=torch.bool)
    bridged_tensor = manager.bridge_attention(memory_tensor, attention_tensor, memory_mask, attention_mask)
    assert bridged_tensor.shape == (10, 20, 128)
    assert bridged_tensor.dtype == memory_tensor.dtype
    assert bridged_tensor.device == memory_tensor.device


def test_memory_allocation_with_different_dtypes():
    """
    Farklı veri tiplerinde bellek tahsis edilebildiğini doğrular.
    """
    manager = MemoryManager(log_level=logging.DEBUG)
    
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        tensor = manager.allocate_memory((10, 20, 64), dtype=dtype)
        assert tensor.dtype == dtype, f"Memory allocation failed for dtype {dtype}"


def test_memory_normalization_with_extreme_values():
    """
    Çok büyük ve küçük değerler içeren tensörlerin normalizasyon işleminin bozulmadığını kontrol eder.
    """
    manager = MemoryManager(normalization_type="batch_norm", log_level=logging.DEBUG)

    tensor = manager.allocate_memory((10, 20, 64))
    tensor.fill_(1e9)  # Aşırı büyük değerler
    normalized_tensor = manager.normalize_memory(tensor)
    assert torch.isfinite(normalized_tensor).all(), "Normalization failed for large values!"

    tensor.fill_(-1e9)  # Aşırı küçük değerler
    normalized_tensor = manager.normalize_memory(tensor)
    assert torch.isfinite(normalized_tensor).all(), "Normalization failed for small values!"


def test_memory_scaling_preserves_structure():
    """
    Ölçeklendirme işleminden önce ve sonra tensörlerin yapısının korunduğunu kontrol eder.
    """
    manager = MemoryManager(scaling_type="min_max", scale_range=(-1, 1), log_level=logging.DEBUG)

    tensor = manager.allocate_memory((10, 20, 64))
    scaled_tensor = manager.scale_memory(tensor)

    assert scaled_tensor.shape == tensor.shape, "Scaling changed tensor shape!"
    assert scaled_tensor.min() >= -1 and scaled_tensor.max() <= 1, "Scaling out of expected range!"


def test_memory_optimizer_does_not_increase_memory():
    """
    Optimizasyon işleminin bellek kullanımını artırmadığını doğrular.
    """
    manager = MemoryManager(log_level=logging.DEBUG)

    tensor = manager.allocate_memory((10, 20, 64))
    memory_before = tensor.element_size() * tensor.nelement()
    
    optimized_tensor = manager.optimize_memory(tensor)
    memory_after = optimized_tensor.element_size() * optimized_tensor.nelement()

    assert memory_after <= memory_before, "Memory optimization increased memory usage!"


def test_bridge_attention_with_various_masks():
    """
    Farklı maske türleri ile attention köprüleme işlemini doğrular.
    """
    manager = MemoryManager(log_level=logging.DEBUG)

    memory_tensor = manager.allocate_memory((10, 20, 64))
    attention_tensor = torch.randn(10, 20, 64)

    # Tamamen sıfırdan oluşan maske
    zero_mask = torch.zeros(10, 20, dtype=torch.bool)

    # Tamamen birden oluşan maske
    one_mask = torch.ones(10, 20, dtype=torch.bool)

    # Rastgele değerlerden oluşan maske
    random_mask = torch.randint(0, 2, (10, 20), dtype=torch.bool)

    for mask in [zero_mask, one_mask, random_mask]:
        bridged_tensor = manager.bridge_attention(memory_tensor, attention_tensor, memory_mask=mask, attention_mask=mask)
        assert bridged_tensor.shape == (10, 20, 128), "Bridge attention failed for mask!"


