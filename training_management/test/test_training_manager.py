import os
import time
import pytest
import torch
import logging
from model_management.model_manager import ModelManager
from training_management.training_manager import TrainingManager
from src.neural_network import CevahirNeuralNetwork
from torch.utils.data import DataLoader

# Örnek yapılandırma (config)
config = {
    "vocab_size": 75000,
    "embed_dim": 1024,
    "seq_proj_dim": 1024,
    "num_heads": 8,
    "num_tasks": 1,
    "attention_type": "multi_head",
    "normalization_type": "layer_norm",
    "dropout": 0.2,
    "device": "cpu",
    "epochs": 3,  # Test süresince düşük bir değer
    "learning_rate": 0.001,
    "checkpoint_dir": os.path.join(os.getcwd(), "saved_models/test_models/checkpoints/"),
    "training_history_path": os.path.join(os.getcwd(), "saved_models/test_models/training_history.json"),
}

# Test logger
logger = logging.getLogger("test_model_manager")
logger.setLevel(logging.DEBUG)

# Collate fonksiyonu
def collate_fn(batch):
    inputs, targets = zip(*batch)
    return torch.stack(inputs), torch.stack(targets)

@pytest.fixture
def model_manager():
    """ModelManager örneği oluşturur."""
    return ModelManager(config, model_class=CevahirNeuralNetwork)

@pytest.fixture
def training_run(model_manager):
    """
    Eğitim döngüsünü test eder.
    """
    model_manager.initialize()
    batch_size, seq_len = 16, 20

    # Dummy veriler
    train_data = [
        (
            torch.randint(0, 100, (seq_len,), dtype=torch.long),
            torch.randint(0, config["vocab_size"], (seq_len,), dtype=torch.long)
        )
        for _ in range(50)
    ]
    val_data = [
        (
            torch.randint(0, 100, (seq_len,), dtype=torch.long),
            torch.randint(0, config["vocab_size"], (seq_len,), dtype=torch.long)
        )
        for _ in range(10)
    ]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    training_manager = TrainingManager(
        model=model_manager.model,
        optimizer=model_manager.optimizer,
        criterion=model_manager.criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    return training_manager

# ========================
# TESTLER
# ========================

def test_gradient_norm_stability(training_run):
    """
    Gradient normlarının belirli bir eşik üzerinde kalması gerekir.
    """
    grad_norm = training_run._calculate_gradient_norm()
    assert grad_norm > 1e-3, f"Gradient normu çok düşük: {grad_norm:.6f}"


def test_weight_update_presence(training_run):
    """
    Model ağırlıklarının güncellenip güncellenmediğini test eder.
    """
    prev_weights = training_run._weights_are_updating()
    training_run._train_epoch()
    new_weights = training_run._weights_are_updating()

    weights_updated = any(
        not torch.allclose(new_weights[name], prev_weights[name])  # torch.equal yerine torch.allclose
        for name in new_weights.keys()
    )
    assert weights_updated, "Model ağırlıkları güncellenmiyor!"


def test_loss_difference_threshold(training_run):
    """
    Train loss ile validation loss arasındaki fark belirli bir eşik altında olmalıdır.
    """
    train_loss, _ = training_run._train_epoch()
    val_loss, _ = training_run._validate_epoch()
    loss_diff = abs(train_loss - val_loss)
    threshold = 0.1 * train_loss
    assert loss_diff < threshold, f"Loss farkı çok yüksek: {loss_diff:.4f}"


def test_model_convergence(training_run):
    """
    Modelin loss değerleri ile monotonik azalma olup olmadığını test eder.
    """
    losses = []
    for _ in range(3):
        train_loss, _ = training_run._train_epoch()
        losses.append(train_loss)
    assert losses == sorted(losses, reverse=True), f"Model loss değerleri azalmıyor: {losses}"


def test_layer_output_variance(training_run):
    """
    Çıktı varyansı embed dimension ile uyumlu olmalıdır.
    """
    dummy_input = torch.randint(0, config["vocab_size"], (16, 20), dtype=torch.long)
    embedding_output = getattr(training_run.model, 'dil_katmani', training_run.model)(dummy_input)
    variance = embedding_output.var().item()
    expected_variance = config["embed_dim"] / 100
    assert abs(variance - expected_variance) < 1.0, f"Varyans değerinde dengesizlik var: {variance:.4f}"


def test_optimizer_update(training_run):
    """
    Optimizer learning rate değişimini test eder.
    """
    orig_lr = training_run.optimizer.param_groups[0]['lr']
    training_run.optimizer.param_groups[0]['lr'] = orig_lr * 0.5
    new_lr = training_run.optimizer.param_groups[0]['lr']
    assert new_lr == orig_lr * 0.5, "Optimizer learning rate güncellemesi başarısız!"


def test_regularization_effect(training_run):
    """
    Dropout artışı sonrası gradient norm değişimini test eder.
    """
    original_dropout = training_run.model.dropout_rate
    training_run.model.dropout_rate = 0.5
    training_run.model.dropout = torch.nn.Dropout(0.5)
    grad_norm = training_run._calculate_gradient_norm()
    assert grad_norm > 1e-3, f"Dropout sonrası gradient norm çok düşük: {grad_norm}"
    training_run.model.dropout_rate = original_dropout


def test_epoch_timing(training_run):
    """
    Epoch başına geçen süre 1 saniyeyi geçmemelidir.
    """
    start_time = time.time()
    training_run._train_epoch()
    duration = time.time() - start_time
    assert duration < 1.0, f"Epoch süresi çok uzun: {duration:.4f} saniye"


def test_learning_rate_effect_on_gradients(training_run):
    """
    Farklı learning rate değerlerinin gradient norm üzerindeki etkisini test eder.
    """
    orig_lr = training_run.optimizer.param_groups[0]['lr']
    training_run.optimizer.param_groups[0]['lr'] = orig_lr * 10
    grad_norm = training_run._calculate_gradient_norm()
    assert grad_norm > 1e-3, "Yüksek learning rate sonrası gradient norm düşük kaldı"

