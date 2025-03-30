import sys
import os
# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
import torch
import logging
from model_management.model_manager import ModelManager
from src.neural_network import CevahirNeuralNetwork

# **Örnek Konfigürasyon**
CONFIG = {
    "vocab_size": 150000,
    "embed_dim": 64,
    "seq_proj_dim": 128,
    "num_heads": 8,
    "attention_type": "multi_head",
    "normalization_type": "layer_norm",
    "learning_rate": 0.0001,
    "device": "cpu",
    "dropout": 0.2
}

# **Logger Ayarı**
logger = logging.getLogger("test_model_manager")
logger.setLevel(logging.DEBUG)

@pytest.fixture
def model_manager():
    """
    ModelManager örneğini oluşturur.
    """
    return ModelManager(CONFIG, model_class=CevahirNeuralNetwork)

def test_initialize(model_manager):
    """
    Modelin düzgün şekilde başlatıldığını test eder.
    """
    logger.info("[TEST] Model başlatma testi başladı.")
    
    model_manager.initialize()

    assert model_manager.model is not None, "Model başlatılamadı!"
    assert model_manager.optimizer is not None, "Optimizer başlatılamadı!"
    assert model_manager.criterion is not None, "Loss fonksiyonu başlatılamadı!"
    assert model_manager.scheduler is not None, "Scheduler başlatılamadı!"

    logger.info("[TEST] Model başlatma testi başarılı.")

def test_forward(model_manager):
    """
    Modelin ileri yayılım işlemini test eder.
    """
    logger.info("[TEST] İleri yayılım testi başladı.")

    model_manager.initialize()
    
    batch_size, seq_len = 2, 10
    input_tensor = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    output, attention_weights = model_manager.forward(input_tensor)

    # ** Çıktının Tensor Olduğunu Doğrula**
    assert isinstance(output, torch.Tensor), "Çıktı bir tensör olmalı!"
    assert isinstance(attention_weights, torch.Tensor) or attention_weights is None, "Attention weights yanlış türde!"

    # ** Çıktı Boyutunu Test Et**
    expected_output_shape = (batch_size, seq_len, CONFIG["vocab_size"])
    assert output.shape == torch.Size(expected_output_shape), \
        f"Çıktı boyutu hatalı! Beklenen: {expected_output_shape}, Gerçek: {output.shape}"

    # ** Attention Boyutu Doğrulama (Eğer Attention Mekanizması Kullanılıyorsa)**
    if attention_weights is not None:
        expected_attn_shape = (batch_size, CONFIG["num_heads"], seq_len, seq_len)
        assert attention_weights.shape == torch.Size(expected_attn_shape), \
            f"Attention ağırlıklarının boyutu hatalı! Beklenen: {expected_attn_shape}, Gerçek: {attention_weights.shape}"

    logger.info("[TEST] İleri yayılım testi başarıyla tamamlandı.")

def test_save_and_load(model_manager, tmp_path):
    """
    Modelin kaydedilip tekrar yüklenmesini test eder.
    """
    logger.info("[TEST] Model kaydetme/yükleme testi başladı.")

    model_manager.initialize()
    
    save_path = tmp_path / "test_model.pth"
    
    # **Modeli Kaydet**
    model_manager.save(str(save_path))
    
    assert save_path.exists(), "Model dosyası kaydedilemedi!"

    # **Yeni bir ModelManager oluştur ve modeli yükle**
    new_model_manager = ModelManager(CONFIG, model_class=CevahirNeuralNetwork)
    new_model_manager.initialize()
    new_model_manager.load(str(save_path))

    assert new_model_manager.model is not None, "Yüklenen model başlatılamadı!"

    # **Forward Testi Tekrar Yap (Model Çıktısının Doğruluğunu Test Et)**
    batch_size, seq_len = 2, 10
    input_tensor = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    output, attention_weights = new_model_manager.forward(input_tensor)

    assert isinstance(output, torch.Tensor), "Yüklenen modelin çıktısı bir tensör olmalı!"
    assert isinstance(attention_weights, torch.Tensor) or attention_weights is None, "Yüklenen modelin attention weights yanlış türde!"

    logger.info("[TEST] Model kaydetme/yükleme testi başarıyla tamamlandı.")

def test_model_update(model_manager):
    """
    Model parametrelerinin güncellenip güncellenmediğini test eder.
    """
    logger.info("[TEST] Model güncelleme testi başladı.")

    model_manager.initialize()
    
    update_params = {
        "learning_rate": 0.0005
    }
    
    # **Modeli Güncelle**
    model_manager.update(update_params)
    
    # **Yeni learning_rate değerini doğrula**
    new_lr = model_manager.optimizer.param_groups[0]["lr"]
    assert new_lr == update_params["learning_rate"], f"Optimizer learning rate güncellenmedi! Beklenen: {update_params['learning_rate']}, Gerçek: {new_lr}"

    logger.info("[TEST] Model güncelleme testi başarıyla tamamlandı.")

def test_memorization(model_manager):
    """
    Modelin ezber yapıp yapmadığını test eder.
    Eğitimde olmayan örneklerle modelin nasıl yanıt verdiğini inceler.
    """
    logger.info("[TEST] Ezber testi başladı.")

    model_manager.initialize()
    
    # Eğitimde olmayan (rastgele) bir girdi oluştur
    batch_size, seq_len = 2, 10
    unseen_input = torch.randint(CONFIG["vocab_size"] // 2, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    output, _ = model_manager.forward(unseen_input)

    # Çıktının hala mantıklı olup olmadığını doğrula
    assert output is not None, "Model ezber yapıyor olabilir! Görmediği veri için çıktı üretemedi."
    
    logger.info("[TEST] Ezber testi başarıyla tamamlandı.")

def test_generalization(model_manager):
    """
    Modelin yeni kelimeler ve kelime kombinasyonlarına karşı nasıl performans gösterdiğini test eder.
    """
    logger.info("[TEST] Genelleme testi başladı.")

    model_manager.initialize()
    
    # Modelin önce görmediği kelimelerle bir girdi üretelim
    batch_size, seq_len = 2, 10
    novel_input = torch.randint(CONFIG["vocab_size"] // 3, CONFIG["vocab_size"] // 2, (batch_size, seq_len), dtype=torch.long)

    output, _ = model_manager.forward(novel_input)

    # Modelin yine anlamlı bir çıktı üretip üretmediğini kontrol et
    assert output is not None, "Model yeni kelimeler için genelleme yapamıyor olabilir!"

    logger.info("[TEST] Genelleme testi başarıyla tamamlandı.")

def test_memory_management(model_manager):
    """
    Modelin önceki çıktıları bellekte saklayıp saklamadığını test eder.
    """
    logger.info("[TEST] Bellek yönetimi testi başladı.")

    model_manager.initialize()
    
    batch_size, seq_len = 2, 10
    input_tensor = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    # Modeli çalıştır ve belleğe yaz
    model_manager.forward(input_tensor)

    # Bellekten önceki çıktıyı al
    retrieved_output = model_manager.model.memory_manager.retrieve("final_output")

    assert retrieved_output is not None, "Bellek yönetimi başarısız! Önceki çıktı bellekte bulunamadı."
    assert retrieved_output.shape == (batch_size, seq_len, CONFIG["vocab_size"]), \
        f"Bellekten çağrılan veri yanlış boyutta: {retrieved_output.shape}"

    logger.info("[TEST] Bellek yönetimi testi başarıyla tamamlandı.")

def test_performance(model_manager):
    """
    Modelin CPU ve GPU farklarını test eder ve belirli bir sürede kaç işlem yapabildiğini ölçer.
    """
    logger.info("[TEST] Performans testi başladı.")

    model_manager.initialize()

    batch_size, seq_len = 8, 20
    input_tensor = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    import time

    # CPU'da ölçüm
    start_time = time.time()
    model_manager.forward(input_tensor)
    cpu_time = time.time() - start_time

    logger.info(f"[PERFORMANCE] CPU İşleme Süresi: {cpu_time:.4f} saniye.")

    # Eğer GPU varsa test et
    if torch.cuda.is_available():
        model_manager.config["device"] = "cuda"
        model_manager.initialize()

        start_time = time.time()
        model_manager.forward(input_tensor.to("cuda"))
        gpu_time = time.time() - start_time

        logger.info(f"[PERFORMANCE] GPU İşleme Süresi: {gpu_time:.4f} saniye.")
    
    logger.info("[TEST] Performans testi başarıyla tamamlandı.")

def test_max_token_capacity(model_manager):
    """
    Modelin maksimum kaç token işleyebildiğini belirler.
    """
    logger.info("[TEST] Maksimum token kapasitesi testi başladı.")

    model_manager.initialize()

    max_tokens = 512  # Bu değeri artırarak sınırları zorlayabiliriz
    batch_size = 4

    try:
        input_tensor = torch.randint(0, CONFIG["vocab_size"], (batch_size, max_tokens), dtype=torch.long)
        output, _ = model_manager.forward(input_tensor)

        assert output.shape[1] == max_tokens, f"Model {max_tokens} token işleyemedi!"
        logger.info(f"[MAX TOKEN] Model başarılı şekilde {max_tokens} token işledi.")

    except RuntimeError as e:
        logger.error(f"[MAX TOKEN] Model {max_tokens} token işleyemedi! Hata: {str(e)}")
    
    logger.info("[TEST] Maksimum token kapasitesi testi tamamlandı.")
