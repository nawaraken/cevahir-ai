"""
model_management/model_manager.py
==================================
Bu dosya, Cevahir Sinir Sistemi projesinde modeli başlatma, kaydetme, yükleme ve güncelleme işlemlerini yönetir.
"""

import sys
import os

#  Proje kök dizinini modül yoluna ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

#  src dizinini ekle
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

print(f" PYTHONPATH Güncellendi: {sys.path}")  # Hangi yolların eklendiğini görmek için
import logging
import torch
import torch.nn as nn
import torch.optim as optim



from model_management.model_initializer import ModelInitializer
from model_management.model_saver import ModelSaver
from model_management.model_loader import ModelLoader
from model_management.model_updater import ModelUpdater
from src.neural_network import CevahirNeuralNetwork  # Cevahir modelini çağır

# **Log yapılandırması**
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
manager_logger = logging.getLogger("ModelManager")


class ModelManager:
    """
    ModelManager:
    ------------------------
    Modelle ilgili tüm işlemleri merkezi olarak yöneten sınıf.
    - Model başlatma, kaydetme, yükleme, güncelleme
    - Modelin ileri yayılım sürecini yönetme
    """

    def __init__(self, config, model_class=CevahirNeuralNetwork):
        """
        ModelManager başlatıcı fonksiyonu.

        Args:
            config (dict): Model yapılandırma parametreleri.
            model_class (type): Kullanılacak model sınıfı.
        """
        self.config = config
        self.model_class = model_class
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.modelUpdater=ModelUpdater
    def initialize(self):
        """
        Model, optimizer, criterion ve scheduler'ı başlatır.
        """
        manager_logger.info(" Model, optimizer, loss fonksiyonu ve scheduler başlatılıyor...")
        try:
            # **CevahirNeuralNetwork başlatılıyor**
            self.model = self.model_class(
                vocab_size=self.config.get("vocab_size", 75000),
                learning_rate=self.config.get("learning_rate",0.0001),
                dropout=self.config.get("dropout", 0.2),
                embed_dim=self.config.get("embed_dim", 1024),
                seq_proj_dim=self.config.get("seq_proj_dim", 1024),
                num_heads=self.config.get("num_heads", 8),
            
                attention_type=self.config.get("attention_type", "multi_head"),
                normalization_type=self.config.get("normalization_type", "layer_norm"),
                log_level=self.config.get("log_level", logging.DEBUG)
            )

            manager_logger.info(f" Model başarıyla başlatıldı: `{self.model.__class__.__name__}`")

            # **Optimizer Ayarı**
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.get("learning_rate", 0.001))
            manager_logger.info(f" Optimizer başlatıldı: `{self.optimizer.__class__.__name__}`")

            # **Loss Fonksiyonu**
            self.criterion = nn.CrossEntropyLoss()
            manager_logger.info(f" Loss fonksiyonu başlatıldı: `{self.criterion.__class__.__name__}`")

            # **Scheduler**
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.get("lr_decay_factor", 0.1),
                patience=self.config.get("lr_decay_patience", 10),
                threshold=self.config.get("lr_threshold", 1e-4),
                verbose=True
            )
            manager_logger.info(" Scheduler başlatıldı: `ReduceLROnPlateau`")

        except Exception as e:
            manager_logger.error(f" Model başlatma sırasında hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError(" Model başlatılamadı.") from e

    def forward(self, inputs, inference=False):
        """
        Modelin ileri yayılım işlemi.

        Args:
            inputs (torch.Tensor): Model giriş tensörü.
            inference (bool): Eğer True ise inference modunda çalışır (eval ve no_grad uygulanır),
                            False ise eğitim modunda çalışır.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Modelin ana çıktısı ve dikkat mekanizması ağırlıkları.
        """
        if self.model is None:
            raise RuntimeError("Model initialize edilmedi. Lütfen önce `initialize()` metodunu çağırın.")

        # Mevcut model modunu logla
        current_mode = "train" if self.model.training else "eval"
        manager_logger.info(f"[FORWARD] İşlem başlamadan önce model modu: {current_mode}")

        try:
            # Eval moduna mı geçiliyor, train moduna mı?
            self.model.train(mode=not inference)

            with torch.set_grad_enabled(not inference):
                output, attention_weights = self.model(inputs)

            # Modelin modunu tekrar kontrol et ve logla
            current_mode_after = "train" if self.model.training else "eval"
            manager_logger.info(f"[FORWARD] İşlem sonrası model modu: {current_mode_after}")

            # **Çıktı doğrulama**
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Çıktı tensör formatında olmalıdır! Alınan: {type(output)}")

            if attention_weights is not None and not isinstance(attention_weights, torch.Tensor):
                raise TypeError(f"Attention weights yanlış türde! Alınan: {type(attention_weights)}")

            # **Boyut kontrolü**
            expected_output_shape = (inputs.shape[0], inputs.shape[1], self.config.get("vocab_size", 75000))
            if output.shape != torch.Size(expected_output_shape):
                manager_logger.warning(f"[FORWARD] Çıktı boyutu hatalı! Beklenen: {expected_output_shape}, Gerçek: {output.shape}")

            manager_logger.info(f"[FORWARD] İleri yayılım tamamlandı. Çıktı boyutu: {output.shape}, Attention boyutu: {attention_weights.shape if attention_weights is not None else 'None'}")

            return output, attention_weights

        except Exception as e:
            manager_logger.error(f"[FORWARD] Model ileri yayılım sırasında hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError(f"[FORWARD] Model ileri yayılım hatası: {str(e)}") from e




    def save(self, save_path=None, epoch=0):
        """
        Modelin ağırlıklarını, optimizer ve scheduler bilgilerini kaydeder.

        Args:
            save_path (str, optional): Modelin kaydedileceği dosya yolu. 
                                    Varsayılan olarak 'saved_models/test_models/cevahir_model.pth' kullanılır.
            epoch (int, optional): Şu anki epoch bilgisi.
        """
        try:
            if save_path is None:
                save_path = os.path.join(os.getcwd(), "saved_models/test_models/cevahir_model.pth")

            # Kaydetme dizinini oluştur
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Tüm model durumunu kaydet
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config  # Yapılandırma bilgilerini de kaydedelim
            }, save_path)

            manager_logger.info(f"Model başarıyla kaydedildi: `{save_path}`")

        except IOError as io_err:
            manager_logger.error(f"Model kaydetme sırasında IO hatası oluştu: {str(io_err)}", exc_info=True)
            raise RuntimeError("Model kaydedilemedi. IO hatası meydana geldi.") from io_err

        except Exception as e:
            manager_logger.error(f"Model kaydetme sırasında hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Model kaydedilemedi.") from e


    def load(self, load_path=None):
        """
        Kaydedilmiş model ağırlıklarını ve eğitim durumunu yükler.

        Args:
            load_path (str, optional): Model ağırlıklarının yükleneceği dosya yolu.
                                    Varsayılan olarak 'saved_models/test_models/cevahir_model.pth' kullanılır.
        """
        try:
            if load_path is None:
                load_path = os.path.join(os.getcwd(), "saved_models/test_models/cevahir_model.pth")

            if not os.path.exists(load_path):
                manager_logger.error(f"Model dosyası bulunamadı: `{load_path}`")
                raise FileNotFoundError(f"Model dosyası bulunamadı: {load_path}")

            # Eğer model başlatılmadıysa başlat
            if self.model is None:
                manager_logger.warning("Model başlatılmadı, otomatik başlatılıyor...")
                self.initialize()

            # Model dosyasını yükle
            checkpoint = torch.load(load_path, map_location=self.config.get("device", "cpu"))

            # Model ağırlıklarını yükle
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Optimizer ve scheduler bilgilerini yükle
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Epoch bilgisini yükle
            self.config['current_epoch'] = checkpoint.get('epoch', 0)

            manager_logger.info(f"Model başarıyla yüklendi: `{load_path}`")
            manager_logger.info(f"Son epoch: {self.config['current_epoch']}")

        except IOError as io_err:
            manager_logger.error(f"Model yükleme sırasında IO hatası oluştu: {str(io_err)}", exc_info=True)
            raise RuntimeError("Model yüklenemedi. IO hatası meydana geldi.") from io_err

        except Exception as e:
            manager_logger.error(f"Model yükleme sırasında hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Model yüklenemedi.") from e



    def update(self, update_params):
        """
        Model ve optimizer parametrelerini günceller.

        Args:
            update_params (dict): Güncelleme parametreleri.
        """
        manager_logger.info(" Model güncelleniyor...")
        try:
            self.modelUpdater.update_model(self.model, update_params)
            self.modelUpdater.update_optimizer(self.optimizer, update_params)
            manager_logger.info(" Model başarıyla güncellendi.")
        except Exception as e:
            manager_logger.error(f" Güncelleme sırasında hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError(" Model güncellenemedi.") from e
