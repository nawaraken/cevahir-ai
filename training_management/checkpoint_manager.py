"""
checkpoint_manager.py
=====================

Bu dosya, Cevahir Sinir Sistemi projesi kapsamında eğitim sırasında modelin ara durumlarını (checkpoint) kaydetmek ve yüklemek için kullanılır.
CheckpointManager sınıfı, model ağırlıklarını, optimizer durumunu ve eğitim geçmişini merkezi bir şekilde yönetir.

Dosya İçeriği:
--------------
1. CheckpointManager Sınıfı:
   - Eğitim sırasında checkpoint kaydetme ve yükleme işlemlerini gerçekleştirir.
   - Epoch başına veya özel durumlarda checkpoint alır.
   - Kaydedilen checkpoint dosyalarının rotasyonunu yönetir.

2. Kullanılan Harici Modüller:
   - `training_logger`: Loglama işlemleri için kullanılır.

3. Örnek Kullanım:
   - Eğitim sırasında model durumunu kaydetmek için `save_checkpoint` metodu çağrılır.
   - Kaydedilen bir durumu geri yüklemek için `load_checkpoint` metodu kullanılır.

Notlar:
------
- Checkpoint dosyaları varsayılan olarak `CHECKPOINT_MODEL` klasörüne kaydedilir.
- Maksimum dosya sayısı aşıldığında en eski dosya otomatik olarak silinir.
"""

import os
import torch
from training_management.training_logger import TrainingLogger
from config.parameters import CHECKPOINT_MODEL, DEVICE

# Logger nesnesi
logger = TrainingLogger()


class CheckpointManager:
    """
    Checkpoint işlemlerini yöneten sınıf.
    """
    def __init__(self, checkpoint_model_dir=CHECKPOINT_MODEL, max_checkpoints=5, device=DEVICE):
        """
        CheckpointManager başlatılır.

        Args:
            checkpoint_model_dir (str): Checkpoint dosyalarının kaydedileceği dizin.
            max_checkpoints (int): Maksimum saklanacak checkpoint dosya sayısı.
            device (str): Modelin çalıştırılacağı cihaz ("cpu" veya "cuda").
        """
        self.checkpoint_model_dir = os.path.abspath(checkpoint_model_dir)

        self.max_checkpoints = max_checkpoints
        self.device = device

        # Checkpoint dizinini kontrol et ve oluştur
        self._ensure_directory_exists(self.checkpoint_model_dir)

    def _ensure_directory_exists(self, directory):
        """ Belirtilen dizinin var olup olmadığını kontrol eder, yoksa oluşturur. """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.log_info(f"Checkpoint dizini oluşturuldu: {directory}")
        except Exception as e:
            logger.log_error(f"Checkpoint dizini oluşturulamadı: {str(e)}")
            raise RuntimeError(f"Checkpoint dizini oluşturulamadı: {directory}")

    def save_checkpoint(self, model, optimizer, epoch, training_history=None, filepath=None):
        """
        Checkpoint dosyası kaydeder.

        Args:
            model (torch.nn.Module): Kaydedilecek model.
            optimizer (torch.optim.Optimizer): Optimizer durumu.
            epoch (int): Mevcut epoch sayısı.
            training_history (dict, optional): Eğitim geçmişi.
            filepath (str, optional): Özel bir checkpoint dosya yolu. Varsayılan olarak checkpoint_dir kullanılır.

        Returns:
            str: Kaydedilen checkpoint dosyasının yolu.
        """
        try:
            # Özel bir dosya yolu belirtilmemişse varsayılan yolu kullan
            if filepath is None:
                filepath = os.path.join(self.checkpoint_model_dir, f"checkpoint_epoch_{epoch}.pth")

            # Checkpoint verileri
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_history": training_history if training_history else {}
            }

            # Checkpoint kaydet
            torch.save(checkpoint, filepath)
            logger.log_info(f"Checkpoint başarıyla kaydedildi: {filepath}")

            # Kaydın doğrulandığını test et
            assert os.path.exists(filepath), f"Checkpoint kaydedilemedi: {filepath}"

            # Eski checkpoint dosyalarını temizle
            self._manage_checkpoint_rotation()
            return filepath

        except Exception as e:
            logger.log_error(f"Checkpoint kaydetme hatası: {str(e)}")
            raise RuntimeError(f"Checkpoint kaydetme hatası: {str(e)}")

    def load_checkpoint(self, model, optimizer, filename):
        """
        Kaydedilen checkpoint dosyasını yükler.

        Args:
            model (torch.nn.Module): Model.
            optimizer (torch.optim.Optimizer): Optimizer.
            filename (str): Yüklenecek checkpoint dosyasının tam yolu.

        Returns:
            dict: Eğitim geçmişi ve epoch bilgileri.
        """
        try:
            # Dosyanın var olup olmadığını kontrol et
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Checkpoint dosyası bulunamadı: {filename}")

            # Checkpoint dosyasını yükle
            checkpoint = torch.load(filename, map_location=torch.device(self.device))
            model.load_state_dict(checkpoint.get("model_state_dict", {}))
            optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))

            epoch = checkpoint.get("epoch", None)
            training_history = checkpoint.get("training_history", {})

            logger.log_info(f"Checkpoint başarıyla yüklendi: {filename} (Epoch: {epoch})")
            return {"epoch": epoch, "training_history": training_history}

        except FileNotFoundError as e:
            logger.log_error(f"Checkpoint dosyası bulunamadı: {filename}")
            raise e

        except KeyError as e:
            logger.log_error(f"Checkpoint dosyasında eksik veri: {str(e)}")
            raise RuntimeError(f"Checkpoint dosyasında eksik veri: {str(e)}")

        except Exception as e:
            logger.log_error(f"Checkpoint yükleme hatası: {str(e)}")
            raise RuntimeError(f"Checkpoint yükleme hatası: {str(e)}")

    def _manage_checkpoint_rotation(self):
        """
        Eski checkpoint dosyalarını silerek maksimum checkpoint sayısını korur.
        """
        try:
            # Klasördeki checkpoint dosyalarını al
            checkpoint_files = sorted(
                [f for f in os.listdir(self.checkpoint_model_dir) if f.endswith(".pth")],
                key=lambda x: os.path.getctime(os.path.join(self.checkpoint_model_dir, x))
            )

            # Maksimum dosya sınırını aşan checkpoint'leri sil
            while len(checkpoint_files) > self.max_checkpoints:
                oldest_file = checkpoint_files.pop(0)
                oldest_file_path = os.path.join(self.checkpoint_model_dir, oldest_file)
                os.remove(oldest_file_path)
                logger.log_info(f"Eski checkpoint dosyası silindi: {oldest_file_path}")

        except FileNotFoundError:
            logger.log_warning("Checkpoint dizini bulunamadı, döngü atlandı.")
        except Exception as e:
            logger.log_error(f"Checkpoint rotasyonu sırasında hata oluştu: {str(e)}")

    def list_checkpoints(self):
        """
        Mevcut checkpoint dosyalarını listeler.

        Returns:
            list: Checkpoint dosyalarının tam yolları.
        """
        try:
            if not os.path.exists(self.checkpoint_model_dir):
                return []
            return [os.path.join(self.checkpoint_model_dir, f) for f in sorted(os.listdir(self.checkpoint_model_dir))]
        except Exception as e:
            logger.log_error(f"Checkpoint listesi alınırken hata oluştu: {str(e)}")
            return []
