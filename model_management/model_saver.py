"""
model_saver.py
Model ve eğitim bilgilerini güvenli bir şekilde kaydetmek için kullanılan bir modül.
"""

import os
import torch
import json
import logging

# Log yapılandırması
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
saver_logger = logging.getLogger("model_saver")


class ModelSaver:
    """
    ModelSaver:
    Model durumunu, optimizer, scheduler ve ek bilgileri kaydetmek için kullanılan bir sınıf.
    """

    @staticmethod
    def save_model(model, optimizer=None, scheduler=None, additional_info=None, save_dir="models/", model_name="model.pth"):
        """
        Model, optimizer, scheduler ve ek bilgileri kaydeder.

        Args:
            model (torch.nn.Module): Kaydedilecek model.
            optimizer (torch.optim.Optimizer, optional): Modelin optimizer'ı.
            scheduler (torch.optim.lr_scheduler, optional): Modelin öğrenme oranı scheduler'ı.
            additional_info (dict, optional): Eğitim bilgileri gibi ek veriler.
            save_dir (str, optional): Model dosyasının kaydedileceği dizin. Varsayılan 'models/'.
            model_name (str, optional): Kaydedilecek model dosyasının adı. Varsayılan 'model.pth'.

        Raises:
            RuntimeError: Kaydetme sırasında hata oluşursa.
        """
        try:
            saver_logger.info("Model kaydetme işlemi başlatılıyor...")

            # Kaydetme dizinini oluştur
            os.makedirs(save_dir, exist_ok=True)

            # Model durumunu kaydet
            model_path = os.path.join(save_dir, model_name)
            model_data = {
                "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict() if optimizer else None,
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "additional_info": additional_info if additional_info else None
            }
            torch.save(model_data, model_path)

            saver_logger.info(f"Model ve ilgili bilgiler başarıyla kaydedildi: {model_path}")

        except Exception as e:
            saver_logger.error(f"Model kaydedilirken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Model kaydedilemedi.") from e

    @staticmethod
    def save_full_model(model, save_dir="models/", model_name="full_model.pth"):
        """
        Modeli tek bir dosyada kaydeder.

        Args:
            model (torch.nn.Module): Kaydedilecek model.
            save_dir (str, optional): Model dosyasının kaydedileceği dizin. Varsayılan 'models/'.
            model_name (str, optional): Kaydedilecek dosyanın adı. Varsayılan 'full_model.pth'.

        Raises:
            RuntimeError: Kaydetme sırasında hata oluşursa.
        """
        try:
            saver_logger.info("Tam model kaydetme işlemi başlatılıyor...")

            # Kaydetme dizinini oluştur
            os.makedirs(save_dir, exist_ok=True)

            # Modeli direkt kaydet
            model_path = os.path.join(save_dir, model_name)
            torch.save(model, model_path)

            saver_logger.info(f"Tam model dosyası başarıyla kaydedildi: {model_path}")

        except Exception as e:
            saver_logger.error(f"Tam model kaydedilirken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Tam model kaydedilemedi.") from e

    @staticmethod
    def save_additional_info(info, save_dir="models/", info_file="additional_info.json"):
        """
        Ek bilgileri JSON formatında kaydeder.

        Args:
            info (dict): Kaydedilecek ek bilgiler.
            save_dir (str, optional): Ek bilgi dosyasının kaydedileceği dizin. Varsayılan 'models/'.
            info_file (str, optional): Ek bilgi dosyasının adı. Varsayılan 'additional_info.json'.

        Raises:
            RuntimeError: Kaydetme sırasında hata oluşursa.
        """
        try:
            saver_logger.info("Ek bilgi kaydetme işlemi başlatılıyor...")

            # Kaydetme dizinini oluştur
            os.makedirs(save_dir, exist_ok=True)

            # JSON dosyasına yazma
            info_path = os.path.join(save_dir, info_file)
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=4)

            saver_logger.info(f"Ek bilgiler başarıyla kaydedildi: {info_path}")

        except Exception as e:
            saver_logger.error(f"Ek bilgiler kaydedilirken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Ek bilgiler kaydedilemedi.") from e
