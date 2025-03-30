"""
model_loader.py
Kaydedilen modelleri ve ilgili bilgileri yükler.
"""

import torch
import os
import json
import logging

# Log yapılandırması
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
loader_logger = logging.getLogger("model_loader")


class ModelLoader:
    """
    ModelLoader:
    Kaydedilen modeli ve ilgili ek bilgileri yüklemek için yardımcı bir sınıf.
    """

    @staticmethod
    def load_model(model_class, model_path, device="cpu"):
        """
        Kaydedilmiş model dosyasını yükler.

        Args:
            model_class (type): Yüklenmek istenen modelin sınıfı.
            model_path (str): Model dosyasının yolu.
            device (str, optional): Modelin yükleneceği cihaz (default: "cpu").

        Returns:
            torch.nn.Module: Yüklenmiş model örneği.
        """
        try:
            loader_logger.info(f"Model yükleme işlemi başlatılıyor: {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

            # Kaydedilmiş model ağırlıklarını yükle
            model_state = torch.load(model_path, map_location=device)

            # Modeli oluştur
            model_instance = model_class().to(device)

            # Modelin ağırlıklarını yükle
            model_instance.load_state_dict(model_state)

            loader_logger.info(f"Model başarıyla yüklendi ve {device} cihazına taşındı.")
            return model_instance

        except FileNotFoundError as fnf_error:
            loader_logger.error(f"Model dosyası bulunamadı: {str(fnf_error)}", exc_info=True)
            raise RuntimeError("Model yükleme sırasında bir dosya hatası oluştu.") from fnf_error

        except Exception as e:
            loader_logger.error(f"Model yükleme sırasında hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Model yüklenemedi.") from e

    @staticmethod
    def load_optimizer(optimizer, optimizer_path):
        """
        Optimizer durum dosyasını yükler.

        Args:
            optimizer (torch.optim.Optimizer): Yüklenecek optimizer örneği.
            optimizer_path (str): Optimizer dosyasının yolu.

        Returns:
            torch.optim.Optimizer: Yüklenmiş optimizer örneği.
        """
        try:
            loader_logger.info(f"Optimizer yükleme işlemi başlatılıyor: {optimizer_path}")

            if not os.path.exists(optimizer_path):
                raise FileNotFoundError(f"Optimizer dosyası bulunamadı: {optimizer_path}")

            state_dict = torch.load(optimizer_path)
            optimizer.load_state_dict(state_dict)

            loader_logger.info("Optimizer başarıyla yüklendi.")
            return optimizer
        except FileNotFoundError as fnf_error:
            loader_logger.error(f"Optimizer dosyası bulunamadı: {str(fnf_error)}", exc_info=True)
            raise RuntimeError("Optimizer yüklenemedi.") from fnf_error
        except Exception as e:
            loader_logger.error(f"Optimizer yükleme sırasında hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Optimizer yüklenemedi.") from e

    @staticmethod
    def load_scheduler(scheduler, scheduler_path):
        """
        Scheduler durum dosyasını yükler.

        Args:
            scheduler (torch.optim.lr_scheduler._LRScheduler): Yüklenecek scheduler örneği.
            scheduler_path (str): Scheduler dosyasının yolu.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: Yüklenmiş scheduler örneği.
        """
        try:
            loader_logger.info(f"Scheduler yükleme işlemi başlatılıyor: {scheduler_path}")

            if not os.path.exists(scheduler_path):
                raise FileNotFoundError(f"Scheduler dosyası bulunamadı: {scheduler_path}")

            state_dict = torch.load(scheduler_path)
            scheduler.load_state_dict(state_dict)

            loader_logger.info("Scheduler başarıyla yüklendi.")
            return scheduler
        except FileNotFoundError as fnf_error:
            loader_logger.error(f"Scheduler dosyası bulunamadı: {str(fnf_error)}", exc_info=True)
            raise RuntimeError("Scheduler yüklenemedi.") from fnf_error
        except Exception as e:
            loader_logger.error(f"Scheduler yükleme sırasında hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Scheduler yüklenemedi.") from e

    @staticmethod
    def load_additional_info(info_path):
        """
        Ek bilgileri JSON formatında yükler.

        Args:
            info_path (str): Ek bilgilerin kaydedildiği dosya yolu.

        Returns:
            dict: Ek bilgiler (ör. eğitim geçmişi, metrikler).
        """
        try:
            loader_logger.info(f"Ek bilgiler yükleniyor: {info_path}")

            if not os.path.exists(info_path):
                raise FileNotFoundError(f"Ek bilgi dosyası bulunamadı: {info_path}")

            with open(info_path, "r") as json_file:
                additional_info = json.load(json_file)

            loader_logger.info("Ek bilgiler başarıyla yüklendi.")
            return additional_info
        except FileNotFoundError as fnf_error:
            loader_logger.error(f"Ek bilgi dosyası bulunamadı: {str(fnf_error)}", exc_info=True)
            raise RuntimeError("Ek bilgiler yüklenemedi.") from fnf_error
        except json.JSONDecodeError as json_error:
            loader_logger.error(f"JSON formatında hata: {str(json_error)}", exc_info=True)
            raise RuntimeError("Ek bilgiler JSON formatında değil.") from json_error
        except Exception as e:
            loader_logger.error(f"Ek bilgiler yüklenirken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Ek bilgiler yüklenemedi.") from e
