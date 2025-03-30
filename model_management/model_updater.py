"""
model_updater.py
Model parametrelerini ve bileşenlerini güncellemek için kullanılan bir modül.
"""

import logging
import torch

# Log yapılandırması
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
updater_logger = logging.getLogger("model_updater")


class ModelUpdater:
    """
    ModelUpdater:
    Model parametrelerini, optimizer ve scheduler yapılandırmalarını güncellemek için kullanılan yardımcı bir sınıf.
    """

    @staticmethod
    def update_model(model, update_params):
        """
        Modelin belirli parametrelerini günceller.

        Args:
            model (torch.nn.Module): Güncellenecek model.
            update_params (dict): Güncellenmesi istenen parametreler.

        Raises:
            AttributeError: Modelde belirtilen parametre bulunamazsa.
        """
        try:
            updater_logger.info("Model güncelleme işlemi başlatılıyor...")

            for name, param in update_params.items():
                if hasattr(model, name):
                    setattr(model, name, param)
                    updater_logger.info(f"{name} parametresi güncellendi.")
                else:
                    updater_logger.warning(f"{name} parametresi modelde bulunamadı. Güncelleme atlandı.")

            # Modelin state_dict güncellenmesi gerekiyorsa
            if "state_dict" in update_params:
                try:
                    model.load_state_dict(update_params["state_dict"])
                    updater_logger.info("Model state_dict başarıyla yüklendi.")
                except RuntimeError as e:
                    updater_logger.error(f"Model state_dict yüklenirken hata oluştu: {e}")
                    raise RuntimeError("Model state_dict güncelleme başarısız oldu.") from e

            updater_logger.info("Model parametre güncellemesi tamamlandı.")
        except Exception as e:
            updater_logger.error(f"Model güncellenirken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Model parametre güncelleme işlemi başarısız oldu.") from e

    @staticmethod
    def update_optimizer(optimizer, update_params):
        """
        Optimizer parametrelerini günceller.

        Args:
            optimizer (torch.optim.Optimizer): Güncellenecek optimizer.
            update_params (dict): Güncellenecek optimizer parametreleri.

        Raises:
            ValueError: Geçersiz bir parametre güncellemesi yapılırsa.
        """
        try:
            updater_logger.info("Optimizer güncelleme işlemi başlatılıyor...")

            # Optimizer öğrenme oranı özel olarak ele alınıyor
            if "learning_rate" in update_params:
                ModelUpdater.update_learning_rate(optimizer, update_params["learning_rate"])

            # Diğer optimizer parametrelerini güncelle
            for group in optimizer.param_groups:
                for key, value in update_params.items():
                    if key in group:
                        group[key] = value
                        updater_logger.info(f"Optimizer parametresi güncellendi: {key} = {value}")
                    else:
                        updater_logger.warning(f"{key} parametresi optimizer'da bulunamadı. Güncelleme atlandı.")

            updater_logger.info("Optimizer parametre güncellemesi tamamlandı.")
        except KeyError as ke:
            updater_logger.error(f"Parametre anahtarı hatalı: {str(ke)}", exc_info=True)
            raise ValueError(f"Geçersiz parametre anahtarı: {str(ke)}") from ke
        except Exception as e:
            updater_logger.error(f"Optimizer güncellenirken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Optimizer güncelleme işlemi başarısız oldu.") from e

    @staticmethod
    def update_scheduler(scheduler, update_params):
        """
        Scheduler parametrelerini günceller.

        Args:
            scheduler (torch.optim.lr_scheduler): Güncellenecek scheduler.
            update_params (dict): Güncelleme parametreleri.

        Raises:
            AttributeError: Scheduler üzerinde belirtilen parametre bulunamazsa.
        """
        try:
            updater_logger.info("Scheduler güncelleme işlemi başlatılıyor...")
            
            if hasattr(scheduler, "optimizer"):
                if "learning_rate" in update_params:
                    ModelUpdater.update_learning_rate(scheduler.optimizer, update_params["learning_rate"])

            for key, value in update_params.items():
                if hasattr(scheduler, key):
                    setattr(scheduler, key, value)
                    updater_logger.info(f"Scheduler parametresi güncellendi: {key} = {value}")
                else:
                    updater_logger.warning(f"{key} parametresi scheduler'da bulunamadı. Güncelleme atlandı.")

            updater_logger.info("Scheduler parametre güncellemesi tamamlandı.")
        except Exception as e:
            updater_logger.error(f"Scheduler güncellenirken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Scheduler güncelleme işlemi başarısız oldu.") from e

    @staticmethod
    def update_learning_rate(optimizer, new_lr):
        """
        Optimizer'ın öğrenme oranını (learning rate) günceller.

        Args:
            optimizer (torch.optim.Optimizer): Güncellenecek optimizer.
            new_lr (float): Yeni öğrenme oranı.

        Raises:
            ValueError: Eğer optimizer üzerinde güncelleme yapılamazsa.
        """
        try:
            updater_logger.info(f"Learning rate güncelleme işlemi başlatılıyor... Yeni learning rate: {new_lr}")
            for group in optimizer.param_groups:
                group["lr"] = new_lr
                updater_logger.info(f"Learning rate güncellendi: {new_lr}")
            updater_logger.info("Learning rate güncellemesi tamamlandı.")
        except Exception as e:
            updater_logger.error(f"Learning rate güncellenirken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Learning rate güncelleme işlemi başarısız oldu.") from e

    @staticmethod
    def bulk_update(optimizer=None, scheduler=None, model=None, update_params=None):
        """
        Model, optimizer ve scheduler için toplu güncelleme işlemi yapar.

        Args:
            optimizer (torch.optim.Optimizer, optional): Güncellenecek optimizer.
            scheduler (torch.optim.lr_scheduler, optional): Güncellenecek scheduler.
            model (torch.nn.Module, optional): Güncellenecek model.
            update_params (dict): Güncelleme parametreleri.

        Raises:
            RuntimeError: Güncelleme sırasında herhangi bir bileşende hata oluşursa.
        """
        if update_params is None:
            update_params = {}

        try:
            updater_logger.info("Toplu güncelleme işlemi başlatılıyor...")
            if optimizer:
                ModelUpdater.update_optimizer(optimizer, update_params)
            if scheduler:
                ModelUpdater.update_scheduler(scheduler, update_params)
            if model:
                ModelUpdater.update_model(model, update_params)
            updater_logger.info("Toplu güncelleme işlemi tamamlandı.")
        except Exception as e:
            updater_logger.error(f"Toplu güncelleme sırasında hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Toplu güncelleme işlemi başarısız oldu.") from e
