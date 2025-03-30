"""
training_scheduler.py
======================

Bu dosya, Cevahir Sinir Sistemi projesi kapsamında eğitim sırasında öğrenme oranını dinamik olarak ayarlamak için kullanılan 
öğrenme oranı planlayıcılarını yönetir. Eğitim sürecinin daha verimli ve optimize çalışmasını sağlar.

Dosya İçeriği:
--------------
1. TrainingScheduler Sınıfı:
   - PyTorch öğrenme oranı planlayıcılarını (scheduler) başlatır ve günceller.
   - Eğitim performansına bağlı olarak dinamik öğrenme oranı ayarlamaları yapar.
   - Gelişmiş loglama ile planlayıcı adımlarını detaylandırır.

2. Kullanılan Harici Modüller:
   - `training_logger.py`: Öğrenme oranındaki değişiklikleri loglar.

3. Örnek Kullanım:
   TrainingScheduler sınıfı, bir uygulamada veya serviste kolayca entegre edilip kullanılabilir. Örnek adımlar:
   - `TrainingScheduler` başlatılır (`__init__`).
   - Scheduler eğitim döngüsü sonunda güncellenir (`step`).
   - Scheduler parametreleri farklı senaryolar için özelleştirilir.

4. Loglama:
   Tüm öğrenme oranı değişiklikleri ve scheduler adımları `training_logger` kullanılarak loglanır.

Notlar:
------
- PyTorch `ReduceLROnPlateau`, `StepLR`, `ExponentialLR` gibi planlayıcılarla çalışır.
- Gerektiğinde yeni planlayıcı türleri kolayca eklenebilir.
"""

import os
import sys
import torch
from training_management.training_logger import TrainingLogger

class TrainingScheduler:
    def __init__(self, optimizer, scheduler_type="ReduceLROnPlateau", **kwargs):
        """
        TrainingScheduler sınıfının başlatılması.

        Args:
            optimizer (torch.optim.Optimizer): PyTorch optimizer nesnesi.
            scheduler_type (str): Kullanılacak planlayıcı türü ("ReduceLROnPlateau", "StepLR", "ExponentialLR" vb.).
            **kwargs: Planlayıcı türüne bağlı ek parametreler.
        """
        self.logger = TrainingLogger()
        self.scheduler_type = scheduler_type
        self.scheduler = self._initialize_scheduler(optimizer, scheduler_type, **kwargs)
        self.logger.log_info(f"{scheduler_type} öğrenme oranı planlayıcısı başarıyla başlatıldı.")
        self.logger.log_debug(f"Scheduler parametreleri: {self.scheduler.__dict__}")

    def _initialize_scheduler(self, optimizer, scheduler_type, **kwargs):
        """
        Belirtilen optimizer ve parametrelerle planlayıcıyı başlatır.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer nesnesi.
            scheduler_type (str): Planlayıcı türü.
            **kwargs: Ek parametreler.

        Returns:
            torch.optim.lr_scheduler: Başlatılmış öğrenme oranı planlayıcı nesnesi.
        """
        if scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get("mode", "min"),
                factor=kwargs.get("factor", 0.5),  # Öğrenme oranını yavaş düşür.
                patience=kwargs.get("patience", 8),  # Daha uzun süre bekleyelim.
                verbose=kwargs.get("verbose", True),
                threshold=kwargs.get("threshold", 1e-4),
                cooldown=kwargs.get("cooldown", 3),  # LR düşüşünden sonra 3 epoch bekle.
                min_lr=kwargs.get("min_lr", 5e-6),  # Öğrenme oranının aşırı düşmesini engelle.
                eps=kwargs.get("eps", 1e-8)
            )
            return scheduler

        elif scheduler_type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get("step_size", 30),
                gamma=kwargs.get("gamma", 0.1)
            )
            return scheduler

        elif scheduler_type == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get("gamma", 0.9)
            )
            return scheduler

        else:
            error_msg = f"Bilinmeyen scheduler türü: {scheduler_type}"
            self.logger.log_error(error_msg)
            raise ValueError(error_msg)

    def step(self, metric=None, gradient_norm=None):
        """
        Öğrenme oranını günceller. ReduceLROnPlateau için metrik gerekli olup, isteğe bağlı olarak gradient norm
        kontrolü de yapılır.

        Args:
            metric (float, optional): ReduceLROnPlateau için doğrulama kaybı gibi bir metrik.
            gradient_norm (float, optional): Eğer gradient norm belirli bir eşik değerin altındaysa LR güncellemesini yavaşlat.
        """
        self.logger.log_debug("Scheduler step metoduna girildi.")
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is None:
                error_msg = "ReduceLROnPlateau için bir metrik sağlanması gerekiyor."
                self.logger.log_error(error_msg)
                raise ValueError(error_msg)
            
            # Gradient Norm kontrolü: çok düşükse LR güncellemesini yapmadan uyarı ver.
            if gradient_norm is not None and gradient_norm < 10:
                self.logger.log_warning(f"Gradient Norm çok düşük ({gradient_norm:.4f}). LR güncellemesi atlanıyor!")
                return

            self.logger.log_debug(f"ReduceLROnPlateau için step öncesi metrik: {metric:.6f}")
            self.scheduler.step(metric)
            self.logger.log_info(f"ReduceLROnPlateau için öğrenme oranı güncellendi. Metrik: {metric:.6f}")
        else:
            self.scheduler.step()
            self.logger.log_info(f"{self.scheduler_type} için öğrenme oranı güncellendi.")

        current_lr = self.get_last_lr()
        self.logger.log_debug(f"Güncelleme sonrası öğrenme oranı: {current_lr:.6f}")
        if current_lr < 1e-5:
            self.logger.log_warning(f"Öğrenme oranı çok düştü! (LR={current_lr:.6f}) Model öğrenmeyi bırakabilir.")

    def get_last_lr(self):
        """
        Mevcut öğrenme oranını döndürür.

        Returns:
            float: Son öğrenme oranı.
        """
        try:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr = self.scheduler.optimizer.param_groups[0]["lr"]
            else:
                lr_list = self.scheduler.get_last_lr()
                lr = lr_list[0] if lr_list else None
                if lr is None:
                    raise RuntimeError("Scheduler'dan öğrenme oranı alınamadı.")
        except Exception as e:
            self.logger.log_error(f"Öğrenme oranı alınırken hata oluştu: {e}")
            raise e

        self.logger.log_info(f"Son öğrenme oranı: {lr:.6f}")
        return lr
