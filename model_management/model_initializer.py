"""
model_initializer.py
Model, optimizer, loss fonksiyonu ve scheduler başlatma işlemleri.
"""

import torch
import logging

# Log yapılandırması
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
initializer_logger = logging.getLogger("model_initializer")


class ModelInitializer:
    """
    ModelInitializer:
    Neural network modeli, optimizer, kayıp fonksiyonu ve scheduler başlatımı için yardımcı bir sınıf.
    """

    @staticmethod
    def _check_config_keys(config, required_keys):
        """
        Gerekli parametrelerin olup olmadığını kontrol eder.

        Args:
            config (dict): Model yapılandırma parametreleri.
            required_keys (list): Gerekli anahtarlar listesi.

        Raises:
            ValueError: Eğer eksik bir parametre varsa hata fırlatır.
        """
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Config içinde eksik parametre(ler) var: {', '.join(missing_keys)}")

    @staticmethod
    def initialize_model(model_class, config):
        """
        Neural network modelini başlatır.

        Args:
            model_class (type): Neural network model sınıfı.
            config (dict): Model yapılandırma parametreleri.

        Returns:
            torch.nn.Module: Eğitim yapılacak model.
        """
        try:
            initializer_logger.info("Model başlatma işlemi başlatıldı...")

            # Gerekli tüm parametrelerin olup olmadığını kontrol et
            ModelInitializer._check_config_keys(config, ["vocab_size", "embed_dim", "seq_proj_dim", 
                                                         "num_heads", "num_tasks", "attention_type", 
                                                         "normalization_type", "device"])

            # Modeli oluştur
            model = model_class(
                vocab_size=config["vocab_size"],
                embed_dim=config["embed_dim"],
                seq_proj_dim=config["seq_proj_dim"],
                num_heads=config["num_heads"],
                num_tasks=config["num_tasks"],
                attention_type=config["attention_type"],
                normalization_type=config["normalization_type"],
            ).to(config["device"])

            initializer_logger.info(f"Model başarıyla başlatıldı: {type(model).__name__}")
            return model

        except Exception as e:
            initializer_logger.error(f"Model başlatılırken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Model başlatılamadı.") from e

    @staticmethod
    def initialize_optimizer(model, config):
        """
        Optimizer'ı başlatır.

        Args:
            model (torch.nn.Module): Eğitim yapılacak model.
            config (dict): Model yapılandırma parametreleri.

        Returns:
            torch.optim.Optimizer: AdamW optimizer.
        """
        try:
            initializer_logger.info("Optimizer başlatma işlemi başlatılıyor...")

            # `weight_decay` kontrolünü opsiyonel hale getiriyoruz.
            ModelInitializer._check_config_keys(config, ["learning_rate"])

            weight_decay = config.get("weight_decay", 0.0)  # Eğer weight_decay eksikse 0.0 olarak ayarla

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=weight_decay,  # Varsayılan olarak 0.0 kullan
                eps=1e-8  # Sayısal kararlılık için epsilon
            )

            initializer_logger.info(f"Optimizer başarıyla başlatıldı: {optimizer.__class__.__name__}")
            return optimizer

        except Exception as e:
            initializer_logger.error(f"Optimizer başlatılırken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Optimizer başlatılamadı.") from e


    @staticmethod
    def initialize_criterion(config):
        """
        Loss fonksiyonunu başlatır.

        Args:
            config (dict): Model yapılandırma parametreleri.

        Returns:
            torch.nn.Module: CrossEntropyLoss.
        """
        try:
            initializer_logger.info("Loss fonksiyonu başlatma işlemi başlatılıyor...")

            # `label_smoothing` zorunlu olmaktan çıktı, default değer verdik
            label_smoothing = config.get("label_smoothing", 0.0)

            criterion = torch.nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                reduction="mean"  # Ortalama kayıp hesaplama
            )

            initializer_logger.info(f"Loss fonksiyonu başarıyla başlatıldı: {criterion.__class__.__name__}")
            return criterion

        except Exception as e:
            initializer_logger.error(f"Loss fonksiyonu başlatılırken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Loss fonksiyonu başlatılamadı.") from e


    @staticmethod
    def initialize_scheduler(optimizer, config):
        """
        Scheduler'ı başlatır.

        Args:
            optimizer (torch.optim.Optimizer): Eğitim yapılacak optimizer.
            config (dict): Model yapılandırma parametreleri.

        Returns:
            torch.optim.lr_scheduler: ReduceLROnPlateau scheduler.
        """
        try:
            initializer_logger.info("Scheduler başlatma işlemi başlatılıyor...")

            # Gerekli parametreleri kontrol et
            ModelInitializer._check_config_keys(config, ["lr_decay_factor", "lr_decay_patience", "lr_threshold"])

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",  # Minimum doğrulama kaybını hedefler
                factor=config["lr_decay_factor"],
                patience=config["lr_decay_patience"],
                threshold=config["lr_threshold"],
                verbose=True
            )

            initializer_logger.info("Scheduler başarıyla başlatıldı: ReduceLROnPlateau")
            return scheduler

        except Exception as e:
            initializer_logger.error(f"Scheduler başlatılırken hata oluştu: {str(e)}", exc_info=True)
            raise RuntimeError("Scheduler başlatılamadı.") from e
