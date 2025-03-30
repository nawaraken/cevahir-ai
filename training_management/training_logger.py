"""
training_logger.py
===================

Bu dosya, Cevahir Sinir Sistemi projesi kapsamında eğitim sırasında metriklerin, hataların ve önemli olayların loglanmasını sağlar. 
Eğitim süreci boyunca yaşanan olayları ayrıntılı bir şekilde kaydederek hata ayıklama ve performans takibi için kullanılabilir.

Dosya İçeriği:
--------------
1. TrainingLogger Sınıfı:
   - Loglama işlemlerini merkezi bir şekilde yönetir.
   - Eğitim sürecindeki kayıpları, doğrulukları ve hataları kaydeder.

2. Kullanılan Harici Modüller:
   - Python `logging` kütüphanesi: Loglama işlevselliği sağlar.

3. Örnek Kullanım:
   - TrainingLogger sınıfı başlatılır.
   - Eğitim metrikleri ve olaylar loglanır (`log_info`, `log_error` vb.).
   - Loglar farklı dosyalara veya rotasyonlu olarak kaydedilir.

Notlar:
------
- Log dosyaları `parameters.py` içinde belirtilen LOGGING_PATH dizinine kaydedilir.
- Maksimum dosya boyutu ve rotasyon parametreleri `parameters.py` üzerinden kontrol edilir.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import logging

from logging.handlers import RotatingFileHandler
from config.parameters import LOGGING_PATH, MAX_LOG_SIZE, BACKUP_COUNT


class TrainingLogger:
    def __init__(self):
        """
        TrainingLogger sınıfının başlatılması. Eğitim süreçlerinde loglama işlemleri için kullanılır.
        """
        # Eğitim loglayıcısı
        self.logger = self._initialize_logger("training_logger", "training.log")
        # Hata loglayıcısı
        self.error_logger = self._initialize_logger("error_logger", "errors.log")

    def _initialize_logger(self, name, filename):
        """
        Loglayıcı nesnesini başlatır.

        Args:
            name (str): Loglayıcı adı.
            filename (str): Log dosyasının adı.

        Returns:
            logging.Logger: Yapılandırılmış log nesnesi.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        log_file_path = os.path.join(LOGGING_PATH, filename)

        # Rotasyonlu dosya handler
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
        )
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)

        # Konsol handler (isteğe bağlı, bilgi seviyesinde)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    # ** Standart `logging` metodlarını ekleyerek uyumluluk sağlayalım**
    def info(self, message):
        """ Standart `info` metodu """
        self.logger.info(message)

    def warning(self, message):
        """ Standart `warning` metodu """
        self.logger.warning(message)

    def error(self, message, exc_info=False):
        """ Standart `error` metodu """
        if exc_info:
            self.error_logger.error(message, exc_info=True)
            self.logger.error(message, exc_info=True)
        else:
            self.error_logger.error(message)
            self.logger.error(message)

    def debug(self, message):
        """ Standart `debug` metodu """
        self.logger.debug(message)

    def critical(self, message):
        """ Standart `critical` metodu """
        self.logger.critical(message)
        self.error_logger.critical(message)

    def log_info(self, message):
        """
        Bilgilendirme mesajlarını loglar.

        Args:
            message (str): Loglanacak mesaj.
        """
        self.logger.info(message)

    def log_warning(self, message):
        """
        Uyarı mesajlarını loglar.

        Args:
            message (str): Loglanacak mesaj.
        """
        self.logger.warning(message)

    def log_error(self, message, exc_info=False):
        """
        Hata mesajlarını loglar.

        Args:
            message (str): Loglanacak hata mesajı.
            exc_info (bool, optional): İstisna bilgisini eklemek için True yapılabilir. Varsayılan False.
        """
        if exc_info:
            self.error_logger.error(message, exc_info=True)
            self.logger.error(message, exc_info=True)
        else:
            self.error_logger.error(message)
            self.logger.error(message)


    def log_metrics(self, epoch, training_loss, validation_loss=None, accuracy=None):
        """
        Eğitim ve doğrulama metriklerini loglar.

        Args:
            epoch (int): Geçerli epoch sayısı.
            training_loss (float): Eğitim kaybı değeri.
            validation_loss (float, optional): Doğrulama kaybı değeri.
            accuracy (float, optional): Doğrulama doğruluk değeri.
        """
        log_message = f"Epoch {epoch} - Training Loss: {training_loss:.4f}"
        if validation_loss is not None:
            log_message += f", Validation Loss: {validation_loss:.4f}"
        if accuracy is not None:
            log_message += f", Accuracy: {accuracy:.2f}%"
        self.log_info(log_message)

    def log_critical(self, message):
        """
        Kritik hataları loglar.

        Args:
            message (str): Loglanacak kritik hata mesajı.
        """
        self.logger.critical(message)
        self.error_logger.critical(message)

    def log_debug(self, message):
        """
        Debug seviyesinde mesajları loglar.

        Args:
            message (str): Loglanacak debug mesajı.
        """
        self.logger.debug(message)
