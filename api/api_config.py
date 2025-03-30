import os
import logging
from config.parameters import LOGGING_PATH, MODEL_SAVE_PATH, DEVICE

class Config:
    """Genel yapılandırma ayarları"""
    DEBUG = True
    TESTING = False
    SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
    JSONIFY_PRETTYPRINT_REGULAR = False  # Üretimde JSON yanıtlarını küçült
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # Varsayılan 16 MB
    LOGGING_PATH = LOGGING_PATH
    MODEL_SAVE_PATH = MODEL_SAVE_PATH
    DEVICE = DEVICE  # Cihaz ayarı (CPU veya GPU)
    LOG_LEVEL = logging.INFO  # Varsayılan log seviyesi

    # Log dosya yolları
    PROCESS_LOG = os.path.join(LOGGING_PATH, 'api_config_process.log')
    ERROR_LOG = os.path.join(LOGGING_PATH, 'errors.log')

    @staticmethod
    def init_logging():
        """Loglama yapılandırmasını başlatır"""
        os.makedirs(Config.LOGGING_PATH, exist_ok=True)  # Log klasörünü oluştur
        logging.basicConfig(
            level=Config.LOG_LEVEL,
            format="%(asctime)s - [%(levelname)s] - %(message)s",
            handlers=[
                logging.FileHandler(Config.ERROR_LOG, mode='a'),
                logging.StreamHandler()
            ]
        )

class DevelopmentConfig(Config):
    """Geliştirme ortamı yapılandırması"""
    DEBUG = True
    TESTING = False
    SECRET_KEY = os.getenv("DEVELOPMENT_SECRET_KEY", "dev_secret_key")
    JSONIFY_PRETTYPRINT_REGULAR = True  # Geliştirme için okunabilir JSON yanıtları
    LOG_LEVEL = logging.DEBUG  # Geliştirme için daha ayrıntılı log seviyesi

class TestingConfig(Config):
    """Test ortamı yapılandırması"""
    DEBUG = True
    TESTING = True
    SECRET_KEY = os.getenv("TESTING_SECRET_KEY", "test_secret_key")
    MAX_CONTENT_LENGTH = int(os.getenv("TESTING_MAX_CONTENT_LENGTH", 8 * 1024 * 1024))  # Test için içerik boyutu sınırı
    LOG_LEVEL = logging.WARNING  # Test ortamında yalnızca uyarılar ve üstü loglanır

class ProductionConfig(Config):
    """Üretim ortamı yapılandırması"""
    DEBUG = True
    TESTING = False
    SECRET_KEY = os.getenv("PRODUCTION_SECRET_KEY", "prod_secret_key")
    JSONIFY_PRETTYPRINT_REGULAR = False  # Üretim ortamında JSON yanıtlarını küçült
    LOG_LEVEL = logging.ERROR  # Üretimde yalnızca hatalar loglanır

def get_config():
    """
    FLASK_ENV ortam değişkenine göre yapılandırmayı döndürür.
    Varsayılan olarak 'development' ortamını kullanır.
    """
    env = os.getenv("FLASK_ENV", "development").lower()
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()

# Uygulama başlatıldığında yapılandırmayı başlat
config = get_config()
config.init_logging()
