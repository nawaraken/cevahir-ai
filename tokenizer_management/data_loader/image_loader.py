import logging
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
import traceback
from typing import Optional

logger = logging.getLogger(__name__)

# ======================================================
#  Desteklenen formatlar
# ======================================================
SUPPORTED_FORMATS = ['.jpeg', '.jpg', '.png', '.bmp', '.gif', '.webp']

# ======================================================
#  Özel Hata Sınıfları
# ======================================================

class ImageLoaderError(Exception):
    pass

class UnsupportedImageFormatError(ImageLoaderError):
    pass

class ModelInitializationError(ImageLoaderError):
    pass

class ImageProcessingError(ImageLoaderError):
    pass

# ======================================================
#  ImageLoader Sınıfı
# ======================================================

class ImageLoader:
    """
    ImageLoader, görüntü dosyalarını yükler ve özellik çıkarımı yapar.
    
    - Özellik çıkarımı için ResNet18, EfficientNet veya MobileNetV3 seçeneklerini destekler.
    - Görüntü tensor formatına çevrilir ve normalize edilir.
    - GPU ve multi-GPU desteklidir.
    """

    def __init__(self, 
                 model_name: str = 'resnet18', 
                 device: Optional[str] = None,
                 use_data_augmentation: bool = False):
        """
        Args:
            model_name (str): Kullanılacak model tipi ('resnet18', 'mobilenet_v3', 'efficientnet_b0').
            device (str): 'cpu' veya 'cuda' olarak belirtilebilir.
            use_data_augmentation (bool): Eğitim için data augmentation kullan. Varsayılan False.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name.lower()
        self.use_data_augmentation = use_data_augmentation

        # Model ve Transform ayarlarını başlat
        self._model = None
        self._transform = None

        logger.info(f" Cihaz: {self.device}")

    # ======================================================
    #  Model Başlatma
    # ======================================================

    def _initialize_model(self):
        if self._model is not None and self._transform is not None:
            return

        try:
            logger.info(f" {self.model_name} modeli yükleniyor...")

            if self.model_name == 'resnet18':
                self._model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                output_size = 512
            elif self.model_name == 'mobilenet_v3':
                self._model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
                output_size = 960
            elif self.model_name == 'efficientnet_b0':
                self._model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
                output_size = 1280
            else:
                raise ValueError(f"Geçersiz model tipi: {self.model_name}")

            #  Son FC katmanını çıkararak özellik çıkarımı moduna al
            self._model = nn.Sequential(*list(self._model.children())[:-1])

            # Multi-GPU desteği
            if torch.cuda.device_count() > 1:
                logger.info(f" {torch.cuda.device_count()} GPU kullanılacak (DataParallel).")
                self._model = nn.DataParallel(self._model)

            self._model.to(self.device)
            self._model.eval()
            logger.info(f" {self.model_name} modeli başarıyla yüklendi ve özellik çıkarımı için hazırlandı.")

            #  Transformasyon Yapısı
            self._transform = self._create_transform()

        except Exception as e:
            logger.error(f" Model başlatılırken hata oluştu: {e}", exc_info=True)
            raise ModelInitializationError(f"Model başlatılırken hata oluştu: {e}")

    # ======================================================
    #  Görüntü Ön İşleme
    # ======================================================

    def _create_transform(self):
        if self.use_data_augmentation:
            logger.info(" Data Augmentation aktifleştirildi.")
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            logger.info(" Normalizasyon başlatıldı.")
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    # ======================================================
    #  Görüntü Yükleme
    # ======================================================

    def load_file(self, file_path: str) -> torch.Tensor:
        if not any(file_path.lower().endswith(fmt) for fmt in SUPPORTED_FORMATS):
            raise UnsupportedImageFormatError(f"Desteklenmeyen dosya formatı: {file_path}")

        try:
            self._initialize_model()

            image = Image.open(file_path).convert('RGB')

            # Görüntüyü tensor formatına çevir
            image_tensor = self._transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self._model(image_tensor)

            # Çıkışı düzleştir
            feature_vector = features.view(features.size(0), -1)[0]

            logger.info(f" Özellik çıkarımı tamamlandı. Çıktı boyutu: {feature_vector.shape}")
            return feature_vector

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("⚠️ CUDA bellek yetersizliği, bellek boşaltılıyor...")
                torch.cuda.empty_cache()

            logger.error(f" Model çalıştırılırken hata oluştu: {e}", exc_info=True)
            raise

        except Exception as e:
            logger.error(f" Görüntü dosyası yüklenirken hata oluştu: {e}", exc_info=True)
            raise ImageProcessingError(f"Görsel dosyası yüklenirken hata oluştu: {e}")

