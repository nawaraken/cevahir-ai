import os
import logging
import numpy as np
import torch
from typing import List, Dict, Any, Type, Optional,Tuple
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
import traceback

# İlgili loader modüllerini içe aktaralım
from tokenizer_management.data_loader.json_loader import JSONLoader, JSONLoaderError
from tokenizer_management.data_loader.txt_loader import TXTLoader
from tokenizer_management.data_loader.docx_loader import DOCXLoader
from tokenizer_management.data_loader.mp3_loader import MP3Loader
from tokenizer_management.data_loader.image_loader import ImageLoader
from tokenizer_management.data_loader.video_loader import VideoLoader
from tokenizer_management.data_loader.data_preprocessor import DataPreprocessor
from tokenizer_management.data_loader.tensorizer import Tensorizer

logger = logging.getLogger(__name__)

# ================================================================
# DataLoaderManager Özel Exception Yapısı
# ================================================================
class DataLoaderError(Exception):
    pass

class FileNotFoundError(DataLoaderError):
    pass

class UnsupportedFormatError(DataLoaderError):
    pass

class DataLoadError(DataLoaderError):
    pass

class DataPreprocessError(DataLoaderError):
    pass

class JSONLoaderError(Exception):
    """Genel JSON yükleyici hatası."""
    pass

# ================================================================
# BaseLoader (Abstract Class)
# ================================================================
class BaseLoader(ABC):
    @abstractmethod
    def load_file(self, file_path: str) -> Any:
        pass

# ================================================================
# Loader Registry (Factory Pattern)
# ================================================================
LOADER_REGISTRY: Dict[str, Type[BaseLoader]] = {}

def register_loader(extension: str):
    def wrapper(cls):
        LOADER_REGISTRY[extension] = cls
        return cls
    return wrapper

# ================================================================
# Dosya Loaderları (JSON, TXT, DOCX, MP3, IMAGE, VIDEO)
# ================================================================
@register_loader('.json')
class RegisteredJSONLoader(JSONLoader):
    pass

@register_loader('.txt')
class RegisteredTXTLoader(TXTLoader):
    pass

@register_loader('.docx')
class RegisteredDOCXLoader(DOCXLoader):
    pass

@register_loader('.mp3')
class RegisteredMP3Loader(MP3Loader):
    pass

@register_loader('.jpg')
@register_loader('.jpeg')
@register_loader('.png')
class RegisteredImageLoader(ImageLoader):
    pass

@register_loader('.mp4')
@register_loader('.avi')
@register_loader('.mov')
class RegisteredVideoLoader(VideoLoader):
    pass

# ================================================================
# DataLoaderManager
# ================================================================
class DataLoaderManager:
    def __init__(self, 
                 data_directory: str, 
                 batch_size: int = 8, 
                 max_length: int = 128, 
                 skip_missing: bool = True):
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.max_length = max_length
        self.skip_missing = skip_missing
        
        self.preprocessor = DataPreprocessor()
        self.tensorizer = Tensorizer(max_length=self.max_length)

    def _get_loader(self, file_extension: str) -> BaseLoader:
        """
        Dosya uzantısına göre uygun loader sınıfını döndürür.

        Args:
            file_extension (str): Dosya uzantısı (ör. 'json', 'csv', 'xml').

        Returns:
            BaseLoader: Yükleyici sınıf nesnesi.

        Raises:
            UnsupportedFormatError: Desteklenmeyen dosya formatı için fırlatılır.
            JSONTypeError: Dosya uzantısı geçersiz tipte ise fırlatılır.
            JSONLoaderError: Yükleyici örneği oluşturulamazsa fırlatılır.
        """
        if not isinstance(file_extension, str):
            logger.error(f"Geçersiz dosya uzantı tipi: {type(file_extension)}")
            raise JSONTypeError(f"Geçersiz dosya uzantı tipi: {type(file_extension)}")

        file_extension = file_extension.lower().strip()
        if not file_extension:
            logger.error("Dosya uzantısı boş veya geçersiz.")
            raise UnsupportedFormatError("Dosya uzantısı boş veya geçersiz.")

        try:
            loader_cls = LOADER_REGISTRY.get(file_extension)
            if loader_cls is None:
                logger.error(f"{file_extension} formatı desteklenmiyor. Desteklenen formatlar: {list(LOADER_REGISTRY.keys())}")
                raise UnsupportedFormatError(f"{file_extension} formatı desteklenmiyor. Desteklenen formatlar: {list(LOADER_REGISTRY.keys())}")

            # Loader sınıfının BaseLoader'dan türetilip türetilmediğini kontrol edelim
            if not hasattr(loader_cls, 'load_file') or not callable(getattr(loader_cls, 'load_file')):
                logger.error(f"Geçersiz loader: {loader_cls}. 'load_file' metodu eksik veya geçersiz.")
                raise JSONLoaderError(f"Geçersiz loader: {loader_cls}. 'load_file' metodu eksik veya geçersiz.")


            # Loader sınıfı yerine instance oluşturulması
            loader_instance = loader_cls()
            logger.info(f"{file_extension} formatı için {loader_cls.__name__} kullanılıyor.")
            return loader_instance

        except KeyError as e:
            logger.error(f"Loader kayıt hatası: {e}")
            raise JSONLoaderError(f"Loader kayıt hatası: {e}") from e

        except TypeError as e:
            logger.error(f"Loader tipi hatası: {e}")
            raise JSONLoaderError(f"Loader tipi hatası: {e}") from e

        except Exception as e:
            logger.error(f"Yükleyici oluşturulurken beklenmedik hata: {e}\n{traceback.format_exc()}")
            raise JSONLoaderError(f"Yükleyici oluşturulurken beklenmedik hata: {e}") from e


    def _determine_modality(self, file_extension: str) -> str:
        if file_extension in ['.json', '.txt', '.docx']:
            return "text"
        elif file_extension == '.mp3':
            return "audio"
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            return "image"
        elif file_extension in ['.mp4', '.avi', '.mov']:
            return "video"
        elif file_extension == '.npy':
            return "tensor"
        else:
            return "unknown"

    def load_data(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.data_directory):
            raise FileNotFoundError(f"Veri dizini bulunamadı: {self.data_directory}")
        
        data_list = []
        skipped_files = 0
        
        for root, _, files in os.walk(self.data_directory):
            for file in files:
                file_path = os.path.join(root, file)

                # Geçersiz veya boş dosya kontrolü
                if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
                    logger.warning(f"Geçersiz veya boş dosya atlanıyor: {file_path}")
                    skipped_files += 1
                    continue

                file_extension = os.path.splitext(file)[1].lower()
                try:
                    # Loader çek - Factory Pattern üzerinden
                    loader = self._get_loader(file_extension)
                    result = loader.load_file(file_path)

                    if not isinstance(result, dict):
                        raise DataLoadError(f"{file_path} → Yüklenen veri bir dict değil!")
                    
                    # Beklenen JSON formatını kontrol et
                    if 'data' not in result:
                        raise DataLoadError(f"{file_path} → 'data' anahtarı eksik!")
                    
                    modality = self._determine_modality(file_extension)
                    result["modality"] = modality
                    result["source_file"] = os.path.basename(file_path)
                    
                    # Dönüşü ortak format haline getir
                    processed_data = {
                        "modality": result["modality"],
                        "data": result["data"],
                        "source_file": result["source_file"]
                    }
                    data_list.append(processed_data)
                    logger.info(f"[✓] {file_path} başarıyla yüklendi. (Modality: {modality})")

                except (UnsupportedFormatError, DataLoadError) as specific_err:
                    logger.error(f"[X] {file_path} → Veri yüklenirken hata oluştu: {specific_err}")
                    if not self.skip_missing:
                        raise
                
                except Exception as general_err:
                    logger.error(f"[X] {file_path} yüklenirken hata: {general_err}", exc_info=True)
                    if not self.skip_missing:
                        raise
        
        if skipped_files > 0:
            logger.warning(f"[!] Toplam {skipped_files} dosya atlandı.")

        logger.info(f"[+] Toplam {len(data_list)} dosya başarıyla işlendi.")
        return data_list




    def _process(self, modality: str, data: Any, file_path: str) -> Any:
        """
        Modality’ye göre merkezi işleme uygular.
        Text için, preprocessor.preprocess_text çağrılır.
        Audio ve video için, tensor dönüşümü uygulanır.
        Image için ise veri direkt döndürülür.
        """
        try:
            if modality == "text":
                if isinstance(data, dict):
                    # JSON anahtarları dinamik yapıdaysa metinleri birleştir
                    combined_text = "\n".join(f"{k}: {v}" for k, v in data.items())
                    logger.debug(f"{file_path} - Birleştirilmiş JSON metni: {combined_text[:100]}...")
                    return self.preprocessor.preprocess_text(combined_text)
                elif isinstance(data, str):
                    logger.debug(f"{file_path} - İşlenen metin: {data[:100]}...")
                    return self.preprocessor.preprocess_text(data)
                else:
                    raise DataLoadError(f"{file_path} - Metin formatı hatalı!")

            elif modality == "audio":
                if isinstance(data, list) or isinstance(data, np.ndarray):
                    logger.debug(f"{file_path} - Ses verisi tensor formatına çevriliyor...")
                    return torch.tensor(data, dtype=torch.float32)
                else:
                    raise DataLoadError(f"{file_path} - Ses verisi yanlış formatta!")

            elif modality == "image":
                logger.debug(f"{file_path} - Görsel verisi işleniyor...")
                return data
            
            elif modality == "video":
                if isinstance(data, list) or isinstance(data, np.ndarray):
                    logger.debug(f"{file_path} - Video verisi tensor formatına çevriliyor...")
                    return torch.tensor(data, dtype=torch.float32)
                else:
                    raise DataLoadError(f"{file_path} - Video verisi yanlış formatta!")
            
            elif modality == "tensor":
                if file_path.endswith('.npy'):
                    try:
                        tensor_data = np.load(file_path)
                        if not isinstance(tensor_data, np.ndarray):
                            raise DataLoadError(f"{file_path} - Tensor verisi yanlış formatta!")
                        logger.debug(f"{file_path} - Tensor verisi tensor formatına çevriliyor...")
                        return torch.tensor(tensor_data, dtype=torch.float32)
                    except Exception as e:
                        raise DataLoadError(f"{file_path} - Tensor yükleme hatası: {e}") from e
                else:
                    raise DataLoadError(f"{file_path} - Tensor dosya formatı hatalı!")
            
            else:
                raise DataLoadError(f"{file_path} için bilinmeyen modality: {modality}")

        except DataLoadError as e:
            logger.error(f"{file_path} işlenirken hata: {e}")
            raise
        except Exception as e:
            logger.error(f"{file_path} - Beklenmeyen hata oluştu: {e}", exc_info=True)
            raise DataLoadError(f"{file_path} - İşleme sırasında beklenmeyen hata: {e}")


    def convert_to_tensor(self, token_ids: List[int], max_length: Optional[int] = None) -> torch.Tensor:
        length = max_length if max_length is not None else self.max_length
        try:
            if token_ids is None or len(token_ids) == 0:
                raise DataLoadError("Tensor verisi boş veya geçersiz!")
            return self.tensorizer.tensorize_text(token_ids, length)
        except Exception as e:
            logger.error(f"convert_to_tensor() sırasında hata: {e}", exc_info=True)
            raise DataLoadError(f"Tensor dönüşümü sırasında hata oluştu: {e}")

    def tensorize_batch(self, batch: List[List[int]], max_length: Optional[int] = None) -> torch.Tensor:
        length = max_length if max_length is not None else self.max_length
        try:
            if batch is None or len(batch) == 0:
                raise DataLoadError("Batch tensor verisi boş veya geçersiz!")
            return self.tensorizer.tensorize_batch_text(batch, length)
        except Exception as e:
            logger.error(f"tensorize_batch() sırasında hata: {e}", exc_info=True)
            raise DataLoadError(f"Batch tensor dönüşümü sırasında hata oluştu: {e}")
        
    def prepare_data_loader(self, tokenized_data: List[Tuple[List[int], List[int]]], drop_last: bool = True) -> Optional[TorchDataLoader]:
        """
        Tokenize edilmiş veriyi TensorDataset üzerinden DataLoader formatına dönüştürür.
        
        Args:
            tokenized_data (List[Tuple[List[int], List[int]]]): (input_ids, target_ids) çiftlerinden oluşan liste.
            drop_last (bool): Son batch eksikse atlama seçeneği.
        
        Returns:
            Optional[TorchDataLoader]: Tensor tabanlı DataLoader döner.
        
        Raises:
            DataLoaderError: Geçersiz tensor formatı veya tensor işlemi hatası durumunda fırlatılır.
        """
        try:
            if not tokenized_data:
                logger.warning("[!] Tokenize edilmiş veri boş geldi!")
                return None

            if not isinstance(tokenized_data, list):
                raise DataLoaderError(f"Geçersiz format: tokenized_data bir liste olmalıdır. Alınan tip: {type(tokenized_data)}")

            inputs, targets = [], []

            for idx, item in enumerate(tokenized_data):
                if not isinstance(item, tuple) or len(item) != 2:
                    raise DataLoaderError(
                        f"Geçersiz format ({idx}. indeks): {item} → Beklenen format (input, target)"
                    )

                input_ids, target_ids = item

                if not isinstance(input_ids, list) or not isinstance(target_ids, list):
                    raise DataLoaderError(
                        f"({idx}. indeks) input veya target list değil → input: {type(input_ids)}, target: {type(target_ids)}"
                    )

                input_tensor = torch.tensor(input_ids, dtype=torch.long)
                target_tensor = torch.tensor(target_ids, dtype=torch.long)

                if input_tensor.size(0) != target_tensor.size(0):
                    raise DataLoaderError(
                        f"Tensor boyut uyumsuzluğu ({idx}. indeks): input_tensor -> {input_tensor.size()}, "
                        f"target_tensor -> {target_tensor.size()}"
                    )

                inputs.append(input_tensor)
                targets.append(target_tensor)

            try:
                inputs_tensor = torch.stack(inputs)
                targets_tensor = torch.stack(targets)
            except RuntimeError as e:
                logger.error(f"[X] Tensor birleştirme hatası: {e}")
                raise DataLoaderError(f"Tensor birleştirme hatası: {e}") from e

            dataset = TensorDataset(inputs_tensor, targets_tensor)

            if len(dataset) == 0:
                raise DataLoaderError("Oluşturulan dataset boş. Tensor yüklemesi başarısız.")

            try:
                data_loader = TorchDataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=drop_last
                )
            except Exception as e:
                logger.error(f"[X] DataLoader oluşturulurken hata oluştu: {e}")
                raise DataLoaderError(f"DataLoader oluşturulurken hata oluştu: {e}") from e

            logger.info(f"[✓] DataLoader başarıyla hazırlandı. Toplam batch sayısı: {len(data_loader)}")
            return data_loader

        except DataLoaderError as e:
            logger.error(f"DataLoader hazırlanırken hata: {e}")
            raise

        except Exception as e:
            logger.error(f"Beklenmedik hata oluştu: {e}\n{traceback.format_exc()}")
            raise DataLoaderError(f"DataLoader hazırlanırken beklenmedik hata: {e}") from e


