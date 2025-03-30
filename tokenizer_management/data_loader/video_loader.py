import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ======================================================
#  Özel Hata Sınıfları
# ======================================================

class VideoLoaderError(Exception):
    pass

class InvalidFrameError(VideoLoaderError):
    pass

class VideoReadError(VideoLoaderError):
    pass

class FileFormatError(VideoLoaderError):
    pass

class FrameProcessError(VideoLoaderError):
    pass

# ======================================================
#  VideoLoader Sınıfı
# ======================================================

class VideoLoader:
    """
    VideoLoader, video dosyalarını yükleyen ve özetleyen bir sınıftır.

    Özellikler:
    - OpenCV kullanarak video dosyasını yükler.
    - Kare senkronizasyonu sağlar.
    - İstenilen kare sayısını eşit aralıklarla seçer.
    - Bellek dostu ve yüksek çözünürlük desteği sunar.
    """

    def __init__(self, desired_frames: int = 16, resize: Optional[tuple] = None):
        """
        Args:
            desired_frames (int): Elde edilmek istenen kare sayısı.
            resize (tuple, optional): Karelerin yeniden boyutlandırılacağı (genişlik, yükseklik) değeri.
        """
        if not isinstance(desired_frames, int) or desired_frames <= 0:
            raise ValueError(f"`desired_frames` pozitif bir tam sayı olmalıdır. Alınan değer: {desired_frames}")

        if resize and (not isinstance(resize, tuple) or len(resize) != 2):
            raise ValueError(f"`resize` değeri (width, height) formatında olmalıdır. Alınan: {resize}")

        self.desired_frames = desired_frames
        self.resize = resize

    # ======================================================
    #  Dosya Yükleme
    # ======================================================

    def load_file(self, file_path: str) -> np.ndarray:
        """
        Belirtilen video dosyasını yükler ve karelerini döndürür.

        Args:
            file_path (str): Yüklenecek video dosyasının yolu.

        Returns:
            np.ndarray: Yüklenen video karelerini içeren numpy array.
        
        Raises:
            FileNotFoundError: Dosya bulunamazsa.
            VideoReadError: Video dosyası okunamazsa.
            InvalidFrameError: Kare okuma hatası.
            FrameProcessError: Kare işleme hatası.
            Exception: Diğer tüm hatalar.
        """
        if not isinstance(file_path, str):
            raise TypeError(f"`file_path` tipi `str` olmalıdır. Alınan: {type(file_path)}")

        if not file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise FileFormatError(f"Geçersiz dosya formatı: {file_path}")

        if not cv2.haveImageReader(file_path):
            raise FileNotFoundError(f"Video dosyası açılamadı veya format desteklenmiyor: {file_path}")

        logger.info(f" Video dosyası açılıyor: {file_path}")

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise VideoReadError(f"Video dosyası açılamadı: {file_path}")

        try:
            #  Toplam kare sayısını alalım
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise VideoReadError(f"Video dosyasındaki toplam kare sayısı geçersiz: {total_frames}")

            logger.info(f" Toplam kare sayısı: {total_frames}")

            #  İstenilen kare sayısına göre indeks hesaplayalım
            num_frames = min(self.desired_frames, total_frames)
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            logger.info(f" Seçilen kare sayısı: {num_frames}")

            frames = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f" Kare {idx} okunamadı, atlanıyor.")
                    continue

                #  BGR → RGB dönüşümünü hızlandırıyoruz
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except cv2.error as e:
                    logger.warning(f" Kare {idx} dönüştürülürken hata oluştu: {e}")
                    continue

                #  Yeniden boyutlandırma işlemi
                if self.resize:
                    try:
                        frame_rgb = cv2.resize(frame_rgb, self.resize, interpolation=cv2.INTER_AREA)
                    except cv2.error as e:
                        logger.warning(f" Kare {idx} yeniden boyutlandırılırken hata oluştu: {e}")
                        continue

                frames.append(frame_rgb)

            if not frames:
                raise InvalidFrameError(f"Video dosyasından geçerli kare okunamadı: {file_path}")

            #  Kareleri numpy array'e dönüştür
            video_summary = np.stack(frames, axis=0)

            if video_summary.shape[0] != num_frames:
                logger.warning(
                    f" Beklenen kare sayısı: {num_frames}, alınan: {video_summary.shape[0]}"
                )

            logger.info(f" Video özetleme tamamlandı -> Çıktı şekli: {video_summary.shape}")

            return video_summary

        except Exception as e:
            logger.error(f" Video dosyası yüklenirken hata oluştu: {e}", exc_info=True)
            raise

        finally:
            cap.release()
            logger.info(f" Video dosyası kapatıldı: {file_path}")

