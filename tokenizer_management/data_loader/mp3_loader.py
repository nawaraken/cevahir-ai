import os
import logging
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import resample

logger = logging.getLogger(__name__)

# ======================================================
#  Özel Hata Sınıfları
# ======================================================

class MP3LoaderError(Exception):
    pass

class FileFormatError(MP3LoaderError):
    pass

class AudioProcessingError(MP3LoaderError):
    pass

class InvalidAudioDataError(MP3LoaderError):
    pass

# ======================================================
#  MP3Loader Sınıfı
# ======================================================

class MP3Loader:
    """
    MP3Loader, MP3 ses dosyalarını yükler ve MFCC özelliklerini çıkarır.
    
    - Gürültü azaltma ve sinyal temizleme desteği
    - Delta ve hızlanma (acceleration) çıkarımı
    - Bellek dostu ve büyük dosya desteği
    """

    def __init__(self, 
                 sr: int = 22050, 
                 n_mfcc: int = 13, 
                 normalize: str = 'z-score', 
                 duration: float = 5.0, 
                 noise_reduction: bool = True):
        """
        Args:
            sr (int): Örnekleme hızı (Hz). Varsayılan 22050 Hz.
            n_mfcc (int): Çıkarılacak MFCC özellik sayısı. Varsayılan 13.
            normalize (str): 'z-score', 'min-max' veya 'none' olabilir.
            duration (float): Ses dosyasının maksimum uzunluğu (saniye).
            noise_reduction (bool): Gürültü azaltma uygulansın mı?
        """
        if not isinstance(sr, int) or sr <= 0:
            raise ValueError(f"`sr` pozitif bir tam sayı olmalıdır, ancak {sr} alındı.")

        if not isinstance(n_mfcc, int) or n_mfcc <= 0:
            raise ValueError(f"`n_mfcc` pozitif bir tam sayı olmalıdır, ancak {n_mfcc} alındı.")

        if normalize not in ['z-score', 'min-max', 'none']:
            raise ValueError(f"normalize yalnızca 'z-score', 'min-max' veya 'none' olabilir. Alınan: {normalize}")

        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"`duration` pozitif bir sayı olmalıdır, ancak {duration} alındı.")

        self.sr = sr
        self.n_mfcc = n_mfcc
        self.normalize = normalize
        self.duration = duration
        self.noise_reduction = noise_reduction

    # ======================================================
    #  Dosya Yükleme
    # ======================================================

    def load_file(self, file_path: str) -> np.ndarray:
        """
        MP3 dosyasını yükler ve MFCC özelliklerini çıkarır.
        
        Args:
            file_path (str): Yüklenecek MP3 dosyasının tam yolu.

        Returns:
            np.ndarray: (n_mfcc,) boyutunda MFCC özellik vektörü.
        
        Raises:
            FileFormatError: Desteklenmeyen format hatası.
            AudioProcessingError: Ses dosyası işlenemediğinde.
            InvalidAudioDataError: Bozuk veya geçersiz dosya.
            Exception: Diğer genel hatalar.
        """
        if not isinstance(file_path, str) or not file_path.endswith('.mp3'):
            raise FileFormatError(f"Geçersiz dosya formatı: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

        logger.info(f" MP3 dosyası yükleniyor: {file_path}")

        try:
            #  Dosyayı yükle
            y, sr = librosa.load(file_path, sr=self.sr, mono=True, duration=self.duration)
            logger.info(f" MP3 dosyası yüklendi -> SR: {sr}, Uzunluk: {len(y)} örnek")

            if len(y) == 0:
                raise InvalidAudioDataError("Ses dosyası boş!")

            #  Gürültü azaltma (Noise Reduction)
            if self.noise_reduction:
                y = self._reduce_noise(y)

            #  Zaman boyutunu sabitle (Padding veya Trimming)
            y = self._fix_length(y)

            #  MFCC özellik çıkarımı
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

            #  Delta (Eğim) ve Acceleration (Hızlanma) çıkarımı
            delta = librosa.feature.delta(mfcc)
            delta_delta = librosa.feature.delta(mfcc, order=2)

            #  Tüm özellikleri birleştir
            mfcc_features = np.concatenate([mfcc, delta, delta_delta], axis=0)

            #  Ortalama al
            mfcc_mean = np.mean(mfcc_features, axis=1)

            #  Normalizasyon
            mfcc_mean = self._normalize(mfcc_mean)

            logger.info(f" Çıkarılan özellik şekli: {mfcc_mean.shape}")

            return mfcc_mean

        except Exception as e:
            logger.error(f" MP3 dosyası işlenirken hata oluştu: {e}", exc_info=True)
            raise AudioProcessingError(f"MP3 dosyası işlenirken hata oluştu: {e}")

    # ======================================================
    #  Gürültü Azaltma
    # ======================================================

    def _reduce_noise(self, y: np.ndarray) -> np.ndarray:
        try:
            noise_sample = y[:self.sr]  # İlk 1 saniyeyi noise örneği olarak al
            y_denoised = y - noise_sample.mean()
            return y_denoised
        except Exception as e:
            logger.warning(f"Gürültü azaltma sırasında hata oluştu: {e}")
            return y

    # ======================================================
    #  Uzunluğu Sabitleme
    # ======================================================

    def _fix_length(self, y: np.ndarray) -> np.ndarray:
        target_length = int(self.sr * self.duration)
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, target_length - len(y)))
        return y

    # ======================================================
    #  Normalizasyon
    # ======================================================

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        if self.normalize == 'z-score':
            scaler = StandardScaler()
            return scaler.fit_transform(data.reshape(-1, 1)).flatten()

        if self.normalize == 'min-max':
            scaler = MinMaxScaler()
            return scaler.fit_transform(data.reshape(-1, 1)).flatten()

        return data
