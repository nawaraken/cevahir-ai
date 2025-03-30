import torch
import logging
import gc  # Garbage Collector (çöp toplayıcı)

class MemoryOptimizer:
    """
    MemoryOptimizer sınıfı, bellek yönetimi ve optimizasyon işlemlerini gerçekleştirir.
    Tensörleri sıkıştırır, gereksiz bellek kullanımını önler ve önbelleği temizler.
    """
    
    _logger_initialized = False  # Sınıf düzeyinde log başlatmanın tek seferlik olup olmadığını kontrol eder.

    def __init__(self, enable_gc=True, log_level=logging.INFO):
        """
        MemoryOptimizer sınıfını başlatır.

        Args:
            enable_gc (bool): Garbage Collector (çöp toplayıcı) etkinleştirilsin mi?
            log_level (int): Log seviyesi (örn. logging.DEBUG, logging.INFO).
        """
        self.enable_gc = enable_gc  # Çöp toplayıcının aktif olup olmadığını belirler
        self.logger = logging.getLogger(self.__class__.__name__)

        if not MemoryOptimizer._logger_initialized:
            self.logger.setLevel(log_level)
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

            MemoryOptimizer._logger_initialized = True  # Log başlatma sadece bir kez çalışır.

        self.logger.info(f"MemoryOptimizer initialized successfully. GC Enabled: {self.enable_gc}")

    def optimize_memory(self, tensor):
        """
        Bellek optimizasyonu işlemlerini gerçekleştirir.
        Tensörü sıkıştırır, bellek tüketimini minimize eder ve işlem süresini loglar.

        Args:
            tensor (torch.Tensor): Optimize edilecek tensör.

        Returns:
            torch.Tensor: Optimize edilmiş tensör.
        
        Raises:
            TypeError: Eğer tensör bir torch.Tensor değilse.
            RuntimeError: Bellek optimizasyonu sırasında hata oluşursa.
        """
        import time
        start_time = time.time()
        try:
            # Giriş doğrulaması
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("[ERROR] Input must be a torch.Tensor.")
            if tensor.numel() == 0:
                raise ValueError("[ERROR] Input tensor is empty.")
            self.logger.debug(f"[optimize_memory] Starting optimization for tensor of shape {tensor.shape} on device {tensor.device}.")
            self._log_memory_usage(tensor)

            # Tensörü hesaplama grafiğinden ayır ve contiguous hale getir
            t_detach_start = time.time()
            optimized_tensor = tensor.detach().clone().contiguous()
            t_detach = time.time()
            self.logger.debug(f"[optimize_memory] Detach and clone time: {t_detach - t_detach_start:.6f} seconds.")

            # Bellek kullanımını logla (son hali)
            self._log_memory_usage(optimized_tensor)
            
            total_time = time.time() - start_time
            self.logger.info(f"[optimize_memory] Memory optimization completed in {total_time:.6f} seconds.")
            self._log_memory_usage(optimized_tensor)
            return optimized_tensor

        except TypeError as e:
            self.logger.error(f"[ERROR] Invalid input for memory optimization: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.critical(f"[ERROR] Failed to optimize memory: {e}", exc_info=True)
            raise RuntimeError(f"Failed to optimize memory: {e}") from e


    def compact_memory(self):
        """
        Bellek alanını sıkıştırarak daha verimli kullanım sağlar.
        Kullanılmayan tensörleri temizler ve CUDA belleğini sıfırlar.
        """
        try:
            #  Kullanılmayan tensörleri sil
            gc.collect()

            #  CUDA kullanılabilir mi kontrol et
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            self.logger.info("[INFO] Memory compacted successfully.")
        except Exception as e:
            self.logger.critical(f"[ERROR] Failed to compact memory: {e}", exc_info=True)
            raise RuntimeError(f"Failed to compact memory: {e}") from e

    def clear_cache(self):
        """
        Bellek önbelleğini temizler ve gereksiz yer kaplayan bellek alanlarını serbest bırakır.
        """
        try:
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info("[INFO] Cache cleared successfully.")
        except Exception as e:
            self.logger.critical(f"[ERROR] Failed to clear cache: {e}", exc_info=True)
            raise RuntimeError(f"Failed to clear cache: {e}") from e

    def release_memory(self, tensor):
        """
        Belleği serbest bırakır ve gereksiz verileri temizler.

        Args:
            tensor (torch.Tensor): Serbest bırakılacak tensör.

        Raises:
            TypeError: Eğer giriş bir torch.Tensor değilse.
            RuntimeError: Bellek serbest bırakma sırasında hata oluşursa.
        """
        try:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("[ERROR] Input must be a torch.Tensor.")
            
            #  Tensörü serbest bırak
            del tensor
            gc.collect()

            #  CUDA Belleğini sıfırla
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info("[INFO] Memory released successfully.")
        except TypeError as e:
            self.logger.error(f"[ERROR] Invalid input for memory release: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.critical(f"[ERROR] Failed to release memory: {e}", exc_info=True)
            raise RuntimeError(f"Failed to release memory: {e}") from e

    def _log_memory_usage(self, tensor):
        """
        Bellek kullanımı ile ilgili loglama yapar.

        Args:
            tensor (torch.Tensor): Bellek kullanımı loglanacak tensör.
        """
        self.logger.info("[INFO] Memory usage logged.")
        self.logger.debug(f"[DEBUG] Tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

    def enforce_strict_gc(self):
        """
        Garbage Collector'ı (çöp toplayıcı) zorla çalıştırarak gereksiz bellek kullanımını sıfırlar.
        """
        try:
            gc.collect()
            self.logger.info("[INFO] Strict garbage collection enforced successfully.")
        except Exception as e:
            self.logger.critical(f"[ERROR] Failed to enforce strict GC: {e}", exc_info=True)
            raise RuntimeError(f"Failed to enforce strict GC: {e}") from e
