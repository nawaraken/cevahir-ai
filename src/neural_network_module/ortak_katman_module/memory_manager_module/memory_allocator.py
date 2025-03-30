import torch
import logging
import gc  # Garbage Collector (çöp toplayıcı)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class MemoryAllocator:
    """
    MemoryAllocator sınıfı, bellek tahsisi, serbest bırakma ve yönetimi işlevlerini sağlar.
    """
    _last_allocation = None  # Önceki tahsis edilen bellek bilgilerini saklar
    _memory_cache = {}  # Tekrar kullanılabilir tensörleri saklar
    _logger_initialized = False  # Loglama sadece bir kez başlatılacak

    def __init__(self, enable_cache=True, device=None, log_level=logging.INFO):
        """
        MemoryAllocator sınıfını başlatır.

        Args:
            enable_cache (bool): Tekrar kullanılabilir bellekleri önbellekte sakla.
            device (str): Tensorlerin çalışacağı cihaz ('cpu', 'cuda' veya None - otomatik belirleme).
            log_level (int): Log seviyesi.
        """
        self.enable_cache = enable_cache
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)

        if not MemoryAllocator._logger_initialized:
            self.logger.setLevel(log_level)
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            MemoryAllocator._logger_initialized = True  # Log başlatma yalnızca bir kez yapılır.

        self.logger.info(f"MemoryAllocator initialized with device={self.device}, enable_cache={self.enable_cache}")

    def allocate_memory(self, size, dtype=torch.float32):
        """
        Belirtilen boyutta ve türde bellek tahsisi yapar.

        Args:
            size (tuple): Tahsis edilecek bellek boyutları.
            dtype (torch.dtype): Bellek türü (varsayılan: torch.float32).

        Returns:
            torch.Tensor: Tahsis edilen bellek tensörü.

        Raises:
            ValueError: Eğer boyutlar geçersizse.
            RuntimeError: Bellek tahsisi sırasında hata oluşursa.
        """
        import time
        start_time = time.time()
        try:
            # 1. Boyut doğrulaması
            self._validate_size(size)
            self.logger.debug(f"[allocate_memory] Size validated: {size}")

            # 2. Önbellek kontrolü
            cache_key = (size, dtype, self.device)
            if self.enable_cache and cache_key in MemoryAllocator._memory_cache:
                self.logger.info(f"[allocate_memory] Reusing cached tensor: size={size}, dtype={dtype}, device={self.device}")
                tensor = MemoryAllocator._memory_cache.pop(cache_key)
                total_time = time.time() - start_time
                self.logger.debug(f"[allocate_memory] Allocation (from cache) completed in {total_time:.6f}s")
                return tensor

            # 3. Yeni tensör tahsisi
            tensor = torch.empty(size, dtype=dtype, device=self.device)
            self.logger.info(f"[allocate_memory] New tensor allocated: size={size}, dtype={dtype}, device={self.device}")
            self._log_tensor_stats(tensor, "Allocated Tensor")

            # 4. Önceki tahsis bilgisi kontrolü
            current_allocation = (size, dtype, self.device)
            if MemoryAllocator._last_allocation != current_allocation:
                self.logger.info(f"[allocate_memory] New allocation details: {current_allocation}")
                MemoryAllocator._last_allocation = current_allocation

            total_time = time.time() - start_time
            self.logger.debug(f"[allocate_memory] Allocation completed in {total_time:.6f}s")
            return tensor

        except ValueError as ve:
            self.logger.error(f"[allocate_memory] Invalid size: {size}. Error: {ve}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"[allocate_memory] Failed to allocate memory: {e}", exc_info=True)
            raise RuntimeError(f"Failed to allocate memory: {e}")

    def release_memory(self, tensor):
        """
        Tahsis edilen belleği serbest bırakır veya önbelleğe alır.

        Args:
            tensor (torch.Tensor): Serbest bırakılacak tensör.

        Raises:
            TypeError: Eğer tensör bir torch.Tensor değilse.
            RuntimeError: Bellek serbest bırakma sırasında hata oluşursa.
        """
        import time
        start_time = time.time()
        try:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Input must be a torch.Tensor.")
            size, dtype, device = tensor.shape, tensor.dtype, tensor.device
            if self.enable_cache:
                MemoryAllocator._memory_cache[(size, dtype, device)] = tensor.detach()
                self.logger.info(f"[release_memory] Tensor cached for reuse: size={size}, dtype={dtype}, device={device}")
            else:
                del tensor
                gc.collect()
                self.logger.info("[release_memory] Tensor released and garbage collector invoked.")
            total_time = time.time() - start_time
            self.logger.debug(f"[release_memory] Memory release completed in {total_time:.6f}s")
        except TypeError as te:
            self.logger.error(f"[release_memory] Invalid input: {te}")
            raise
        except Exception as e:
            self.logger.error(f"[release_memory] Failed to release memory: {e}", exc_info=True)
            raise RuntimeError(f"Failed to release memory: {e}")

    def clear_cache(self):
        """
        Bellek önbelleğini temizler ve tüm saklanan tensörleri serbest bırakır.
        """
        try:
            num_keys = len(MemoryAllocator._memory_cache)
            MemoryAllocator._memory_cache.clear()
            gc.collect()
            self.logger.info(f"[clear_cache] Memory cache cleared successfully. {num_keys} keys removed.")
        except Exception as e:
            self.logger.error(f"[clear_cache] Failed to clear memory cache: {e}", exc_info=True)
            raise RuntimeError(f"Failed to clear memory cache: {e}")

    def check_memory_usage(self):
        """
        GPU belleği kullanımını ve önbellek durumunu kontrol eder.
        """
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            self.logger.info(f"[check_memory_usage] GPU Memory - Allocated: {memory_allocated:.2f} MB, Reserved: {memory_reserved:.2f} MB")
        cache_size = sum(t.numel() * t.element_size() for t in MemoryAllocator._memory_cache.values()) / (1024 ** 2)
        self.logger.info(f"[check_memory_usage] CPU Memory Cache Size: {cache_size:.2f} MB")

    def _validate_size(self, size):
        """
        Bellek boyutlarını doğrular.

        Args:
            size (tuple): Doğrulanacak boyutlar.

        Raises:
            ValueError: Eğer boyutlar geçersizse.
        """
        if not isinstance(size, tuple) or not all(isinstance(dim, int) and dim > 0 for dim in size):
            raise ValueError("Size must be a tuple of positive integers.")

    def _log_tensor_stats(self, tensor, tensor_name="Tensor"):
        """
        Tensörün istatistiklerini loglar.

        Args:
            tensor (torch.Tensor): Loglanacak tensör.
            tensor_name (str): Tensörün adı.
        """
        self.logger.debug(
            f"[{tensor_name} Stats] shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, "
            f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, "
            f"mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}"
        )
