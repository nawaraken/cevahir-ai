import torch
import logging
import gc  # Garbage Collector (çöp toplayıcı)

# Proje kök dizinini ekleme (gerekirse)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from .memory_manager_module.memory_utils_module.memory_initializer import MemoryInitializer
from .memory_manager_module.memory_allocator import MemoryAllocator
from .memory_manager_module.memory_attention_bridge import MemoryAttentionBridge
from .memory_manager_module.memory_optimizer import MemoryOptimizer

class MemoryManager:
    """
    MemoryManager, bellek tahsisi, başlatma, optimizasyon ve dikkat köprüsü işlemlerini yönetir.
    Overfitting'i önlemek için dinamik veri saklama ve tekrar eden veriyi önleme mekanizması içerir.
    Ek olarak, her adımda tensor istatistikleri (min, max, mean, std) loglanarak sürecin
    detaylı takibi sağlanır.
    """
    _logger_initialized = False  

    def __init__(self, init_type="xavier", expand_tensors=False, log_level=logging.INFO, enable_gc=True):
        """
        MemoryManager sınıfını başlatır.

        Args:
            init_type (str): Başlangıç türü (örn. "xavier", "he", "normal").
            expand_tensors (bool): Tensor boyutlarının genişletilip genişletilmeyeceğini belirler.
            log_level (int): Log seviyesi (örn. logging.DEBUG, logging.INFO).
            enable_gc (bool): Çöp toplayıcının aktif olup olmadığını belirler.
        """
        self.expand_tensors = expand_tensors
        self.memory_allocated = False
        self.enable_gc = enable_gc
        self.memory_storage = {}  # Bellek deposu
        
        # Logger tanımlama
        self.logger = logging.getLogger(self.__class__.__name__)
        if not MemoryManager._logger_initialized:
            self.logger.setLevel(log_level)
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            MemoryManager._logger_initialized = True  

        # Modül başlatma
        try:
            self.initializer = MemoryInitializer(init_type=init_type, log_level=log_level)
            self.allocator = MemoryAllocator(log_level=log_level)
            self.attention_bridge = MemoryAttentionBridge(log_level=log_level)
            self.optimizer = MemoryOptimizer(log_level=log_level)
            self.logger.info("MemoryManager successfully initialized.")
        except Exception as e:
            self.logger.critical(f"[ERROR] MemoryManager initialization failed: {e}", exc_info=True)
            raise RuntimeError("MemoryManager initialization failed.") from e

    def allocate_memory(self, size, dtype=torch.float32, device='cpu'):
        """Bellek tahsisi yapar."""
        try:
            self.memory_allocated = True  
            tensor = self.allocator.allocate_memory(size, dtype, device)
            self.logger.debug(f"Allocated memory: size={size}, dtype={dtype}, device={device}")
            self.logger.debug(f"Allocated tensor stats - shape: {tensor.shape}, min: {tensor.min().item():.6f}, "
                              f"max: {tensor.max().item():.6f}, mean: {tensor.mean().item():.6f}, std: {tensor.std().item():.6f}")
            return tensor
        except Exception as e:
            self.logger.error(f"[ERROR] allocate_memory() failed: {e}", exc_info=True)
            raise

    def initialize_memory(self, tensor):
        """Bellek alanını başlatır."""
        try:
            tensor = self.initializer.initialize_memory(tensor)
            self.logger.debug(f"Initialized memory: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            self.logger.debug(f"Initialized tensor stats - min: {tensor.min().item():.6f}, max: {tensor.max().item():.6f}, "
                              f"mean: {tensor.mean().item():.6f}, std: {tensor.std().item():.6f}")
            return tensor
        except Exception as e:
            self.logger.error(f"[ERROR] initialize_memory() failed: {e}", exc_info=True)
            raise

    def optimize_memory(self, tensor):
        """Bellek alanını optimize eder."""
        try:
            tensor = self.optimizer.optimize_memory(tensor)
            self.logger.debug(f"Optimized memory: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            self.logger.debug(f"Optimized tensor stats - min: {tensor.min().item():.6f}, max: {tensor.max().item():.6f}, "
                              f"mean: {tensor.mean().item():.6f}, std: {tensor.std().item():.6f}")
            return tensor
        except Exception as e:
            self.logger.error(f"[ERROR] optimize_memory() failed: {e}", exc_info=True)
            raise

    def bridge_attention(self, memory_tensor, attention_tensor):
        """
        Bellek ve dikkat mekanizmaları arasında köprü kurar.
        """
        try:
            self.validate_tensor(memory_tensor)
            self.validate_tensor(attention_tensor)

            combined = self.attention_bridge.bridge_attention(memory_tensor, attention_tensor)
            self.logger.debug(f"Bridged attention: shape={combined.shape}, dtype={combined.dtype}")
            self.logger.debug(f"Bridged tensor stats - min: {combined.min().item():.6f}, max: {combined.max().item():.6f}, "
                              f"mean: {combined.mean().item():.6f}, std: {combined.std().item():.6f}")
            return combined
        except Exception as e:
            self.logger.error(f"[ERROR] bridge_attention() failed: {e}", exc_info=True)
            raise

    def store(self, key, tensor):
        """
        Bellek deposuna bir tensörü kaydeder. Eğer aynı anahtar altında
        saklanan tensör ile yeni tensör aynı ise, üzerine yazmayı önler.

        Args:
            key (str): Tensörün saklanacağı anahtar.
            tensor (torch.Tensor): Saklanacak tensör.

        Raises:
            TypeError: Eğer tensör geçerli bir torch.Tensor değilse.
            ValueError: Eğer tensör beklenen boyutlara sahip değilse.
            RuntimeError: Saklama işlemi sırasında beklenmeyen bir hata oluşursa.
        """
        import time
        t_start = time.time()
        try:
            # Giriş tensörünü doğrula
            self.validate_tensor(tensor)
            self.logger.debug(f"[store] Input tensor for key '{key}' validated: shape={tensor.shape}, dtype={tensor.dtype}")

            # Eğer aynı anahtarda zaten bir tensör varsa, aynı olup olmadığını kontrol et
            if key in self.memory_storage:
                existing_tensor = self.memory_storage[key]
                if torch.equal(existing_tensor, tensor):
                    self.logger.warning(f"[store] Redundant tensor detected for key '{key}'. Overwriting prevented.")
                    return

            # Tensörü detach edip, clone'layarak sakla (hesaplama grafiğinden ayırmak için)
            stored_tensor = tensor.detach().clone()
            self.memory_storage[key] = stored_tensor

            # Saklanan tensörün temel istatistiklerini logla
            self.logger.info(f"[store] Tensor stored with key '{key}': shape={stored_tensor.shape}, dtype={stored_tensor.dtype}")
            self.logger.debug(
                f"[store] Stored tensor stats: min={stored_tensor.min().item():.6f}, "
                f"max={stored_tensor.max().item():.6f}, mean={stored_tensor.mean().item():.6f}, "
                f"std={stored_tensor.std().item():.6f}"
            )
            t_elapsed = time.time() - t_start
            self.logger.debug(f"[store] store() completed in {t_elapsed:.6f} seconds.")
        except Exception as e:
            self.logger.error(f"[store] Failed to store tensor for key '{key}': {e}", exc_info=True)
            raise



    def retrieve(self, key):
        """
        Bellekte kaydedilen tensörü geri döndürür.
        """
        import time
        start_time = time.time()
        try:
            if key not in self.memory_storage:
                self.logger.error(f"[ERROR] Key '{key}' not found in memory.")
                raise KeyError(f"Key '{key}' not found in memory.")

            tensor = self.memory_storage[key]
            self.logger.info(f"Tensor retrieved from memory with key: {key}, shape: {tensor.shape}, dtype: {tensor.dtype}")
            self.logger.debug(
                f"Retrieved tensor stats - min: {tensor.min().item():.6f}, max: {tensor.max().item():.6f}, "
                f"mean: {tensor.mean().item():.6f}, std: {tensor.std().item():.6f}"
            )
            total_time = time.time() - start_time
            self.logger.debug(f"retrieve() completed in {total_time:.6f} seconds.")
            return tensor
        except Exception as e:
            self.logger.error(f"[ERROR] retrieve() failed for key '{key}': {e}", exc_info=True)
            raise


    def validate_tensor(self, tensor):
        """
        Tensörün geçerliliğini kontrol eder.

        Raises:
            TypeError: Eğer tensör torch.Tensor değilse.
            ValueError: Eğer tensör boyutları geçersizse.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("[ERROR] Input must be a torch.Tensor.")
        if tensor.dim() < 2:
            raise ValueError("[ERROR] Tensor must have at least 2 dimensions.")

    def check_redundancy(self, old_tensor, new_tensor):
        """
        Aynı tensörün tekrar saklanmasını önler.

        Returns:
            bool: Eğer eski ve yeni tensörler aynıysa True döndürür.
        """
        return torch.equal(old_tensor, new_tensor)

    def clear_memory(self):
        """
        Bellekteki tüm tensörleri temizler.
        """
        num_keys = len(self.memory_storage)
        self.memory_storage.clear()
        if self.enable_gc:
            gc.collect()
        self.logger.info(f"Memory cleared successfully. {num_keys} keys removed.")

    def enforce_strict_gc(self):
        """
        Garbage Collector'ı zorla çalıştırarak gereksiz bellek kullanımını sıfırlar.
        """
        try:
            gc.collect()
            self.logger.info("Strict garbage collection enforced successfully.")
        except Exception as e:
            self.logger.critical(f"[ERROR] Failed to enforce strict GC: {e}", exc_info=True)
            raise RuntimeError(f"Failed to enforce strict GC: {e}") from e
