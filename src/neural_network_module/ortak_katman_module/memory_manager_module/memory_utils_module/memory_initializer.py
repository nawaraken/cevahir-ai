import logging
import torch

class MemoryInitializer:
    """
    MemoryInitializer sınıfı, bellek işlemleri için gerekli başlangıç değerlerini sağlar.
    """
    def __init__(self, init_type="xavier", device=None, low_precision=False, log_level=logging.INFO):
        """
        MemoryInitializer sınıfını başlatır.

        Args:
            init_type (str): Başlangıç türü ("xavier", "he", "normal", "orthogonal", "truncated_normal", "lecun", "uniform").
            device (str): Tensorlerin çalışacağı cihaz ('cpu', 'cuda' veya None - otomatik belirleme).
            low_precision (bool): Hafızayı optimize etmek için düşük hassasiyetli başlatma (float16) kullan.
            log_level (int): Log seviyesi.
        """
        self.init_type = init_type.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.low_precision = low_precision
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(f"MemoryInitializer initialized with init_type={self.init_type}, device={self.device}, low_precision={self.low_precision}")

    def initialize_memory(self, tensor):
        """
        Bellek için gerekli başlangıç değerlerini sağlar.

        Args:
            tensor (torch.Tensor): Başlatılacak tensör.

        Returns:
            torch.Tensor: Başlatılmış tensör.

        Raises:
            TypeError, ValueError: Giriş tensörü doğrulama hatası.
            RuntimeError: Başlatma sırasında beklenmeyen bir hata oluşursa.
        """
        import time
        start_time = time.time()
        try:
            # 1. Giriş doğrulaması
            self.validate_initialization(tensor)
            self.logger.debug(f"[initialize_memory] Input tensor validated: shape={tensor.shape}, dtype={tensor.dtype}")

            # 2. Tensörü belirtilen cihaza taşı
            tensor = tensor.to(self.device)
            self.logger.debug(f"[initialize_memory] Tensor moved to device: {self.device}")

            # 3. İstenen precision'a göre veri tipi belirle
            dtype = torch.float16 if self.low_precision else torch.float32

            # 4. Başlatma tipine göre uygun yöntemi uygula
            if self.init_type == "xavier":
                torch.nn.init.xavier_uniform_(tensor)
                self.logger.debug("[initialize_memory] Xavier uniform initialization applied.")
            elif self.init_type == "he":
                torch.nn.init.kaiming_uniform_(tensor, nonlinearity='relu')
                self.logger.debug("[initialize_memory] Kaiming uniform (He) initialization applied.")
            elif self.init_type == "normal":
                torch.nn.init.normal_(tensor, mean=0, std=0.02)
                self.logger.debug("[initialize_memory] Normal initialization applied.")
            elif self.init_type == "orthogonal":
                torch.nn.init.orthogonal_(tensor)
                self.logger.debug("[initialize_memory] Orthogonal initialization applied.")
            elif self.init_type == "truncated_normal":
                self._truncated_normal_(tensor, mean=0, std=0.02)
                self.logger.debug("[initialize_memory] Truncated normal initialization applied.")
            elif self.init_type == "lecun":
                torch.nn.init.kaiming_normal_(tensor, nonlinearity='linear')
                self.logger.debug("[initialize_memory] Lecun initialization (kaiming normal with linear nonlinearity) applied.")
            elif self.init_type == "uniform":
                torch.nn.init.uniform_(tensor, a=-0.1, b=0.1)
                self.logger.debug("[initialize_memory] Uniform initialization applied.")
            else:
                raise ValueError(f"Invalid initialization type: {self.init_type}")

            # 5. Tensörü istenen veri tipine dönüştür
            tensor = tensor.to(dtype)
            self.logger.debug(f"[initialize_memory] Tensor converted to dtype: {dtype}")

            total_time = time.time() - start_time
            self.logger.info(f"[initialize_memory] Memory initialization completed in {total_time:.6f} seconds.")
            self.log_initialization(tensor)
            return tensor

        except (TypeError, ValueError) as ve:
            self.logger.error(f"[initialize_memory] Memory initialization validation error: {ve}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"[initialize_memory] Unexpected error during memory initialization: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize memory: {e}")


    def validate_initialization(self, tensor):
        """
        Bellek başlatma için giriş doğrulaması yapar.

        Args:
            tensor (torch.Tensor): Doğrulanacak tensör.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if tensor.dim() < 2:
            raise ValueError("Tensor must have at least 2 dimensions.")
        if tensor.numel() == 0:
            raise ValueError("Tensor must have at least one element.")
        if tensor.requires_grad:
            self.logger.warning("Tensor has requires_grad=True, consider detaching before initialization.")

    def log_initialization(self, tensor):
        """
        Bellek başlatma sırasında log kaydı tutar.

        Args:
            tensor (torch.Tensor): Loglanacak tensör.
        """
        self.logger.debug(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
        self.logger.debug(f"Tensor mean: {tensor.mean().item():.6f}, std: {tensor.std().item():.6f}")

    @staticmethod
    def _truncated_normal_(tensor, mean=0, std=1):
        """
        Truncated normal dağılım ile tensör başlatma.

        Args:
            tensor (torch.Tensor): Başlatılacak tensör.
            mean (float): Ortalama.
            std (float): Standart sapma.
        """
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.mul_(std).add_(mean)
