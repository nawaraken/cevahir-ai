import torch
import logging

class ResidualInitializer:
    """
    ResidualInitializer sınıfı, tensörler üzerinde artık bağlantı işlemlerini başlatmak için gerekli olan işlemleri gerçekleştirir.
    """

    def __init__(self, log_level=logging.INFO):
        """
        ResidualInitializer sınıfını başlatır.

        Args:
            log_level (int): Log seviyesi (örn. logging.DEBUG, logging.INFO).
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Tekrarlanan logları engellemek için bir set kullan
        if not hasattr(self.logger, "seen_messages"):
            self.logger.seen_messages = set()

        self._log_once("ResidualInitializer initialized.")

    def _log_once(self, message):
        """
        Aynı log mesajlarının tekrar tekrar yazılmasını önler.

        Args:
            message (str): Log mesajı.
        """
        if message not in self.logger.seen_messages:
            self.logger.info(message)
            self.logger.seen_messages.add(message)

    def initialize(self, tensor, method="clone"):
        """
        Artık bağlantı işlemi için tensörleri başlatır.

        Args:
            tensor (torch.Tensor): Başlatılacak tensör.
            method (str): Başlatma yöntemi. Varsayılan: "clone".

        Returns:
            torch.Tensor: Başlatılmış tensör.
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.error("Input must be a torch.Tensor.")
            raise TypeError("Input must be a torch.Tensor.")

        if tensor.numel() == 0:
            self.logger.error("Tensor cannot be empty.")
            raise ValueError("Tensor cannot be empty.")

        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

        try:
            if method == "clone":
                initialized_tensor = self._clone_initialize(tensor)
                initialization_type = "Clone Initialization"
            elif method == "zeros":
                initialized_tensor = self._zeros_initialize(tensor)
                initialization_type = "Zeros Initialization"
            elif method == "random":
                initialized_tensor = self._random_initialize(tensor)
                initialization_type = "Random Initialization"
            else:
                self.logger.error(f"Unsupported initialization method: {method}")
                raise ValueError(f"Unsupported initialization method: {method}")

            self._log_initialization(tensor, initialized_tensor, initialization_type)
            return initialized_tensor

        except Exception as e:
            self.logger.error(f"Error in {method} initialization: {e}")
            raise

    def _clone_initialize(self, tensor):
        """Mevcut tensörü klonlayarak başlatır."""
        return tensor.clone().detach()

    def _zeros_initialize(self, tensor):
        """Tüm elemanları sıfır olan bir tensör oluşturur."""
        return torch.zeros_like(tensor)

    def _random_initialize(self, tensor):
        """Rastgele değerlere sahip bir tensör oluşturur."""
        return torch.randn_like(tensor)

    def _log_initialization(self, original_tensor, initialized_tensor, initialization_type):
        """
        Başlatma işlemleri sırasında loglama yapar.

        Args:
            original_tensor (torch.Tensor): Orijinal tensör.
            initialized_tensor (torch.Tensor): Başlatılmış tensör.
            initialization_type (str): Başlatma türü.
        """
        self._log_once(f"{initialization_type} completed.")
        self.logger.debug(f"Original tensor shape: {original_tensor.shape}, dtype: {original_tensor.dtype}, device: {original_tensor.device}")
        self.logger.debug(f"Initialized tensor shape: {initialized_tensor.shape}, dtype: {initialized_tensor.dtype}, device: {initialized_tensor.device}")
