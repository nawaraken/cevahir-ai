import torch
import logging

class TensorNormalizer:
    """
    TensorNormalizer sınıfı, tensörler üzerinde farklı normalizasyon işlemlerini gerçekleştirir.
    """

    def __init__(self, log_level=logging.INFO):
        """
        TensorNormalizer sınıfını başlatır.

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

        self._log_once(f"TensorNormalizer initialized with log_level={log_level}")

    def _log_once(self, message):
        """
        Aynı log mesajlarının tekrar tekrar yazılmasını önler.

        Args:
            message (str): Log mesajı
        """
        if message not in self.logger.seen_messages:
            self.logger.info(message)
            self.logger.seen_messages.add(message)

    def normalize_batch(self, tensor):
        """
        Batch normalizasyon işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Normalizasyon yapılacak tensör.

        Returns:
            torch.Tensor: Normalizasyon yapılmış tensör.
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.error("Input must be a torch.Tensor.")
            raise TypeError("Input must be a torch.Tensor.")

        if tensor.numel() == 0:
            self.logger.error("Attempted to normalize an empty tensor.")
            raise ValueError("Cannot normalize an empty tensor.")

        self.logger.debug("Starting Batch Normalization...")
        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True)
        normalized_tensor = (tensor - mean) / (std + 1e-5)

        self._log_normalization(tensor, normalized_tensor, "Batch Normalization")
        return normalized_tensor

    def normalize_layer(self, tensor):
        """
        Layer normalizasyon işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Normalizasyon yapılacak tensör.

        Returns:
            torch.Tensor: Normalizasyon yapılmış tensör.
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.error("Input must be a torch.Tensor.")
            raise TypeError("Input must be a torch.Tensor.")

        if tensor.numel() == 0:
            self.logger.error("Attempted to normalize an empty tensor.")
            raise ValueError("Cannot normalize an empty tensor.")

        self.logger.debug("Starting Layer Normalization...")
        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

        mean = tensor.mean(dim=-1, keepdim=True)
        std = tensor.std(dim=-1, keepdim=True)
        normalized_tensor = (tensor - mean) / (std + 1e-5)

        self._log_normalization(tensor, normalized_tensor, "Layer Normalization")
        return normalized_tensor

    def normalize_instance(self, tensor):
        """
        Instance normalizasyon işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Normalizasyon yapılacak tensör.

        Returns:
            torch.Tensor: Normalizasyon yapılmış tensör.
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.error("Input must be a torch.Tensor.")
            raise TypeError("Input must be a torch.Tensor.")

        if tensor.numel() == 0:
            self.logger.error("Attempted to normalize an empty tensor.")
            raise ValueError("Cannot normalize an empty tensor.")

        self.logger.debug("Starting Instance Normalization...")
        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

        mean = tensor.mean(dim=(2, 3), keepdim=True)
        std = tensor.std(dim=(2, 3), keepdim=True)
        normalized_tensor = (tensor - mean) / (std + 1e-5)

        self._log_normalization(tensor, normalized_tensor, "Instance Normalization")
        return normalized_tensor

    def normalize_group(self, tensor, num_groups):
        """
        Group normalizasyon işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Normalizasyon yapılacak tensör.
            num_groups (int): Grup sayısı.

        Returns:
            torch.Tensor: Normalizasyon yapılmış tensör.
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.error("Input must be a torch.Tensor.")
            raise TypeError("Input must be a torch.Tensor.")

        if tensor.numel() == 0:
            self.logger.error("Attempted to normalize an empty tensor.")
            raise ValueError("Cannot normalize an empty tensor.")

        self.logger.debug("Starting Group Normalization...")
        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

        original_shape = tensor.shape
        if tensor.dim() == 2:  # Eğer tensör 2 boyutluysa
            tensor = tensor.unsqueeze(2).unsqueeze(3)  # 4 boyutlu hale getir
        elif tensor.dim() != 4:
            self.logger.error("Unsupported tensor dimensionality for group normalization.")
            raise ValueError("Unsupported tensor dimensionality for group normalization.")

        N, C, H, W = tensor.shape
        G = num_groups

        # **Hata kontrolü: G, C'yi tam bölmelidir**
        if C % G != 0:
            self.logger.error(f"Invalid num_groups={G}. It must be a divisor of C={C}.")
            raise ValueError(f"Invalid num_groups={G}. It must be a divisor of C={C}.")

        tensor = tensor.view(N, G, C // G, H, W)
        mean = tensor.mean(dim=(2, 3, 4), keepdim=True)

        # **Güncellenmiş Standart Sapma Hesaplaması**
        std = tensor.std(dim=(2, 3, 4), keepdim=True, unbiased=False)  # unbiased=False eklenerek DoF hatası önlenir

        tensor = (tensor - mean) / (std + 1e-5)
        normalized_tensor = tensor.view(N, C, H, W)

        if original_shape != normalized_tensor.shape:
            normalized_tensor = normalized_tensor.view(original_shape)

        self._log_normalization(tensor, normalized_tensor, "Group Normalization")
        return normalized_tensor


    def _log_normalization(self, original_tensor, normalized_tensor, normalization_type):
        """
        Normalizasyon işlemleri sırasında loglama yapar.

        Args:
            original_tensor (torch.Tensor): Orijinal tensör.
            normalized_tensor (torch.Tensor): Normalizasyon yapılmış tensör.
            normalization_type (str): Normalizasyon türü.
        """
        self._log_once(f"{normalization_type} completed successfully.")
        self.logger.debug(f"Original tensor shape: {original_tensor.shape}, dtype: {original_tensor.dtype}, device: {original_tensor.device}")
        self.logger.debug(f"Normalized tensor shape: {normalized_tensor.shape}, dtype: {normalized_tensor.dtype}, device: {normalized_tensor.device}")
