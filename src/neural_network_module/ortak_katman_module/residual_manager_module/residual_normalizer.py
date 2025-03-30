import torch
import logging

class ResidualNormalizer:
    """
    ResidualNormalizer sınıfı, tensörler üzerinde normalizasyon işlemlerini gerçekleştirir.
    """

    def __init__(self, log_level=logging.INFO):
        """
        ResidualNormalizer sınıfını başlatır.

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

        self._log_once("ResidualNormalizer initialized.")

    def _log_once(self, message):
        """
        Aynı log mesajlarının tekrar tekrar yazılmasını önler.

        Args:
            message (str): Log mesajı.
        """
        if message not in self.logger.seen_messages:
            self.logger.info(message)
            self.logger.seen_messages.add(message)

    def normalize(self, tensor, method="standard"):
        """
        Normalizasyon işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Normalizasyon yapılacak tensör.
            method (str): Normalizasyon yöntemi. Varsayılan: "standard".

        Returns:
            torch.Tensor: Normalizasyon yapılmış tensör.
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.error("Input must be a torch.Tensor.")
            raise TypeError("Input must be a torch.Tensor.")

        if tensor.numel() == 0:
            self.logger.error("Tensor cannot be empty.")
            raise ValueError("Tensor cannot be empty.")

        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

        try:
            if method == "standard":
                normalized_tensor = self._standard_normalize(tensor)
                normalization_type = "Standard Normalization"
            elif method == "min_max":
                normalized_tensor = self._min_max_normalize(tensor)
                normalization_type = "Min-Max Normalization"
            elif method == "robust":
                normalized_tensor = self._robust_normalize(tensor)
                normalization_type = "Robust Normalization"
            else:
                self.logger.error(f"Unsupported normalization method: {method}")
                raise ValueError(f"Unsupported normalization method: {method}")

            self._log_normalization(tensor, normalized_tensor, normalization_type)
            return normalized_tensor

        except Exception as e:
            self.logger.error(f"Error in {method} normalization: {e}")
            raise

    def _standard_normalize(self, tensor):
        """Standart normalizasyon uygular (ortalama 0, standart sapma 1)."""
        mean, std = tensor.mean(), tensor.std()

        if std == 0:
            self.logger.warning("Standard deviation is zero. Returning zero tensor to avoid division by zero.")
            return torch.zeros_like(tensor)

        return (tensor - mean) / (std + 1e-5)

    def _min_max_normalize(self, tensor, feature_range=(0, 1)):
        """Min-Max ölçeklendirme uygular."""
        min_val, max_val = feature_range
        tensor_min, tensor_max = tensor.min(), tensor.max()

        if tensor_max == tensor_min:
            self.logger.warning("Tensor min and max values are equal. Returning zero tensor to avoid division by zero.")
            return torch.zeros_like(tensor)

        scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-5)
        return scaled_tensor * (max_val - min_val) + min_val

    def _robust_normalize(self, tensor):
        """Robust ölçeklendirme uygular (medyan ve IQR kullanır)."""
        tensor_median = tensor.median()
        q1 = tensor.kthvalue(max(1, int(0.25 * tensor.numel())), dim=0)[0]
        q3 = tensor.kthvalue(max(1, int(0.75 * tensor.numel())), dim=0)[0]

        if (q3 - q1).abs().sum() == 0:
            self.logger.warning("Tensor interquartile range is zero. Returning zero tensor to avoid division by zero.")
            return torch.zeros_like(tensor)

        return (tensor - tensor_median) / (q3 - q1 + 1e-5)

    def _log_normalization(self, original_tensor, normalized_tensor, normalization_type):
        """
        Normalizasyon işlemleri sırasında loglama yapar.

        Args:
            original_tensor (torch.Tensor): Orijinal tensör.
            normalized_tensor (torch.Tensor): Normalizasyon yapılmış tensör.
            normalization_type (str): Normalizasyon türü.
        """
        self._log_once(f"{normalization_type} completed.")
        self.logger.debug(f"Original tensor shape: {original_tensor.shape}, dtype: {original_tensor.dtype}, device: {original_tensor.device}")
        self.logger.debug(f"Normalized tensor shape: {normalized_tensor.shape}, dtype: {normalized_tensor.dtype}, device: {normalized_tensor.device}")
