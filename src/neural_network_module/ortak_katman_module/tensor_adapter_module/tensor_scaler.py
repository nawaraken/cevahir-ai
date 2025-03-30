import torch
import logging

class TensorScaler:
    """
    TensorScaler sınıfı, tensörler üzerinde ölçeklendirme işlemlerini gerçekleştirir.
    """

    def __init__(self, log_level=logging.INFO):
        """
        TensorScaler sınıfını başlatır.

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

        self._log_once("TensorScaler initialized.")

    def _log_once(self, message):
        """
        Aynı log mesajlarının tekrar tekrar yazılmasını önler.

        Args:
            message (str): Log mesajı
        """
        if message not in self.logger.seen_messages:
            self.logger.info(message)
            self.logger.seen_messages.add(message)

    def scale_min_max(self, tensor, feature_range=(0, 1)):
        """
        Min-Max ölçeklendirme işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Ölçeklendirilecek tensör.
            feature_range (tuple): Min-Max ölçeklendirme için kullanılacak aralık (varsayılan: (0, 1)).

        Returns:
            torch.Tensor: Ölçeklendirilmiş tensör.
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.error("Input is not a tensor.")
            raise TypeError("Input must be a torch.Tensor.")

        if tensor.numel() == 0:
            self.logger.error("Attempted to scale an empty tensor.")
            raise ValueError("Cannot scale an empty tensor.")

        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

        min_val, max_val = feature_range
        tensor_min, tensor_max = tensor.min(), tensor.max()
        
        if tensor_min == tensor_max:
            self.logger.warning("Min-Max scaling attempted on a constant tensor. Returning zeros.")
            return torch.zeros_like(tensor)

        scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        scaled_tensor = scaled_tensor * (max_val - min_val) + min_val

        self.logger.debug(f"Output tensor shape after Min-Max Scaling: {scaled_tensor.shape}, dtype: {scaled_tensor.dtype}, device: {scaled_tensor.device}")
        self._log_once("Min-Max Scaling applied.")

        return scaled_tensor

    def scale_standard(self, tensor):
        """
        Standart ölçeklendirme işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Ölçeklendirilecek tensör.

        Returns:
            torch.Tensor: Ölçeklendirilmiş tensör.
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.error("Input is not a tensor.")
            raise TypeError("Input must be a torch.Tensor.")

        if tensor.numel() == 0:
            self.logger.error("Attempted to scale an empty tensor.")
            raise ValueError("Cannot scale an empty tensor.")

        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

        tensor_mean, tensor_std = tensor.mean(), tensor.std()
        
        if tensor_std == 0:
            self.logger.warning("Standard scaling attempted on a zero-variance tensor. Returning zeros.")
            return torch.zeros_like(tensor)

        scaled_tensor = (tensor - tensor_mean) / tensor_std

        self.logger.debug(f"Output tensor shape after Standard Scaling: {scaled_tensor.shape}, dtype: {scaled_tensor.dtype}, device: {scaled_tensor.device}")
        self._log_once("Standard Scaling applied.")

        return scaled_tensor

    def scale_robust(self, tensor):
        """
        Robust ölçeklendirme işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Ölçeklendirilecek tensör.

        Returns:
            torch.Tensor: Ölçeklendirilmiş tensör.
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.error("Input is not a tensor.")
            raise TypeError("Input must be a torch.Tensor.")

        if tensor.numel() == 0:
            self.logger.error("Attempted to scale an empty tensor.")
            raise ValueError("Cannot scale an empty tensor.")

        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

        tensor_median = tensor.median()
        q1 = tensor.kthvalue(int(0.25 * tensor.numel()))[0]
        q3 = tensor.kthvalue(int(0.75 * tensor.numel()))[0]

        if q1 == q3:
            self.logger.warning("Robust scaling attempted on a tensor with no interquartile range. Returning zeros.")
            return torch.zeros_like(tensor)

        scaled_tensor = (tensor - tensor_median) / (q3 - q1)

        self.logger.debug(f"Output tensor shape after Robust Scaling: {scaled_tensor.shape}, dtype: {scaled_tensor.dtype}, device: {scaled_tensor.device}")
        self._log_once("Robust Scaling applied.")

        return scaled_tensor
