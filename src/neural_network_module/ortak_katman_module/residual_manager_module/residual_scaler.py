import torch
import logging

class ResidualScaler:
    """
    ResidualScaler sınıfı, tensörler üzerinde ölçeklendirme işlemlerini gerçekleştirir.
    """

    def __init__(self, log_level=logging.INFO):
        """
        ResidualScaler sınıfını başlatır.

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

        self._log_once("ResidualScaler initialized.")

    def _log_once(self, message):
        """
        Aynı log mesajlarının tekrar tekrar yazılmasını önler.

        Args:
            message (str): Log mesajı.
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
            self.logger.error("Input must be a torch.Tensor.")
            raise TypeError("Input must be a torch.Tensor.")

        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

        min_val, max_val = feature_range
        tensor_min = tensor.min()
        tensor_max = tensor.max()

        if tensor_max == tensor_min:
            self.logger.warning("Tensor min and max values are equal. Returning zero tensor to avoid division by zero.")
            scaled_tensor = torch.zeros_like(tensor)
        else:
            scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-5)
            scaled_tensor = scaled_tensor * (max_val - min_val) + min_val

        self._log_scaling(tensor, scaled_tensor, "Min-Max Scaling")
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
            self.logger.error("Input must be a torch.Tensor.")
            raise TypeError("Input must be a torch.Tensor.")

        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

        tensor_mean = tensor.mean()
        tensor_std = tensor.std()

        if tensor_std == 0:
            self.logger.warning("Tensor standard deviation is zero. Returning zero tensor to avoid division by zero.")
            scaled_tensor = torch.zeros_like(tensor)
        else:
            scaled_tensor = (tensor - tensor_mean) / (tensor_std + 1e-5)

        self._log_scaling(tensor, scaled_tensor, "Standard Scaling")
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
            self.logger.error("Input must be a torch.Tensor.")
            raise TypeError("Input must be a torch.Tensor.")

        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

        tensor_median = torch.median(tensor)

        try:
            q1 = tensor.quantile(0.25)
            q3 = tensor.quantile(0.75)
        except RuntimeError as e:
            self.logger.error(f"Error in computing quartiles: {e}")
            raise

        iqr = q3 - q1
        if iqr.abs().sum() == 0:
            self.logger.warning("Tensor interquartile range is zero. Returning zero tensor to avoid division by zero.")
            scaled_tensor = torch.zeros_like(tensor)
        else:
            scaled_tensor = (tensor - tensor_median) / (iqr + 1e-5)

        self._log_scaling(tensor, scaled_tensor, "Robust Scaling")
        return scaled_tensor

    def _log_scaling(self, original_tensor, scaled_tensor, scaling_type):
        """
        Ölçeklendirme işlemleri sırasında loglama yapar.

        Args:
            original_tensor (torch.Tensor): Orijinal tensör.
            scaled_tensor (torch.Tensor): Ölçeklendirilmiş tensör.
            scaling_type (str): Ölçeklendirme türü.
        """
        self._log_once(f"{scaling_type} completed.")
        self.logger.debug(f"Original tensor shape: {original_tensor.shape}, dtype: {original_tensor.dtype}, device: {original_tensor.device}")
        self.logger.debug(f"Scaled tensor shape: {scaled_tensor.shape}, dtype: {scaled_tensor.dtype}, device: {scaled_tensor.device}")
