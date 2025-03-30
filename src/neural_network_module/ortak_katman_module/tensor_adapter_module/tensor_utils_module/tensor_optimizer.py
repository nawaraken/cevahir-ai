import torch
import torch.nn as nn
import logging

class TensorOptimizer:
    """
    TensorOptimizer sınıfı, tensörler üzerinde optimizasyon işlemlerini gerçekleştirir.
    """
    _logger_initialized = False  

    def __init__(self, learning_rate, log_level=logging.INFO):
        """
        TensorOptimizer başlatma fonksiyonu.

        Args:
            learning_rate (float): Optimizasyon için öğrenme oranı.
            log_level (int): Log seviyesi (örn. logging.DEBUG, logging.INFO).
        """
        if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
            raise ValueError(f"[ERROR] Invalid learning_rate: {learning_rate}. Must be a positive float.")

        self.learning_rate = float(learning_rate)  # Öğrenme oranını güvenli bir şekilde float'a çevir.
        self.logger = logging.getLogger(self.__class__.__name__)

        if not TensorOptimizer._logger_initialized:
            self.logger.setLevel(log_level)
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            TensorOptimizer._logger_initialized = True  

        self.logger.info(f"TensorOptimizer initialized with learning_rate={self.learning_rate}, log_level={log_level}")


    def _validate_learning_rate(self, lr):
        """Öğrenme oranını doğrular."""
        if not isinstance(lr, (float, int)) or lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}. Must be a positive float.")
        return float(lr)

    def _validate_tensors(self, *tensors):
        """Giriş tensörlerini doğrular."""
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Input must be a torch.Tensor, but got {type(tensor)}")
            if tensor.numel() == 0:
                raise ValueError(f"Tensor is empty with shape {tensor.shape}")

    def _log_tensor_stats(self, tensor, name="Tensor"):
        """Tensörün istatistiklerini loglar."""
        self.logger.debug(
            f"{name} stats: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, "
            f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}"
        )

    def _initialize_momentum(self, tensor):
        """Adam optimizasyonu için momentum ve hız değişkenlerini başlatır."""
        momentum = torch.zeros_like(tensor, dtype=torch.float32, device=tensor.device)
        self.logger.debug("Momentum tensor initialized.")
        self._log_tensor_stats(momentum, name="Momentum")
        return momentum

    def optimize(self, tensor, gradients, method="sgd", **kwargs):
        """
        Public optimize metodu; 'sgd' veya 'adam' yöntemine göre uygun optimizasyonu gerçekleştirir.
        Ek parametreler (örneğin, beta1, beta2, epsilon, t) yalnızca Adam için geçerlidir.

        Args:
            tensor (torch.Tensor): Optimizasyon yapılacak tensör.
            gradients (torch.Tensor): Tensörün gradyanları.
            method (str): Kullanılacak optimizasyon yöntemi ("sgd" veya "adam").
            **kwargs: Adam parametreleri için ek argümanlar (beta1, beta2, epsilon, t).

        Returns:
            torch.Tensor: Optimizasyon sonrası tensör.

        Raises:
            ValueError, TypeError: Giriş doğrulaması veya desteklenmeyen yöntem durumunda.
            RuntimeError: Optimizasyon sırasında oluşan beklenmeyen hatalar.
        """
        import time
        t_start = time.time()
        try:
            # Giriş tensörlerini doğrula
            self._validate_tensors(tensor, gradients)
            method = method.lower()
            self.logger.info(f"Starting optimization using method: {method}")
            self._log_tensor_stats(tensor, name="Input tensor")
            self._log_tensor_stats(gradients, name="Gradients")

            # Seçilen optimizasyon yöntemine göre işlemi uygula
            if method == "sgd":
                optimized_tensor = self.optimize_sgd(tensor, gradients)
            elif method == "adam":
                beta1 = kwargs.get("beta1", 0.9)
                beta2 = kwargs.get("beta2", 0.999)
                epsilon = kwargs.get("epsilon", 1e-8)
                t = kwargs.get("t", 1)
                optimized_tensor = self.optimize_adam(tensor, gradients, beta1, beta2, epsilon, t)
            else:
                raise ValueError(f"Unsupported optimization method: {method}")

            total_time = time.time() - t_start
            self.logger.info(f"Optimization using {method} completed in {total_time:.6f} seconds.")
            self._log_tensor_stats(optimized_tensor, name="Optimized tensor")
            return optimized_tensor

        except Exception as e:
            self.logger.error(f"Error in optimize() with method {method}: {e}", exc_info=True)
            raise


    def optimize_sgd(self, tensor, gradients):
        """Stokastik Gradient Descent (SGD) optimizasyon işlemini gerçekleştirir."""
        self.logger.info("Performing SGD optimization.")
        return self._optimize(
            tensor, gradients,
            lambda: tensor - self.learning_rate * gradients,
            "SGD"
        )

    def optimize_adam(self, tensor, gradients, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):
        """Adam optimizasyon işlemini gerçekleştirir."""
        self.logger.info("Performing Adam optimization.")
        m = self._initialize_momentum(tensor)
        v = self._initialize_momentum(tensor)
        return self._optimize(
            tensor, gradients,
            lambda: self._adam_update(tensor, gradients, m, v, beta1, beta2, epsilon, t),
            "Adam"
        )

    def _adam_update(self, tensor, gradients, m, v, beta1, beta2, epsilon, t):
        """Adam optimizasyon güncellemesi."""
        self.logger.debug("Starting Adam update.")
        self._log_tensor_stats(m, name="m before update")
        self._log_tensor_stats(v, name="v before update")
        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * (gradients ** 2)
        self.logger.debug("Momentum (m) and velocity (v) updated.")
        self._log_tensor_stats(m, name="m after update")
        self._log_tensor_stats(v, name="v after update")
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        self.logger.debug("Bias-corrected m_hat and v_hat computed.")
        self._log_tensor_stats(m_hat, name="m_hat")
        self._log_tensor_stats(v_hat, name="v_hat")
        update = self.learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
        self._log_tensor_stats(update, name="Update step")
        updated_tensor = tensor - update
        self.logger.debug("Adam update applied to tensor.")
        return updated_tensor

    def optimize(self, tensor, gradients, method="sgd", **kwargs):
        """
        Public optimize metodu; 'sgd' veya 'adam' yöntemine göre uygun optimizasyonu çağırır.
        Ek parametreler (örneğin, beta1, beta2, epsilon, t) sadece adam için geçerlidir.
        """
        if method.lower() == "sgd":
            return self.optimize_sgd(tensor, gradients)
        elif method.lower() == "adam":
            beta1 = kwargs.get("beta1", 0.9)
            beta2 = kwargs.get("beta2", 0.999)
            epsilon = kwargs.get("epsilon", 1e-8)
            t = kwargs.get("t", 1)
            return self.optimize_adam(tensor, gradients, beta1, beta2, epsilon, t)
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
