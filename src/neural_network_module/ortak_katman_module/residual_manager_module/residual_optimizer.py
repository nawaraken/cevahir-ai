import torch
import logging

class ResidualOptimizer:
    """
    ResidualOptimizer sınıfı, tensörler üzerinde optimizasyon işlemlerini gerçekleştirir.
    """

    def __init__(self, learning_rate=0.0001, log_level=logging.INFO):
        """
        ResidualOptimizer sınıfını başlatır.

        Args:
            learning_rate (float): Optimizasyon için öğrenme oranı.
            log_level (int): Log seviyesi (örn. logging.DEBUG, logging.INFO).
        """
        self.learning_rate = learning_rate
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

        self._log_once(f"ResidualOptimizer initialized with learning_rate={learning_rate}")

    def _log_once(self, message):
        """
        Aynı log mesajlarının tekrar tekrar yazılmasını önler.

        Args:
            message (str): Log mesajı.
        """
        if message not in self.logger.seen_messages:
            self.logger.info(message)
            self.logger.seen_messages.add(message)

    def optimize(self, tensor, gradients):
        """
        Optimizasyon işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Optimizasyon yapılacak tensör.
            gradients (torch.Tensor): Tensörün gradyanları.

        Returns:
            torch.Tensor: Optimizasyon sonrası tensör.
        """
        if not isinstance(tensor, torch.Tensor) or not isinstance(gradients, torch.Tensor):
            self.logger.error("Both inputs must be torch.Tensors.")
            raise TypeError("Both inputs must be torch.Tensors.")

        if tensor.shape != gradients.shape:
            self.logger.error("Tensor and gradients must have the same shape.")
            raise ValueError("Tensor and gradients must have the same shape.")

        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
        self.logger.debug(f"Gradients shape: {gradients.shape}, dtype: {gradients.dtype}, device: {gradients.device}")

        try:
            optimized_tensor = tensor - self.learning_rate * gradients
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            raise

        self.logger.debug(f"Optimized tensor shape: {optimized_tensor.shape}, dtype: {optimized_tensor.dtype}, device: {optimized_tensor.device}")
        self._log_once("Optimization completed successfully.")
        return optimized_tensor

    def optimize_with_momentum(self, tensor, gradients, velocity, momentum=0.9):
        """
        Momentum tabanlı optimizasyon işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Optimizasyon yapılacak tensör.
            gradients (torch.Tensor): Tensörün gradyanları.
            velocity (torch.Tensor): Momentum için hız tensörü.
            momentum (float): Momentum katsayısı (varsayılan: 0.9).

        Returns:
            tuple: (Optimizasyon sonrası tensör, güncellenmiş hız tensörü).
        """
        if not isinstance(tensor, torch.Tensor) or not isinstance(gradients, torch.Tensor) or not isinstance(velocity, torch.Tensor):
            self.logger.error("All inputs must be torch.Tensors.")
            raise TypeError("All inputs must be torch.Tensors.")

        if tensor.shape != gradients.shape or tensor.shape != velocity.shape:
            self.logger.error("Tensor, gradients, and velocity must have the same shape.")
            raise ValueError("Tensor, gradients, and velocity must have the same shape.")

        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
        self.logger.debug(f"Gradients shape: {gradients.shape}, dtype: {gradients.dtype}, device: {gradients.device}")
        self.logger.debug(f"Velocity shape: {velocity.shape}, dtype: {velocity.dtype}, device: {velocity.device}")

        try:
            velocity = momentum * velocity + self.learning_rate * gradients
            optimized_tensor = tensor - velocity
        except Exception as e:
            self.logger.error(f"Error during momentum optimization: {e}")
            raise

        self.logger.debug(f"Optimized tensor shape: {optimized_tensor.shape}, dtype: {optimized_tensor.dtype}, device: {optimized_tensor.device}")
        self.logger.debug(f"Updated velocity shape: {velocity.shape}, dtype: {velocity.dtype}, device: {velocity.device}")
        self._log_once("Momentum-based optimization completed successfully.")
        return optimized_tensor, velocity

    def optimize_with_adaptive_lr(self, tensor, gradients, prev_gradients, beta=0.99):
        """
        Adaptive öğrenme oranı ile optimizasyon işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Optimizasyon yapılacak tensör.
            gradients (torch.Tensor): Tensörün güncel gradyanları.
            prev_gradients (torch.Tensor): Önceki iterasyondaki gradyanlar.
            beta (float): Öğrenme oranı adaptasyon katsayısı (varsayılan: 0.99).

        Returns:
            torch.Tensor: Optimizasyon sonrası tensör.
        """
        if not isinstance(tensor, torch.Tensor) or not isinstance(gradients, torch.Tensor) or not isinstance(prev_gradients, torch.Tensor):
            self.logger.error("All inputs must be torch.Tensors.")
            raise TypeError("All inputs must be torch.Tensors.")

        if tensor.shape != gradients.shape or tensor.shape != prev_gradients.shape:
            self.logger.error("Tensor, gradients, and prev_gradients must have the same shape.")
            raise ValueError("Tensor, gradients, and prev_gradients must have the same shape.")

        self.logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
        self.logger.debug(f"Gradients shape: {gradients.shape}, dtype: {gradients.dtype}, device: {gradients.device}")
        self.logger.debug(f"Previous gradients shape: {prev_gradients.shape}, dtype: {prev_gradients.dtype}, device: {prev_gradients.device}")

        try:
            adaptive_lr = self.learning_rate / (1 + beta * torch.norm(gradients - prev_gradients))
            optimized_tensor = tensor - adaptive_lr * gradients
        except Exception as e:
            self.logger.error(f"Error during adaptive learning rate optimization: {e}")
            raise

        self.logger.debug(f"Optimized tensor shape: {optimized_tensor.shape}, dtype: {optimized_tensor.dtype}, device: {optimized_tensor.device}")
        self._log_once("Adaptive learning rate optimization completed successfully.")
        return optimized_tensor

    def _log_optimization(self, original_tensor, gradients, optimized_tensor):
        """
        Optimizasyon işlemleri sırasında loglama yapar.

        Args:
            original_tensor (torch.Tensor): Orijinal tensör.
            gradients (torch.Tensor): Tensörün gradyanları.
            optimized_tensor (torch.Tensor): Optimizasyon sonrası tensör.
        """
        self._log_once("Optimization completed.")
        self.logger.debug(f"Original tensor shape: {original_tensor.shape}, dtype: {original_tensor.dtype}, device: {original_tensor.device}")
        self.logger.debug(f"Gradients shape: {gradients.shape}, dtype: {gradients.dtype}, device: {gradients.device}")
        self.logger.debug(f"Optimized tensor shape: {optimized_tensor.shape}, dtype: {optimized_tensor.dtype}, device: {optimized_tensor.device}")
