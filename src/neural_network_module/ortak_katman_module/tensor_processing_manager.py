import torch
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from .tensor_adapter_module.tensor_utils_module.tensor_initializer import TensorInitializer
from .tensor_adapter_module.tensor_utils_module.tensor_optimizer import TensorOptimizer
from .tensor_adapter_module.tensor_projection import TensorProjection

class TensorProcessingManager:
    """
    TensorProcessingManager, tensör işlemlerini yöneten merkezi bir sınıftır.
    **ÖNEMLİ:** Residual Connection kaldırıldı, çünkü NeuralLayerProcessor zaten bu işlemi yapıyor.
    """

    _logger_initialized = False  

    def __init__(self, learning_rate, input_dim=None, output_dim=None, num_tasks=None, log_level=logging.INFO):
        self.input_dim = self._validate_dimension(input_dim, "input_dim")
        self.output_dim = self._validate_dimension(output_dim, "output_dim")
        self.num_tasks = self._validate_tasks(num_tasks)
        self.learning_rate = self._validate_learning_rate(learning_rate)  # Doğrulama fonksiyonunu ekledik

        self.logger = logging.getLogger(self.__class__.__name__)
        if not TensorProcessingManager._logger_initialized:
            self.logger.setLevel(log_level)
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            TensorProcessingManager._logger_initialized = True  

        self.logger.info(f"TensorProcessingManager initialized with input_dim={self.input_dim}, "
                         f"output_dim={self.output_dim}, num_tasks={self.num_tasks}, "
                         f"learning_rate={self.learning_rate}, log_level={log_level}.")

        # Modülleri başlat
        self.initializer = TensorInitializer(log_level)
        self.tensor_optimizer = TensorOptimizer(self.learning_rate, log_level)  # Güncellenmiş değer ile çağırıyoruz

        if self.input_dim and self.output_dim:
            self.projection = TensorProjection(self.input_dim, self.output_dim, num_tasks=self.num_tasks, log_level=log_level)
            self.logger.info("Projection layer successfully initialized.")
        else:
            self.projection = None
            self.logger.warning("Projection layer is not initialized due to missing input_dim or output_dim.")

    def _validate_learning_rate(self, learning_rate):
        """
        Öğrenme oranının (learning_rate) geçerli olup olmadığını kontrol eder.
        """
        if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
            raise ValueError(f"[ERROR] learning_rate must be a positive number, but got {learning_rate}.")
        return float(learning_rate)  # Eğer geçerliyse float olarak geri döndür


    def _validate_dimension(self, value, name):
        if value is None:
            return None
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"[ERROR] {name} must be a positive integer, but got {value}.")
        return value

    def _validate_tasks(self, num_tasks):
        if num_tasks is None or num_tasks <= 0:
            return 1
        if not isinstance(num_tasks, int):
            raise ValueError(f"[ERROR] num_tasks must be an integer, but got {num_tasks}.")
        return num_tasks

    def initialize(self, shape, method="zeros"):
        """Tensörleri başlatır."""
        try:
            self.logger.debug(f"Initializing tensor with shape: {shape} using method: {method}")
            # Yönteme göre uygun initializer metodunu çağırıyoruz.
            if method == "zeros":
                initialized_tensor = self.initializer.initialize_zeros(shape)
            elif method == "ones":
                initialized_tensor = self.initializer.initialize_ones(shape)
            elif method == "random":
                initialized_tensor = self.initializer.initialize_random(shape)
            elif method == "normal":
                initialized_tensor = self.initializer.initialize_normal(shape)
            elif method == "uniform":
                initialized_tensor = self.initializer.initialize_uniform(shape)
            elif method == "xavier":
                initialized_tensor = self.initializer.initialize_xavier(shape)
            elif method == "kaiming":
                initialized_tensor = self.initializer.initialize_kaiming(shape)
            else:
                raise ValueError(f"Invalid initialization method: {method}")
            self._log_execution("Initialization", initialized_tensor)
            return initialized_tensor
        except Exception as e:
            self.logger.error(f"[ERROR] initialize() failed: {str(e)}", exc_info=True)
            raise

    def optimize(self, tensor, gradients, method="sgd", **kwargs):
        """Tensörleri optimize eder."""
        try:
            self._validate_tensor(tensor, "input tensor")
            self._validate_tensor(gradients, "gradients")
            
            self.logger.debug(f"Optimizing tensor using method: {method}")
            self._log_tensor_stats(tensor, "Before Optimization")
            self._log_tensor_stats(gradients, "Gradients")

            # Yönteme göre ilgili optimize fonksiyonunu çağır
            if method.lower() == "sgd":
                optimized_tensor = self.tensor_optimizer.optimize_sgd(tensor, gradients)
            elif method.lower() == "adam":
                beta1 = kwargs.get("beta1", 0.9)
                beta2 = kwargs.get("beta2", 0.999)
                epsilon = kwargs.get("epsilon", 1e-8)
                t = kwargs.get("t", 1)
                optimized_tensor = self.tensor_optimizer.optimize_adam(tensor, gradients, beta1, beta2, epsilon, t)
            else:
                raise ValueError(f"Unsupported optimization method: {method}")

            self._log_execution("Optimization", optimized_tensor)
            return optimized_tensor
        except Exception as e:
            self.logger.error(f"[ERROR] optimize() failed: {str(e)}", exc_info=True)
            raise


    def project(self, tensor):
        """
        Projeksiyon işlemini gerçekleştirir.
        Giriş tensörünün geçerliliğini kontrol eder, projeksiyon katmanını uygular ve
        işlem sürelerini ile çıktı istatistiklerini detaylı loglar.

        Args:
            tensor (torch.Tensor): Projeksiyona tabi tutulacak tensör.

        Returns:
            torch.Tensor: Projeksiyon sonrası tensör.

        Raises:
            RuntimeError: Projeksiyon katmanı tanımlı değilse.
            Exception: Projeksiyon işlemi sırasında oluşan diğer hatalar.
        """
        import time
        t_start = time.time()
        try:
            # 1. Giriş tensörünü doğrula
            self._validate_tensor(tensor, "input tensor for projection")
            self.logger.debug(f"[PROJECT] Starting projection. Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
            self._log_tensor_stats(tensor, "Before Projection")
            t_before = time.time()

            # 2. Projeksiyon katmanının mevcut olduğundan emin ol
            if self.projection is None:
                raise RuntimeError("Projection layer is not initialized. Cannot project tensor.")

            # 3. Projeksiyon işlemini gerçekleştir
            projected_tensor = self.projection.project(tensor)

            t_after = time.time()
            self.logger.debug(f"[PROJECT] Projection layer output shape: {projected_tensor.shape}")
            self._log_tensor_stats(projected_tensor, "After Projection")
            self._log_execution("Projection", projected_tensor)
            self.logger.debug(f"[PROJECT] Projection operation took: {t_after - t_before:.4f} seconds")
            
            total_time = time.time() - t_start
            self.logger.debug(f"[PROJECT] Total projection processing time: {total_time:.4f} seconds")
            return projected_tensor

        except Exception as e:
            self.logger.error(f"[PROJECT] Projection failed: {e}", exc_info=True)
            raise



    def _validate_tensor(self, tensor, name):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"[ERROR] {name} must be a torch.Tensor, but got {type(tensor)}.")
        if tensor.numel() == 0:
            raise ValueError(f"[ERROR] {name} is empty with shape {tensor.shape}.")

    def _log_tensor_stats(self, tensor, tensor_name="Tensor"):
        self.logger.debug(
            f"{tensor_name} stats: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, "
            f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}"
        )

    def _log_execution(self, operation, tensor):
        self.logger.info(f"{operation} completed successfully.")
        # Artık INFO seviyesinde loglamayarak testlerin beklentisini karşılayalım:
        self.logger.info(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
        self._log_tensor_stats(tensor, tensor_name=f"{operation} output")
