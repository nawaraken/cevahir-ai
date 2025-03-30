import torch
import logging
from .parallel_execution_module.parallel_utils_module.load_balancer import LoadBalancer
from .parallel_execution_module.parallel_utils_module.parallel_optimizer import ParallelOptimizer
from .parallel_execution_module.parallel_utils_module.task_scheduler import TaskScheduler
from .parallel_execution_module.parallel_initializer import ParallelInitializer
from .parallel_execution_module.parallel_scaler import ParallelScaler


class ParallelExecutionManager:
    """
    ParallelExecutionManager, tüm paralel işlemleri yöneten merkezi bir sınıftır.
    """
    def __init__(self, num_tasks: int, task_dim: int, learning_rate: float = 0.0005, scale_range: tuple = (0, 1), log_level=logging.INFO):
        """
        Paralel işlemleri yöneten sınıf.

        Args:
            num_tasks (int): Paralel görev sayısı.
            task_dim (int): Görevlerin çalışacağı boyut.
            learning_rate (float): Öğrenme oranı.
            scale_range (tuple): Min-Max ölçeklendirme aralığı.
            log_level (int): Log seviyesi.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.initializer = ParallelInitializer(num_tasks, task_dim, log_level)
        self.scaler = ParallelScaler(num_tasks, task_dim, scale_range, log_level)
        self.scheduler = TaskScheduler(num_tasks, task_dim, log_level)
        self.load_balancer = LoadBalancer(num_tasks, task_dim, log_level)
        self.optimizer = ParallelOptimizer(num_tasks, task_dim, learning_rate, log_level)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(f"ParallelExecutionManager initialized with num_tasks={num_tasks}, task_dim={task_dim}, learning_rate={learning_rate}")

    def _validate_tensor(self, tensor: torch.Tensor, name: str):
        """
        Tensörün geçerli olup olmadığını kontrol eder.

        Args:
            tensor (torch.Tensor): Kontrol edilecek tensör.
            name (str): Tensörün adı.

        Raises:
            TypeError: Eğer giriş bir tensör değilse.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Invalid tensor type for {name}. Expected torch.Tensor but got {type(tensor)}")

        tensor = tensor.to(self.device)
        self.logger.debug(f"{name} tensor validated. Shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

    def initialize(self, tensor: torch.Tensor):
        """
        Paralel işlemleri başlatır.

        Args:
            tensor (torch.Tensor): İşlenecek tensör.

        Returns:
            list: Başlatılan paralel görevler.
        """
        self._validate_tensor(tensor, "initialize")

        if tensor.numel() == 0 or torch.isnan(tensor).any() or torch.isinf(tensor).any():
            self.logger.error("Invalid tensor detected in initialize()")
            raise ValueError("Input tensor cannot be empty, NaN, or Inf.")

        tensor = tensor.to(self.device)

        try:
            tasks = self.initializer.initialize_tasks(tensor)
            if not tasks:
                raise ValueError("initialize_tasks() returned an empty list.")
            
            self._log_execution("Initialization", tensor, tasks)
            return tasks

        except Exception as e:
            self.logger.error(f"Error in initialize(): {str(e)}", exc_info=True)
            raise RuntimeError("Initialization failed due to an internal error.") from e

    def scale(self, tensor: torch.Tensor, method: str = "min_max"):
        """
        Tensörü ölçeklendirir.

        Args:
            tensor (torch.Tensor): Ölçeklendirilecek tensör.
            method (str): Kullanılacak ölçeklendirme yöntemi.

        Returns:
            torch.Tensor: Ölçeklendirilmiş tensör.
        """
        self._validate_tensor(tensor, "scale")

        try:
            if method == "min_max":
                scaled_tensor = self.scaler.scale_min_max(tensor)
            elif method == "standard":
                scaled_tensor = self.scaler.scale_standard(tensor)
            elif method == "robust":
                scaled_tensor = self.scaler.scale_robust(tensor)
            else:
                raise ValueError(f"Unsupported scaling method: {method}")

            self._log_execution("Scaling", tensor, [scaled_tensor])
            return scaled_tensor

        except Exception as e:
            self.logger.error(f"Error in scale(): {str(e)}", exc_info=True)
            raise RuntimeError(f"Scaling failed with method {method}.")

    def schedule(self, tensor: torch.Tensor):
        """
        Görevleri zamanlar.

        Args:
            tensor (torch.Tensor): Zamanlanacak tensör.

        Returns:
            list: Zamanlanmış görevler.
        """
        self._validate_tensor(tensor, "schedule")

        try:
            tasks = self.scheduler.schedule_tasks(tensor)
            if not tasks:
                raise ValueError("schedule_tasks() returned an empty list.")

            self._log_execution("Scheduling", tensor, tasks)
            return tasks

        except Exception as e:
            self.logger.error(f"Error in schedule(): {str(e)}", exc_info=True)
            raise RuntimeError("Task scheduling failed.")

    def balance(self, tensor: torch.Tensor):
        """
        Yük dengeleme işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Dengeleme yapılacak tensör.

        Returns:
            list: Dengelenmiş görevler.
        """
        self._validate_tensor(tensor, "balance")

        try:
            tasks = self.load_balancer.balance_load(tensor)
            if not tasks:
                raise ValueError("balance_load() returned an empty list.")

            self._log_execution("Load Balancing", tensor, tasks)
            return tasks

        except Exception as e:
            self.logger.error(f"Error in balance(): {str(e)}", exc_info=True)
            raise RuntimeError("Load balancing failed.")

    def optimize(self, tensor: torch.Tensor, gradients: torch.Tensor):
        """
        Tensörleri optimize eder.

        Args:
            tensor (torch.Tensor): Optimizasyon yapılacak tensör.
            gradients (torch.Tensor): Tensörün gradyanları.

        Returns:
            torch.Tensor: Optimizasyon sonrası tensör.
        """
        self._validate_tensor(tensor, "optimize")
        self._validate_tensor(gradients, "gradients")

        try:
            if tensor.shape != gradients.shape:
                raise ValueError("Tensor and gradient shapes do not match.")

            optimized_tensor = self.optimizer.optimize_tasks(tensor, gradients)
            self._log_execution("Optimization", tensor, [optimized_tensor])
            return optimized_tensor

        except Exception as e:
            self.logger.error(f"Error in optimize(): {str(e)}", exc_info=True)
            raise RuntimeError("Optimization failed.")

    def _log_execution(self, operation: str, original_tensor: torch.Tensor, result: list):
        """
        İşlem detaylarını loglar.

        Args:
            operation (str): İşlem adı.
            original_tensor (torch.Tensor): Orijinal tensör.
            result (list): İşlem sonucu.
        """
        self.logger.info(f"{operation} completed.")
        for i, task in enumerate(result):
            if isinstance(task, torch.Tensor):
                self.logger.debug(f"Task {i} shape: {task.shape}, dtype: {task.dtype}, device: {task.device}")
