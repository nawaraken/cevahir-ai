import torch
import logging

class ParallelOptimizer:
    """
    ParallelOptimizer sınıfı, tensörler üzerinde paralel işlemleri optimize etmek için gerekli olan işlemleri gerçekleştirir.
    """
    def __init__(self, num_tasks, task_dim, learning_rate=0.0005, log_level=logging.INFO):
        """
        ParallelOptimizer sınıfını başlatır.

        Args:
            num_tasks (int): Paralel işlemler için görev sayısı.
            task_dim (int): Görevlerin çalışacağı boyut.
            learning_rate (float): Optimizasyon için öğrenme oranı.
            log_level (int): Log seviyesi (örn. logging.DEBUG, logging.INFO).
        """
        self.num_tasks = num_tasks
        self.task_dim = task_dim
        self.learning_rate = learning_rate

        # Logger Ayarları
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(f"ParallelOptimizer initialized with num_tasks={num_tasks}, task_dim={task_dim}, learning_rate={learning_rate}")

    def optimize_tasks(self, tensor, gradients):
        """
        Paralel işlemleri optimize etmek için görevleri optimize eder.

        Args:
            tensor (torch.Tensor): Optimizasyon yapılacak tensör.
            gradients (torch.Tensor): Tensörün gradyanları.

        Returns:
            torch.Tensor: Optimizasyon sonrası tensör.
        """
        self._validate_inputs(tensor, gradients)

        # Tensor'un cihazını otomatik algıla (CPU/GPU)
        device = tensor.device
        self.logger.info(f"Optimizing tasks on device: {device}")

        # Tensör ve gradyanları eşit parçalara böl
        tensor_tasks = self._split_tensor(tensor)
        grad_tasks = self._split_tensor(gradients)

        optimized_tasks = [
            task - self.learning_rate * grad_task
            for task, grad_task in zip(tensor_tasks, grad_tasks)
        ]

        # Optimizasyon sonrası tensörü geri birleştir
        optimized_tensor = torch.cat(optimized_tasks, dim=self.task_dim)
        self._log_optimization(tensor, gradients, optimized_tensor)
        return optimized_tensor

    def _split_tensor(self, tensor):
        """
        Tensörü paralel görevler için böler.

        Args:
            tensor (torch.Tensor): Bölünecek tensör.

        Returns:
            list: Bölünmüş tensör görevleri.
        """
        total_size = tensor.size(self.task_dim)

        # Eğer num_tasks total_size'ten büyükse, ayarla
        if self.num_tasks > total_size:
            self.logger.warning(f"num_tasks ({self.num_tasks}) is greater than tensor size ({total_size}). Adjusting to {total_size}.")
            self.num_tasks = total_size

        # Eğer tam bölünebiliyorsa, torch.chunk() kullan
        if total_size % self.num_tasks == 0:
            return list(torch.chunk(tensor, self.num_tasks, dim=self.task_dim))
        else:
            task_size = total_size // self.num_tasks
            remainder = total_size % self.num_tasks
            tasks = []
            start = 0

            for i in range(self.num_tasks):
                extra = 1 if i < remainder else 0  # İlk birkaç parçaya fazladan 1 ekle
                end = start + task_size + extra
                tasks.append(tensor.narrow(self.task_dim, start, end - start))
                start = end

            return tasks

    def _validate_inputs(self, tensor, gradients):
        """
        Giriş tensörlerinin geçerli olup olmadığını kontrol eder.

        Args:
            tensor (torch.Tensor): Optimizasyon yapılacak tensör.
            gradients (torch.Tensor): Tensörün gradyanları.

        Raises:
            TypeError: Eğer girişler torch.Tensor değilse hata fırlatır.
            ValueError: Eğer tensörlerin boyutları uyuşmuyorsa hata fırlatır.
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.error(f"Invalid tensor type: {type(tensor)}. Expected torch.Tensor.")
            raise TypeError("Input tensor must be a torch.Tensor.")

        if not isinstance(gradients, torch.Tensor):
            self.logger.error(f"Invalid gradients type: {type(gradients)}. Expected torch.Tensor.")
            raise TypeError("Gradients must be a torch.Tensor.")

        if tensor.shape != gradients.shape:
            self.logger.error(f"Shape mismatch: tensor {tensor.shape} vs gradients {gradients.shape}")
            raise ValueError(f"Tensor and gradients must have the same shape: {tensor.shape} vs {gradients.shape}")

        if len(tensor.shape) == 0:
            self.logger.error("Tensor is a scalar, cannot be optimized.")
            raise ValueError("Tensor must have at least one dimension to be optimized.")

    def _log_optimization(self, original_tensor, gradients, optimized_tensor):
        """
        Optimizasyon işlemleri sırasında loglama yapar.

        Args:
            original_tensor (torch.Tensor): Orijinal tensör.
            gradients (torch.Tensor): Tensörün gradyanları.
            optimized_tensor (torch.Tensor): Optimizasyon sonrası tensör.
        """
        self.logger.info("Optimization completed.")
        self.logger.debug(f"Original tensor shape: {original_tensor.shape}, dtype: {original_tensor.dtype}, device: {original_tensor.device}")
        self.logger.debug(f"Gradients shape: {gradients.shape}, dtype: {gradients.dtype}, device: {gradients.device}")
        self.logger.debug(f"Optimized tensor shape: {optimized_tensor.shape}, dtype: {optimized_tensor.dtype}, device: {optimized_tensor.device}")
