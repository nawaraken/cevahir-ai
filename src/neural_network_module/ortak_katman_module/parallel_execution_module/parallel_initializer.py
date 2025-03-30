import torch
import logging

class ParallelInitializer:
    """
    ParallelInitializer sınıfı, tensörler üzerinde paralel işlemleri başlatmak için gerekli olan işlemleri gerçekleştirir.
    """
    def __init__(self, num_tasks, task_dim, log_level=logging.INFO):
        """
        ParallelInitializer sınıfını başlatır.

        Args:
            num_tasks (int): Paralel işlemler için görev sayısı.
            task_dim (int): Görevlerin çalışacağı boyut.
            log_level (int): Log seviyesi (örn. logging.DEBUG, logging.INFO).
        """
        self.num_tasks = num_tasks
        self.task_dim = task_dim

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def initialize_tasks(self, tensor):
        """
        Paralel işlemleri başlatmak için görevleri başlatır.

        Args:
            tensor (torch.Tensor): Paralel işlemler için kullanılacak tensör.

        Returns:
            list: Paralel görevler.
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.error("Input must be a torch.Tensor.")
            raise TypeError("Input must be a torch.Tensor.")

        if tensor.dim() <= self.task_dim:
            self.logger.error(f"Invalid task_dim: {self.task_dim}. Tensor dimensions: {tensor.dim()}.")
            raise ValueError(f"Invalid task_dim: {self.task_dim}. Tensor dimensions: {tensor.dim()}.")

        total_size = tensor.size(self.task_dim)

        # Eğer num_tasks total_size'i tam bölemiyorsa, dinamik bölme yap.
        if total_size < self.num_tasks:
            self.logger.warning(f"num_tasks ({self.num_tasks}) is greater than tensor size ({total_size}). Adjusting to {total_size}.")
            self.num_tasks = total_size

        # Eğer bölünebiliyorsa torch.chunk kullanarak daha performanslı bölme
        if total_size % self.num_tasks == 0:
            tasks = list(torch.chunk(tensor, self.num_tasks, dim=self.task_dim))
        else:
            task_size = total_size // self.num_tasks
            remainder = total_size % self.num_tasks
            tasks = []

            start = 0
            for i in range(self.num_tasks):
                extra = 1 if i < remainder else 0  # İlk birkaç parçaya fazladan 1 eleman ekle
                end = start + task_size + extra
                tasks.append(tensor.narrow(self.task_dim, start, end - start))
                start = end

        self.log_initialization(tensor, tasks)
        return tasks

    def log_initialization(self, original_tensor, tasks):
        """
        Başlatma işlemleri sırasında loglama yapar.

        Args:
            original_tensor (torch.Tensor): Orijinal tensör.
            tasks (list): Paralel görevler.
        """
        self.logger.info("Parallel initialization completed.")
        self.logger.debug(f"Original tensor shape: {original_tensor.shape}, dtype: {original_tensor.dtype}, device: {original_tensor.device}")

        for i, task in enumerate(tasks):
            self.logger.debug(f"Task {i} tensor shape: {task.shape}, dtype: {task.dtype}, device: {task.device}")
