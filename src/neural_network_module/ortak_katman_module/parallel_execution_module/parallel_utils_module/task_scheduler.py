import torch
import logging

class TaskScheduler:
    """
    TaskScheduler sınıfı, paralel görevleri zamanlamak için gerekli işlemleri gerçekleştirir.
    """
    def __init__(self, num_tasks, task_dim, log_level=logging.INFO):
        """
        TaskScheduler sınıfını başlatır.

        Args:
            num_tasks (int): Paralel işlemler için görev sayısı.
            task_dim (int): Görevlerin çalışacağı boyut.
            log_level (int): Log seviyesi (örn. logging.DEBUG, logging.INFO).
        """
        self.num_tasks = num_tasks
        self.task_dim = task_dim

        # Logger ayarları
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(f"TaskScheduler initialized with num_tasks={num_tasks}, task_dim={task_dim}")

    def schedule_tasks(self, tensor):
        """
        Görevleri zamanlamak için kullanılır.

        Args:
            tensor (torch.Tensor): Zamanlanacak tensör.

        Returns:
            list: Zamanlanmış paralel görevler.
        """
        self._validate_tensor(tensor)

        # Tensor'un cihazını otomatik algıla (CPU/GPU)
        device = tensor.device
        self.logger.info(f"Scheduling tasks on device: {device}")

        # task_dim'in geçerli olup olmadığını kontrol et
        if self.task_dim >= len(tensor.shape):
            self.logger.error(f"task_dim={self.task_dim} is out of range for tensor shape {tensor.shape}")
            raise IndexError(f"task_dim={self.task_dim} is out of range for tensor with shape {tensor.shape}")

        tasks = self._split_tensor(tensor)
        self._log_scheduling(tensor, tasks)
        return tasks

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

    def _validate_tensor(self, tensor):
        """
        Tensörün geçerli olup olmadığını kontrol eder.

        Args:
            tensor (torch.Tensor): Kontrol edilecek tensör.

        Raises:
            TypeError: Eğer giriş bir tensör değilse hata fırlatır.
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.error(f"Invalid input type: {type(tensor)}. Expected torch.Tensor.")
            raise TypeError("Input must be a torch.Tensor.")

        if len(tensor.shape) == 0:
            self.logger.error("Tensor is a scalar, cannot be scheduled.")
            raise ValueError("Tensor must have at least one dimension to be scheduled.")

    def _log_scheduling(self, original_tensor, tasks):
        """
        Zamanlama işlemleri sırasında loglama yapar.

        Args:
            original_tensor (torch.Tensor): Orijinal tensör.
            tasks (list): Zamanlanmış paralel görevler.
        """
        self.logger.info("Task scheduling completed.")
        self.logger.debug(f"Original tensor shape: {original_tensor.shape}, dtype: {original_tensor.dtype}, device: {original_tensor.device}")

        for i, task in enumerate(tasks):
            self.logger.debug(f"Task {i} tensor shape: {task.shape}, dtype: {task.dtype}, device: {task.device}")
