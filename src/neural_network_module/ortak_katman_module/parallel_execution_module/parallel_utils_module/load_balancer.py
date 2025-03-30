import torch
import logging

class LoadBalancer:
    """
    LoadBalancer sınıfı, tensörler üzerinde paralel işlemleri yük dengeleyici olarak yönetir.
    """
    def __init__(self, num_tasks, task_dim, log_level=logging.INFO):
        """
        LoadBalancer sınıfını başlatır.

        Args:
            num_tasks (int): Paralel işlemler için görev sayısı.
            task_dim (int): Görevlerin çalışacağı boyut.
            log_level (int): Log seviyesi (örn. logging.DEBUG, logging.INFO).
        """
        self.num_tasks = num_tasks
        self.task_dim = task_dim

        # Logger Ayarları
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(f"LoadBalancer initialized with num_tasks={num_tasks}, task_dim={task_dim}")

    def balance_load(self, tensor):
        """
        Yük dengeleme işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Dengeleme işlemi yapılacak tensör.

        Returns:
            list: Dengelenmiş paralel görevler.
        """
        self._validate_input(tensor)

        # Tensor'un cihazını otomatik algıla (CPU/GPU)
        device = tensor.device
        self.logger.info(f"Balancing tasks on device: {device}")

        # Tensörü paralel işlemler için eşit parçalara böl
        tasks = self._split_tensor(tensor)
        self._log_balancing(tensor, tasks)
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

    def _validate_input(self, tensor):
        """
        Giriş tensörünün geçerli olup olmadığını kontrol eder.

        Args:
            tensor (torch.Tensor): Dengeleme işlemi yapılacak tensör.

        Raises:
            TypeError: Eğer giriş torch.Tensor değilse hata fırlatır.
            ValueError: Eğer tensörün boyutu sıfırsa hata fırlatır.
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.error(f"Invalid tensor type: {type(tensor)}. Expected torch.Tensor.")
            raise TypeError("Input tensor must be a torch.Tensor.")

        if len(tensor.shape) == 0:
            self.logger.error("Tensor is a scalar, cannot be balanced.")
            raise ValueError("Tensor must have at least one dimension to be balanced.")

        if self.task_dim >= len(tensor.shape):
            self.logger.error(f"Invalid task_dim={self.task_dim} for tensor with shape {tensor.shape}")
            raise ValueError(f"task_dim={self.task_dim} is out of range for tensor with shape {tensor.shape}")

    def _log_balancing(self, original_tensor, tasks):
        """
        Yük dengeleme işlemi sırasında loglama yapar.

        Args:
            original_tensor (torch.Tensor): Orijinal tensör.
            tasks (list): Dengelenmiş paralel görevler.
        """
        self.logger.info("Load balancing completed.")
        self.logger.debug(f"Original tensor shape: {original_tensor.shape}, dtype: {original_tensor.dtype}, device: {original_tensor.device}")

        for i, task in enumerate(tasks):
            self.logger.debug(f"Task {i} tensor shape: {task.shape}, dtype: {task.dtype}, device: {task.device}")
