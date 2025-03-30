import torch
import logging
from typing import Callable, Optional

class ParallelScaler:
    """
    ParallelScaler sınıfı, tensörler üzerinde paralel ölçeklendirme işlemlerini gerçekleştirir.
    """
    def __init__(self, num_tasks, task_dim, scale_range=(0, 1), log_level=logging.INFO):
        """
        ParallelScaler sınıfını başlatır.

        Args:
            num_tasks (int): Paralel işlemler için görev sayısı.
            task_dim (int): Görevlerin çalışacağı boyut.
            scale_range (tuple): Min-Max ölçeklendirme için kullanılacak aralık (varsayılan: (0, 1)).
            log_level (int): Log seviyesi (örn. logging.DEBUG, logging.INFO).
        """
        self.num_tasks = num_tasks
        self.task_dim = task_dim
        self.scale_range = scale_range

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def scale_min_max(self, tensor):
        """
        Min-Max ölçeklendirme işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Ölçeklendirilecek tensör.

        Returns:
            list: Paralel ölçeklendirilmiş görevler.
        """
        self._validate_tensor(tensor, "Min-Max Scaling")
        min_val, max_val = self.scale_range
        tasks = self._split_tensor(tensor)
        scaled_tasks = []

        for task in tasks:
            task_min = task.min()
            task_max = task.max()

            if task_max == task_min:  # Sayısal kararsızlık önleme
                self.logger.warning(f"Task has constant values, skipping Min-Max scaling.")
                scaled_tasks.append(task.clone())  # Değişiklik yapmadan ekle
                continue

            scaled_task = (task - task_min) / (task_max - task_min)
            scaled_task = scaled_task * (max_val - min_val) + min_val
            scaled_tasks.append(scaled_task)

        self._log_scaling(tensor, scaled_tasks, "Min-Max Scaling")
        return scaled_tasks

    def scale_standard(self, tensor):
        """
        Standart ölçeklendirme işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Ölçeklendirilecek tensör.

        Returns:
            list: Paralel ölçeklendirilmiş görevler.
        """
        self._validate_tensor(tensor, "Standard Scaling")
        tasks = self._split_tensor(tensor)
        scaled_tasks = []

        for task in tasks:
            task_mean = task.mean()
            task_std = task.std()

            if task_std == 0:  # Sayısal kararsızlık önleme
                self.logger.warning(f"Task has zero variance, skipping Standard scaling.")
                scaled_tasks.append(task.clone())  # Değişiklik yapmadan ekle
                continue

            scaled_task = (task - task_mean) / task_std
            scaled_tasks.append(scaled_task)

        self._log_scaling(tensor, scaled_tasks, "Standard Scaling")
        return scaled_tasks

    def scale_robust(self, tensor):
        """
        Robust ölçeklendirme işlemini gerçekleştirir.

        Args:
            tensor (torch.Tensor): Ölçeklendirilecek tensör.

        Returns:
            list: Paralel ölçeklendirilmiş görevler.
        """
        self._validate_tensor(tensor, "Robust Scaling")
        tasks = self._split_tensor(tensor)
        scaled_tasks = []

        for task in tasks:
            task_median = task.median()
            q1 = torch.quantile(task, 0.25)
            q3 = torch.quantile(task, 0.75)
            iqr = q3 - q1

            if torch.all(iqr == 0):  # Tüm elemanların sıfır olup olmadığını kontrol et
                self.logger.warning(f"Task has zero interquartile range, skipping Robust scaling.")
                scaled_tasks.append(task.clone())  # Değişiklik yapmadan ekle
                continue

            scaled_task = (task - task_median) / iqr
            scaled_tasks.append(scaled_task)

        self._log_scaling(tensor, scaled_tasks, "Robust Scaling")
        return scaled_tasks

    def scale_custom(self, tensor: torch.Tensor, custom_func: Callable[[torch.Tensor], torch.Tensor], **kwargs):
        """
        Kullanıcı tanımlı fonksiyon ile özel ölçeklendirme yapar.

        Args:
            tensor (torch.Tensor): Ölçeklendirilecek tensör.
            custom_func (Callable): Kullanıcının tanımladığı dönüşüm fonksiyonu.
            kwargs: Ek parametreler (custom_func içinde kullanılabilir).

        Returns:
            list: Paralel ölçeklendirilmiş görevler.
        """
        self._validate_tensor(tensor, "Custom Scaling")
        tasks = self._split_tensor(tensor)
        scaled_tasks = []

        for task in tasks:
            try:
                scaled_task = custom_func(task, **kwargs)
                scaled_tasks.append(scaled_task)
            except Exception as e:
                self.logger.error(f"Custom scaling function failed: {e}")
                scaled_tasks.append(task.clone())  # Başarısız olursa orijinal veriyi ekle

        self._log_scaling(tensor, scaled_tasks, "Custom Scaling")
        return scaled_tasks

    def _split_tensor(self, tensor):
        """
        Tensörü paralel görevler için böler.

        Args:
            tensor (torch.Tensor): Bölünecek tensör.

        Returns:
            list: Bölünmüş tensör görevleri.
        """
        total_size = tensor.size(self.task_dim)

        if self.num_tasks > total_size:
            self.logger.warning(f"num_tasks ({self.num_tasks}) is greater than tensor size ({total_size}). Adjusting to {total_size}.")
            self.num_tasks = total_size

        if total_size % self.num_tasks == 0:
            return list(torch.chunk(tensor, self.num_tasks, dim=self.task_dim))
        else:
            task_size = total_size // self.num_tasks
            remainder = total_size % self.num_tasks
            tasks = []
            start = 0

            for i in range(self.num_tasks):
                extra = 1 if i < remainder else 0
                end = start + task_size + extra
                tasks.append(tensor.narrow(self.task_dim, start, end - start))
                start = end

            return tasks

    def _validate_tensor(self, tensor, scaling_type):
        """
        Tensörün geçerli bir PyTorch tensörü olup olmadığını kontrol eder.

        Args:
            tensor (torch.Tensor): Kontrol edilecek tensör.
            scaling_type (str): Ölçeklendirme türü.

        Raises:
            TypeError: Eğer giriş bir tensör değilse hata fırlatır.
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.error(f"{scaling_type} - Invalid input type: {type(tensor)}. Expected torch.Tensor.")
            raise TypeError("Input must be a torch.Tensor.")

    def _log_scaling(self, original_tensor, scaled_tasks, scaling_type):
        """
        Ölçeklendirme işlemleri sırasında loglama yapar.

        Args:
            original_tensor (torch.Tensor): Orijinal tensör.
            scaled_tasks (list): Ölçeklendirilmiş görevler.
            scaling_type (str): Ölçeklendirme türü.
        """
        self.logger.info(f"{scaling_type} completed.")
        self.logger.debug(f"Original tensor shape: {original_tensor.shape}, dtype: {original_tensor.dtype}, device: {original_tensor.device}")

        for i, task in enumerate(scaled_tasks):
            self.logger.debug(f"Task {i} tensor shape: {task.shape}, dtype: {task.dtype}, device: {task.device}")
