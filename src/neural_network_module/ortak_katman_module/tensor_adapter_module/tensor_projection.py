import torch
import torch.nn as nn
import logging

class TensorProjection:
    """
    TensorProjection sınıfı, tensörler üzerinde projeksiyon işlemlerini gerçekleştirir.
    """

    _logger_initialized = False  

    def __init__(self, input_dim, output_dim, num_tasks=None, log_level=logging.INFO):
        """
        TensorProjection başlatma fonksiyonu.

        Args:
            input_dim (int): Giriş boyutu.
            output_dim (int): Tam çıkış boyutu.
            num_tasks (int, optional): Paralel işlem için görev sayısı.
            log_level (int): Log seviyesi.
        """
        self.input_dim = self._validate_dimension(input_dim, "input_dim")
        self.output_dim = self._validate_dimension(output_dim, "output_dim")

        # Görev sayısını doğrula ve çıkış boyutunu hesapla
        self.num_tasks = self._validate_tasks(num_tasks)
        self.effective_output_dim = self._calculate_effective_output_dim()

        # Projeksiyon katmanını oluştur
        self.projection_layer = nn.Linear(self.input_dim, self.effective_output_dim)

        # Xavier başlatma
        nn.init.xavier_uniform_(self.projection_layer.weight)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(self.effective_output_dim)

        # Logger yapılandırması
        self.logger = logging.getLogger(self.__class__.__name__)
        if not TensorProjection._logger_initialized:
            self.logger.setLevel(log_level)
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            TensorProjection._logger_initialized = True  

        self.logger.info(f"TensorProjection initialized with input_dim={self.input_dim}, output_dim={self.output_dim}, "
                         f"num_tasks={self.num_tasks}, effective_output_dim={self.effective_output_dim}.")

    def _validate_dimension(self, value, name):
        """Boyut değerlerini doğrular."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"[ERROR] {name} must be a positive integer, but got {value}.")
        return value

    def _validate_tasks(self, num_tasks):
        """Paralel görev sayısını doğrular ve varsayılan değeri belirler."""
        if num_tasks is None or num_tasks <= 0:
            return 1
        if not isinstance(num_tasks, int):
            raise ValueError(f"[ERROR] num_tasks must be an integer, but got {num_tasks}.")
        return num_tasks

    def _calculate_effective_output_dim(self):
        """Çıkış boyutunu num_tasks ile bölerek hesaplar."""
        if self.num_tasks > 1 and self.output_dim % self.num_tasks != 0:
            self.logger.warning(f"output_dim={self.output_dim} is not evenly divisible by num_tasks={self.num_tasks}. Rounding may occur.")
        return self.output_dim // self.num_tasks

    def project(self, tensor):
        """
        Projeksiyon işlemini gerçekleştirir.
        
        Bu metod, giriş tensörünü doğrular, uygun cihaz ve veri tipine dönüştürür,
        projeksiyon katmanını uygular, LayerNorm ile normalize eder ve işlem sürelerini
        ve ara istatistikleri detaylı olarak loglar.

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
            self.logger.debug(f"[PROJECT] Starting projection. Input tensor: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            self._log_tensor_stats(tensor, "Before Projection")
            
            # 2. Tensörün uygun veri tipine ve cihaza dönüştürülmesi
            device_target = self.projection_layer.weight.device
            tensor = tensor.to(dtype=torch.float32, device=device_target)
            t_conversion = time.time()
            self.logger.debug(f"[PROJECT] Tensor conversion completed in {t_conversion - t_start:.4f} seconds.")
            
            # 3. Projeksiyon katmanının kontrolü
            if self.projection_layer is None:
                raise RuntimeError("Projection layer is not initialized. Cannot perform projection.")
            
            # 4. Projeksiyon işleminin uygulanması
            projected_tensor = self.projection_layer(tensor)
            t_projection = time.time()
            self.logger.debug(f"[PROJECT] Projection layer execution time: {t_projection - t_conversion:.4f} seconds.")
            
            # 5. Layer Normalization işlemi
            projected_tensor = self.layer_norm(projected_tensor)
            t_norm = time.time()
            self.logger.debug(f"[PROJECT] LayerNorm execution time: {t_norm - t_projection:.4f} seconds.")
            
            # 6. Çıktı istatistiklerini loglama
            self._log_tensor_stats(projected_tensor, "After Projection")
            self._log_execution("Projection", projected_tensor)
            self.logger.debug(f"[PROJECT] Projection operation took: {t_norm - t_start:.4f} seconds")
            
            total_time = time.time() - t_start
            self.logger.info(f"[PROJECT] Total projection processing time: {total_time:.4f} seconds.")
            return projected_tensor

        except Exception as e:
            self.logger.error(f"[PROJECT] Projection failed: {e}", exc_info=True)
            raise

    def _validate_tensor(self, tensor, name="Tensor"):
        """
        Giriş tensörünü doğrular.

        Args:
            tensor (torch.Tensor): Doğrulanacak tensör.
            name (str): Tensörün loglama ve hata mesajlarında kullanılacak adı.

        Raises:
            TypeError: Eğer tensör bir torch.Tensor değilse.
            ValueError: Eğer tensör boşsa veya yanlış boyutlara sahipse.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"[ERROR] {name} must be a torch.Tensor, but got {type(tensor)}.")
        if tensor.numel() == 0:
            raise ValueError(f"[ERROR] {name} is empty with shape {tensor.shape}.")
        if tensor.size(-1) != self.input_dim:
            raise ValueError(f"[ERROR] {name} last dimension must be {self.input_dim}, but got {tensor.size(-1)}.")

    def _log_tensor_stats(self, tensor, stage_name="Tensor"):
        """
        Tensor istatistiklerini loglar.

        Args:
            tensor (torch.Tensor): Loglanacak tensör.
            stage_name (str): Loglama sırasında kullanılacak aşama adı.
        """
        try:
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            mean_val = tensor.mean().item()
            std_val = tensor.std().item()
            self.logger.debug(f"{stage_name} Stats -> shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}, "
                              f"min: {min_val:.4f}, max: {max_val:.4f}, mean: {mean_val:.4f}, std: {std_val:.4f}")
        except Exception as e:
            self.logger.error(f"Error logging tensor stats for {stage_name}: {e}", exc_info=True)

    def _log_execution(self, operation, tensor):
        """
        İşlem tamamlandığında temel çıktı bilgilerini loglar.

        Args:
            operation (str): Yapılan işlemin adı.
            tensor (torch.Tensor): İşlem sonrası tensör.
        """
        self.logger.info(f"{operation} completed successfully.")
        self.logger.info(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
        self._log_tensor_stats(tensor, stage_name=f"{operation} output")
