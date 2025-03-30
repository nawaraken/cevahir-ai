import torch
import torch.nn as nn
import logging

class TensorInitializer:
    """
    TensorInitializer sınıfı, tensörler üzerinde çeşitli başlatma işlemlerini gerçekleştirir.
    """
    _logger_initialized = False  

    def __init__(self, log_level=logging.INFO):
        """
        TensorInitializer başlatma fonksiyonu.

        Args:
            log_level (int): Log seviyesi (örn. logging.DEBUG, logging.INFO).
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        if not TensorInitializer._logger_initialized:
            self.logger.setLevel(log_level)
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            TensorInitializer._logger_initialized = True  

        self.logger.info(f"TensorInitializer initialized with log_level={log_level}")

    def _validate_shape(self, shape):
        """
        Tensor başlatma için geçerli şekil olup olmadığını doğrular.

        Args:
            shape (tuple): Tensör şekli.

        Raises:
            ValueError: Geçersiz şekil durumu.
            TypeError: Eğer shape yanlış bir tipteyse.
        """
        if not isinstance(shape, tuple) or not all(isinstance(dim, int) and dim > 0 for dim in shape):
            raise ValueError(f"Invalid tensor shape: {shape}. Shape must be a tuple of positive integers.")

    def _log_tensor_stats(self, tensor, message="Tensor stats"):
        """
        Tensörün istatistiklerini loglar.

        Args:
            tensor (torch.Tensor): Loglanacak tensör.
            message (str): Log mesajı.
        """
        self.logger.debug(
            f"{message}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, "
            f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}"
        )

    def _initialize_tensor(self, shape, method, init_fn=None, *args, **kwargs):
        """
        Tüm tensör başlatma işlemlerini tek bir fonksiyon altında yönetir.

        Args:
            shape (tuple): Tensör şekli.
            method (str): Başlatma yöntemi.
            init_fn (callable, optional): Tensor başlatma işlemini yapan fonksiyon. Eğer None ise,
                                          tensör doğrudan oluşturulur.
            *args, **kwargs: Başlatıcıya iletilecek ek parametreler.

        Returns:
            torch.Tensor: Başlatılmış tensör.
        """
        try:
            self._validate_shape(shape)
            tensor = torch.empty(shape)
            if init_fn is not None:
                init_fn(tensor, *args, **kwargs)
            self.logger.info(f"{method} completed successfully.")
            self._log_tensor_stats(tensor, message=f"{method} result")
            return tensor
        except Exception as e:
            self.logger.error(f"Error in {method}: {e}", exc_info=True)
            raise

    def initialize_zeros(self, shape):
        """Tensörü sıfırlarla başlatır."""
        self._validate_shape(shape)
        tensor = torch.zeros(shape)
        self.logger.info("Zeros Initialization completed successfully.")
        self._log_tensor_stats(tensor, message="Zeros Initialization result")
        return tensor

    def initialize_ones(self, shape):
        """Tensörü birlerle başlatır."""
        self._validate_shape(shape)
        tensor = torch.ones(shape)
        self.logger.info("Ones Initialization completed successfully.")
        self._log_tensor_stats(tensor, message="Ones Initialization result")
        return tensor

    def initialize_random(self, shape):
        """Tensörü rastgele değerlerle başlatır."""
        return self._initialize_tensor(shape, "Random Initialization", lambda t: t.copy_(torch.rand(shape)))

    def initialize_normal(self, shape, mean=0.0, std=1.0):
        """Tensörü normal dağılıma göre başlatır."""
        return self._initialize_tensor(shape, "Normal Initialization", lambda t: t.copy_(torch.normal(mean, std, size=shape)))

    def initialize_uniform(self, shape, min_val=0.0, max_val=1.0):
        """Tensörü uniform dağılıma göre başlatır."""
        return self._initialize_tensor(shape, "Uniform Initialization", lambda t: t.uniform_(min_val, max_val))

    def initialize_xavier(self, shape, gain=1.0):
        """Tensörü Xavier başlatma yöntemi ile başlatır."""
        return self._initialize_tensor(shape, "Xavier Initialization", torch.nn.init.xavier_uniform_, gain=gain)

    def initialize_kaiming(self, shape, mode="fan_in", nonlinearity="relu"):
        """Tensörü Kaiming başlatma yöntemi ile başlatır."""
        return self._initialize_tensor(shape, "Kaiming Initialization", torch.nn.init.kaiming_uniform_, mode=mode, nonlinearity=nonlinearity)
