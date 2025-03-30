import torch
import torch.nn as nn


class AttentionInitializer:
    """
    Dikkat mekanizmaları için parametre başlatıcı sınıfı.
    """

    def __init__(self, initialization_type="xavier", seed=None, verbose=False):
        """
        Başlatıcı sınıfını yapılandırır.

        Args:
            initialization_type (str): Başlatma tipi ("xavier", "he", "uniform", "normal", "constant").
            seed (int, optional): Rastgele başlatma için kullanılacak tohum değeri.
            verbose (bool): Loglama ve detaylı bilgi çıktılarını etkinleştirme seçeneği.
        """
        self.initialization_type = initialization_type.lower()
        self.verbose = verbose

        # Tohum belirleme (isteğe bağlı)
        if seed is not None:
            torch.manual_seed(seed)
            if self.verbose:
                print(f"[AttentionInitializer] Rastgele tohum ayarlandı: {seed}")

        # Desteklenen başlatma türleri
        self.supported_initializations = ["xavier", "he", "uniform", "normal", "constant"]

        if self.initialization_type not in self.supported_initializations:
            raise ValueError(
                f"Geçersiz başlatma tipi: {self.initialization_type}. "
                f"Desteklenen türler: {', '.join(self.supported_initializations)}"
            )

    def initialize_weights(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Verilen tensörün değerlerini, belirtilen başlatma (initialization) yöntemine göre başlatır.
        
        Args:
            tensor (torch.Tensor): Başlatılacak tensör.
        
        Returns:
            torch.Tensor: Başlatılmış tensör.
        
        Raises:
            TypeError: Eğer verilen veri bir tensör değilse.
            RuntimeError: Başlatma sırasında beklenmeyen bir hata oluşursa.
            ValueError: Başlatma sonrası tensörde NaN veya sonsuz değerler bulunursa.
        """
        import time
        t_start = time.time()
        
        # Giriş tip ve geçerlilik kontrolü
        if not isinstance(tensor, torch.Tensor):
            error_msg = "Başlatma için verilen veri bir PyTorch tensörü olmalıdır."
            self.logger.error(error_msg)
            raise TypeError(error_msg)
        
        self.logger.debug(f"[AttentionInitializer] Başlatma işlemi başlatılıyor. Giriş tensörü şekli: {tensor.shape}")
        
        try:
            # Belirtilen başlatma yöntemine göre tensörü başlatma
            if self.initialization_type == "xavier":
                nn.init.xavier_uniform_(tensor)
            elif self.initialization_type == "he":
                nn.init.kaiming_uniform_(tensor, nonlinearity="relu")
            elif self.initialization_type == "uniform":
                nn.init.uniform_(tensor, a=-0.1, b=0.1)
            elif self.initialization_type == "normal":
                nn.init.normal_(tensor, mean=0.0, std=0.02)
            elif self.initialization_type == "constant":
                nn.init.constant_(tensor, 0.1)
            else:
                error_msg = f"Desteklenmeyen başlatma tipi: {self.initialization_type}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            self.logger.error(f"[AttentionInitializer] Tensör başlatma sırasında hata: {e}", exc_info=True)
            raise RuntimeError(f"Tensör başlatma sırasında hata oluştu: {str(e)}") from e

        # Başlatma sonrası geçerlilik kontrolü
        try:
            if not torch.isfinite(tensor).all():
                error_msg = "Başlatılmış tensör NaN veya sonsuz değerler içeriyor."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            self.logger.error(f"[AttentionInitializer] Başlatma sonrası tensör kontrolü sırasında hata: {e}", exc_info=True)
            raise

        t_end = time.time()
        elapsed = t_end - t_start
        self.logger.debug(f"[AttentionInitializer] Tensör {self.initialization_type} yöntemi ile {elapsed:.6f} saniyede başlatıldı.")
        
        if self.verbose:
            self.logger.debug(
                f"[AttentionInitializer] Başlatılmış tensör istatistikleri: Şekil: {tensor.shape}, "
                f"Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}, Mean: {tensor.mean().item():.6f}"
            )
            print(f"[AttentionInitializer] Tensör {self.initialization_type} yöntemi ile başlatıldı: Şekil {tensor.shape}")

        return tensor


    def initialize_tensors(self, tensor):
        """
        Tensörlerin tümünü başlatır.

        Args:
            tensor (torch.Tensor): Başlatılacak tensör.

        Returns:
            torch.Tensor: Başlatılmış tensör.
        """
        return self.initialize_weights(tensor)

    def initialize_param_matrix(self, input_dim, output_dim):
        """
        Ağırlık matrislerini başlatır.

        Args:
            input_dim (int): Giriş boyutu.
            output_dim (int): Çıkış boyutu.

        Returns:
            torch.Tensor: Başlatılmış ağırlık matrisi.
        """
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Giriş ve çıkış boyutları pozitif olmalıdır.")

        weights = torch.empty(input_dim, output_dim)
        initialized_weights = self.initialize_weights(weights)

        if self.verbose:
            print(f"[AttentionInitializer] Parametre matrisi başlatıldı: {initialized_weights.shape}")
        return initialized_weights

    def initialize_bias(self, size):
        """
        Bias vektörünü başlatır.

        Args:
            size (int): Bias vektörünün boyutu.

        Returns:
            torch.Tensor: Başlatılmış bias vektörü.
        """
        if size <= 0:
            raise ValueError("Bias boyutu pozitif olmalıdır.")

        bias = torch.zeros(size)

        if self.verbose:
            print(f"[AttentionInitializer] Bias başlatıldı: Şekil {bias.shape}")
        return bias

    def __call__(self, tensor):
        """
        Tensörleri başlatmak için çağrılabilir metot.

        Args:
            tensor (torch.Tensor): Başlatılacak tensör.

        Returns:
            torch.Tensor: Başlatılmış tensör.
        """
        return self.initialize_weights(tensor)

    def log_initialization_details(self, tensor, description=""):
        """
        Başlatma süreci hakkında bilgi verir.

        Args:
            tensor (torch.Tensor): Başlatılmış tensör.
            description (str): Tensörün açıklaması (isteğe bağlı).
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Loglama için verilen veri bir PyTorch tensörü olmalıdır.")

        if self.verbose:
            print(f"[AttentionInitializer] {description} başlatıldı:")
            print(f"  - Şekil: {tensor.shape}")
            print(f"  - Min Değer: {tensor.min().item():.4f}")
            print(f"  - Max Değer: {tensor.max().item():.4f}")
            print(f"  - Ortalama Değer: {tensor.mean().item():.4f}")

    def validate_tensor(self, tensor):
        """
        Tensörün başlatma sonrası geçerliliğini kontrol eder.

        Args:
            tensor (torch.Tensor): Kontrol edilecek tensör.

        Returns:
            bool: Tensör geçerli mi?
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Kontrol için verilen veri bir PyTorch tensörü olmalıdır.")

        if torch.isnan(tensor).any():
            raise ValueError("Tensör NaN değerler içeriyor.")
        if torch.isinf(tensor).any():
            raise ValueError("Tensör sonsuz değerler içeriyor.")
        return True
