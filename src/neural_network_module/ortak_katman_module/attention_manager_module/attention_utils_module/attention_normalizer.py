import torch
import torch.nn as nn
from training_management.training_logger import TrainingLogger
logger = TrainingLogger()


class AttentionNormalizer(nn.Module):
    """
    Dikkat mekanizmaları için tensör normalizasyon sınıfı.
    """

    def __init__(self, normalization_type="layer_norm", embed_dim=None, seq_len=None, eps=None, verbose=False,momentum=None):
        """
        AttentionNormalizer sınıfını başlatır.

        Args:
            normalization_type (str): Normalizasyon tipi ("layer_norm", "batch_norm", "group_norm", "instance_norm").
            embed_dim (int, optional): Katman veya özellik boyutu (LayerNorm, GroupNorm, InstanceNorm için kullanılır).
            seq_len (int, optional): Sekans boyutu (BatchNorm için kullanılır).
            eps (float, optional): Sayısal kararlılığı artırmak için küçük bir değer.
            verbose (bool): Detaylı loglama etkinleştirme seçeneği.

        Raises:
            ValueError: Geçersiz parametreler veya normalizasyon türleri için hata mesajı.
        """
        super(AttentionNormalizer, self).__init__()
        # Global logger'ı kullanarak örnek seviyesinde logger oluştur
        self.logger = logger  # Global logger'ın tanımlı olduğu varsayılıyor
        self.logger.debug("MultiHeadAttention __init__ çağrıldı.")
        self.momentum = momentum

        # Tip doğrulama
        try:
            if not isinstance(normalization_type, str):
                raise TypeError("`normalization_type` bir string olmalıdır.")
            if not isinstance(verbose, bool):
                raise TypeError("`verbose` bir bool türünde olmalıdır.")
            if eps is not None and not isinstance(eps, (float, int)):
                raise TypeError("`eps` bir float veya int türünde olmalıdır.")
        except Exception as e:
            print(f"[Hata] Tip doğrulama sırasında bir hata oluştu: {e}")
            raise

        self.normalization_type = normalization_type.lower()
        self.verbose = verbose

        # Varsayılan eps değerlerini tanımlama
        try:
            default_eps = {
                "layer_norm": 1e-5,
                "batch_norm": 1e-5,
                "group_norm": 1e-5,
                "instance_norm": 1e-5
            }
            self.eps = eps if eps is not None else default_eps.get(self.normalization_type, 1e-6)
        except Exception as e:
            print(f"[Hata] eps değeri atanırken bir hata oluştu: {e}")
            raise

        # Desteklenen normalizasyon türleri
        self.supported_normalizations = ["layer_norm", "batch_norm", "group_norm", "instance_norm"]

        # Geçersiz normalizasyon türü kontrolü
        try:
            if self.normalization_type not in self.supported_normalizations:
                raise ValueError(
                    f"Geçersiz normalizasyon tipi: '{self.normalization_type}'. "
                    f"Desteklenen türler: {', '.join(self.supported_normalizations)}."
                )
        except Exception as e:
            print(f"[Hata] Geçersiz normalizasyon türü: {e}")
            raise

        # Parametre doğrulama ve ayarlama
        try:
            if self.normalization_type == "layer_norm":
                if embed_dim is None:
                    raise ValueError("LayerNorm için 'embed_dim' belirtilmelidir.")
                self.embed_dim = embed_dim

            elif self.normalization_type == "batch_norm":
                if seq_len is None:
                    raise ValueError("BatchNorm için 'seq_len' belirtilmelidir.")
                self.seq_len = seq_len

            elif self.normalization_type in ["group_norm", "instance_norm"]:
                if embed_dim is None:
                    raise ValueError(f"{self.normalization_type} için 'embed_dim' belirtilmelidir.")
                self.embed_dim = embed_dim
        except Exception as e:
            print(f"[Hata] Parametre doğrulama sırasında bir hata oluştu: {e}")
            raise

        # Normalizasyon modülünü başlat
        try:
            self.normalizer = self._initialize_normalizer(embed_dim=embed_dim, seq_len=seq_len)
        except Exception as e:
            print(f"[Hata] Normalizasyon modülü başlatılırken bir hata oluştu: {e}")
            raise

        # verbose modu: detaylı loglama
        if self.verbose:
            print(
                f"[AttentionNormalizer] '{self.normalization_type}' normalizasyonu başarıyla başlatıldı. "
                f"eps={self.eps}, embed_dim={embed_dim}, seq_len={seq_len}."
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Giriş tensörü üzerinde normalizasyon işlemini gerçekleştirir.
        - Giriş tensörü 3B (batch_size, seq_len, embed_dim) veya 4B (batch_size, num_heads, seq_len, head_dim) olabilir.
        - 4B tensörler, normalizasyon işlemi için uygun forma dönüştürülür; normalizasyon sonrası orijinal forma geri döndürülür.
        
        Args:
            x (torch.Tensor): Giriş tensörü.
        
        Returns:
            torch.Tensor: Normalize edilmiş tensör, orijinal giriş şekline uyarlanmış.
        
        Raises:
            ValueError, TypeError: Giriş doğrulaması veya dönüşüm aşamalarında oluşan hatalarda.
            RuntimeError: Normalizasyon sonrası tensörde beklenmeyen durumlarda.
        """
        import time
        t_start = time.time()
        try:
            # Adım 1: Giriş doğrulaması ve orijinal şeklin kaydedilmesi
            self.validate_input(x)
            original_shape = x.shape
            self.logger.debug(f"[AttentionNormalizer] Giriş doğrulandı: Şekil={x.shape}, dtype={x.dtype}")
        except Exception as e:
            self.logger.error(f"[AttentionNormalizer] Giriş doğrulaması başarısız: {e}", exc_info=True)
            raise

        t_validation = time.time()

        # Adım 2: Tensörü normalizasyon için uygun formata dönüştür
        try:
            if x.ndim == 4:
                # 4B tensör: [batch_size, num_heads, seq_len, head_dim]
                batch_size, num_heads, seq_len, head_dim = x.size()
                embed_dim = num_heads * head_dim
                self.logger.debug(f"[AttentionNormalizer] 4B tensör algılandı: batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
                if self.normalization_type == "batch_norm":
                    # BatchNorm: tensör [batch_size * seq_len, embed_dim]
                    x_transformed = x.permute(0, 2, 1, 3).reshape(batch_size * seq_len, embed_dim)
                elif self.normalization_type in ["group_norm", "instance_norm"]:
                    # Group/InstanceNorm: tensör [batch_size, embed_dim, seq_len] (sonrasında 3B olarak uygulanır)
                    x_transformed = x.reshape(batch_size, embed_dim, seq_len).permute(0, 2, 1)
                else:
                    # LayerNorm: [batch_size, seq_len, embed_dim]
                    x_transformed = x.reshape(batch_size, seq_len, embed_dim)
                self.logger.debug(f"[AttentionNormalizer] 4B tensör 3B formata dönüştürüldü: {x_transformed.shape}")
            elif x.ndim == 3:
                self.logger.debug(f"[AttentionNormalizer] 3B tensör algılandı: {x.shape}")
                x_transformed = x
            else:
                raise ValueError(f"Giriş tensörü yalnızca 3B veya 4B olabilir. Alınan: {x.ndim}B, Şekil: {x.shape}")
        except Exception as e:
            self.logger.error(f"[AttentionNormalizer] Tensör dönüşüm aşamasında hata: {e}", exc_info=True)
            raise

        t_transform = time.time()

        # Adım 3: Normalizasyon işlemini uygula
        try:
            normalized_x = self._apply_normalization(x_transformed)
            self.logger.debug(f"[AttentionNormalizer] Normalizasyon uygulandı. Çıkış tensör şekli: {normalized_x.shape}")
        except Exception as e:
            self.logger.error(f"[AttentionNormalizer] Normalizasyon sırasında hata: {e}", exc_info=True)
            raise

        t_normalization = time.time()

        # Adım 4: Orijinal forma geri dönüş
        try:
            if original_shape != normalized_x.shape:
                # Eğer orijinal giriş 4B ise, sonucu 4B'ye dönüştür
                if len(original_shape) == 4:
                    # Beklenen 4B şekil: (batch_size, num_heads, seq_len, head_dim)
                    if normalized_x.ndim in [2, 3] or normalized_x.shape[0] != batch_size:
                        if self.normalization_type == "batch_norm":
                            normalized_x = normalized_x.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
                        elif self.normalization_type in ["group_norm", "instance_norm"]:
                            normalized_x = normalized_x.permute(0, 2, 1).reshape(batch_size, num_heads, seq_len, head_dim)
                        else:
                            normalized_x = normalized_x.reshape(batch_size, num_heads, seq_len, head_dim)
                        self.logger.debug(f"[AttentionNormalizer] 4B tensör orijinal forma geri dönüştürüldü: {normalized_x.shape}")
                # Eğer orijinal giriş 2B ise, squeeze işlemi
                elif len(original_shape) == 2:
                    normalized_x = normalized_x.squeeze(0)
                    self.logger.debug(f"[AttentionNormalizer] 2B tensör orijinal forma geri dönüştürüldü: {normalized_x.shape}")
        except Exception as e:
            self.logger.error(f"[AttentionNormalizer] Orijinal forma dönüş sırasında hata: {e}", exc_info=True)
            raise

        t_reconstruct = time.time()

        # Adım 5: Çıkış tensörünü kontrol et
        try:
            if normalized_x.shape != original_shape:
                raise ValueError(
                    f"Normalizasyon sonrası tensör boyutu, orijinal giriş boyutuyla uyuşmuyor. "
                    f"Orijinal: {original_shape}, Normalized: {normalized_x.shape}"
                )
            self.logger.debug(f"[AttentionNormalizer] Çıkış tensörü doğrulandı: {normalized_x.shape}")
        except Exception as e:
            self.logger.error(f"[AttentionNormalizer] Çıkış doğrulama hatası: {e}", exc_info=True)
            raise

        total_time = time.time() - t_start
        self.logger.info(f"[AttentionNormalizer] Forward pass tamamlandı: Toplam süre: {total_time:.6f} s")
        self.logger.debug(
            f"[AttentionNormalizer] Zaman Ölçümleri: Doğrulama: {(t_validation - t_start):.4f}s, "
            f"Dönüşüm: {(t_transform - t_validation):.4f}s, Normalizasyon: {(t_normalization - t_transform):.4f}s, "
            f"Yeniden Yapılandırma: {(t_reconstruct - t_normalization):.4f}s"
        )
        return normalized_x





    def _initialize_normalizer(self, embed_dim: int = None, seq_len: int = None) -> nn.Module:
        """
        Normalizasyon türüne göre uygun modülü başlatır.

        Args:
            embed_dim (int, optional): Katman veya özellik boyutu. (LayerNorm, GroupNorm, InstanceNorm için gereklidir.)
            seq_len (int, optional): Sekans boyutu. (BatchNorm için gereklidir.)

        Returns:
            nn.Module: Uygun normalizasyon modülü.

        Raises:
            ValueError: Geçersiz veya eksik parametre durumunda hata.
            RuntimeError: Desteklenmeyen normalizasyon türü durumunda hata.
        """
        try:
            # Desteklenen normalizasyon türlerini tanımla
            supported_types = {
                'layer_norm': self._initialize_layer_norm,
                'batch_norm': self._initialize_batch_norm,
                'group_norm': self._initialize_group_norm,
                'instance_norm': self._initialize_instance_norm
            }

            # Verbose modda detaylı loglama
            if self.verbose:
                print(
                    f"[AttentionNormalizer] `_initialize_normalizer` çağrıldı: "
                    f"normalization_type={self.normalization_type}, embed_dim={embed_dim}, seq_len={seq_len}, eps={self.eps}"
                )

            # Geçersiz normalizasyon türü kontrolü
            if self.normalization_type not in supported_types:
                raise RuntimeError(
                    f"Geçersiz normalizasyon türü: '{self.normalization_type}'. "
                    f"Desteklenen türler: {', '.join(supported_types.keys())}."
                )

            # Parametre doğrulama
            if self.normalization_type in ["layer_norm", "group_norm", "instance_norm"]:
                self._validate_positive_param("embed_dim", embed_dim)
            elif self.normalization_type == "batch_norm":
                if seq_len is None:
                    raise ValueError("BatchNorm için `seq_len` parametresi belirtilmelidir.")
                self._validate_positive_param("seq_len", seq_len)

            # GroupNorm uyum kontrolü
            if self.normalization_type == "group_norm" and embed_dim:
                num_groups = self._calculate_num_groups(embed_dim)
                if embed_dim % num_groups != 0:
                    if self.verbose:
                        print(
                            f"[AttentionNormalizer] Uyarı: embed_dim ({embed_dim}) ve num_groups ({num_groups}) uyumsuz. "
                            f"Grup sayısı varsayılan olarak 1'e ayarlandı."
                        )
                    num_groups = 1

            # Uygun başlatma metodunu çağır
            normalizer = supported_types[self.normalization_type](embed_dim, seq_len)

            if self.verbose:
                print(f"[AttentionNormalizer] Normalizer başarıyla başlatıldı: Type={type(normalizer)}")

            return normalizer

        except ValueError as ve:
            # Parametre doğrulama hatası
            print(f"[Hata] Parametre doğrulama hatası: {ve}")
            raise
        except RuntimeError as re:
            # Geçersiz normalizasyon türü hatası
            print(f"[Hata] Desteklenmeyen normalizasyon türü: {re}")
            raise
        except Exception as e:
            # Beklenmeyen hata
            print(f"[Hata] `_initialize_normalizer` metodunda beklenmeyen bir hata oluştu: {e}")
            raise





    def _initialize_layer_norm(self, embed_dim, seq_len=None):
        """
        Layer Normalization için modülü başlatır.

        Args:
            embed_dim (int): Özellik boyutu.
            seq_len (int, optional): Sekans boyutu (kullanılmıyor, sadece uyumluluk için).

        Returns:
            nn.LayerNorm: Layer Normalization modülü.

        Raises:
            ValueError: embed_dim geçerli değilse.
        """
        try:
            # embed_dim değerini doğrula
            self._validate_positive_param("embed_dim", embed_dim)

            # LayerNorm modülünü başlat
            normalizer = nn.LayerNorm(normalized_shape=embed_dim, eps=self.eps)

            if self.verbose:
                print(f"[AttentionNormalizer] LayerNorm başlatıldı: embed_dim={embed_dim}, eps={self.eps}")

            return normalizer

        except ValueError as ve:
            print(f"[Hata] LayerNorm başlatılırken bir hata oluştu: {ve}")
            raise
        except Exception as e:
            print(f"[Hata] LayerNorm başlatılırken beklenmeyen bir hata oluştu: {e}")
            raise

    def _initialize_batch_norm(self, embed_dim: int, seq_len: int) -> nn.BatchNorm1d:
        """
        Batch Normalization için modülü başlatır.

        Args:
            embed_dim (int): Özellik boyutu.
            seq_len (int): Sekans boyutu.

        Returns:
            nn.BatchNorm1d: Batch Normalization modülü.

        Raises:
            ValueError: embed_dim veya seq_len geçerli değilse.
        """
        try:
            # Giriş doğrulama: seq_len pozitif olmalı
            if seq_len is None or seq_len <= 0:
                raise ValueError("BatchNorm için 'seq_len' pozitif bir değer olmalıdır.")

            # Giriş doğrulama: embed_dim pozitif olmalı
            embed_dim = embed_dim or seq_len
            if embed_dim <= 0:
                raise ValueError("'embed_dim' pozitif bir değer olmalıdır.")

            # EPS ve momentum değerlerini dinamik olarak belirle
            eps_value = self.eps if self.eps is not None else 1e-3  # Daha kararlı bir EPS değeri
            momentum_value = 0.9  # Daha hızlı adaptasyon için momentum artırıldı

            # BatchNorm modülünü başlat
            normalizer = nn.BatchNorm1d(
                num_features=embed_dim,
                eps=eps_value,           # Optimize edilmiş EPS değeri
                momentum=momentum_value, # Optimize edilmiş momentum değeri
                
                track_running_stats=True # `running_mean` ve `running_var` izlenir
            )

            # Verbose modunda başlatma bilgisi yazdır
            if self.verbose:
                print(
                    f"[AttentionNormalizer] BatchNorm başlatıldı: "
                    f"num_features={embed_dim}, seq_len={seq_len}, "
                    f"eps={eps_value}, momentum={momentum_value}, track_running_stats=True"
                )

            return normalizer

        except ValueError as ve:
            # Parametre doğrulama hatası
            print(f"[Hata] BatchNorm başlatılırken doğrulama hatası: {ve}")
            raise
        except Exception as e:
            # Beklenmeyen hatalar
            print(f"[Hata] BatchNorm başlatılırken beklenmeyen bir hata oluştu: {e}")
            raise


    def _initialize_group_norm(self, embed_dim, seq_len=None):
        """
        Group Normalization için modülü başlatır.

        Args:
            embed_dim (int): Özellik boyutu.
            seq_len (int, optional): Sekans boyutu (kullanılmıyor, sadece uyumluluk için).

        Returns:
            nn.GroupNorm: Group Normalization modülü.

        Raises:
            ValueError: embed_dim geçerli değilse veya num_groups uyumu sağlanamıyorsa.
        """
        try:
            # Pozitif parametre doğrulaması
            self._validate_positive_param("embed_dim", embed_dim)

            # NumGroups hesaplama
            num_groups = self._calculate_num_groups(embed_dim)

            # Uyumsuzluk kontrolü
            if embed_dim % num_groups != 0:
                if self.verbose:
                    print(
                        f"[AttentionNormalizer] Uyarı: embed_dim ({embed_dim}) ve num_groups ({num_groups}) tam uyumlu değil. "
                        "Grup sayısı 1 olarak ayarlanıyor."
                    )
                num_groups = 1  # Uyumsuzluk durumunda varsayılan grup sayısı

            # GroupNorm modülünü başlat
            normalizer = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim, eps=self.eps)

            if self.verbose:
                print(f"[AttentionNormalizer] GroupNorm başlatıldı: num_groups={num_groups}, "
                    f"embed_dim={embed_dim}, eps={self.eps}")

            return normalizer

        except ValueError as ve:
            print(f"[Hata] Parametre doğrulama hatası: {ve}")
            raise
        except Exception as e:
            print(f"[Hata] `_initialize_group_norm` sırasında beklenmeyen bir hata oluştu: {e}")
            raise



    def _initialize_instance_norm(self, embed_dim: int, seq_len: int = None) -> nn.InstanceNorm1d:
        """
        Instance Normalization için modülü başlatır.

        Args:
            embed_dim (int): Özellik boyutu.
            seq_len (int, optional): Sekans boyutu (isteğe bağlı, uyumluluk için belirtilmiştir).

        Returns:
            nn.InstanceNorm1d: Instance Normalization modülü.

        Raises:
            ValueError: embed_dim geçerli değilse veya negatif bir değer ise.
            RuntimeError: InstanceNorm başlatma sırasında beklenmeyen bir hata oluşursa.
        """
        try:
            if self.verbose:
                print(f"[AttentionNormalizer] InstanceNorm başlatılıyor: embed_dim={embed_dim}, seq_len={seq_len}")

            # Parametre doğrulama: embed_dim kontrolü
            if embed_dim is None or embed_dim <= 0:
                raise ValueError(f"'embed_dim' pozitif bir değer olmalıdır. Sağlanan değer: {embed_dim}")

            # EPS değerini dinamik olarak optimize et
            eps_value = self.eps if self.eps is not None else 1e-3  # Varsayılan EPS
            if eps_value <= 0:
                raise ValueError(f"'eps' pozitif bir sayı olmalıdır. Sağlanan değer: {eps_value}")

            # Verbose mod: EPS bilgisi
            if self.verbose:
                print(f"[AttentionNormalizer] EPS değeri belirlendi: {eps_value}")

            # InstanceNorm parametreleri doğrulama ve başlatma
            normalizer = nn.InstanceNorm1d(
                num_features=embed_dim,
                eps=eps_value,
                affine=True,
                track_running_stats=True
            )

            # Çıkış doğrulama
            if not isinstance(normalizer, nn.InstanceNorm1d):
                raise RuntimeError(
                    "InstanceNorm modülü başlatılırken bir hata oluştu. "
                    "Dönen nesne nn.InstanceNorm1d türünde değil."
                )

            # Verbose mod: Başlatma logu
            if self.verbose:
                print(
                    f"[AttentionNormalizer] InstanceNorm başarıyla başlatıldı: "
                    f"embed_dim={embed_dim}, eps={eps_value}, affine=True, track_running_stats=True"
                )

            return normalizer

        except ValueError as ve:
            # Parametre doğrulama hatası
            print(f"[Hata] InstanceNorm başlatılırken doğrulama hatası: {ve}")
            raise

        except RuntimeError as re:
            # Çalışma zamanı hatası
            print(f"[Hata] InstanceNorm başlatılırken çalışma zamanı hatası: {re}")
            raise

        except Exception as e:
            # Beklenmeyen hata
            print(f"[Hata] InstanceNorm başlatılırken beklenmeyen bir hata oluştu: {e}")
            raise


    def _calculate_num_groups(self, embed_dim):
        """
        GroupNorm için num_groups değerini dinamik olarak hesaplar.

        Args:
            embed_dim (int): Özellik boyutu.

        Returns:
            int: Uygun grup sayısı.
        """
        num_groups = max(1, min(embed_dim // 16, 32))  # Dinamik num_groups hesaplama
        if embed_dim % num_groups != 0:
            if self.verbose:
                print(f"[AttentionNormalizer] `embed_dim` ({embed_dim}) ve `num_groups` ({num_groups}) uyumsuz. "
                    f"Grup sayısı 1 olarak ayarlanıyor.")
            num_groups = 1  # Uyumsuzluk durumunda grup sayısı 1 olarak ayarlanır
        return num_groups


    def _validate_positive_param(self, param_name, value, allow_zero=False):
        """
        Pozitif veya sıfır değer doğrulama metodu.

        Args:
            param_name (str): Doğrulanan parametrenin adı.
            value (int or float): Doğrulanacak değer.
            allow_zero (bool): Değerin sıfır olmasına izin verilip verilmediği. Varsayılan: False.

        Raises:
            ValueError: Değer pozitif değilse veya sıfır olması izin verilen durumda negatifse.
        """
        try:
            # Değerin None olmaması kontrolü
            if value is None:
                raise ValueError(f"{param_name} belirtilmelidir, ancak None değeri sağlandı.")

            # Pozitiflik kontrolü
            if allow_zero:
                if value < 0:
                    raise ValueError(f"{param_name} sıfırdan küçük olamaz. Sağlanan değer: {value}.")
            else:
                if value <= 0:
                    raise ValueError(f"{param_name} pozitif bir değer olmalı. Sağlanan değer: {value}.")

            # EPS kontrolü (spesifik olarak EPS için bir kontrol gerekiyorsa)
            if param_name.lower() == "eps" and value < 1e-6:
                raise ValueError(
                    f"{param_name} (EPS) çok küçük bir değer. Stabilite için 1e-6 veya daha büyük bir değer önerilir. "
                    f"Sağlanan değer: {value}."
                )

            # Verbose loglama
            if self.verbose:
                print(f"[AttentionNormalizer] {param_name} doğrulandı: {value} (allow_zero={allow_zero})")

        except ValueError as ve:
            print(f"[Hata] {param_name} doğrulama hatası: {ve}")
            raise
        except Exception as e:
            print(f"[Hata] `_validate_positive_param` metodunda beklenmeyen bir hata oluştu: {e}")
            raise


    def _apply_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Belirtilen normalizasyon işlemini uygular.

        Args:
            x (torch.Tensor): Normalizasyon uygulanacak tensör.

        Returns:
            torch.Tensor: Normalizasyon işlemi sonrası tensör.

        Raises:
            ValueError: Geçersiz normalizasyon türü veya tensör boyutları durumunda hata.
            RuntimeError: Normalizasyon türü desteklenmiyorsa hata.
        """
        try:
            # Giriş doğrulama
            self.validate_input(x)
            if self.verbose:
                print(
                    f"[AttentionNormalizer] Normalizasyon başlıyor: tensör şekli={x.shape}, tür={self.normalization_type}"
                )

            # NaN/Inf kontrolü (fallback mekanizması için)
            if not torch.isfinite(x).all():
                raise ValueError("Giriş tensörü NaN veya Inf değerler içeriyor. Normalizasyon uygulanamaz.")

            # Normalizasyon türüne göre işlemi seç
            normalized_x = None
            if self.normalization_type == "layer_norm":
                normalized_x = self._apply_layer_norm(x)
            elif self.normalization_type == "batch_norm":
                normalized_x = self._apply_batch_norm(x)
            elif self.normalization_type == "group_norm":
                normalized_x = self._apply_group_norm(x)
            elif self.normalization_type == "instance_norm":
                normalized_x = self._apply_instance_norm(x)
            else:
                raise RuntimeError(
                    f"Geçersiz normalizasyon türü: {self.normalization_type}. "
                    f"Desteklenen türler: 'layer_norm', 'batch_norm', 'group_norm', 'instance_norm'."
                )

            # Normalizasyon sonrası NaN/Inf kontrolü
            if not torch.isfinite(normalized_x).all():
                raise RuntimeError("Normalizasyon sonrası tensörde NaN veya Inf değerler bulundu.")

            # Çıkış tensörünün ortalama ve standart sapmasını kontrol et
            output_mean = normalized_x.mean()
            output_std = normalized_x.std()
            if self.verbose:
                print(
                    f"[AttentionNormalizer] Normalizasyon tamamlandı: "
                    f"Çıkış Ortalama={output_mean.item():.6f}, Çıkış Standart Sapma={output_std.item():.6f}, "
                    f"Çıkış Şekli={normalized_x.shape}"
                )

            return normalized_x

        except ValueError as ve:
            # Değer hatası durumunda loglama
            print(f"[Hata] Normalizasyon işlemi sırasında doğrulama hatası: {ve}")
            raise
        except RuntimeError as re:
            # Desteklenmeyen normalizasyon türü hatası
            print(f"[Hata] Normalizasyon türü hatası: {re}")
            raise
        except Exception as e:
            # Beklenmeyen hatalar için genel loglama
            print(f"[Hata] `_apply_normalization` metodunda beklenmeyen bir hata oluştu: {e}")
            raise


    def _apply_layer_norm(self, x):
        """
        Layer Normalization işlemini uygular.
        Args:
            x (torch.Tensor): Giriş tensörü.
        Returns:
            torch.Tensor: LayerNorm uygulanmış tensör.
        """
        try:
            normalized_x = self.normalizer(x)
            if self.verbose:
                print(f"[AttentionNormalizer] LayerNorm başarıyla uygulandı: Çıkış Şekli={normalized_x.shape}")
            return normalized_x
        except Exception as e:
            raise RuntimeError(f"LayerNorm uygulanırken bir hata oluştu: {e}")


    def _apply_batch_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batch Normalization işlemini uygular.

        Args:
            x (torch.Tensor): Giriş tensörü.

        Returns:
            torch.Tensor: BatchNorm uygulanmış tensör.

        Raises:
            ValueError: Giriş tensörü boyutları veya embed_dim uyumsuzluğu durumunda.
            RuntimeError: BatchNorm uygulanırken bir hata oluşursa.
        """
        try:
            if self.verbose:
                print(f"[AttentionNormalizer] BatchNorm başlatılıyor: Giriş Şekli={x.shape}")

            # Giriş tensörünün boyutlarını kontrol et
            if x.ndim not in (3, 4):
                raise ValueError(
                    f"BatchNorm yalnızca 3D veya 4D tensörler üzerinde çalışır. Ancak giriş {x.ndim}D bulundu."
                )

            # EPS ve Momentum Değerlerini Optimize Et
            eps_value = self.eps if self.eps is not None else 1e-3
            momentum_value = 0.9

            # 3D tensör: [batch_size, seq_len, embed_dim]
            if x.ndim == 3:
                batch_size, seq_len, embed_dim = x.size()

                # embed_dim ve num_features uyumu kontrolü
                if self.normalizer.num_features != embed_dim:
                    raise ValueError(
                        f"BatchNorm için embed_dim ({embed_dim}) ile num_features ({self.normalizer.num_features}) uyuşmuyor."
                    )

                # Tensör dönüşümü ve normalizasyon
                x = x.transpose(1, 2)  # [batch_size, embed_dim, seq_len]
                normalized_x = self.normalizer(x)  # [batch_size, embed_dim, seq_len]
                normalized_x = normalized_x.transpose(1, 2)  # [batch_size, seq_len, embed_dim]

            # 4D tensör: [batch_size, num_heads, seq_len, head_dim]
            elif x.ndim == 4:
                batch_size, num_heads, seq_len, head_dim = x.size()
                embed_dim = num_heads * head_dim

                # embed_dim ve num_features uyumu kontrolü
                if self.normalizer.num_features != embed_dim:
                    raise ValueError(
                        f"BatchNorm için embed_dim ({embed_dim}) ile num_features ({self.normalizer.num_features}) uyuşmuyor."
                    )

                # Tensör dönüşümü ve normalizasyon
                x = x.permute(0, 2, 1, 3).reshape(batch_size * seq_len, embed_dim)  # [batch_size * seq_len, embed_dim]
                normalized_x = self.normalizer(x)  # [batch_size * seq_len, embed_dim]
                normalized_x = (
                    normalized_x.reshape(batch_size, seq_len, num_heads, head_dim)
                    .permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
                )

            # Sayısal Kararlılık Kontrolleri
            if not torch.isfinite(normalized_x).all():
                raise RuntimeError(
                    "BatchNorm uygulandıktan sonra tensörde NaN veya Inf değerleri bulundu."
                )

            # Çıkış Tensörünün Ortalama ve Standart Sapmasını Kontrol Et
            output_mean = normalized_x.mean(dim=(-1, -2), keepdim=True)
            output_std = normalized_x.std(dim=(-1, -2), keepdim=True)

            if self.verbose:
                print(
                    f"[AttentionNormalizer] Çıkış Ortalama: {output_mean.mean().item():.6f}, "
                    f"Standart Sapma: {output_std.mean().item():.6f}"
                )

            if self.verbose:
                print(f"[AttentionNormalizer] BatchNorm başarıyla uygulandı: Çıkış Şekli={normalized_x.shape}")

            return normalized_x

        except ValueError as ve:
            print(f"[Hata] BatchNorm uygulama sırasında doğrulama hatası: {ve}")
            raise
        except RuntimeError as re:
            print(f"[Hata] BatchNorm işlemi sırasında bir hata oluştu: {re}")
            raise
        except Exception as e:
            print(f"[Hata] BatchNorm uygulanırken beklenmeyen bir hata oluştu: {e}")
            raise

    def _apply_group_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Group Normalization işlemini uygular.

        Args:
            x (torch.Tensor): Giriş tensörü.

        Returns:
            torch.Tensor: GroupNorm uygulanmış tensör.

        Raises:
            ValueError: Giriş tensörü boyutları veya embed_dim uyumsuzluğu durumunda.
            RuntimeError: GroupNorm uygulanırken bir hata oluşursa.
        """
        try:
            if self.verbose:
                print(f"[AttentionNormalizer] GroupNorm başlatılıyor: Giriş Şekli={x.shape}")

            # Giriş doğrulaması
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Giriş tensörü bir PyTorch tensörü olmalıdır. Sağlanan tür: {type(x)}.")
            if x.ndim not in [3, 4]:
                raise ValueError(
                    f"GroupNorm yalnızca 3D veya 4D tensörler üzerinde çalışır. Ancak giriş {x.ndim}D bulundu."
                )

            # Grup sayısını hesaplama ve kontrol
            def calculate_num_groups(embed_dim):
                # Dinamik grup sayısı hesaplama
                num_groups = max(1, min(embed_dim // 16, 32))  # Grup sayısı 16 ile sınırlı
                if embed_dim % num_groups != 0:
                    if self.verbose:
                        print(
                            f"[AttentionNormalizer] embed_dim={embed_dim} ve num_groups={num_groups} uyumsuz. "
                            f"Grup sayısı varsayılan olarak 1'e ayarlandı."
                        )
                    num_groups = 1  # Uyumsuzluk durumunda grup sayısı 1 olarak ayarlanır
                return num_groups

            # 3D tensör: [batch_size, seq_len, embed_dim]
            if x.ndim == 3:
                batch_size, seq_len, embed_dim = x.size()

                # Grup sayısını hesapla
                num_groups = calculate_num_groups(embed_dim)

                # GroupNorm modülünü yeniden yapılandır
                if self.normalizer.num_groups != num_groups:
                    self.normalizer.num_groups = num_groups
                    if self.verbose:
                        print(
                            f"[AttentionNormalizer] embed_dim={embed_dim} ve num_groups={num_groups} ile "
                            f"GroupNorm yeniden yapılandırıldı."
                        )

                # Normalize et
                x = x.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]
                normalized_x = self.normalizer(x)  # [batch_size, embed_dim, seq_len]
                normalized_x = normalized_x.permute(0, 2, 1)  # [batch_size, seq_len, embed_dim]

            # 4D tensör: [batch_size, num_heads, seq_len, head_dim]
            elif x.ndim == 4:
                batch_size, num_heads, seq_len, head_dim = x.size()
                embed_dim = num_heads * head_dim

                # Grup sayısını hesapla
                num_groups = calculate_num_groups(embed_dim)

                # GroupNorm modülünü yeniden yapılandır
                if self.normalizer.num_groups != num_groups:
                    self.normalizer.num_groups = num_groups
                    if self.verbose:
                        print(
                            f"[AttentionNormalizer] embed_dim={embed_dim} ve num_groups={num_groups} ile "
                            f"GroupNorm yeniden yapılandırıldı."
                        )

                # Normalize et
                x = x.reshape(batch_size, embed_dim, seq_len).permute(0, 2, 1)  # [batch_size, seq_len, embed_dim]
                normalized_x = self.normalizer(x)  # [batch_size, seq_len, embed_dim]
                normalized_x = normalized_x.permute(0, 2, 1).reshape(batch_size, num_heads, seq_len, head_dim)

            else:
                raise ValueError(
                    f"GroupNorm yalnızca 3D veya 4D tensörler üzerinde çalışır. Ancak giriş {x.ndim}D bulundu."
                )

            # Çıkış tensörünün ortalama ve standart sapmasını kontrol et
            if self.verbose:
                output_mean = normalized_x.mean(dim=(-1, -2), keepdim=True)
                output_std = normalized_x.std(dim=(-1, -2), keepdim=True)
                print(
                    f"[AttentionNormalizer] Çıkış Ortalaması: {output_mean.mean().item():.6f}, "
                    f"Çıkış Standart Sapması: {output_std.mean().item():.6f}"
                )

            if self.verbose:
                print(f"[AttentionNormalizer] GroupNorm başarıyla uygulandı: Çıkış Şekli={normalized_x.shape}")
            return normalized_x

        except ValueError as ve:
            print(f"[Hata] GroupNorm uygulama sırasında doğrulama hatası: {ve}")
            raise
        except Exception as e:
            print(f"[Hata] GroupNorm uygulanırken beklenmeyen bir hata oluştu: {e}")
            raise



    def _apply_instance_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Instance Normalization işlemini uygular.

        Args:
            x (torch.Tensor): Giriş tensörü.

        Returns:
            torch.Tensor: InstanceNorm uygulanmış tensör.

        Raises:
            ValueError: Giriş tensörü boyutları geçersizse.
            TypeError: Giriş tensörü bir PyTorch tensörü değilse.
            RuntimeError: InstanceNorm uygulanırken bir hata oluşursa.
        """
        try:
            if self.verbose:
                print(f"[AttentionNormalizer] InstanceNorm başlatılıyor: Giriş Şekli={x.shape}")

            # Giriş doğrulama
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Giriş tensörü bir PyTorch tensörü olmalıdır. Sağlanan tür: {type(x)}.")
            if x.ndim not in (3, 4):
                raise ValueError(
                    f"InstanceNorm yalnızca 3D veya 4D tensörler üzerinde çalışır. Ancak giriş {x.ndim}D bulundu."
                )

            # 3D ve 4D tensörler için işleme
            if x.ndim == 3:  # 3D tensör: [batch_size, seq_len, embed_dim]
                batch_size, seq_len, embed_dim = x.size()

                # Boyut doğrulama
                if any(dim <= 0 for dim in (batch_size, seq_len, embed_dim)):
                    raise ValueError(
                        f"Giriş tensör boyutları geçersiz: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}. "
                        "Tüm boyutların pozitif bir değer olması gereklidir."
                    )

                if self.verbose:
                    print(
                        f"[AttentionNormalizer] 3D tensör işleniyor: batch_size={batch_size}, "
                        f"seq_len={seq_len}, embed_dim={embed_dim}"
                    )

                # Tensör dönüşümü ve InstanceNorm uygulama
                x = x.transpose(1, 2)  # [batch_size, embed_dim, seq_len]
                normalized_x = self.normalizer(x)  # [batch_size, embed_dim, seq_len]
                normalized_x = normalized_x.transpose(1, 2)  # [batch_size, seq_len, embed_dim]

            elif x.ndim == 4:  # 4D tensör: [batch_size, num_heads, seq_len, head_dim]
                batch_size, num_heads, seq_len, head_dim = x.size()
                embed_dim = num_heads * head_dim

                # Boyut doğrulama
                if any(dim <= 0 for dim in (batch_size, num_heads, seq_len, head_dim, embed_dim)):
                    raise ValueError(
                        f"Giriş tensör boyutları geçersiz: batch_size={batch_size}, num_heads={num_heads}, "
                        f"seq_len={seq_len}, head_dim={head_dim}, embed_dim={embed_dim}. "
                        "Tüm boyutların pozitif bir değer olması gereklidir."
                    )

                if self.verbose:
                    print(
                        f"[AttentionNormalizer] 4D tensör işleniyor: batch_size={batch_size}, "
                        f"num_heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}, embed_dim={embed_dim}"
                    )

                # Tensör dönüşümü ve InstanceNorm uygulama
                x = x.reshape(batch_size, embed_dim, seq_len).transpose(1, 2)  # [batch_size, seq_len, embed_dim]
                normalized_x = self.normalizer(x)  # [batch_size, seq_len, embed_dim]
                normalized_x = normalized_x.transpose(1, 2).reshape(batch_size, num_heads, seq_len, head_dim)

            else:
                raise ValueError(
                    f"InstanceNorm yalnızca 3D veya 4D tensörler üzerinde çalışır. Ancak giriş {x.ndim}D bulundu."
                )

            # Çıkış doğrulama ve loglama
            if not torch.isfinite(normalized_x).all():
                raise RuntimeError(
                    "InstanceNorm uygulandıktan sonra tensörde NaN veya Inf değerleri bulundu."
                )

            output_mean = normalized_x.mean(dim=(-1, -2), keepdim=True)
            output_std = normalized_x.std(dim=(-1, -2), keepdim=True)

            if self.verbose:
                print(
                    f"[AttentionNormalizer] Çıkış Ortalama: {output_mean.mean().item():.6f}, "
                    f"Standart Sapma: {output_std.mean().item():.6f}"
                )
                print(f"[AttentionNormalizer] InstanceNorm başarıyla uygulandı: Çıkış Şekli={normalized_x.shape}")

            return normalized_x

        except ValueError as ve:
            print(f"[Hata] InstanceNorm uygulama sırasında doğrulama hatası: {ve}")
            raise
        except TypeError as te:
            print(f"[Hata] InstanceNorm uygulama sırasında tür hatası: {te}")
            raise
        except Exception as e:
            print(f"[Hata] InstanceNorm uygulanırken beklenmeyen bir hata oluştu: {e}")
            raise



    def validate_input(self, x: torch.Tensor) -> None:
        """
        Giriş tensörünün şekil, tip ve değer doğrulamasını yapar.

        Args:
            x (torch.Tensor): Doğrulanacak tensör.

        Raises:
            TypeError: Tensör tipi geçerli değilse.
            ValueError: Tensör şekli geçersizse, boyut uyumsuzluğu varsa veya NaN/Inf değerleri içeriyorsa.
        """
        try:
            # Giriş türü kontrolü
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Giriş bir PyTorch tensörü olmalıdır. Sağlanan tür: {type(x)}.")

            # Giriş tensörü boyutları (3D veya 4D olmalı)
            if x.ndim not in (3, 4):
                raise ValueError(
                    f"Giriş tensörünün boyutları yalnızca 3D veya 4D olabilir. "
                    f"Sağlanan boyut: {x.ndim}. Beklenen boyutlar: "
                    f"[3D: (batch_size, seq_len, embed_dim) veya 4D: (batch_size, num_heads, seq_len, head_dim)]."
                )

            # Pozitif boyut kontrolü
            invalid_dims = [dim for dim in x.shape if dim <= 0]
            if invalid_dims:
                raise ValueError(
                    f"Giriş tensör boyutları geçersiz: {x.shape}. "
                    f"Tüm boyutların pozitif bir değer olması gereklidir. Geçersiz boyutlar: {invalid_dims}."
                )

            # NaN/Inf değer kontrolü
            if not torch.isfinite(x).all():
                raise ValueError(
                    "Giriş tensörü geçersiz değerler içeriyor. "
                    "Tensör yalnızca sonlu (finite) değerler içermelidir. NaN veya Inf değerler bulundu."
                )

            # Tensörün 3D olması durumu (batch_size, seq_len, embed_dim)
            if x.ndim == 3:
                batch_size, seq_len, embed_dim = x.size()
                if batch_size < 1:
                    raise ValueError("3D tensör için batch_size değeri en az 1 olmalıdır.")
                if seq_len < 2:
                    raise ValueError("3D tensör için seq_len değeri en az 2 olmalıdır.")
                if embed_dim <= 0:
                    raise ValueError("3D tensör için embed_dim pozitif bir değer olmalıdır.")

                if self.verbose:
                    print(
                        f"[AttentionNormalizer] 3D tensör doğrulandı: "
                        f"batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}."
                    )

            # Tensörün 4D olması durumu (batch_size, num_heads, seq_len, head_dim)
            elif x.ndim == 4:
                batch_size, num_heads, seq_len, head_dim = x.size()
                if batch_size < 1:
                    raise ValueError("4D tensör için batch_size değeri en az 1 olmalıdır.")
                if num_heads < 1:
                    raise ValueError("4D tensör için num_heads değeri en az 1 olmalıdır.")
                if seq_len < 2:
                    raise ValueError("4D tensör için seq_len değeri en az 2 olmalıdır.")
                if head_dim <= 0:
                    raise ValueError("4D tensör için head_dim pozitif bir değer olmalıdır.")

                # 4D tensör için embed_dim doğrulaması (num_heads * head_dim)
                embed_dim = num_heads * head_dim
                if embed_dim <= 0:
                    raise ValueError(
                        f"4D tensör için embed_dim geçersiz: {embed_dim}. "
                        f"num_heads ({num_heads}) ve head_dim ({head_dim}) ile uyumlu olmalıdır."
                    )

                if self.verbose:
                    print(
                        f"[AttentionNormalizer] 4D tensör doğrulandı: "
                        f"batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}."
                    )

            # Loglama (özet bilgi)
            if self.verbose:
                print(f"[AttentionNormalizer] Giriş doğrulandı: Şekil={x.shape}, Tür={x.dtype}.")

        except (TypeError, ValueError) as e:
            # Hata durumunda loglama
            print(f"[Hata] Giriş doğrulama hatası: {e}")
            raise
        except Exception as e:
            # Beklenmeyen hata durumunda loglama
            print(f"[Hata] `validate_input` metodunda beklenmeyen bir hata oluştu: {e}")
            raise





    def extra_repr(self):
        """
        Sınıfın özetini verir.

        Returns:
            str: Sınıf özet metni.
        """
        try:
            # Temel parametreleri içeren özet metni
            repr_str = (
                f"normalization_type={self.normalization_type}, "
                f"eps={self.eps}, "
                f"verbose={self.verbose}, "
                f"supported_normalizations={self.supported_normalizations}"
            )

            # embed_dim değeri varsa ekle
            if hasattr(self.normalizer, 'normalized_shape') and self.normalizer.normalized_shape is not None:
                repr_str += f", embed_dim={self.normalizer.normalized_shape}"

            # seq_len değeri varsa ekle
            if hasattr(self.normalizer, 'num_features') and self.normalizer.num_features is not None:
                repr_str += f", seq_len={self.normalizer.num_features}"

            # Loglama (verbose mod aktifse)
            if self.verbose:
                print(f"[AttentionNormalizer] Parametreler: {repr_str}")

            return repr_str

        except AttributeError as ae:
            print(f"[Hata] Özet oluşturulurken bir özellik bulunamadı: {ae}")
            raise
        except Exception as e:
            print(f"[Hata] `extra_repr` metodunda beklenmeyen bir hata oluştu: {e}")
            raise
