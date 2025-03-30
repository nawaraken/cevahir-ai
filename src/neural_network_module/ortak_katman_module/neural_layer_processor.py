import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import time
import torch.nn as nn
import logging
from .attention_manager_module.multi_head_attention import MultiHeadAttention
from .attention_manager_module.self_attention import SelfAttention
from .attention_manager_module.cross_attention import CrossAttention
from .attention_manager_module.attention_optimizer import AttentionOptimizer
from .attention_manager_module.attention_utils_module.attention_initializer import AttentionInitializer
from .attention_manager_module.attention_utils_module.attention_normalizer import AttentionNormalizer
from .attention_manager_module.attention_utils_module.attention_scaler import AttentionScaler

class NeuralLayerProcessor(nn.Module):
    def __init__(self, 
                embed_dim, 
                num_heads, 
                dropout,
                attention_type="multi_head", 
                debug=False, 
                normalization_type="layer_norm", 
                scaling_strategy="sqrt", 
                scaling_method="softmax", 
                log_level=logging.INFO, 
                seed=None,
                eps=1e-5, 
                scale_factor=1.0,
                scaling_factor=None, 
                clip_range=None, 
                normalize_method="softmax",
                verbose=False, 
                num_groups=None):

        super(NeuralLayerProcessor, self).__init__()

        # **  Parametre Doğrulama **
        self._validate_parameters(embed_dim, num_heads, attention_type, normalization_type, scaling_strategy, scaling_method, clip_range, dropout)

        # **  Rastgelelik Sabitleme **
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # **  Özelliklerin Tanımlanması **
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.debug = debug
        self.normalization_type = normalization_type
        self.scaling_strategy = scaling_strategy
        self.scaling_method = scaling_method
        self.eps = eps
        self.scale_factor = scale_factor
        self.scaling_factor = scaling_factor if scaling_factor is not None else scale_factor
        self.clip_range = clip_range
        self.verbose = verbose
        self.num_groups = num_groups
        self.dropout=dropout
        self.normalize_method = normalize_method  # Normalize method eklendi

        # ** Varsayılan Clipping Değeri Tanımlama **
        self.clip_value = clip_range[1] if clip_range else None  # Varsayılan olarak None olabilir

        # **  Logger Tanımlama **
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        if not self.logger.hasHandlers():  # Tekrarlı handler eklememek için
            self.logger.addHandler(handler)

        # **  Modüllerin Başlatılması **
        try:
            self.multi_head_attention = self._initialize_module(
                MultiHeadAttention, embed_dim, num_heads, self.dropout, normalization_type, debug
            )
            self.self_attention = self._initialize_module(
                SelfAttention, embed_dim, num_heads, dropout, normalization_type, debug, eps=eps, num_groups=num_groups
            )
            self.cross_attention = self._initialize_module(
                CrossAttention, embed_dim, num_heads, dropout, attention_scaling=True,
                normalization_type=normalization_type, scaling_strategy=scaling_strategy, debug=debug
            )
        except Exception as e:
            self.logger.error(f"[ERROR] Attention modules could not be initialized: {e}")
            raise RuntimeError(f"Attention module initialization failed: {e}")

        # **  Yardımcı Modüllerin Başlatılması **
        try:
            self.attention_optimizer = AttentionOptimizer(  # Önceki `self.optimizer` yanlış tanımlanmıştı
                epsilon=eps, 
                verbose=verbose, 
                default_scaling_method=scaling_method,
                default_clipping_value=self.clip_value
            )
            self.initializer = AttentionInitializer(
                initialization_type="xavier", seed=seed, verbose=verbose
            )
            self.normalizer = AttentionNormalizer(
                normalization_type=normalization_type, embed_dim=embed_dim, eps=eps,
                verbose=verbose, momentum=0.9
            )
            self.attention_scaler = AttentionScaler(
                scale_factor=scale_factor, clip_range=clip_range, verbose=verbose, num_heads=num_heads
            )
        except Exception as e:
            self.logger.error(f"[ERROR] Auxiliary components could not be initialized: {e}")
            raise RuntimeError(f"Auxiliary components initialization failed: {e}")

        # ** Dropout Katmanının Tanımlanması **
        self.dropout_layer = nn.Dropout(self.dropout)  # self.dropout_layer tanımlandı


    def _validate_parameters(self, embed_dim, num_heads, attention_type, normalization_type, 
                            scaling_strategy, scaling_method, clip_range, dropout, eps=1e-5, 
                            scale_factor=1.0, verbose=False, num_groups=None):
        """
        Parametrelerin doğruluğunu kontrol eder ve hataları loglar.

        Args:
            embed_dim (int): Gömme boyutu.
            num_heads (int): Çok başlık sayısı.
            attention_type (str): Dikkat türü.
            normalization_type (str): Normalizasyon türü.
            scaling_strategy (str): Ölçeklendirme stratejisi.
            scaling_method (str): Ölçeklendirme yöntemi.
            clip_range (tuple, optional): Kırpma aralığı.
            dropout (float): Dropout oranı.
            eps (float): Sayısal kararlılık için epsilon değeri.
            scale_factor (float): Ölçeklendirme çarpanı.
            verbose (bool): Loglama durumu.
            num_groups (int, optional): Grup sayısı (GroupNorm için).

        Raises:
            ValueError: Geçersiz parametre değeri durumunda.
            TypeError: Yanlış veri türü kullanımı durumunda.
        """

        #  **Tür Kontrolleri**
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"[ERROR] embed_dim hatalı: {embed_dim}. embed_dim pozitif bir tamsayı olmalıdır.")

        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"[ERROR] num_heads hatalı: {num_heads}. num_heads pozitif bir tamsayı olmalıdır.")

        if not isinstance(eps, float) or eps <= 0:
            raise ValueError(f"[ERROR] eps hatalı: {eps}. eps pozitif bir ondalık sayı olmalıdır.")

        if not isinstance(scale_factor, (int, float)) or scale_factor <= 0:
            raise ValueError(f"[ERROR] scale_factor hatalı: {scale_factor}. Pozitif bir sayı olmalıdır.")

        if not isinstance(verbose, bool):
            raise TypeError(f"[ERROR] verbose hatalı: {verbose}. Boolean (True/False) olmalıdır.")

        if num_groups is not None and (not isinstance(num_groups, int) or num_groups <= 0):
            raise ValueError(f"[ERROR] num_groups hatalı: {num_groups}. Pozitif bir tamsayı olmalıdır veya None olabilir.")

        #  **embed_dim ve num_heads Uyum Kontrolü**
        if embed_dim % num_heads != 0:
            raise ValueError(f"[ERROR] embed_dim ({embed_dim}) çok başlık sayısına ({num_heads}) tam bölünemiyor. "
                            "embed_dim, num_heads ile tam bölünebilecek bir değer olmalıdır.")

        #  **Geçerli Seçeneklerin Kontrolü**
        valid_attention_types = ["multi_head", "self", "cross"]
        if attention_type not in valid_attention_types:
            raise ValueError(f"[ERROR] attention_type hatalı: {attention_type}. Geçerli türler: {valid_attention_types}")

        valid_normalization_types = ["layer_norm", "batch_norm", "instance_norm", "group_norm"]
        if normalization_type not in valid_normalization_types:
            raise ValueError(f"[ERROR] normalization_type hatalı: {normalization_type}. Geçerli türler: {valid_normalization_types}")

        valid_scaling_strategies = ["sqrt", "linear", "none"]
        if scaling_strategy not in valid_scaling_strategies:
            raise ValueError(f"[ERROR] scaling_strategy hatalı: {scaling_strategy}. Geçerli stratejiler: {valid_scaling_strategies}")

        valid_scaling_methods = ["softmax", "sigmoid", "zscore"]
        if scaling_method not in valid_scaling_methods:
            raise ValueError(f"[ERROR] scaling_method hatalı: {scaling_method}. Geçerli yöntemler: {valid_scaling_methods}")

        #  **Dropout Kontrolü**
        if not isinstance(dropout, float) or not (0.0 <= dropout <= 1.0):
            raise ValueError(f"[ERROR] dropout hatalı: {dropout}. Dropout 0 ile 1 arasında bir float değer olmalıdır.")

        #  **clip_range Kontrolü**
        if clip_range is not None:
            if not (isinstance(clip_range, tuple) and len(clip_range) == 2 and all(isinstance(x, (int, float)) for x in clip_range)):
                raise ValueError(f"[ERROR] clip_range hatalı: {clip_range}. Geçerli bir (min, max) aralığı belirtilmelidir.")
            if clip_range[0] > clip_range[1]:
                raise ValueError(f"[ERROR] clip_range'in ilk değeri ({clip_range[0]}) ikinci değerinden ({clip_range[1]}) büyük olamaz.")

        #  **Tüm kontrolleri başarıyla geçtiyse işlem tamam!**
        return True

    def _initialize_module(self, module_cls, *args, **kwargs):
        """
        Modül başlatma için yardımcı metod.

        Args:
            module_cls (type): Başlatılacak modül sınıfı.
            *args: Modül için pozisyonel argümanlar.
            **kwargs: Modül için anahtar argümanlar.

        Returns:
            nn.Module: Başlatılmış modül.

        Raises:
            ValueError: Geçersiz giriş parametresi.
            RuntimeError: Modül başlatılamazsa.
        """
        #  **Geçerli bir modül sınıfı olup olmadığını kontrol et**
        if module_cls is None:
            raise ValueError("[ERROR] module_cls cannot be None.")

        if not isinstance(module_cls, type) or not issubclass(module_cls, nn.Module):
            raise TypeError(f"[ERROR] {module_cls} geçerli bir PyTorch modülü değil.")

        try:
            # ** Modülün __init__ metodunun aldığı argümanları belirle**
            param_names = module_cls.__init__.__code__.co_varnames[1:]

            # ** Çakışan argümanları kaldır**
            overlapping_params = [param for param in param_names[:len(args)] if param in kwargs]
            for param in overlapping_params:
                kwargs.pop(param)

            # ** Negatif veya sıfır değer kontrolü**
            if args and args[0] <= 0:
                raise ValueError(f"[ERROR] {module_cls.__name__} başlatılamadı: embed_dim negatif veya sıfır olamaz.")

            # ** Modülü başlat**
            module = module_cls(*args, **kwargs)

            # ** Doğrulama: Modül gerçekten başlatıldı mı?**
            if not isinstance(module, nn.Module):
                raise TypeError(f"[ERROR] {module_cls.__name__} başlatılamadı: Dönüş tipi nn.Module değil.")

            # ** Loglama (debug modu aktifse)**
            if self.debug:
                self.logger.debug(f"[INFO] {module_cls.__name__} başarıyla başlatıldı. Args: {args}, Kwargs: {kwargs}")

            return module

        except ZeroDivisionError:
            raise ValueError(f"[ERROR] {module_cls.__name__} başlatılamadı: num_heads değeri 0 olamaz.")

        except TypeError as e:
            raise TypeError(f"[ERROR] {module_cls.__name__} başlatılamadı: Yanlış türde argüman. Hata: {e}")

        except Exception as e:
            raise RuntimeError(f"[ERROR] {module_cls.__name__} başlatılamadı. Args: {args}, Kwargs: {kwargs}, Error: {e}")

    def initialize_attention(self, inputs):
        """
        Dikkat mekanizması için gerekli ilk değerleri hazırlar.

        Args:
            inputs (torch.Tensor): Girdi tensörü.

        Returns:
            torch.Tensor: Başlatılmış tensör.
        
        Raises:
            TypeError: Girdiler torch.Tensor değilse.
            ValueError: Girdi boyutları doğru değilse.
        """
        # Girdi doğrulama
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("[ERROR] 'inputs' must be a torch.Tensor.")
        if inputs.dim() < 2:
            raise ValueError("[ERROR] 'inputs' must have at least 2 dimensions.")

        # Loglama: İşlemin başladığını bildir
        if self.verbose:
            print("[INFO] Attention initialization started.")
            print(f"[DEBUG] Input shape: {inputs.shape}, dtype: {inputs.dtype}")

        # Başlatma işlemi
        try:
            initialized_inputs = self.initializer.initialize_weights(inputs)
        except Exception as e:
            # Hata durumunda loglama
            raise RuntimeError(f"[ERROR] Failed to initialize attention: {e}")

        # Loglama: İşlemin tamamlandığını bildir
        if self.verbose:
            print("[INFO] Attention initialization completed successfully.")
            print(f"[DEBUG] Initialized tensor shape: {initialized_inputs.shape}, dtype: {initialized_inputs.dtype}")

        return initialized_inputs

    def normalize_attention(self, inputs):
        """
        Girdi tensörünü normalizasyon işlemlerine tabi tutar.

        Args:
            inputs (torch.Tensor): Girdi tensörü.

        Returns:
            torch.Tensor: Normalizasyon sonrası tensör.

        Raises:
            TypeError: Eğer 'inputs' bir torch.Tensor değilse.
            ValueError: Eğer tensörün boyutları doğru değilse.
            RuntimeError: Normalizasyon işlemi başarısız olursa.
        """
        # Girdi doğrulama
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("[ERROR] 'inputs' must be a torch.Tensor.")
        if inputs.dim() < 2:
            raise ValueError("[ERROR] 'inputs' must have at least 2 dimensions. Received shape: {}".format(inputs.shape))

        # Loglama: İşlem başlangıcı
        if self.verbose:
            print("[INFO] Normalization process started.")
            print(f"[DEBUG] Input tensor shape: {inputs.shape}, dtype: {inputs.dtype}")

        # Normalizasyon işlemi
        try:
            normalized_inputs = self.normalizer.forward(inputs)
        except Exception as e:
            # Hata durumunda loglama
            raise RuntimeError(f"[ERROR] Failed to normalize inputs: {e}")

        # Loglama: İşlem tamamlandı
        if self.verbose:
            print("[INFO] Normalization process completed successfully.")
            print(f"[DEBUG] Normalized tensor shape: {normalized_inputs.shape}, dtype: {normalized_inputs.dtype}")

        return normalized_inputs

    def optimize_attention(self, outputs):
        """
        Dikkat mekanizmasının çıktısını optimize eder.

        Args:
            outputs (torch.Tensor): Dikkat çıktısı tensörü.

        Returns:
            torch.Tensor: Optimize edilmiş çıktı.

        Raises:
            TypeError: Eğer 'outputs' bir torch.Tensor değilse.
            ValueError: Eğer tensörün boyutları beklenenden farklıysa.
            RuntimeError: Optimize işlemi başarısız olursa.
        """
        # Girdi doğrulama
        if not isinstance(outputs, torch.Tensor):
            raise TypeError("[ERROR] 'outputs' must be a torch.Tensor.")
        if outputs.dim() < 2:
            raise ValueError(f"[ERROR] 'outputs' must have at least 2 dimensions. Received shape: {outputs.shape}")

        # Loglama: İşlemin başladığını bildir
        if self.verbose:
            print("[INFO] Optimization process started.")
            print(f"[DEBUG] Output tensor shape before optimization: {outputs.shape}, dtype: {outputs.dtype}")

        # Optimize işlemi
        try:
            optimized_outputs = self.attention_optimizer.forward(outputs)

            # Optimize edilen çıktının boyutlarını doğrula
            if optimized_outputs.shape != outputs.shape:
                raise ValueError(f"[ERROR] Optimized tensor shape mismatch. "
                                f"Expected: {outputs.shape}, Got: {optimized_outputs.shape}")

            # Loglama: İşlem tamamlandı
            if self.verbose:
                print("[INFO] Optimization completed successfully.")
                print(f"[DEBUG] Optimized tensor shape: {optimized_outputs.shape}, dtype: {optimized_outputs.dtype}")

            return optimized_outputs

        except Exception as e:
            # Hata durumunda loglama
            error_message = f"[ERROR] Failed to optimize attention outputs: {e}"
            if self.verbose:
                print(error_message)
            raise RuntimeError(error_message)

    def forward(self, query, key=None, value=None, mask=None):
        """
        Seçilen dikkat mekanizmasını uygular ve çıktı ile dikkat ağırlıklarını döndürür.
        Bu metod; giriş doğrulaması, ilgili attention modülünün çalıştırılması,
        (varsa) residual bağlantı eklenmesi, ardından attention_scaler, normalizer ve dropout
        aşamalarının uygulanmasını içerir.
        
        Args:
            query (torch.Tensor): Sorgu tensörü (beklenen şekil: [batch, seq_len, embed_dim]).
            key (torch.Tensor, optional): Anahtar tensörü (aynı boyutlarda).
            value (torch.Tensor, optional): Değer tensörü (aynı boyutlarda).
            mask (torch.Tensor, optional): Maske tensörü.
        
        Returns:
            tuple: (final_output, attn_weights) – İşlenmiş çıktı tensörü ve (varsa) dikkat ağırlıkları.
        
        Raises:
            ValueError, TypeError, RuntimeError: Giriş doğrulaması veya işlem sırasında oluşan hatalar.
        """
        import time
        t_start = time.time() if self.verbose else None

        try:
            # 1. Girdi doğrulaması
            self._validate_tensor(query, name='query', expected_dim=3)
            if key is not None:
                self._validate_tensor(key, name='key', expected_dim=3)
            if value is not None:
                self._validate_tensor(value, name='value', expected_dim=3)
            if mask is not None:
                self._validate_tensor(mask, name='mask')
            self.logger.debug(f"[FORWARD] Query doğrulaması başarılı: {query.shape}")
        except Exception as e:
            self.logger.error(f"[FORWARD] Girdi doğrulaması başarısız: {e}", exc_info=True)
            raise

        if self.verbose:
            t_input = time.time()

        try:
            # 2. Dikkat mekanizması seçimi
            if self.attention_type == "multi_head":
                self.logger.debug("[FORWARD] Multi-Head Attention uygulanıyor.")
                result = self._apply_multi_head_attention(query, key, value, mask)
            elif self.attention_type == "self":
                self.logger.debug("[FORWARD] Self-Attention uygulanıyor.")
                result = self._apply_self_attention(query, mask)
            elif self.attention_type == "cross":
                self.logger.debug("[FORWARD] Cross-Attention uygulanıyor.")
                result = self._apply_cross_attention(query, key, value, mask)
            else:
                raise ValueError(f"[ERROR] Geçersiz attention_type: {self.attention_type}")

            if self.verbose:
                t_attention = time.time()

            # 3. Çıktıların ayrılması: (attn_output ve (varsa) attn_weights)
            if isinstance(result, tuple):
                attn_output, attn_weights = result
            else:
                attn_output = result
                attn_weights = None

            # --- Ek Kontroller: Dikkat Ağırlıkları ---
            if attn_weights is not None:
                min_aw = attn_weights.min().item()
                max_aw = attn_weights.max().item()
                mean_aw = attn_weights.mean().item()
                std_aw = attn_weights.std().item()
                self.logger.debug(
                    f"[FORWARD] Dikkat Ağırlıkları -> shape: {attn_weights.shape}, "
                    f"min: {min_aw:.6f}, max: {max_aw:.6f}, mean: {mean_aw:.6f}, std: {std_aw:.6f}"
                )
                # Eğer tüm değerler çok düşük (ör. min değer neredeyse 0) veya standart sapma çok düşükse uyarı ver
                if std_aw < 1e-5:
                    self.logger.warning("[FORWARD] Dikkat ağırlıklarının dağılımı çok dar (std < 1e-5).")
                if max_aw < 1e-4:
                    self.logger.warning("[FORWARD] Dikkat ağırlıkları tümüyle çok düşük (max < 1e-4).")
            else:
                self.logger.warning("[FORWARD] Dikkat ağırlıkları None!")

            # 4. Residual bağlantı: Eğer attn_output ve query aynı boyutta ise
            if attn_output.shape == query.shape:
                combined_output = query + attn_output
                self.logger.debug("[FORWARD] Residual bağlantı eklendi: Query ile attention çıktısı toplandı.")
            else:
                combined_output = attn_output
                self.logger.debug("[FORWARD] Residual bağlantı uygulanmadı (boyut uyumsuzluğu).")

            # 5. Ölçeklendirme ve Normalizasyon
            t_norm_scale_start = time.time()
            try:
                scaled_output = self.attention_scaler(combined_output)
                self.logger.debug(f"[FORWARD] AttentionScaler sonrası çıktı: {scaled_output.shape}")
            except Exception as e:
                self.logger.error(f"[FORWARD] Hata: AttentionScaler işlemi başarısız: {e}", exc_info=True)
                raise

            try:
                normalized_output = self.normalizer(scaled_output)
                self.logger.debug(f"[FORWARD] Normalizer sonrası çıktı: {normalized_output.shape}")
            except Exception as e:
                self.logger.error(f"[FORWARD] Hata: Normalizer işlemi başarısız: {e}", exc_info=True)
                raise

            if self.verbose:
                t_norm_scale = time.time()

            # 6. Dropout uygulaması
            t_dropout_start = time.time()
            final_output = self.dropout_layer(normalized_output)
            self.logger.debug(f"[FORWARD] Dropout sonrası çıktı: {final_output.shape}")
            if self.verbose:
                t_dropout = time.time()

            # Ek kontrol: Son çıktının istatistiklerini kontrol edelim
            final_min = final_output.min().item()
            final_max = final_output.max().item()
            final_mean = final_output.mean().item()
            final_std = final_output.std().item()
            if final_std < 1e-3:
                self.logger.warning("[FORWARD] Son çıktı dağılımı çok sıkışık (std < 1e-3).")
            self.logger.debug(
                f"[FORWARD] Final Output Stats -> min: {final_min:.4f}, max: {final_max:.4f}, "
                f"mean: {final_mean:.4f}, std: {final_std:.4f}"
            )

            if self.verbose:
                t_end = time.time()
                self.logger.info(f"[FORWARD] {self.attention_type} attention ile forward pass başarıyla tamamlandı.")
                self.logger.debug(f"[FORWARD] Final çıktı şekli: {final_output.shape}, dtype: {final_output.dtype}")
                self.logger.debug(
                    f"[FORWARD] Zaman Ölçümleri: Input Validation: {(t_input - t_start):.4f}s, "
                    f"Attention Seçimi: {(t_attention - t_input):.4f}s, "
                    f"Norm & Scaling: {(t_norm_scale - t_norm_scale_start):.4f}s, "
                    f"Dropout: {(t_dropout - t_norm_scale):.4f}s, "
                    f"Total: {(t_end - t_start):.4f}s"
                )

            return final_output, attn_weights

        except ValueError as e:
            error_message = f"[VALUE ERROR] Forward pass, geçersiz giriş nedeniyle başarısız: {e}"
            if self.verbose:
                self.logger.error(error_message, exc_info=True)
            raise
        except TypeError as e:
            error_message = f"[TYPE ERROR] Forward pass, tip uyuşmazlığı nedeniyle başarısız: {e}"
            if self.verbose:
                self.logger.error(error_message, exc_info=True)
            raise
        except RuntimeError as e:
            error_message = f"[RUNTIME ERROR] Forward pass, çalışma zamanı hatası nedeniyle başarısız: {e}"
            if self.verbose:
                self.logger.error(error_message, exc_info=True)
            raise
        except Exception as e:
            error_message = f"[UNEXPECTED ERROR] Forward pass beklenmedik bir hata nedeniyle başarısız: {e}"
            if self.verbose:
                self.logger.error(error_message, exc_info=True)
            raise RuntimeError(error_message)




    def _apply_multi_head_attention(self, query, key, value, mask):
        """
        Multi-head attention işlemini uygular.
        """
        attention_output, attn_weights = self.multi_head_attention(query, key, value, mask, return_attention_weights=True)
        return attention_output, attn_weights

    def _apply_self_attention(self, query, mask):
        """
        Self-attention işlemini uygular.
        """
        attention_output, attn_weights = self.self_attention(query, mask=mask)
        return attention_output, attn_weights
    
    def _apply_cross_attention(self, query, key, value, mask):
        """
        Cross-attention işlemini uygular.
        """
        if mask is not None:
            self.logger.debug(f"Original mask shape: {mask.shape}")
            if mask.dim() == 4:
                mask = mask.squeeze(1).squeeze(1)
            elif mask.dim() == 3:
                mask = mask.squeeze(1)
            if mask.dim() != 2:
                raise ValueError("[ERROR] key_padding_mask must be 2D after squeezing.")
            self.logger.debug(f"Squeezed mask shape: {mask.shape}")
        
        attention_output, attn_weights = self.cross_attention(query, key, value, key_padding_mask=mask)
        return attention_output, attn_weights

    def _validate_tensor(self, tensor, name="tensor", expected_dim=None, expected_dtype=None, min_shape=None, required=True):
        """
        Bir tensörü doğrular ve hataları loglar.

        Args:
            tensor (torch.Tensor): Doğrulanacak tensör.
            name (str): Tensörün adı (loglama için).
            expected_dim (int, optional): Beklenen boyut sayısı.
            expected_dtype (torch.dtype, optional): Beklenen tensör veri türü.
            min_shape (tuple, optional): Tensörün minimum boyutları.
            required (bool): Tensörün zorunlu olup olmadığını belirtir.

        Raises:
            TypeError: Eğer tensör bir torch.Tensor değilse.
            ValueError: Eğer tensörün boyutları beklenenden farklıysa.
            ValueError: Eğer tensörün veri türü beklenenden farklıysa.
            ValueError: Eğer tensörün boyutları minimum boyutlardan küçükse.
        """
        # Girdi doğrulama
        if tensor is None:
            if required:
                raise ValueError(f"[ERROR] '{name}' is required but got None.")
            else:
                if self.verbose:
                    print(f"[INFO] '{name}' tensor is optional and not provided.")
                return

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"[ERROR] '{name}' must be a torch.Tensor.")

        # Beklenen boyut sayısı doğrulama
        if expected_dim is not None and tensor.dim() < expected_dim:
            raise ValueError(f"[ERROR] '{name}' must have at least {expected_dim} dimensions. Received shape: {tensor.shape}")

        # Beklenen veri türü doğrulama
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            raise ValueError(f"[ERROR] '{name}' must have dtype {expected_dtype}, but got {tensor.dtype}.")

        # Minimum boyutlar doğrulama
        if min_shape is not None:
            if any(tensor.size(i) < min_shape[i] for i in range(len(min_shape))):
                raise ValueError(f"[ERROR] '{name}' must have minimum shape {min_shape}. Received shape: {tensor.shape}")

        # Loglama: Başarılı doğrulama
        if self.verbose:
            print(f"[INFO] '{name}' tensor validation successful.")
            print(f"[DEBUG] {name} - Shape: {tensor.shape}, Dtype: {tensor.dtype}")

