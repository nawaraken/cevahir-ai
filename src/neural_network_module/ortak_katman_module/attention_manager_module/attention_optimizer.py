import torch
import torch.nn as nn
from enum import Enum

class ScalingMethod(Enum):
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
    ZSCORE = "zscore"
    SQRT = "sqrt"

class AttentionOptimizer:
    """
    Dikkat mekanizmaları için optimizasyon sınıfı.
    """

    def __init__(self, epsilon=1e-9, verbose=False, default_scaling_method=ScalingMethod.SOFTMAX, default_clipping_value=None):
        """
        AttentionOptimizer sınıfını başlatır.

        Args:
            epsilon (float): Sayısal kararlılık için küçük bir değer.
            verbose (bool): Hata ayıklama ve loglama için detaylı bilgi seçeneği.
            default_scaling_method (ScalingMethod veya str): Varsayılan dikkat normalizasyon yöntemi 
                                                            (ScalingMethod Enum veya string).
            default_clipping_value (float): Varsayılan maksimum değer (None veya pozitif bir sayı).
        """
        # Girdi doğrulama: epsilon
        if not isinstance(epsilon, (float, int)) or epsilon <= 0:
            raise ValueError("[ERROR] Epsilon değeri sıfırdan büyük bir float veya int olmalıdır.")
        
        # Girdi doğrulama: verbose
        if not isinstance(verbose, bool):
            raise TypeError("[ERROR] Verbose parametresi bir boolean olmalıdır.")
        
        # Girdi doğrulama: default_scaling_method
        if isinstance(default_scaling_method, str):
            try:
                default_scaling_method = ScalingMethod(default_scaling_method.lower())
            except ValueError:
                raise ValueError(
                    f"[ERROR] Bilinmeyen scaling method: '{default_scaling_method}'. "
                    f"Geçerli seçenekler: {[e.value for e in ScalingMethod]}"
                )
        elif not isinstance(default_scaling_method, ScalingMethod):
            raise TypeError(
                "[ERROR] default_scaling_method bir ScalingMethod Enum değeri veya geçerli bir string olmalıdır."
            )

        # Girdi doğrulama: default_clipping_value
        if default_clipping_value is not None:
            if not isinstance(default_clipping_value, (float, int)) or default_clipping_value < 0:
                raise ValueError(
                    "[ERROR] default_clipping_value sıfırdan büyük bir float/int olmalıdır veya None bırakılmalıdır."
                )

        # Varsayılan değer atamaları
        self.epsilon = float(epsilon)
        self.verbose = verbose
        self.default_scaling_method = default_scaling_method
        self.default_clipping_value = float(default_clipping_value) if default_clipping_value is not None else 1.0

        # Loglama: Başlangıç bilgileri (verbose modunda)
        if self.verbose:
            print("[INFO] AttentionOptimizer başlatıldı:")
            print(f"  - Epsilon: {self.epsilon}")
            print(f"  - Verbose modu: {self.verbose}")
            print(f"  - Varsayılan Ölçeklendirme Yöntemi: {self.default_scaling_method.value}")
            print(f"  - Varsayılan Clipping Değeri: {self.default_clipping_value}")

        # Ek loglama: Parametre doğrulama
        if self.verbose:
            print("[DEBUG] Girdi parametreleri doğrulandı ve sınıf başarıyla başlatıldı.")

        # Uyarılar: default_clipping_value atanmamışsa varsayılan değer hakkında bilgi
        if default_clipping_value is None and self.verbose:
            print("[WARNING] default_clipping_value belirtilmedi. Varsayılan değer: 1.0 kullanılıyor.")



    def __repr__(self):
        """
        Sınıfın özet bilgilerini döndürür.

        Returns:
            str: Sınıf özet metni.
        """
        return (
            f"AttentionOptimizer(epsilon={self.epsilon}, verbose={self.verbose}, "
            f"default_scaling_method={self.default_scaling_method.value}, "
            f"default_clipping_value={self.default_clipping_value})"
        )


    def log_tensor_info(self, tensor, name="Tensor", verbose_level=1):
        """
        Tensor ile ilgili detaylı bilgileri loglar.

        Args:
            tensor (torch.Tensor): İncelenecek tensör.
            name (str): Tensör adı.
            verbose_level (int): Loglama detay seviyesi (1: Özet, 2: Ayrıntılı, 3: Derinlemesine).
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"[ERROR] {name} bir PyTorch tensörü olmalıdır. Bulunan tür: {type(tensor)}")

        try:
            # Tensör genel bilgileri
            tensor_info = {
                "Shape": tensor.shape,
                "Dim": tensor.dim(),
                "NumElements": tensor.numel(),
                "Min": tensor.min().item() if tensor.numel() > 0 else None,
                "Max": tensor.max().item() if tensor.numel() > 0 else None,
                "Mean": tensor.mean().item() if tensor.numel() > 0 else None,
                "Std": tensor.std().item() if tensor.numel() > 0 else None,
            }

            # Büyük tensörler için uyarı
            if tensor_info["NumElements"] > 1e6:
                tensor_info["Warning"] = (
                    f"Tensör çok büyük ({tensor_info['NumElements']} eleman). Performans kaybını önlemek için "
                    "daha küçük tensörlerle çalışmayı düşünebilirsiniz."
                )

            # Detaylı loglama (verbose_level 2 ve üzeri için)
            if verbose_level >= 2:
                tensor_info["FirstFewElements"] = tensor.flatten()[:10].tolist()
                tensor_info["HasNaN"] = torch.isnan(tensor).any().item()
                tensor_info["HasInf"] = torch.isinf(tensor).any().item()

                if tensor_info["HasNaN"]:
                    nan_indices = torch.nonzero(torch.isnan(tensor), as_tuple=True)
                    tensor_info["NaNIndices"] = nan_indices

                if tensor_info["HasInf"]:
                    inf_indices = torch.nonzero(torch.isinf(tensor), as_tuple=True)
                    tensor_info["InfIndices"] = inf_indices

            # Çok ayrıntılı loglama (verbose_level 3 için)
            if verbose_level >= 3:
                tensor_info["Histogram"] = torch.histc(tensor, bins=10).tolist()

            # Loglama çıktısı
            if self.verbose:
                print(f"[INFO] {name} Bilgileri:")
                for key, value in tensor_info.items():
                    print(f"  {key}: {value}")

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] {name} tensör bilgileri loglanırken hata oluştu: {e}")

        finally:
            # NaN ve sonsuz değerlerin temizlenmesine yönelik uyarılar
            if torch.isnan(tensor).any():
                print(f"[WARNING] {name} tensöründe NaN değerler bulundu.")
            if torch.isinf(tensor).any():
                print(f"[WARNING] {name} tensöründe sonsuz değerler bulundu.")
   
    def normalize_attention(self, attention_scores, method=None):
        """
        Dikkat tensörlerini normalize eder.

        Args:
            attention_scores (torch.Tensor): Dikkat çıktısı tensörü (boyut: batch_size, num_heads, seq_len, seq_len).
            method (str veya ScalingMethod, optional): Normalizasyon yöntemi ('softmax', 'sigmoid', 'zscore').
                                                    Varsayılan olarak self.default_scaling_method kullanılır.

        Returns:
            torch.Tensor: Normalize edilmiş dikkat tensörleri.
        """
        # Giriş doğrulama
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError("[ERROR] 'attention_scores' bir PyTorch tensörü olmalıdır.")
        
        # NaN ve sonsuz değer kontrolü
        if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
            if self.verbose:
                print("[WARNING] Attention scores içinde NaN veya sonsuz değerler bulundu ve temizleniyor.")
            attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e9, neginf=-1e9)
        
        # Normalizasyon yöntemi belirleme
        if method is None:
            method = self.default_scaling_method
        elif isinstance(method, str):
            try:
                method = ScalingMethod(method.lower())
            except ValueError:
                raise ValueError(
                    f"[ERROR] Geçersiz normalizasyon yöntemi: '{method}'. "
                    f"Geçerli yöntemler: {[e.value for e in ScalingMethod]}"
                )
        elif not isinstance(method, ScalingMethod):
            raise TypeError("[ERROR] 'method', ScalingMethod Enum veya geçerli bir string olmalıdır.")
        
        # Normalizasyon işlemi
        try:
            if method == ScalingMethod.SOFTMAX:
                normalized_scores = torch.softmax(attention_scores, dim=-1)
            elif method == ScalingMethod.SIGMOID:
                normalized_scores = torch.sigmoid(attention_scores)
            elif method == ScalingMethod.ZSCORE:
                mean = attention_scores.mean(dim=-1, keepdim=True)
                std = attention_scores.std(dim=-1, keepdim=True)
                zero_std_indices = (std == 0)

                if zero_std_indices.any():
                    if self.verbose:
                        print(
                            f"[WARNING] Z-Score yöntemi kullanılırken sıfır standart sapma tespit edildi. "
                            f"Tespit edilen eleman sayısı: {zero_std_indices.sum().item()}. "
                            "Bu değerler için alternatif bir normalizasyon uygulanıyor."
                        )
                    std[zero_std_indices] = self.epsilon  # Sıfır sapma değerlerine epsilon atanıyor.
                
                normalized_scores = (attention_scores - mean) / (std + self.epsilon)
            else:
                raise ValueError(
                    f"[ERROR] Desteklenmeyen normalizasyon yöntemi: '{method}'. "
                    f"Geçerli yöntemler: {[e.value for e in ScalingMethod]}"
                )
        except Exception as e:
            raise RuntimeError(f"[ERROR] Normalizasyon işlemi sırasında hata oluştu: {e}")
        
        # Çıktı doğrulama ve loglama
        if self.verbose:
            if torch.isnan(normalized_scores).any():
                print("[ERROR] Normalizasyon sonrası NaN değerler bulundu.")
            if torch.isinf(normalized_scores).any():
                print("[ERROR] Normalizasyon sonrası sonsuz değerler bulundu.")
            if not torch.isnan(normalized_scores).any() and not torch.isinf(normalized_scores).any():
                print("[INFO] Normalizasyon başarılı bir şekilde tamamlandı.")

        # Tensör bilgilerini loglama
        self.log_tensor_info(normalized_scores, f"Normalized Attention ({method.value})")
        
        return normalized_scores

    def mask_attention(self, attention_scores, attention_mask=None, mask_type="default"):
        """
        Dikkat tensörlerine maske uygular.

        Args:
            attention_scores (torch.Tensor): Dikkat çıktısı tensörü (boyut: batch_size, num_heads, seq_len, seq_len).
            attention_mask (torch.Tensor, optional): Maske tensörü. Varsayılan olarak None.
            mask_type (str, optional): Maske türü ('default', 'causal'). Varsayılan 'default'.

        Returns:
            torch.Tensor: Maskelenmiş dikkat tensörleri.
        """
        # Giriş doğrulama
        MaskingHelper.validate_attention_inputs(attention_scores, attention_mask, mask_type)

        if mask_type == "default":
            if self.verbose:
                print("[INFO] Varsayılan maske uygulanıyor...")
            attention_scores = MaskingHelper.apply_default_mask(attention_scores, attention_mask, verbose=self.verbose)

        elif mask_type == "causal":
            if self.verbose:
                print("[INFO] Causal maske uygulanıyor...")
            attention_scores = MaskingHelper.apply_causal_mask(attention_scores, verbose=self.verbose)

        # NaN ve sonsuz değerlerin temizlenmesi
        attention_scores = MaskingHelper.clean_tensor(attention_scores, verbose=self.verbose)

        # Çıktı doğrulama ve loglama
        if self.verbose:
            self.log_tensor_info(attention_scores, f"Masked Attention ({mask_type})")

        return attention_scores


    def scale_attention(self, attention_scores, scaling_factor=None, adaptive=False, embed_dim=None, num_heads=None):
        """
        Dikkat tensörlerini ölçeklendirir.

        Args:
            attention_scores (torch.Tensor): Dikkat çıktısı tensörü (boyut: batch_size, num_heads, seq_len, seq_len).
            scaling_factor (float, optional): Ölçeklendirme için kullanılan çarpan.
            adaptive (bool, optional): Ölçeklendirme faktörünün dinamik hesaplanıp hesaplanmayacağını belirler. Varsayılan False.
            embed_dim (int, optional): Gömme boyutu. Dinamik ölçeklendirme için gereklidir.
            num_heads (int, optional): Çok başlıklı dikkat mekanizmasındaki başlık sayısı. Dinamik ölçeklendirme için gereklidir.

        Returns:
            torch.Tensor: Ölçeklendirilmiş dikkat tensörleri.
        """
        # Giriş doğrulama
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError("[ERROR] 'attention_scores' bir PyTorch tensörü olmalıdır.")

        # Ölçeklendirme faktörünü kontrol et
        if adaptive:
            # Adaptif ölçeklendirme için gerekli parametrelerin doğrulanması
            if embed_dim is None or num_heads is None:
                raise ValueError("[ERROR] Adaptive scaling için 'embed_dim' ve 'num_heads' sağlanmalıdır.")
            try:
                # Dinamik ölçeklendirme faktörünün hesaplanması
                scaling_factor = torch.sqrt(torch.tensor(embed_dim / num_heads, dtype=torch.float32))
                if self.verbose:
                    print(f"[INFO] Adaptive scaling kullanılıyor. Hesaplanan scaling_factor: {scaling_factor:.4f}")
            except Exception as e:
                raise RuntimeError(f"[ERROR] Adaptive scaling factor hesaplanırken hata oluştu: {e}")
        elif scaling_factor is None or scaling_factor <= 0:
            raise ValueError("[ERROR] Scaling factor pozitif bir değer olmalıdır.")

        # Ölçeklendirme işlemi
        try:
            attention_scores = attention_scores / scaling_factor
            if self.verbose:
                print(f"[INFO] Attention scores, scaling_factor={scaling_factor:.4f} ile ölçeklendirildi.")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Attention scores ölçeklendirilirken hata oluştu: {e}")

        # NaN ve sonsuz değer kontrolü
        if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
            if self.verbose:
                print("[WARNING] Ölçeklendirme sonrası tensörde NaN veya sonsuz değerler bulundu. Temizleniyor...")
            attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e9, neginf=-1e9)
            if self.verbose:
                print("[INFO] NaN ve sonsuz değerler temizlendi.")

        # Çıktı doğrulama ve loglama
        if self.verbose:
            if torch.isnan(attention_scores).any():
                print("[ERROR] Ölçeklendirme sonrası NaN değerler bulundu.")
            if torch.isinf(attention_scores).any():
                print("[ERROR] Ölçeklendirme sonrası sonsuz değerler bulundu.")
            self.log_tensor_info(attention_scores, "Scaled Attention")

        return attention_scores



    def clip_attention(self, attention_scores, clip_value=None):
        """
        Dikkat tensörlerini belirli bir maksimum değere sınırlar (clipping).

        Args:
            attention_scores (torch.Tensor): Dikkat çıktısı tensörü (boyut: batch_size, num_heads, seq_len, seq_len).
            clip_value (float, optional): Maksimum değer. Varsayılan olarak default_clipping_value kullanılır.

        Returns:
            torch.Tensor: Clip edilmiş dikkat tensörleri.
        """
        # Giriş doğrulama
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError("[ERROR] 'attention_scores' bir PyTorch tensörü olmalıdır.")

        # clip_value kontrolü ve varsayılan değer atanması
        if clip_value is None:
            clip_value = self.default_clipping_value
            if self.verbose:
                print(f"[INFO] clip_value belirtilmedi. Varsayılan değer ({clip_value}) kullanılacak.")

        # clip_value geçerlilik kontrolü
        if not isinstance(clip_value, (float, int)) or clip_value <= 0:
            raise ValueError("[ERROR] 'clip_value' pozitif bir sayı olmalıdır.")

        # Clipping işlemi
        try:
            attention_scores = torch.clamp(attention_scores, min=-clip_value, max=clip_value)
            if self.verbose:
                print(f"[INFO] Dikkat tensörleri clip_value={clip_value} ile sınırlandırıldı.")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Clipping işlemi sırasında hata oluştu: {e}")

        # NaN ve sonsuz değer kontrolü
        if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
            if self.verbose:
                print("[WARNING] Clipping sonrası tensörde NaN veya sonsuz değerler bulundu. Temizleniyor...")
            attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=clip_value, neginf=-clip_value)
            if self.verbose:
                print("[INFO] NaN ve sonsuz değerler temizlendi.")

        # Çıktı doğrulama ve loglama
        if self.verbose:
            self.log_tensor_info(attention_scores, "Clipped Attention")
            if torch.isnan(attention_scores).any():
                print("[ERROR] Clip işlemi sonrası NaN değerler bulundu.")
            if torch.isinf(attention_scores).any():
                print("[ERROR] Clip işlemi sonrası sonsuz değerler bulundu.")

        return attention_scores

    def optimize(self, attention_scores, attention_mask=None, scaling_factor=None, clip_value=None, 
                        normalize_method=None, mask_type="default"):
        """
        Dikkat tensörlerini optimize eder.

        Args:
            attention_scores (torch.Tensor): Dikkat çıktısı tensörü
                                            (boyut: batch_size, num_heads, seq_len, seq_len).
            attention_mask (torch.Tensor, optional): Maske tensörü.
            scaling_factor (float, optional): Ölçeklendirme için kullanılan çarpan.
            clip_value (float, optional): Dikkat değerlerini sınırlandırmak için maksimum değer.
            normalize_method (str, optional): Normalizasyon yöntemi ('softmax', 'sigmoid', 'zscore').
            mask_type (str, optional): Maske türü ('default', 'causal').

        Returns:
            torch.Tensor: Optimize edilmiş dikkat tensörleri.
        """
        # 1. Giriş tensörlerini doğrula
        if not self.validate_attention_scores(attention_scores):
            raise ValueError("[ERROR] Attention scores contain NaN or infinity values before optimization.")

        if self.verbose:
            print("[INFO] Başlangıç tensörü doğrulandı. Optimization başlıyor.")

        # 2. Maske Uygulama
        if attention_mask is not None:
            try:
                attention_scores = self.mask_attention(attention_scores, attention_mask, mask_type)
                if self.verbose:
                    print(f"[INFO] {mask_type} maskesi dikkat tensörlerine başarıyla uygulandı.")
            except Exception as e:
                raise RuntimeError(f"[ERROR] Mask uygulaması sırasında hata oluştu: {e}")

        # 3. Ölçeklendirme
        if scaling_factor is not None:
            try:
                attention_scores = self.scale_attention(attention_scores, scaling_factor)
                if self.verbose:
                    print(f"[INFO] Dikkat tensörleri {scaling_factor} ölçek faktörü ile ölçeklendirildi.")
            except Exception as e:
                raise RuntimeError(f"[ERROR] Ölçeklendirme sırasında hata oluştu: {e}")

        # 4. Clipping
        if clip_value is not None:
            try:
                attention_scores = self.clip_attention(attention_scores, clip_value)
                if self.verbose:
                    print(f"[INFO] Dikkat tensörleri {clip_value} clip değeri ile sınırlandırıldı.")
            except Exception as e:
                raise RuntimeError(f"[ERROR] Clipping işlemi sırasında hata oluştu: {e}")

        # 5. Normalize Etme
        try:
            optimized_attention = self.normalize_attention(attention_scores, method=normalize_method)
            if self.verbose:
                print(f"[INFO] Dikkat tensörleri {normalize_method or self.default_scaling_method} ile normalize edildi.")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Normalizasyon işlemi sırasında hata oluştu: {e}")

        # 6. Optimize edilen tensörü doğrula
        if not self.validate_attention_scores(optimized_attention):
            raise ValueError("[ERROR] Optimize edilen dikkat tensörleri NaN veya sonsuz değer içeriyor.")

        if self.verbose:
            print("[INFO] Optimize edilen tensörler doğrulandı ve geçerli.")

        # 7. Loglama (Opsiyonel)
        if self.verbose:
            self.log_tensor_info(optimized_attention, "Optimized Attention")

        return optimized_attention
    def forward(self, attention_scores, attention_mask=None, scaling_factor=None, 
                clip_value=None, normalize_method=None, mask_type="default"):
        """
        Dikkat tensörlerini optimize eden ana `forward` metodu.
        `optimize` metodunu çağırarak işlemi gerçekleştirir.

        Args:
            attention_scores (torch.Tensor): Dikkat çıktısı tensörü
            attention_mask (torch.Tensor, optional): Maske tensörü
            scaling_factor (float, optional): Ölçeklendirme çarpanı
            clip_value (float, optional): Clipleme değeri
            normalize_method (str, optional): Normalizasyon metodu ('softmax', 'sigmoid', 'zscore')
            mask_type (str, optional): Maske türü ('default', 'causal')

        Returns:
            torch.Tensor: Optimize edilmiş dikkat tensörü
        """
        return self.optimize(
            attention_scores, attention_mask, scaling_factor, clip_value, normalize_method, mask_type
        )

    def check_for_nan(self, attention_scores, replace_with_zero=True):
        """
        Dikkat tensöründe NaN değerlerini kontrol eder ve isteğe bağlı olarak temizler.

        Args:
            attention_scores (torch.Tensor): Kontrol edilecek dikkat tensörü.
            replace_with_zero (bool, optional): NaN değerlerini sıfır ile değiştirme seçeneği. Varsayılan True.

        Returns:
            tuple: (bool, torch.Tensor) - NaN değerler varsa True, aksi halde False. Temizlenmiş tensör döner.
        """
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError("[ERROR] Attention scores must be a PyTorch tensor.")

        nan_exists = torch.isnan(attention_scores).any()

        if nan_exists:
            nan_indices = torch.nonzero(torch.isnan(attention_scores), as_tuple=True)
            if self.verbose:
                print("[WARNING] Attention tensor contains NaN values.")
                print(f"[DETAILS] NaN found at indices: {nan_indices}")

            if replace_with_zero:
                attention_scores = torch.nan_to_num(attention_scores, nan=0.0)
                if self.verbose:
                    print("[INFO] NaN values replaced with 0.")

        return nan_exists.item(), attention_scores

    def check_for_inf(self, attention_scores, replace_with_max=True, clip_value=1e9):
        """
        Dikkat tensöründe sonsuz değerlerini kontrol eder ve isteğe bağlı olarak temizler.

        Args:
            attention_scores (torch.Tensor): Kontrol edilecek dikkat tensörü.
            replace_with_max (bool, optional): Sonsuz değerlerini clip_value ile değiştirme seçeneği. Varsayılan True.
            clip_value (float, optional): Sonsuz değerler için kullanılacak maksimum değer. Varsayılan 1e9.

        Returns:
            tuple: (bool, torch.Tensor) - Sonsuz değerler varsa True, aksi halde False. Temizlenmiş tensör döner.
        """
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError("[ERROR] Attention scores must be a PyTorch tensor.")

        inf_exists = torch.isinf(attention_scores).any()

        if inf_exists:
            inf_indices = torch.nonzero(torch.isinf(attention_scores), as_tuple=True)
            if self.verbose:
                print("[WARNING] Attention tensor contains infinity values.")
                print(f"[DETAILS] Infinity found at indices: {inf_indices}")

            if replace_with_max:
                attention_scores = torch.nan_to_num(attention_scores, posinf=clip_value, neginf=-clip_value)
                if self.verbose:
                    print(f"[INFO] Infinity values replaced with ±{clip_value}.")

        return inf_exists.item(), attention_scores


    def validate_attention_scores(self, attention_scores):
        """
        Dikkat tensörlerini doğrular ve sorun olup olmadığını kontrol eder.

        Args:
            attention_scores (torch.Tensor): Kontrol edilecek dikkat tensörü.

        Returns:
            bool: Geçerli ise True, aksi halde False.
        """
        try:
            # Giriş türü doğrulama
            if not isinstance(attention_scores, torch.Tensor):
                raise TypeError("[ERROR] Attention scores must be a PyTorch tensor.")

            # Şekil doğrulama
            if attention_scores.dim() not in [3, 4]:
                raise ValueError(
                    f"[ERROR] Invalid tensor dimensions: {attention_scores.dim()}. Expected: 3D or 4D tensor."
                )

            # NaN kontrolü
            if torch.isnan(attention_scores).any():
                nan_indices = torch.nonzero(torch.isnan(attention_scores), as_tuple=True)
                raise ValueError(f"[ERROR] Attention scores contain NaN values at indices: {nan_indices}")

            # Sonsuzluk kontrolü
            if torch.isinf(attention_scores).any():
                inf_indices = torch.nonzero(torch.isinf(attention_scores), as_tuple=True)
                raise ValueError(f"[ERROR] Attention scores contain infinite values at indices: {inf_indices}")

            # Tensör değer aralığı kontrolü
            min_val, max_val = attention_scores.min().item(), attention_scores.max().item()
            if self.verbose:
                print(f"[INFO] Tensor value range: Min: {min_val}, Max: {max_val}")

            # Büyük tensörlerde örnekleme ile kontrol
            if attention_scores.numel() > 1e6:
                sampled_indices = torch.randint(0, attention_scores.numel(), (1000,))
                sampled_values = attention_scores.view(-1)[sampled_indices]
                if torch.isnan(sampled_values).any():
                    raise ValueError("[ERROR] Sampled values contain NaN values.")
                if torch.isinf(sampled_values).any():
                    raise ValueError("[ERROR] Sampled values contain infinite values.")
                if self.verbose:
                    print("[INFO] Large tensor validated using sampling.")

            # Tüm kontroller başarılıysa True döndür
            return True

        except (TypeError, ValueError) as e:
            if self.verbose:
                print(f"{e}")
            return False


    def extra_repr(self):
        """
        Sınıfın özet bilgilerini döndürür.

        Returns:
            str: Sınıf özet metni.
        """
        summary = (
            f"epsilon={self.epsilon}, "
            f"verbose={self.verbose}, "
            f"default_scaling_method={self.default_scaling_method}, "
            f"default_clipping_value={self.default_clipping_value}, "
            f"supported_methods=['softmax', 'sigmoid', 'zscore']"
        )
        return summary


class MaskingHelper:
    """
    Dikkat tensörlerine maske uygulama işlemlerini yöneten ileri seviye yardımcı sınıf.
    """


    @staticmethod
    def validate_attention_inputs(attention_scores, attention_mask, mask_type):
        """
        Dikkat tensörlerine maske uygulanmadan önce giriş doğrulama işlemini yapar.

        Args:
            attention_scores (torch.Tensor): Dikkat tensörleri.
            attention_mask (torch.Tensor veya None): Maske tensörü.
            mask_type (str): Maske türü ('default' veya 'causal').

        Raises:
            ValueError: Eğer giriş verilerinde uyuşmazlık veya hata varsa.
        """
        # attention_scores doğrulama
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError("[ERROR] 'attention_scores' bir PyTorch tensörü olmalıdır.")
        if attention_scores.dim() < 2:
            raise ValueError("[ERROR] 'attention_scores' en az 2 boyutlu olmalıdır.")

        # mask_type doğrulama
        if mask_type not in ["default", "causal"]:
            raise ValueError(f"[ERROR] Geçersiz maske türü: {mask_type}. Geçerli türler: ['default', 'causal']")

        # attention_mask doğrulama (sadece 'default' için gereklidir)
        if mask_type == "default":
            if attention_mask is None:
                raise ValueError("[ERROR] 'default' maske türü için 'attention_mask' gereklidir.")
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError("[ERROR] 'attention_mask' bir PyTorch tensörü olmalıdır.")
            if attention_scores.shape != attention_mask.shape:
                raise ValueError(
                    f"[ERROR] 'attention_scores' ve 'attention_mask' boyutları uyumsuz: "
                    f"{attention_scores.shape} ve {attention_mask.shape}"
                )

        print("[INFO] Giriş doğrulama başarılı.")


    @staticmethod
    def validate_attention_scores(attention_scores):
        """
        Dikkat tensörünü doğrular ve uyumluluğunu kontrol eder.

        Args:
            attention_scores (torch.Tensor): Doğrulama yapılacak tensör.

        Raises:
            TypeError: Eğer attention_scores bir tensör değilse.
            ValueError: Eğer attention_scores NaN veya sonsuz değerler içeriyorsa.
        """
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError("[ERROR] 'attention_scores' bir PyTorch tensörü olmalıdır.")
        if attention_scores.dim() < 2:
            raise ValueError("[ERROR] 'attention_scores' en az 2 boyutlu olmalıdır.")
        if torch.isnan(attention_scores).any():
            raise ValueError("[ERROR] 'attention_scores' NaN değerler içeriyor.")
        if torch.isinf(attention_scores).any():
            raise ValueError("[ERROR] 'attention_scores' sonsuz değerler içeriyor.")
        print("[INFO] 'attention_scores' doğrulandı ve uyumlu.")

    @staticmethod
    def create_causal_mask(seq_len, device):
        """
        Causal maskeyi oluşturur.

        Args:
            seq_len (int): Sekans uzunluğu.
            device (torch.device): Cihaz bilgisi (CPU veya GPU).

        Returns:
            torch.Tensor: Causal mask tensörü.
        """
        if not isinstance(seq_len, int) or seq_len <= 0:
            raise ValueError("[ERROR] 'seq_len' pozitif bir tamsayı olmalıdır.")
        if not isinstance(device, torch.device):
            raise TypeError("[ERROR] 'device' bir torch.device olmalıdır.")

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        print(f"[INFO] Causal mask başarıyla oluşturuldu. Boyut: {causal_mask.shape}")
        return causal_mask

    @staticmethod
    def apply_default_mask(attention_scores, attention_mask, verbose=False):
        """
        Varsayılan maskeyi uygular.

        Args:
            attention_scores (torch.Tensor): Dikkat tensörleri.
            attention_mask (torch.Tensor): Maske tensörü.
            verbose (bool, optional): Hata ayıklama ve bilgi loglama seçeneği.

        Returns:
            torch.Tensor: Maskelenmiş dikkat tensörleri.
        """
        if attention_mask is None:
            raise ValueError("[ERROR] Varsayılan maskeyi uygulamak için 'attention_mask' gereklidir.")
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("[ERROR] 'attention_mask' bir PyTorch tensörü olmalıdır.")
        if attention_scores.shape != attention_mask.shape:
            raise ValueError(
                f"[ERROR] 'attention_scores' ve 'attention_mask' boyutları uyumsuz: "
                f"{attention_scores.shape} ve {attention_mask.shape}"
            )

        masked_scores = attention_scores.masked_fill(~attention_mask, float("-inf"))

        if verbose:
            print("[INFO] Varsayılan maske başarıyla uygulandı.")
            print(f"[DETAILS] Mask applied to attention_scores with shape: {attention_scores.shape}")

        return masked_scores

    @staticmethod
    def apply_causal_mask(attention_scores, verbose=False):
        """
        Causal maskeyi uygular.

        Args:
            attention_scores (torch.Tensor): Dikkat tensörleri.
            verbose (bool, optional): Hata ayıklama ve bilgi loglama seçeneği.

        Returns:
            torch.Tensor: Maskelenmiş dikkat tensörleri.
        """
        seq_len = attention_scores.size(-1)
        causal_mask = MaskingHelper.create_causal_mask(seq_len, attention_scores.device)
        masked_scores = attention_scores.masked_fill(causal_mask, float("-inf"))

        if verbose:
            print("[INFO] Causal maske başarıyla uygulandı.")
            print(f"[DETAILS] Mask applied to attention_scores with shape: {attention_scores.shape}")

        return masked_scores

    @staticmethod
    def clean_tensor(tensor, nan_value=0.0, posinf_value=1e6, neginf_value=-1e6, verbose=False):
        """
        Tensördeki NaN ve sonsuz değerleri temizler.

        Args:
            tensor (torch.Tensor): Temizleme yapılacak tensör.
            nan_value (float, optional): NaN değerler için kullanılacak değer. Varsayılan 0.0.
            posinf_value (float, optional): Pozitif sonsuz değerler için kullanılacak değer. Varsayılan 1e6.
            neginf_value (float, optional): Negatif sonsuz değerler için kullanılacak değer. Varsayılan -1e6.
            verbose (bool, optional): Hata ayıklama ve bilgi loglama seçeneği.

        Returns:
            torch.Tensor: Temizlenmiş tensör.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("[ERROR] Temizlenecek veri bir PyTorch tensörü olmalıdır.")

        cleaned_tensor = torch.nan_to_num(tensor, nan=nan_value, posinf=posinf_value, neginf=neginf_value)

        if verbose:
            print("[INFO] Tensor temizleme işlemi tamamlandı.")
            print(f"[DETAILS] Temizlenen tensor min: {cleaned_tensor.min()}, max: {cleaned_tensor.max()}")

        return cleaned_tensor

    @staticmethod
    def log_tensor_details(tensor, label="Tensor"):
        """
        Tensörün detaylarını loglar.

        Args:
            tensor (torch.Tensor): Detayları loglanacak tensör.
            label (str, optional): Tensör için etiket. Varsayılan 'Tensor'.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("[ERROR] Loglanacak veri bir PyTorch tensörü olmalıdır.")

        print(f"[INFO] {label} Detayları:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Min: {tensor.min().item()}")
        print(f"  Max: {tensor.max().item()}")
        print(f"  Mean: {tensor.mean().item()}")
        print(f"  Std: {tensor.std().item()}")
