import torch
import torch.nn as nn
from training_management.training_logger import TrainingLogger
logger = TrainingLogger()


class AttentionScaler(nn.Module):
    """
    Dikkat çıktılarının ölçeklenmesi ve düzenlenmesi için bir sınıf.
    """

    def __init__(self, scale_factor=1.0, clip_range=None, verbose=False, num_heads=None):
        """
        AttentionScaler sınıfını başlatır.

        Args:
            scale_factor (float): Dikkat çıktısını ölçeklemek için kullanılan çarpan.
                                Pozitif bir sayı olmalıdır.
            clip_range (tuple[float, float] | None, optional): Ölçeklenmiş değerleri kırpmak için
                                                            alt ve üst limit (min, max).
                                                            None ise kırpma yapılmaz.
            verbose (bool): Detaylı loglama seçeneği. Varsayılan olarak False.
            num_heads (int, optional): Başlık sayısı (4D dönüşüm için gereklidir).
                                    3D tensörler üzerinde işlem yaparken belirtilmelidir.

        Raises:
            ValueError: scale_factor pozitif bir sayı değilse.
            ValueError: clip_range, geçerli bir çift değer (min, max) içermiyorsa.
            ValueError: num_heads pozitif bir tam sayı değilse veya None değilse.
        """
        super(AttentionScaler, self).__init__()
        # Global logger'ı kullanarak örnek seviyesinde logger oluştur
        self.logger = logger  # Global logger'ın tanımlı olduğu varsayılıyor
        self.logger.debug("MultiHeadAttention __init__ çağrıldı.")
        # Ölçekleme faktörünü doğrula
        if not isinstance(scale_factor, (int, float)) or scale_factor <= 0:
            raise ValueError(f"'scale_factor' pozitif bir sayı olmalıdır. Bulunan: {scale_factor}")
        self.scale_factor = float(scale_factor)

        # Kırpma aralığını doğrula
        if clip_range is not None:
            if not isinstance(clip_range, tuple) or len(clip_range) != 2:
                raise ValueError(
                    f"'clip_range' bir çift (min, max) değer içeren tuple olmalıdır. "
                    f"Bulunan: {clip_range}"
                )
            min_val, max_val = clip_range
            if not (isinstance(min_val, (int, float)) and isinstance(max_val, (int, float))):
                raise ValueError(
                    f"'clip_range' içindeki değerler sayısal olmalıdır. Bulunan: {clip_range}"
                )
            if min_val >= max_val:
                raise ValueError(
                    f"'clip_range' içindeki min ({min_val}) değeri, max ({max_val}) değerinden küçük olmalıdır."
                )
        self.clip_range = clip_range

        # Detaylı loglama kontrolü
        if not isinstance(verbose, bool):
            raise TypeError(f"'verbose' bir boolean değeri olmalıdır. Bulunan: {verbose}")
        self.verbose = verbose

        # num_heads kontrolü
        if num_heads is not None:
            if not isinstance(num_heads, int) or num_heads <= 0:
                raise ValueError(
                    f"'num_heads', pozitif bir tam sayı olmalıdır. Bulunan: {num_heads}"
                )
        self.num_heads = num_heads

        # Loglama
        if self.verbose:
            print(f"[AttentionScaler] Başarılı şekilde başlatıldı:")
            print(f"  - scale_factor: {self.scale_factor}")
            print(f"  - clip_range: {self.clip_range}")
            print(f"  - verbose: {self.verbose}")
            print(f"  - num_heads: {self.num_heads}")


    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Dikkat tensörlerini ölçekler ve isteğe bağlı olarak yeniden normalize eder.
        
        Args:
            attention_scores (torch.Tensor): Dikkat mekanizması çıktısı 
                (2D: [seq_len, embed_dim] veya
                3D: [batch_size, seq_len, embed_dim] veya
                4D: [batch_size, num_heads, seq_len, head_dim]).
                
        Returns:
            torch.Tensor: Ölçeklenmiş (ve gerekiyorsa yeniden normalize edilmiş) dikkat çıktıları.
            
        Raises:
            Exception: Herhangi bir adımda oluşan hata ayrıntılarıyla.
        """
        import time
        start_time = time.time()
        original_ndim = attention_scores.ndim

        # Adım 1: Girdi doğrulaması
        try:
            self.validate_tensor(attention_scores)
            if self.verbose:
                self.logger.debug(f"[AttentionScaler] Original input shape: {attention_scores.shape} (ndim: {original_ndim})")
        except Exception as e:
            self.logger.error(f"[AttentionScaler] Input validation failed: {e}", exc_info=True)
            raise

        # Adım 2: Gerekirse 2D tensörü 3D'ye dönüştür
        try:
            if original_ndim == 2:
                attention_scores = attention_scores.unsqueeze(0)
                if self.verbose:
                    self.logger.debug(f"[AttentionScaler] Converted 2D tensor to 3D: {attention_scores.shape}")
        except Exception as e:
            self.logger.error(f"[AttentionScaler] Error converting 2D to 3D: {e}", exc_info=True)
            raise

        # Adım 3: Eğer tensör 3B ise, num_heads bilgisiyle 4B tensöre dönüştür.
        try:
            if attention_scores.ndim == 3:
                if self.num_heads is None:
                    raise ValueError(f"[AttentionScaler] For 3D input, 'num_heads' must be specified. Input shape: {attention_scores.shape}")
                if self.verbose:
                    self.logger.debug(f"[AttentionScaler] 3D tensor detected, converting to 4D using num_heads.")
                attention_scores = self._convert_3d_to_4d(attention_scores)
                if self.verbose:
                    self.logger.debug(f"[AttentionScaler] Converted to 4D: {attention_scores.shape}")
        except Exception as e:
            self.logger.error(f"[AttentionScaler] Error converting 3D to 4D: {e}", exc_info=True)
            raise

        # Adım 4: Ölçekleme işlemi
        try:
            scaled_attention = attention_scores * self.scale_factor
            if self.verbose:
                self.logger.debug(f"[AttentionScaler] Applied scaling factor: {self.scale_factor}")
                self.logger.debug(f"[AttentionScaler] After scaling -> min: {scaled_attention.min().item():.6f}, "
                                    f"max: {scaled_attention.max().item():.6f}, mean: {scaled_attention.mean().item():.6f}")
        except Exception as e:
            self.logger.error(f"[AttentionScaler] Error during scaling: {e}", exc_info=True)
            raise

        # Adım 5: Opsiyonel kırpma (clip_range)
        try:
            if self.clip_range is not None:
                min_val, max_val = self.clip_range
                scaled_attention = torch.clamp(scaled_attention, min=min_val, max=max_val)
                if self.verbose:
                    self.logger.debug(f"[AttentionScaler] Applied clipping with range: [{min_val}, {max_val}]")
        except Exception as e:
            self.logger.error(f"[AttentionScaler] Error during clipping: {e}", exc_info=True)
            raise

        # Adım 6: Opsiyonel yeniden normalize etme (örn. eğer dikkat ağırlıklarının toplamı 1 olması isteniyorsa)
        try:
            if hasattr(self, "re_normalize") and self.re_normalize:
                # Yeniden normalize: her satırdaki toplamı 1 yap
                attn_sum = scaled_attention.sum(dim=-1, keepdim=True)
                scaled_attention = scaled_attention / (attn_sum + 1e-8)
                if self.verbose:
                    self.logger.debug(f"[AttentionScaler] Re-normalized scaled attention. New sum (mean): {scaled_attention.sum(dim=-1).mean().item():.6f}")
        except Exception as e:
            self.logger.error(f"[AttentionScaler] Error during re-normalization: {e}", exc_info=True)
            raise

        # Adım 7: Orijinal boyutlara geri dönüş
        try:
            if original_ndim == 3:
                if self.verbose:
                    self.logger.debug(f"[AttentionScaler] Converting 4D tensor back to 3D.")
                scaled_attention = self._convert_4d_to_3d(scaled_attention)
                if self.verbose:
                    self.logger.debug(f"[AttentionScaler] Converted back to 3D: {scaled_attention.shape}")
            elif original_ndim == 2:
                scaled_attention = scaled_attention.squeeze(0)
                if self.verbose:
                    self.logger.debug(f"[AttentionScaler] Converted back to 2D: {scaled_attention.shape}")
        except Exception as e:
            self.logger.error(f"[AttentionScaler] Error converting back to original dimensions: {e}", exc_info=True)
            raise

        # Adım 8: Son çıktı tensörünü doğrula
        try:
            self.validate_tensor(scaled_attention)
            if self.verbose:
                self.logger.debug(f"[AttentionScaler] Final output shape: {scaled_attention.shape}")
        except Exception as e:
            self.logger.error(f"[AttentionScaler] Final output validation failed: {e}", exc_info=True)
            raise

        total_time = time.time() - start_time
        self.logger.info(f"[AttentionScaler] Forward pass completed in {total_time:.6f} seconds.")
        return scaled_attention





    def _convert_3d_to_4d(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        3D tensörü 4D tensöre dönüştürür.

        Args:
            attention_scores (torch.Tensor): 3D tensör (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: 4D tensör (batch_size, num_heads, seq_len, head_dim).

        Raises:
            TypeError: Eğer giriş tensörü PyTorch tensörü değilse.
            ValueError: Eğer tensör boyutları beklenenden farklıysa veya num_heads/embed_dim ilişkisi geçersizse.
            RuntimeError: Dönüşüm sırasında beklenmeyen bir hata oluşursa.
        """
        # 1. Giriş doğrulaması
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError(
                f"Giriş tensörü bir PyTorch tensörü olmalıdır. Bulunan tip: {type(attention_scores)}"
            )
        if attention_scores.ndim != 3:
            raise ValueError(
                f"3D tensör bekleniyor, ancak {attention_scores.ndim}D tensör bulundu. "
                f"Tensör boyutları: {attention_scores.shape}"
            )
        if self.num_heads is None or not isinstance(self.num_heads, int) or self.num_heads <= 0:
            raise ValueError(
                f"'num_heads', pozitif bir tam sayı olmalıdır. Bulunan: {self.num_heads}"
            )

        # 2. Tensör boyutlarını ayrıştır
        batch_size, seq_len, embed_dim = attention_scores.size()
        if embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}), num_heads ({self.num_heads}) ile tam bölünmelidir. "
                f"Bölünme sağlanamadı."
            )

        # 3. Başlık başına boyut hesaplama
        head_dim = embed_dim // self.num_heads
        if head_dim <= 0:
            raise ValueError(
                f"Başlık başına boyut (head_dim) sıfırdan büyük olmalıdır. Bulunan: {head_dim}"
            )

        # 4. 3D'den 4D'ye dönüşüm işlemi
        try:
            # 3D tensörü [batch_size, seq_len, num_heads, head_dim] boyutlarına dönüştür
            reshaped_tensor = attention_scores.view(batch_size, seq_len, self.num_heads, head_dim)
            # Tensör eksenlerini yeniden düzenle: [batch_size, num_heads, seq_len, head_dim]
            attention_scores_4d = reshaped_tensor.permute(0, 2, 1, 3).contiguous()
        except RuntimeError as re:
            raise RuntimeError(
                f"3D'den 4D'ye dönüşüm sırasında bir hata oluştu: {re}. "
                f"Giriş tensörü boyutları: {attention_scores.shape}, num_heads={self.num_heads}, "
                f"head_dim={head_dim}"
            ) from re

        # 5. Çıkış doğrulaması
        expected_shape = (batch_size, self.num_heads, seq_len, head_dim)
        if attention_scores_4d.shape != expected_shape:
            raise ValueError(
                f"Dönüşüm sonrası tensör boyutları beklenenden farklı: {attention_scores_4d.shape}. "
                f"Beklenen: {expected_shape}"
            )

        # 6. Loglama (isteğe bağlı)
        if self.verbose:
            print(f"[AttentionScaler] 3D'den 4D'ye dönüşüm başarılı.")
            print(f"Çıkış tensörü boyutları: {attention_scores_4d.shape}")

        # 7. Dönüştürülmüş tensörü döndür
        return attention_scores_4d

    def _convert_4d_to_3d(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        4D tensörü 3D tensöre dönüştürür.

        Args:
            attention_scores (torch.Tensor): 4D tensör (batch_size, num_heads, seq_len, head_dim).

        Returns:
            torch.Tensor: 3D tensör (batch_size, seq_len, embed_dim).

        Raises:
            TypeError: Eğer giriş tensörü PyTorch tensörü değilse.
            ValueError: Eğer tensör boyutları beklenen formatta değilse veya geçersizse.
            RuntimeError: Dönüşüm sırasında beklenmeyen bir hata oluşursa.
        """
        # 1. Giriş doğrulaması
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError(
                f"Giriş tensörü bir PyTorch tensörü olmalıdır. Bulunan tip: {type(attention_scores)}"
            )
        if attention_scores.ndim != 4:
            raise ValueError(
                f"4D tensör bekleniyor, ancak {attention_scores.ndim}D tensör bulundu. "
                f"Tensör boyutları: {attention_scores.shape}"
            )

        # 2. Tensör boyutlarının ayrıştırılması
        batch_size, num_heads, seq_len, head_dim = attention_scores.size()

        if num_heads <= 0 or head_dim <= 0:
            raise ValueError(
                f"num_heads ve head_dim sıfırdan büyük olmalıdır. "
                f"Bulunan: num_heads={num_heads}, head_dim={head_dim}"
            )

        # embed_dim hesaplama
        embed_dim = num_heads * head_dim
        if embed_dim <= 0:
            raise ValueError(
                f"embed_dim sıfırdan büyük olmalıdır. Bulunan: embed_dim={embed_dim}. "
                f"Kontrol edin: num_heads={num_heads}, head_dim={head_dim}"
            )

        # 3. 4D'den 3D'ye dönüşüm işlemi
        try:
            # Permute ile eksenleri yeniden düzenleme
            if self.verbose:
                print(f"[AttentionScaler] 4D'den 3D'ye dönüşüm başlatılıyor.")
            reshaped_tensor = attention_scores.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]

            # 3D tensör oluşturma
            attention_scores_3d = reshaped_tensor.view(batch_size, seq_len, embed_dim)  # [batch_size, seq_len, embed_dim]
        except RuntimeError as re:
            raise RuntimeError(
                f"4D'den 3D'ye dönüşüm sırasında hata oluştu: {re}. "
                f"Tensör boyutları: batch_size={batch_size}, seq_len={seq_len}, "
                f"num_heads={num_heads}, head_dim={head_dim}"
            ) from re

        # 4. Çıkış doğrulaması
        expected_shape = (batch_size, seq_len, embed_dim)
        if attention_scores_3d.shape != expected_shape:
            raise ValueError(
                f"Dönüşüm sonrası tensör boyutları beklenenden farklı: {attention_scores_3d.shape}. "
                f"Beklenen: {expected_shape}"
            )

        # 5. Loglama (isteğe bağlı)
        if self.verbose:
            print(f"[AttentionScaler] 4D'den 3D'ye dönüşüm başarılı.")
            print(f"Giriş boyutları: {attention_scores.shape}, Çıkış boyutları: {attention_scores_3d.shape}")

        # 6. Dönüştürülmüş tensörü döndür
        return attention_scores_3d

    def validate_tensor(self, attention_scores: torch.Tensor):
        """
        Tensörün geçerliliğini kontrol eder.

        Args:
            attention_scores (torch.Tensor): Kontrol edilecek tensör.

        Raises:
            TypeError: Eğer tensör, bir PyTorch tensörü değilse.
            ValueError: Eğer tensör boyutları beklenenden farklıysa veya geçersizse.
        """
        # 1. Tensör Tipi Kontrolü
        if not isinstance(attention_scores, torch.Tensor):
            raise TypeError(
                f"Giriş bir PyTorch tensörü olmalıdır. Bulunan tip: {type(attention_scores)}"
            )

        # 2. Tensör Boyut Kontrolü
        if attention_scores.ndim not in [3, 4]:
            raise ValueError(
                f"Tensör boyutları 3D veya 4D olmalıdır, ancak {attention_scores.ndim}D bulundu. "
                f"Tensör boyutları: {attention_scores.size()}."
            )

        # 3. Negatif veya Sıfır Boyut Kontrolü
        invalid_dims = [dim for dim in attention_scores.size() if dim <= 0]
        if invalid_dims:
            raise ValueError(
                f"Tensör boyutları sıfırdan büyük olmalıdır, ancak geçersiz boyut(lar): {invalid_dims} bulundu. "
                f"Tensör boyutları: {attention_scores.size()}."
            )

        # 4. 4D Tensör İçin Ek Kontroller
        if attention_scores.ndim == 4:
            batch_size, num_heads, seq_len, head_dim = attention_scores.size()

            # 4D Tensör Boyutlarının Geçerlilik Kontrolü
            if batch_size <= 0 or num_heads <= 0 or seq_len <= 0 or head_dim <= 0:
                raise ValueError(
                    f"4D tensörlerde tüm boyutlar sıfırdan büyük olmalıdır. "
                    f"Bulunan: batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}."
                )

            # num_heads ve head_dim Kontrolleri
            if self.num_heads is not None and num_heads != self.num_heads:
                raise ValueError(
                    f"Tensör num_heads ({num_heads}), beklenen num_heads ({self.num_heads}) ile eşleşmiyor."
                )
            if self.num_heads is not None and head_dim != attention_scores.size(-1):
                raise ValueError(
                    f"Tensördeki head_dim ({head_dim}), beklenen değer ({attention_scores.size(-1)}) ile eşleşmiyor."
                )

        # 5. Loglama (isteğe bağlı)
        if self.verbose:
            print(f"[validate_tensor] Tensör doğrulama başarılı: {attention_scores.size()}")

        # 6. Ek Doğrulama ve Loglama
        if attention_scores.ndim == 3:
            batch_size, seq_len, embed_dim = attention_scores.size()
            if embed_dim <= 0:
                raise ValueError(
                    f"3D tensörde embed_dim sıfırdan büyük olmalıdır. Bulunan: embed_dim={embed_dim}."
                )
            if self.verbose:
                print(f"[validate_tensor] 3D Tensör doğrulama başarılı: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}")

        elif attention_scores.ndim == 4:
            if self.verbose:
                print(f"[validate_tensor] 4D Tensör doğrulama başarılı: batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")


    def extra_repr(self) -> str:
        """
        Sınıfın özet bilgilerini döndürür.

        Returns:
            str: Sınıf özet metni.
        """
        # Ölçek faktörü bilgisi
        scale_factor_info = f"scale_factor={self.scale_factor:.3f}"

        # Kırpma aralığı bilgisi (None kontrolü dahil)
        if self.clip_range is None:
            clip_range_info = "clip_range=None"
        else:
            clip_range_info = f"clip_range=({self.clip_range[0]:.3f}, {self.clip_range[1]:.3f})"

        # Verbose (loglama) bilgisi
        verbose_info = f"verbose={'Enabled' if self.verbose else 'Disabled'}"

        # num_heads bilgisi (None kontrolü dahil)
        if self.num_heads is None:
            num_heads_info = "num_heads=None"
        else:
            num_heads_info = f"num_heads={self.num_heads}"

        # Temsil metninin birleştirilmesi
        return ", ".join([scale_factor_info, clip_range_info, verbose_info, num_heads_info])
