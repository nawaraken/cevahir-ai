import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    """
    Sekans içindeki dikkat mekanizması sınıfı.
    Farklı normalizasyon türlerini destekler ve hata ayıklama özellikleri içerir.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.3, normalization_type="layer_norm", 
                num_groups=None, eps=1e-5, debug=False):
        """
        SelfAttention sınıfını başlatır.

        Args:
            embed_dim (int): Gömme boyutu.
            num_heads (int): Çok başlık sayısı.
            dropout (float): Dropout oranı.
            normalization_type (str): Normalizasyon türü. Desteklenenler: 'layer_norm', 'batch_norm', 'group_norm', 'instance_norm'.
            num_groups (int, optional): GroupNorm için grup sayısı.
            eps (float, optional): Sayısal kararlılık için epsilon değeri.
            debug (bool): Hata ayıklama modu.
        """
        super(SelfAttention, self).__init__()

        # Gömme boyutu ve çok başlık kontrolü
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Gömme boyutu ({embed_dim}) çok başlık sayısına ({num_heads}) tam bölünemiyor. "
                f"Gömme boyutunun çok başlık sayısına bölünebilir bir değer olması gerekir."
            )

        if embed_dim <= 0:
            raise ValueError(f"Gömme boyutu ({embed_dim}) pozitif bir sayı olmalıdır.")
        if num_heads <= 0:
            raise ValueError(f"Çok başlık sayısı ({num_heads}) pozitif bir sayı olmalıdır.")

        # Özelliklerin atanması
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.normalization_type = normalization_type
        self.num_groups = num_groups
        self.eps = eps
        self.debug = debug

        # Debug logu
        if self.debug:
            print(f"SelfAttention başlatılıyor: embed_dim={embed_dim}, num_heads={num_heads}, "
                f"dropout={dropout}, normalization_type={normalization_type}, num_groups={num_groups}, eps={eps}")

        # Projeksiyon katmanları
        try:
            self.query_proj = nn.Linear(embed_dim, embed_dim)
            self.key_proj = nn.Linear(embed_dim, embed_dim)
            self.value_proj = nn.Linear(embed_dim, embed_dim)
        except Exception as e:
            raise RuntimeError(f"Sorgu, Anahtar veya Değer projeksiyon katmanları oluşturulurken hata oluştu: {str(e)}")

        # Çıktı projeksiyonu
        try:
            self.out_proj = nn.Linear(embed_dim, embed_dim)
        except Exception as e:
            raise RuntimeError(f"Çıktı projeksiyon katmanı oluşturulurken hata oluştu: {str(e)}")

        # Dropout katmanları
        try:
            self.attn_dropout = nn.Dropout(dropout)
            self.final_dropout = nn.Dropout(dropout)
        except Exception as e:
            raise RuntimeError(f"Dropout katmanları oluşturulurken hata oluştu: {str(e)}")

        # Dinamik normalizasyon modülü
        try:
            self.norm = self.initialize_normalizer(
                normalization_type=normalization_type,
                embed_dim=embed_dim,
                num_groups=num_groups,
                eps=eps
            )
        except ValueError as ve:
            raise ValueError(f"Normalizasyon modülü oluşturulurken doğrulama hatası: {str(ve)}")
        except Exception as e:
            raise RuntimeError(f"Normalizasyon modülü başlatılırken hata oluştu: {str(e)}")

        # Debug modu için başarı mesajı
        if self.debug:
            print("SelfAttention başarıyla başlatıldı!")



    def initialize_normalizer(self, normalization_type, embed_dim, num_groups=None, eps=1e-5):
        """
        Dinamik normalizasyon modülü oluşturur.

        Args:
            normalization_type (str): Normalizasyon türü. Desteklenen tipler: ['layer_norm', 'batch_norm', 'group_norm', 'instance_norm'].
            embed_dim (int): Gömme boyutu.
            num_groups (int, optional): GroupNorm için grup sayısı. Varsayılan None.
            eps (float, optional): Sayısal kararlılık için epsilon değeri. Varsayılan 1e-5.

        Returns:
            nn.Module: Uygun normalizasyon modülü.

        Raises:
            ValueError: Geçersiz normalizasyon tipi veya GroupNorm için uygunsuz grup sayısı verilirse.
        """
        if self.debug:
            print(f"initialize_normalizer çağrıldı: normalization_type={normalization_type}, "
                f"embed_dim={embed_dim}, num_groups={num_groups}, eps={eps}")

        # Desteklenen normalizasyon türlerini tanımla
        supported_norms = ['layer_norm', 'batch_norm', 'group_norm', 'instance_norm']

        if normalization_type not in supported_norms:
            raise ValueError(f"Geçersiz normalizasyon tipi: {normalization_type}. "
                            f"Desteklenen tipler: {supported_norms}")

        if normalization_type == "layer_norm":
            return self._initialize_layer_norm(embed_dim, eps)

        if normalization_type == "batch_norm":
            return self._initialize_batch_norm(embed_dim, eps)

        if normalization_type == "group_norm":
            return self._initialize_group_norm(embed_dim, num_groups, eps)

        if normalization_type == "instance_norm":
            return self._initialize_instance_norm(embed_dim, eps)

        # Ek güvenlik: Bu koda asla ulaşılmamalı.
        raise RuntimeError("Desteklenmeyen bir normalizasyon tipi işlendi.")

    def _initialize_layer_norm(self, embed_dim, eps):
        """
        LayerNorm modülünü başlatır ve ileri seviye doğrulama yapar.

        Args:
            embed_dim (int): Gömme boyutu.
            eps (float): Sayısal kararlılık için epsilon değeri.

        Returns:
            nn.LayerNorm: Başlatılmış LayerNorm modülü.

        Raises:
            ValueError: Eğer embed_dim geçersiz bir değer ise.
        """
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"LayerNorm için embed_dim pozitif bir tam sayı olmalıdır. Verilen: {embed_dim}")
        if not isinstance(eps, (float, int)) or eps <= 0:
            raise ValueError(f"LayerNorm için eps pozitif bir sayı olmalıdır. Verilen: {eps}")
        if self.debug:
            print(f"LayerNorm başlatılıyor... embed_dim={embed_dim}, eps={eps}")
        return nn.LayerNorm(embed_dim, eps=eps)


    def _initialize_batch_norm(self, embed_dim, eps):
        """
        BatchNorm modülünü başlatır ve ileri seviye doğrulama yapar.

        Args:
            embed_dim (int): Gömme boyutu.
            eps (float): Sayısal kararlılık için epsilon değeri.

        Returns:
            nn.BatchNorm1d: Başlatılmış BatchNorm modülü.

        Raises:
            ValueError: Eğer embed_dim veya eps geçersiz bir değer ise.
        """
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"BatchNorm için embed_dim pozitif bir tam sayı olmalıdır. Verilen: {embed_dim}")
        if not isinstance(eps, (float, int)) or eps <= 0:
            raise ValueError(f"BatchNorm için eps pozitif bir sayı olmalıdır. Verilen: {eps}")
        if self.debug:
            print(f"BatchNorm başlatılıyor... embed_dim={embed_dim}, eps={eps}")
        return nn.BatchNorm1d(embed_dim, eps=eps)


    def _initialize_group_norm(self, embed_dim, num_groups, eps):
        """
        GroupNorm modülünü başlatır ve ileri seviye doğrulama yapar.

        Args:
            embed_dim (int): Gömme boyutu.
            num_groups (int): Grup sayısı.
            eps (float): Sayısal kararlılık için epsilon değeri.

        Returns:
            nn.GroupNorm: Başlatılmış GroupNorm modülü.

        Raises:
            ValueError: Eğer embed_dim, num_groups veya eps geçersiz bir değer ise.
        """
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"GroupNorm için embed_dim pozitif bir tam sayı olmalıdır. Verilen: {embed_dim}")
        if not isinstance(num_groups, int) or num_groups <= 0:
            raise ValueError(f"GroupNorm için num_groups pozitif bir tam sayı olmalıdır. Verilen: {num_groups}")
        if embed_dim % num_groups != 0:
            raise ValueError(f"GroupNorm için embed_dim ({embed_dim}) num_groups ({num_groups}) ile tam bölünemiyor.")
        if not isinstance(eps, (float, int)) or eps <= 0:
            raise ValueError(f"GroupNorm için eps pozitif bir sayı olmalıdır. Verilen: {eps}")
        if self.debug:
            print(f"GroupNorm başlatılıyor... embed_dim={embed_dim}, num_groups={num_groups}, eps={eps}")
        return nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim, eps=eps)


    def _initialize_instance_norm(self, embed_dim, eps):
        """
        InstanceNorm modülünü başlatır ve ileri seviye doğrulama yapar.

        Args:
            embed_dim (int): Gömme boyutu.
            eps (float): Sayısal kararlılık için epsilon değeri.

        Returns:
            nn.InstanceNorm1d: Başlatılmış InstanceNorm modülü.

        Raises:
            ValueError: Eğer embed_dim veya eps geçersiz bir değer ise.
        """
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"InstanceNorm için embed_dim pozitif bir tam sayı olmalıdır. Verilen: {embed_dim}")
        if not isinstance(eps, (float, int)) or eps <= 0:
            raise ValueError(f"InstanceNorm için eps pozitif bir sayı olmalıdır. Verilen: {eps}")
        if self.debug:
            print(f"InstanceNorm başlatılıyor... embed_dim={embed_dim}, eps={eps}")
        return nn.InstanceNorm1d(embed_dim, eps=eps)




    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        Ölçeklenmiş nokta çarpımı dikkat mekanizmasını uygular.

        Args:
            query (torch.Tensor): Sorgu tensörü (batch_size, num_heads, seq_len, head_dim).
            key (torch.Tensor): Anahtar tensörü (batch_size, num_heads, seq_len, head_dim).
            value (torch.Tensor): Değer tensörü (batch_size, num_heads, seq_len, head_dim).
            mask (torch.Tensor, optional): Maske tensörü (batch_size, 1, 1, seq_len).

        Returns:
            torch.Tensor: Dikkat mekanizması çıktısı (batch_size, num_heads, seq_len, head_dim).

        Raises:
            RuntimeError: Hatalı giriş verisi veya işlem sırasında hata oluştuğunda.
        """
        try:
            # 1. Girdi boyutları doğrulama
            self.validate_attention_inputs(query, key, value)

            # 2. Giriş tensörlerinde NaN/Sonsuz değer temizleme
            query = torch.nan_to_num(query, nan=0.0, posinf=1e9, neginf=-1e9)
            key = torch.nan_to_num(key, nan=0.0, posinf=1e9, neginf=-1e9)
            value = torch.nan_to_num(value, nan=0.0, posinf=1e9, neginf=-1e9)

            # 3. Ölçeklenmiş skor hesaplama
            scores = self.calculate_scaled_scores(query, key)

            # 4. Maske kontrolü ve uygulanması
            if mask is not None:
                scores = self.apply_mask(scores, mask)

            # 5. Dikkat ağırlıklarını normalize et
            attn_weights = self.normalize_attention_weights(scores)

            # 6. Çıkış hesaplama
            output = self.calculate_attention_output(attn_weights, value)

            # 7. Çıkış tensöründe NaN/Sonsuz temizleme
            output = torch.nan_to_num(output, nan=0.0, posinf=1e9, neginf=-1e9)

            # 8. Debug bilgisi
            if self.debug:
                print(f"[Debug] Final Output Shape: {output.shape}")
                print(f"[Debug] Max Output Value: {output.max().item():.4f}, Min Output Value: {output.min().item():.4f}")

            return output

        except ValueError as ve:
            error_message = f"scaled_dot_product_attention işleminde doğrulama hatası: {str(ve)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)

        except RuntimeError as re:
            error_message = f"scaled_dot_product_attention işleminde çalışma zamanı hatası: {str(re)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)

        except Exception as e:
            error_message = f"scaled_dot_product_attention işleminde beklenmeyen hata: {str(e)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)


    def validate_attention_inputs(self, query, key, value):
        """
        Query, key ve value tensörlerinin boyutlarını ve tutarlılığını doğrular.

        Args:
            query (torch.Tensor): Sorgu tensörü.
            key (torch.Tensor): Anahtar tensörü.
            value (torch.Tensor): Değer tensörü.

        Raises:
            ValueError: Eğer tensör boyutları veya değerleri uyumsuzsa.
        """
        if query.size(-1) != key.size(-1) or query.size(-1) != value.size(-1):
            raise ValueError("Query, Key ve Value tensörlerinin son boyutları eşleşmelidir.")
        if query.size(0) != key.size(0) or query.size(0) != value.size(0):
            raise ValueError("Query, Key ve Value tensörlerinin batch boyutları eşleşmelidir.")
        if query.size(1) != key.size(1) or query.size(1) != value.size(1):
            raise ValueError("Query, Key ve Value tensörlerinin başlık boyutları eşleşmelidir.")
        if self.debug:
            print(f"[Debug] Girdi boyutları doğrulandı: Query={query.shape}, Key={key.shape}, Value={value.shape}")

    def calculate_scaled_scores(self, query, key):
        """
        Query ve Key tensörlerinden ölçeklenmiş dikkat skorlarını hesaplar.

        Args:
            query (torch.Tensor): Sorgu tensörü (batch_size, num_heads, seq_len, head_dim).
            key (torch.Tensor): Anahtar tensörü (batch_size, num_heads, seq_len, head_dim).

        Returns:
            torch.Tensor: Ölçeklenmiş dikkat skorları (batch_size, num_heads, seq_len, seq_len).

        Raises:
            RuntimeError: Girdi tensörleri geçersizse veya hesaplama sırasında hata oluşursa.
        """
        try:
            # 1. Girdi doğrulama
            if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
                raise ValueError("Hem 'query' hem de 'key' birer torch.Tensor olmalıdır.")
            
            if query.size(-1) != key.size(-1):
                raise ValueError(f"'query' ve 'key' tensörlerinin son boyutları eşleşmiyor: "
                                f"query.size(-1)={query.size(-1)}, key.size(-1)={key.size(-1)}")

            if query.dim() != 4 or key.dim() != 4:
                raise ValueError(f"'query' ve 'key' tensörleri 4 boyutlu olmalıdır. "
                                f"Query.dim={query.dim()}, Key.dim={key.dim()}")

            # 2. NaN ve sonsuz değerlerin temizlenmesi
            query = torch.nan_to_num(query, nan=0.0, posinf=1e9, neginf=-1e9)
            key = torch.nan_to_num(key, nan=0.0, posinf=1e9, neginf=-1e9)

            # 3. Ölçeklenmiş skor hesaplama
            d_k = query.size(-1)
            scale_factor = torch.sqrt(torch.tensor(d_k, dtype=query.dtype, device=query.device) + self.eps)
            
            # Skor hesaplama
            scores = torch.matmul(query, key.transpose(-2, -1)) / scale_factor

            # 4. NaN ve sonsuz değerlerin temizlenmesi (scores için)
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)

            # 5. Debug loglama
            if self.debug:
                self.debug_log(scores, "Scaled Scores")

            # 6. NaN/Sonsuz kontrolü sonrası güvenlik doğrulaması
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                raise RuntimeError("Hesaplanan dikkat skorları NaN veya sonsuz değer içeriyor.")

            return scores

        except ValueError as ve:
            error_message = f"calculate_scaled_scores giriş doğrulama hatası: {str(ve)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)

        except Exception as e:
            error_message = f"calculate_scaled_scores sırasında beklenmeyen hata: {str(e)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)



    def calculate_attention_output(self, attn_weights, value):
        """
        Normalize edilmiş dikkat ağırlıklarını kullanarak çıkış hesaplar.

        Args:
            attn_weights (torch.Tensor): Normalize edilmiş dikkat ağırlıkları.
            value (torch.Tensor): Değer tensörü.

        Returns:
            torch.Tensor: Dikkat mekanizması çıktısı.
        """
        output = torch.matmul(attn_weights, value)
        if self.debug:
            print(f"[Debug] Attention output computed. Shape: {output.shape}")
        return output


    def apply_mask(self, scores, mask):
        """
        Maske tensörünü dikkat skorlarına uygular.

        Args:
            scores (torch.Tensor): Dikkat skorları (batch_size, num_heads, seq_len, seq_len).
            mask (torch.Tensor, optional): Maske tensörü (batch_size, 1, 1, seq_len).

        Returns:
            torch.Tensor: Maskelenmiş dikkat skorları.

        Raises:
            ValueError: Maske tensörünün boyutu 4 değilse.
        """
        try:
            if mask is not None:
                if mask.dim() != 4:
                    raise ValueError(f"Maske tensörü 4 boyutlu olmalıdır, ancak {mask.dim()} boyutlu tespit edildi.")
                
                # Maske tipi uyumluluğu ve cihaz taşıma
                mask = mask.to(dtype=torch.float32, device=scores.device)

                # Maske uygulaması
                scores = scores.masked_fill(mask == 0, float('-inf'))

            if self.debug:
                self.debug_log(scores, "Masked Scores")

            return scores

        except ValueError as ve:
            error_message = f"apply_mask işleminde doğrulama hatası: {str(ve)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)

        except Exception as e:
            error_message = f"apply_mask işleminde beklenmeyen hata: {str(e)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)

    def normalize_attention_weights(self, scores):
        """
        Dikkat skorlarını normalize eder ve dropout uygular.

        Args:
            scores (torch.Tensor): Dikkat skorları (batch_size, num_heads, seq_len, seq_len).

        Returns:
            torch.Tensor: Normalize edilmiş dikkat ağırlıkları.

        Raises:
            RuntimeError: Eğer skorlar NaN veya sonsuz içeriyorsa.
        """
        try:
            # 1. Girdi doğrulaması
            if not isinstance(scores, torch.Tensor):
                raise ValueError(f"'scores' bir torch.Tensor olmalı, ancak {type(scores)} tespit edildi.")
            
            if scores.dim() != 4:
                raise ValueError(f"'scores' tensörünün boyutu 4 olmalı, ancak {scores.dim()} boyutlu tespit edildi.")

            # 2. NaN/Sonsuz temizliği (giriş skorları)
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)

            # 3. Softmax işlemi
            attn_weights = torch.softmax(scores, dim=-1)

            # 4. Normalize edilmiş skorları kontrol ve temizleme
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=1e9, neginf=-1e9)

            # 5. Dropout işlemi
            attn_weights = self.attn_dropout(attn_weights)

            # 6. Sayısal kararlılık kontrolü
            if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
                raise RuntimeError("Normalize edilmiş dikkat ağırlıkları hala NaN veya sonsuz değer içeriyor.")

            # 7. Debug loglama
            if self.debug:
                self.debug_log(attn_weights, "Attention Weights")

            return attn_weights

        except Exception as e:
            error_message = f"normalize_attention_weights işleminde hata oluştu: {str(e)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)

        

    def apply_normalization(self, x):
        """
        Girdi tensörüne seçilen normalizasyon türüne göre işlem uygular.

        Args:
            x (torch.Tensor): Girdi tensörü (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Normalizasyon uygulanmış tensör.

        Raises:
            ValueError: Eğer tensör boyutları veya normalizasyon tipi geçerli değilse.
            RuntimeError: Normalizasyon işlemi sırasında beklenmeyen bir hata oluşursa.
        """
        try:
            # Girdi tensörünün geçerliliğini kontrol et
            self._validate_input_tensor(x)

            # Normalizasyon türüne göre ilgili işlevi çağır
            if self.normalization_type == "batch_norm":
                return self._apply_batch_norm(x)
            elif self.normalization_type == "instance_norm":
                return self._apply_instance_norm(x)
            elif self.normalization_type == "layer_norm":
                return self._apply_layer_norm(x)
            elif self.normalization_type == "group_norm":
                return self._apply_group_norm(x)
            else:
                raise ValueError(
                    f"Geçersiz normalizasyon tipi: {self.normalization_type}. "
                    f"Desteklenen tipler: ['layer_norm', 'batch_norm', 'group_norm', 'instance_norm']"
                )
        except Exception as e:
            error_message = f"apply_normalization işleminde beklenmeyen hata oluştu: {str(e)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)


    def _validate_input_tensor(self, x):
        """
        Girdi tensörünün geçerliliğini kontrol eder.

        Args:
            x (torch.Tensor): Girdi tensörü.

        Raises:
            ValueError: Eğer tensör geçerli değilse.
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError(
                f"Giriş 'x' bir torch.Tensor olmalı, ancak {type(x).__name__} türünde tespit edildi."
            )
        if x.dim() != 3:
            raise ValueError(
                f"Giriş tensörü 3 boyutlu olmalı (batch_size, seq_len, embed_dim), "
                f"ancak {x.dim()} boyutlu tespit edildi."
            )
        if self.debug:
            print(f"Girdi tensörü geçerli: {x.shape}")


    def _apply_batch_norm(self, x):
        """
        Batch Normalization uygular.

        Args:
            x (torch.Tensor): Girdi tensörü.

        Returns:
            torch.Tensor: Normalizasyon uygulanmış tensör.
        """
        if self.debug:
            print("BatchNorm seçildi, tensör boyutları dönüştürülüyor.")
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        if self.debug:
            print(f"BatchNorm sonrası çıktı boyutu: {x.shape}")
        return x


    def _apply_instance_norm(self, x):
        """
        Instance Normalization uygular.

        Args:
            x (torch.Tensor): Girdi tensörü.

        Returns:
            torch.Tensor: Normalizasyon uygulanmış tensör.
        """
        if self.debug:
            print("InstanceNorm seçildi, tensör boyutları dönüştürülüyor.")
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        if self.debug:
            print(f"InstanceNorm sonrası çıktı boyutu: {x.shape}")
        return x


    def _apply_layer_norm(self, x):
        """
        Layer Normalization uygular.

        Args:
            x (torch.Tensor): Girdi tensörü.

        Returns:
            torch.Tensor: Normalizasyon uygulanmış tensör.
        """
        if self.debug:
            print("LayerNorm seçildi, tensör direkt normalizasyon işlemine alınıyor.")
        x = self.norm(x)
        if self.debug:
            print(f"LayerNorm sonrası çıktı boyutu: {x.shape}")
        return x


    def _apply_group_norm(self, x):
        """
        Group Normalization uygular.

        Args:
            x (torch.Tensor): Girdi tensörü.

        Returns:
            torch.Tensor: Normalizasyon uygulanmış tensör.

        Raises:
            RuntimeError: Eğer giriş boyutları 'num_groups' ile uyumlu değilse ve otomatik ayarlama başarısız olursa.
        """
        embed_dim = x.size(-1)  # Tensörün embed boyutu (kanal sayısı)

        # Grup uyumluluğunu kontrol et ve ayarla
        if embed_dim % self.num_groups != 0:
            # Dinamik grup sayısını hesapla
            adjusted_num_groups = max(1, min(self.num_groups, embed_dim))
            while embed_dim % adjusted_num_groups != 0 and adjusted_num_groups > 1:
                adjusted_num_groups -= 1

            if adjusted_num_groups < 1:
                raise RuntimeError(
                    f"GroupNorm hatası: embed_dim={embed_dim}, num_groups={self.num_groups}. "
                    f"Geçerli bir grup sayısı bulunamadı."
                )

            # Yeni grup sayısını ata ve logla
            if self.debug:
                print(
                    f"[Uyarı] embed_dim={embed_dim}, num_groups={self.num_groups} uyumsuz. "
                    f"Yeni grup sayısı: {adjusted_num_groups}"
                )
            self.num_groups = adjusted_num_groups
            self.norm = torch.nn.GroupNorm(self.num_groups, embed_dim, eps=self.eps)

        # Debug loglama
        if self.debug:
            print(f"GroupNorm uygulanıyor: embed_dim={embed_dim}, num_groups={self.num_groups}")

        try:
            # Tensör eksenlerini GroupNorm'a uygun hale getir
            x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
            x = self.norm(x)  # GroupNorm işlemi
            x = x.transpose(1, 2)  # (batch_size, seq_len, embed_dim)

            if self.debug:
                print(f"GroupNorm sonrası çıktı boyutu: {x.shape}")
        except Exception as e:
            raise RuntimeError(
                f"GroupNorm sırasında bir hata oluştu: embed_dim={embed_dim}, num_groups={self.num_groups}. "
                f"Hata: {str(e)}"
            )

        # Sayısal kararlılık kontrolü
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError(
                "GroupNorm sonrası tensör NaN veya sonsuz değer içeriyor. "
                f"embed_dim={embed_dim}, num_groups={self.num_groups}"
            )

        return x


    def debug_log(self, tensor, name):
        """
        Tüm ara tensörleri ve hataları loglar.

        Args:
            tensor (torch.Tensor): İncelenecek tensör.
            name (str): Tensörün adı.

        Raises:
            ValueError: Eğer tensör boşsa veya geçersiz tipteyse.
        """
        try:
            if tensor is None:
                raise ValueError(f"{name} tensörü None türünde.")
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} tensörü torch.Tensor türünde değil.")

            # Log bilgileri
            print(
                f"[Debug] {name} - Shape: {tensor.shape}, "
                f"NaN: {torch.isnan(tensor).any()}, Inf: {torch.isinf(tensor).any()}, "
                f"Max: {tensor.max().item():.4f}, Min: {tensor.min().item():.4f}"
            )

        except ValueError as ve:
            error_message = f"debug_log işleminde doğrulama hatası: {str(ve)}"
            print(f"[Error] {error_message}")
            raise RuntimeError(error_message)

        except Exception as e:
            error_message = f"debug_log işleminde beklenmeyen hata: {str(e)}"
            print(f"[Error] {error_message}")
            raise RuntimeError(error_message)

    def forward(self, x, mask=None):
        """
        Sekans içindeki dikkat mekanizmasını uygular.

        Args:
            x (torch.Tensor): Girdi tensörü (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): Maske tensörü (batch_size, seq_len veya batch_size, 1, seq_len).

        Returns:
            torch.Tensor: Dikkat mekanizması çıktısı (batch_size, seq_len, embed_dim).

        Raises:
            RuntimeError: Beklenmeyen bir hata oluşursa veya doğrulama hatası meydana gelirse.
        """
        try:
            # 1. Giriş doğrulaması
            self.validate_inputs(x, mask)

            if self.debug:
                print(f"Giriş tensörü boyutu: {x.shape}")
                if mask is not None:
                    print(f"Maske boyutu: {mask.shape}")

            # NaN ve sonsuz değer kontrolü ve temizleme
            x = torch.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9)

            # 2. Projeksiyon işlemleri (query, key, value tensörleri oluşturulur)
            query, key, value = self.project_inputs(x)

            if self.debug:
                print(f"Projeksiyon sonrası boyutlar: Query={query.shape}, Key={key.shape}, Value={value.shape}")

            # 3. Maske işlemi
            if mask is not None:
                mask = self.process_mask(mask, x.size(1))
                if self.debug:
                    print(f"İşlenmiş maske boyutu: {mask.shape}")

            # 4. Dikkat mekanizması
            attn_output = self.scaled_dot_product_attention(query, key, value, mask)

            if self.debug:
                print(f"Dikkat mekanizması çıktı boyutu: {attn_output.shape}")

            # NaN ve sonsuz değer kontrolü (dikkat mekanizması çıkışı)
            attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=1e9, neginf=-1e9)

            # 5. Çıktıyı birleştirme
            attn_output = self.combine_heads(attn_output, x.size(0), x.size(1))

            if self.debug:
                print(f"Başlıklar birleştirildi: {attn_output.shape}")

            # NaN ve sonsuz değer kontrolü (birleştirilmiş çıktı)
            attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=1e9, neginf=-1e9)

            # 6. Normalizasyon işlemi
            normalized_output = self.apply_normalization(attn_output)

            if self.debug:
                print(f"Normalize edilmiş çıktı boyutu: {normalized_output.shape}")

            # NaN ve sonsuz değer kontrolü (normalize edilmiş çıktı)
            normalized_output = torch.nan_to_num(normalized_output, nan=0.0, posinf=1e9, neginf=-1e9)

            # 7. Çıkış projeksiyonu, residual bağlantı ve dropout
            output = self.final_dropout(self.out_proj(normalized_output) + x)

            if self.debug:
                print(f"Son çıktı boyutu: {output.shape}")

            # NaN ve sonsuz değer kontrolü (nihai çıktı)
            output = torch.nan_to_num(output, nan=0.0, posinf=1e9, neginf=-1e9)

            # NaN/Sonsuz kontrolü sonrası güvenlik doğrulaması
            if torch.isnan(output).any() or torch.isinf(output).any():
                raise RuntimeError("Model çıktısı NaN veya sonsuz değer içeriyor.")

            # 8. Çıkışı döndürme
            return output

        except ValueError as ve:
            error_message = f"SelfAttention 'forward' işlemi sırasında doğrulama hatası: {str(ve)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)

        except Exception as e:
            error_message = f"SelfAttention 'forward' işlemi sırasında beklenmeyen hata: {str(e)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)


    def project_inputs(self, x):
        """
        Girdi tensörlerini sorgu, anahtar ve değer tensörlerine projekte eder.

        Args:
            x (torch.Tensor): Girdi tensörü (batch_size, seq_len, embed_dim).

        Returns:
            tuple: Sorgu, anahtar ve değer tensörleri (batch_size, num_heads, seq_len, head_dim).

        Raises:
            RuntimeError: Projeksiyon işlemi sırasında bir hata oluşursa.
            ValueError: Girdi tensörü boyutları uygun değilse.
        """
        try:
            # 1. Girdi boyut kontrolü
            if x.dim() != 3:
                raise ValueError(f"Girdi tensörünün boyutu 3 olmalı, ancak {x.dim()} bulundu. "
                                f"(Beklenen: [batch_size, seq_len, embed_dim])")

            batch_size, seq_len, embed_dim = x.size()

            # 2. Gömme boyutu kontrolü
            if embed_dim != self.embed_dim:
                raise ValueError(f"Gömme boyutu {embed_dim}, beklenen {self.embed_dim} ile eşleşmiyor.")

            if self.debug:
                print(f"Projeksiyon giriş tensörü boyutları: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}")

            # 3. Sorgu, Anahtar ve Değer projeksiyonları
            query = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            key = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            value = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # 4. Projeksiyon sonrası boyut kontrolü
            if query.size(-1) != self.head_dim or key.size(-1) != self.head_dim or value.size(-1) != self.head_dim:
                raise ValueError(f"Projeksiyon işlemi sonucunda beklenmeyen bir boyut tespit edildi: "
                                f"Query={query.size()}, Key={key.size()}, Value={value.size()}")

            if self.debug:
                print(f"Projeksiyon sonrası boyutlar: Query={query.shape}, Key={key.shape}, Value={value.shape}")

            return query, key, value

        except ValueError as ve:
            error_message = f"Projeksiyon giriş doğrulama hatası: {str(ve)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)

        except Exception as e:
            error_message = f"Projeksiyon işlemleri sırasında beklenmeyen hata: {str(e)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)



    def process_mask(self, mask, seq_len):
        """
        Maskeyi uygun formata dönüştürür.

        Args:
            mask (torch.Tensor): Orijinal maske tensörü.
            seq_len (int): Giriş tensörünün sekans uzunluğu.

        Returns:
            torch.Tensor: Uygun formata dönüştürülmüş maske tensörü.

        Raises:
            RuntimeError: İşleme sırasında hata oluşursa.
            ValueError: Maske boyutu veya değerleri geçersizse.
        """
        try:
            # 1. Maske geçerlilik kontrolü
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"Maske tensörü bir torch.Tensor olmalıdır, ancak {type(mask)} tespit edildi.")

            if mask.dtype not in [torch.bool, torch.float32, torch.float64]:
                raise ValueError(f"Maske tensörünün tipi 'bool', 'float32' veya 'float64' olmalıdır, ancak {mask.dtype} bulundu.")

            if self.debug:
                print(f"Maske tensörünün orijinal boyutları: {mask.shape}, dtype={mask.dtype}")

            # 2. Maske boyut kontrolü ve dönüşüm
            if mask.dim() == 2:
                # 2 boyutlu maske için [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                if mask.size(1) != seq_len:
                    raise ValueError(f"2 boyutlu maske için sekans uzunluğu ({mask.size(1)}), giriş sekans uzunluğuyla ({seq_len}) eşleşmiyor.")
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 4:
                # 4 boyutlu maske için [batch_size, num_heads, seq_len, seq_len]
                if mask.size(-1) != seq_len:
                    raise ValueError(f"4 boyutlu maske için son boyut ({mask.size(-1)}), giriş sekans uzunluğuyla ({seq_len}) eşleşmiyor.")
            else:
                raise ValueError(f"Desteklenmeyen maske boyutu: {mask.dim()}. Sadece 2 veya 4 boyutlu maskeler desteklenir.")

            if self.debug:
                print(f"Maske tensörünün işlenmiş boyutları: {mask.shape}")

            # 3. Maske tipi uyumluluğu (float32 formatına dönüştürme)
            mask = mask.to(dtype=torch.float32)

            return mask

        except ValueError as ve:
            error_message = f"Maske giriş doğrulama hatası: {str(ve)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)

        except Exception as e:
            error_message = f"Maske işleme sırasında beklenmeyen hata: {str(e)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)

    def combine_heads(self, attn_output, batch_size, seq_len):
        """
        Dikkat mekanizması çıktısını birleştirir.

        Args:
            attn_output (torch.Tensor): Dikkat mekanizması çıktısı (batch_size, num_heads, seq_len, head_dim).
            batch_size (int): Batch boyutu.
            seq_len (int): Sekans uzunluğu.

        Returns:
            torch.Tensor: Birleştirilmiş çıktı (batch_size, seq_len, embed_dim).

        Raises:
            RuntimeError: Çıktıyı birleştirme sırasında hata oluşursa.
        """
        try:
            # 1. Girdi doğrulama
            if not isinstance(attn_output, torch.Tensor):
                raise ValueError(f"attn_output bir torch.Tensor olmalıdır, ancak {type(attn_output)} tespit edildi.")

            if attn_output.dim() != 4:
                raise ValueError(f"attn_output tensörünün boyutu 4 olmalıdır, ancak {attn_output.dim()} boyutlu tespit edildi.")

            num_heads, head_dim = attn_output.size(1), attn_output.size(-1)
            if self.embed_dim != num_heads * head_dim:
                raise ValueError(
                    f"Embed boyutu ({self.embed_dim}), num_heads ({num_heads}) ve head_dim ({head_dim}) çarpımıyla eşleşmiyor."
                )

            if self.debug:
                print(f"combine_heads - Girdi boyutları: {attn_output.shape}, batch_size={batch_size}, seq_len={seq_len}")

            # 2. Başlıkları birleştirme işlemi
            combined_output = (
                attn_output.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
                .contiguous()  # Bellekte ardışık yapıya dönüştür
                .view(batch_size, seq_len, self.embed_dim)  # (batch_size, seq_len, embed_dim)
            )

            if self.debug:
                print(f"combine_heads - Birleştirilmiş çıktı boyutları: {combined_output.shape}")

            return combined_output

        except ValueError as ve:
            error_message = f"combine_heads giriş doğrulama hatası: {str(ve)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)

        except Exception as e:
            error_message = f"combine_heads sırasında beklenmeyen hata: {str(e)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)



    def validate_inputs(self, x, mask=None):
        """
        Giriş tensörlerini doğrular.

        Args:
            x (torch.Tensor): Girdi tensörü (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): Maske tensörü (batch_size, seq_len).

        Raises:
            RuntimeError: Eğer tensör boyutları, türleri veya değerleri geçerli değilse.
        """
        try:
            # 1. Gömme boyut kontrolü
            if x.size(-1) != self.embed_dim:
                raise ValueError(f"Gömme boyutu {x.size(-1)} beklenen {self.embed_dim} değil.")

            # 2. Mask kontrolü (varsa)
            if mask is not None:
                # Maske tipi kontrolü
                if mask.dtype not in [torch.float32, torch.bool]:
                    raise ValueError(f"Maskenin tipi 'float32' veya 'bool' olmalıdır, ancak {mask.dtype} bulundu.")

                # Maske boyut kontrolü
                if mask.dim() not in [2, 4]:
                    raise ValueError(f"Maske boyutu {mask.dim()} geçerli değil. Desteklenen boyutlar: [2, 4].")

                # Batch boyutu eşleşme kontrolü
                if mask.size(0) != x.size(0):
                    raise ValueError(f"Maskenin batch boyutu ({mask.size(0)}), giriş tensörünün batch boyutuyla ({x.size(0)}) eşleşmiyor.")

                # Sekans boyutu kontrolü
                if mask.dim() == 2 and mask.size(1) != x.size(1):
                    raise ValueError(f"Maskenin sekans boyutu ({mask.size(1)}), giriş tensörünün sekans boyutuyla ({x.size(1)}) eşleşmiyor.")

                # 4 boyutlu maske için sekans uzunluğu kontrolü
                if mask.dim() == 4 and mask.size(-1) != x.size(1):
                    raise ValueError(f"Maskenin son boyutu ({mask.size(-1)}), giriş tensörünün sekans boyutuyla ({x.size(1)}) eşleşmiyor.")

            # 3. Tensör tür kontrolü
            if not torch.is_floating_point(x):
                raise ValueError(f"Giriş tensörü {x.dtype} türünde. Beklenen tür: float32 veya float64.")

            # 4. Normalizasyon türü kontrolü
            valid_norm_types = ["layer_norm", "batch_norm", "group_norm", "instance_norm"]
            if self.normalization_type not in valid_norm_types:
                raise ValueError(f"Geçersiz normalizasyon tipi: {self.normalization_type}. Desteklenen tipler: {valid_norm_types}")

            # Debug log
            if self.debug:
                print(f"validate_inputs - Giriş doğrulama başarılı. x boyutları: {x.shape}, "
                    f"mask boyutları: {mask.shape if mask is not None else None}")

        except ValueError as ve:
            error_message = f"validate_inputs giriş doğrulama hatası: {str(ve)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)

        except Exception as e:
            error_message = f"validate_inputs sırasında beklenmeyen hata: {str(e)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)



    def extra_repr(self):
        """
        Sınıfın özet bilgisini döndürür.

        Returns:
            str: Sınıf özet metni.
        """
        try:
            # Özet bilgi dizesi
            summary = (
                f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
                f"dropout={self.attn_dropout.p}, debug={self.debug}, "
                f"normalization_type={self.normalization_type}, "
                f"num_groups={self.num_groups}, eps={self.eps}"
            )

            # Ek doğrulama veya bilgi eklenebilir
            if self.debug:
                print(f"extra_repr çağrıldı. Özet bilgi: {summary}")

            return summary

        except Exception as e:
            error_message = f"extra_repr sırasında beklenmeyen hata: {str(e)}"
            if self.debug:
                print(f"[Error] {error_message}")
            raise RuntimeError(error_message)
