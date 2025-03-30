import torch
import torch.nn as nn
import logging

class CrossAttention(nn.Module):
    """
    Katmanlar arası dikkat mekanizması sınıfı.
    Farklı katmanlardan gelen girişleri birleştirerek dikkati uygular.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.3, attention_scaling=True, 
                normalization_type="layer_norm", scaling_strategy="sqrt", debug=False):
        """
        CrossAttention sınıfını başlatır.

        Args:
            embed_dim (int): Giriş tensörlerinin gömme boyutu.
            num_heads (int): Çok başlıklı dikkat mekanizmasındaki başlık sayısı.
            dropout (float): Dikkat mekanizmasında dropout oranı.
            attention_scaling (bool): Dikkat skorlarını ölçeklendirme seçeneği.
            normalization_type (str): Normalizasyon türü ('layer_norm', 'batch_norm', 'instance_norm', 'group_norm').
            scaling_strategy (str): Ölçeklendirme stratejisi ('sqrt', 'linear', 'none').
            debug (bool): Hata ayıklama modu.
        """
        super(CrossAttention, self).__init__()

        # Parametre doğrulama
        if embed_dim <= 0:
            raise ValueError(f"Embed_dim ({embed_dim}) sıfırdan büyük olmalıdır.")
        if num_heads <= 0:
            raise ValueError(f"Num_heads ({num_heads}) sıfırdan büyük olmalıdır.")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embed_dim ({embed_dim}) num_heads ({num_heads}) ile tam bölünemiyor. "
                f"Embed_dim, num_heads ile tam bölünebilmelidir."
            )
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"Dropout oranı ({dropout}) 0 ile 1 arasında olmalıdır.")
        if normalization_type not in ["layer_norm", "batch_norm", "instance_norm", "group_norm"]:
            raise ValueError(
                f"Geçersiz normalizasyon türü: {normalization_type}. "
                f"Geçerli türler: 'layer_norm', 'batch_norm', 'instance_norm', 'group_norm'."
            )
        if scaling_strategy not in ["sqrt", "linear", "none"]:
            raise ValueError(
                f"Geçersiz ölçeklendirme stratejisi: {scaling_strategy}. "
                f"Geçerli türler: 'sqrt', 'linear', 'none'."
            )

        # Parametrelerin atanması
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_scaling = attention_scaling
        self.normalization_type = normalization_type
        self.scaling_strategy = scaling_strategy
        self.debug = debug

        # Çok başlıklı dikkat mekanizması
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Projeksiyon ve dropout
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.final_dropout = nn.Dropout(dropout)

        # Dinamik normalizasyon
        if normalization_type == "layer_norm":
            self.norm = nn.LayerNorm(embed_dim)
        elif normalization_type == "batch_norm":
            self.norm = nn.BatchNorm1d(embed_dim)
        elif normalization_type == "instance_norm":
            self.norm = nn.InstanceNorm1d(embed_dim)
        elif normalization_type == "group_norm":
            self.norm = nn.GroupNorm(num_groups=max(1, embed_dim // self.head_dim), num_channels=embed_dim)

        # Logger initialization
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Hata ayıklama loglama
        if debug:
            self.logger.debug("[DEBUG] CrossAttention başlatıldı:")
            self.logger.debug(f"  Embed_dim: {embed_dim}, Num_heads: {num_heads}, Dropout: {dropout}")
            self.logger.debug(f"  Normalization_type: {normalization_type}, Scaling_strategy: {scaling_strategy}")
            self.logger.debug(f"  Attention_scaling: {attention_scaling}, Debug: {debug}")

        # Ek hata yakalama mekanizması
        try:
            self.validate_internal_config()
        except ValueError as e:
            raise ValueError(f"[CONFIG VALIDATION ERROR] {e}")

    def validate_internal_config(self):
        """
        Dahili yapılandırmaları doğrular.

        Raises:
            ValueError: Eğer dahili yapılandırmalarda bir uyumsuzluk varsa.
        """
        if self.head_dim <= 0:
            raise ValueError(
                f"Head_dim ({self.head_dim}) sıfırdan büyük olmalıdır. Embed_dim ({self.embed_dim}) ve "
                f"Num_heads ({self.num_heads}) değerlerini kontrol edin."
            )

    def forward(self, query, key, value, key_padding_mask=None, attention_mask=None):
        """
        Katmanlar arası dikkat mekanizmasını uygular.

        Args:
            query (torch.Tensor): Sorgu tensörü (batch_size, seq_len, embed_dim).
            key (torch.Tensor): Anahtar tensörü (batch_size, seq_len, embed_dim).
            value (torch.Tensor): Değer tensörü (batch_size, seq_len, embed_dim).
            key_padding_mask (torch.Tensor, optional): Maske tensörü (batch_size, seq_len).
            attention_mask (torch.Tensor, optional): Dikkat maskesi (seq_len, seq_len).

        Returns:
            torch.Tensor: Katmanlar arası dikkat çıktısı (batch_size, seq_len, embed_dim).
            torch.Tensor: Dikkat ağırlıkları.
        """
        # 1. Girdi Doğrulama
        try:
            self.validate_inputs(query, key, value)
        except ValueError as e:
            raise ValueError(f"[INPUT VALIDATION ERROR] {e}")

        # 2. Hata Ayıklama: Tensör Boyut ve Cihaz Bilgisi
        if self.debug:
            print(f"[DEBUG] Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")
            print(f"[DEBUG] Query device: {query.device}, Key device: {key.device}, Value device: {value.device}")

        # 3. Maske İşleme
        if key_padding_mask is not None:
            self.logger.debug(f"Original key_padding_mask shape: {key_padding_mask.shape}")
            if key_padding_mask.dim() == 3:
                key_padding_mask = key_padding_mask.squeeze(1)
            elif key_padding_mask.dim() == 4:
                key_padding_mask = key_padding_mask.squeeze(1).squeeze(1)
            self.logger.debug(f"Squeezed key_padding_mask shape: {key_padding_mask.shape}")
            if key_padding_mask.dim() != 2:
                raise ValueError(f"[ERROR] key_padding_mask must be 2D, but got {key_padding_mask.dim()}D tensor instead.")

        key_padding_mask, attention_mask = self.process_attention_masks(key_padding_mask, attention_mask, query.size(1))
        if self.debug:
            print(f"[DEBUG] Key Padding Mask: {key_padding_mask}")
            print(f"[DEBUG] Attention Mask: {attention_mask}")

        # 4. Çok Başlıklı Dikkat Mekanizması
        try:
            attn_output, attn_weights = self.multihead_attn(
                query, key, value,
                key_padding_mask=key_padding_mask,
                attn_mask=attention_mask
            )
        except Exception as e:
            raise RuntimeError(f"[ATTENTION MECHANISM ERROR] Dikkat mekanizması sırasında hata: {e}")

        # 5. Hata Ayıklama: Dikkat Ağırlıkları
        if self.debug:
            print(f"[DEBUG] Attention Weights - Min: {attn_weights.min().item()}, Max: {attn_weights.max().item()}")

        # 6. Projeksiyon ve Dropout İşlemleri
        try:
            attn_output = self.final_dropout(self.output_proj(attn_output))
        except Exception as e:
            raise RuntimeError(f"[PROJECTION ERROR] Projeksiyon ve dropout işlemleri sırasında hata: {e}")

        # 7. Residual Bağlantı ve Normalizasyon
        try:
            residual_output = attn_output + query
            normalized_output = self.norm(residual_output)
        except Exception as e:
            raise RuntimeError(f"[NORMALIZATION ERROR] Residual bağlantı veya normalizasyon sırasında hata: {e}")

        # 8. Dikkat Skorlarını Ölçeklendirme (Opsiyonel)
        if self.attention_scaling:
            scaling_factor = self.calculate_scaling_factor()
            normalized_output = normalized_output / scaling_factor
            if self.debug:
                print(f"[DEBUG] Scaling Factor: {scaling_factor}")

        # 9. NaN veya Sonsuz Değerleri Temizleme
        normalized_output = torch.nan_to_num(normalized_output, nan=0.0, posinf=1e9, neginf=-1e9)

        # 10. Hata Ayıklama: Çıkış Tensörü Kontrolü
        if self.debug:
            self.check_tensor_values(normalized_output)
            print(f"[DEBUG] Normalized Output Shape: {normalized_output.shape}")

        # 11. Sonuç Döndürme
        return normalized_output, attn_weights

    def process_attention_masks(self, key_padding_mask, attention_mask, seq_len):
        if key_padding_mask is not None:
            # Eğer key_padding_mask 4D ise, 2D'ye sıkıştır
            if key_padding_mask.dim() == 4:
                key_padding_mask = key_padding_mask.squeeze(1).squeeze(1)
            # Eğer key_padding_mask 3D ise, 2D'ye sıkıştır
            elif key_padding_mask.dim() == 3:
                key_padding_mask = key_padding_mask.squeeze(1)
            # key_padding_mask'in 2D olduğundan emin ol
            if key_padding_mask.dim() != 2:
                raise ValueError(f"key_padding_mask must be 2D, but got {key_padding_mask.dim()}D tensor instead.")
        
        if attention_mask is not None:
            attention_mask = attention_mask.expand(seq_len, seq_len)

        return key_padding_mask, attention_mask


    def calculate_scaling_factor(self):
        if self.scaling_strategy == "sqrt":
            return torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32, device=self.norm.weight.device))
        elif self.scaling_strategy == "linear":
            return torch.tensor(self.embed_dim, dtype=torch.float32, device=self.norm.weight.device)
        elif self.scaling_strategy == "none":
            return torch.tensor(1.0, dtype=torch.float32, device=self.norm.weight.device)
        else:
            raise ValueError(f"Unsupported scaling strategy: {self.scaling_strategy}")


    def validate_inputs(self, query, key, value):
        """
        Girdi tensörlerinin boyutlarını, cihazlarını ve özelliklerini doğrular.

        Args:
            query (torch.Tensor): Sorgu tensörü.
            key (torch.Tensor): Anahtar tensörü.
            value (torch.Tensor): Değer tensörü.

        Raises:
            ValueError: Eğer tensör boyutları, cihazları veya türleri uyumsuzsa.
        """
        # Gömme boyutu kontrolü
        if query.size(-1) != self.embed_dim:
            raise ValueError(
                f"[INPUT ERROR] Query embed_dim uyumsuz: {query.size(-1)} != {self.embed_dim}. "
                f"Query tensörünün son boyutu embed_dim ile eşleşmelidir."
            )
        if key.size(-1) != self.embed_dim:
            raise ValueError(
                f"[INPUT ERROR] Key embed_dim uyumsuz: {key.size(-1)} != {self.embed_dim}. "
                f"Key tensörünün son boyutu embed_dim ile eşleşmelidir."
            )
        if value.size(-1) != self.embed_dim:
            raise ValueError(
                f"[INPUT ERROR] Value embed_dim uyumsuz: {value.size(-1)} != {self.embed_dim}. "
                f"Value tensörünün son boyutu embed_dim ile eşleşmelidir."
            )

        # Batch boyutu kontrolü
        if query.size(0) != key.size(0) or key.size(0) != value.size(0):
            raise ValueError(
                f"[INPUT ERROR] Batch boyutları eşleşmiyor: Query={query.size(0)}, "
                f"Key={key.size(0)}, Value={value.size(0)}. "
                f"Tüm tensörlerin batch boyutları eşleşmelidir."
            )

        # Sekans boyutu kontrolü
        if query.size(1) != key.size(1):
            raise ValueError(
                f"[INPUT ERROR] Sekans boyutları eşleşmiyor: Query={query.size(1)}, "
                f"Key={key.size(1)}. Query ve Key tensörlerinin sekans boyutları aynı olmalıdır."
            )

        # Cihaz kontrolü
        if query.device != key.device or key.device != value.device:
            raise ValueError(
                f"[INPUT ERROR] Tensör cihazları uyumsuz: Query={query.device}, "
                f"Key={key.device}, Value={value.device}. "
                f"Tüm tensörler aynı cihazda olmalıdır."
            )

        # Tür kontrolü
        if query.dtype != key.dtype or key.dtype != value.dtype:
            raise ValueError(
                f"[INPUT ERROR] Tensör türleri uyumsuz: Query={query.dtype}, "
                f"Key={key.dtype}, Value={value.dtype}. "
                f"Tüm tensörler aynı türde olmalıdır."
            )

        # Hata Ayıklama: Loglama
        if self.debug:
            print("[DEBUG] Girdi tensörlerinin detayları:")
            print(f"[DEBUG] Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")
            print(f"[DEBUG] Query device: {query.device}, Key device: {key.device}, Value device: {value.device}")
            print(f"[DEBUG] Query dtype: {query.dtype}, Key dtype: {key.dtype}, Value dtype: {value.dtype}")


    def check_tensor_values(self, *tensors):
        """
        Girdi tensörlerinde NaN veya sonsuz değerleri kontrol eder ve hata durumunda loglama yapar.

        Args:
            *tensors (torch.Tensor): Kontrol edilecek tensörler.

        Raises:
            ValueError: Eğer tensörlerde NaN veya sonsuz değer bulunursa.
        """
        for idx, tensor in enumerate(tensors):
            # Hata ayıklama: Tensör özelliklerini loglama
            if self.debug:
                print(f"[DEBUG] Tensor {idx} -> Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}")
            
            # NaN kontrolü
            if torch.isnan(tensor).any():
                nan_indices = torch.nonzero(torch.isnan(tensor), as_tuple=True)
                raise ValueError(
                    f"[INPUT ERROR] Tensor {idx} içinde NaN değer bulundu. "
                    f"İlk NaN konumu: {nan_indices}. "
                    f"Tensör özellikleri -> Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}."
                )

            # Sonsuz değer kontrolü
            if torch.isinf(tensor).any():
                inf_indices = torch.nonzero(torch.isinf(tensor), as_tuple=True)
                raise ValueError(
                    f"[INPUT ERROR] Tensor {idx} içinde sonsuz değer bulundu. "
                    f"İlk sonsuz konumu: {inf_indices}. "
                    f"Tensör özellikleri -> Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}."
                )
            
            # Loglama: Tensör değer aralıkları
            if self.debug:
                print(
                    f"[DEBUG] Tensor {idx} -> Min: {tensor.min().item()}, Max: {tensor.max().item()}, "
                    f"Mean: {tensor.mean().item()}, Std: {tensor.std().item()}"
                )



    def extra_repr(self):
        """
        Sınıfın özet bilgilerini verir.

        Returns:
            str: Sınıf özet metni.
        """
        repr_str = (
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"dropout={getattr(self.final_dropout, 'p', 'N/A')}, "  # Güncellendi
            f"attention_scaling={self.attention_scaling}, "
            f"normalization_type={self.normalization_type}, "
            f"scaling_strategy={self.scaling_strategy}, "
            f"debug={self.debug}"
        )
        return repr_str
