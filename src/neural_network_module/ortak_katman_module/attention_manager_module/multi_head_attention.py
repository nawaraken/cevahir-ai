import torch.nn as nn
import torch
import torch.nn.functional as F
import time
import math
from training_management.training_logger import TrainingLogger
logger = TrainingLogger()

class MultiHeadAttention(nn.Module):
    """
    Çok başlıklı dikkat mekanizması sınıfı.
    Giriş tensörleri üzerinde farklı başlıklar altında dikkat hesaplaması yapar.
    """

    def __init__(self, embed_dim, num_heads, dropout, normalization_type="layer_norm", debug=False):
        """
        MultiHeadAttention sınıfını başlatır.

        Args:
            embed_dim (int): Giriş tensörlerinin gömme boyutu.
            num_heads (int): Dikkat başlığı sayısı.
            dropout (float): Tüm katmanlarda ortak kullanılacak Dropout oranı.
            normalization_type (str): Normalizasyon türü ('layer_norm', 'batch_norm', 'instance_norm', 'group_norm').
            debug (bool): Hata ayıklama modu.
        """
        super(MultiHeadAttention, self).__init__()

        # **Kontroller: embed_dim num_heads'e tam bölünebilir mi?**
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Gömme boyutu ({embed_dim}) çok başlık sayısına ({num_heads}) tam bölünemiyor. "
                f"embed_dim, num_heads ile tam bölünebilecek bir değer olmalıdır."
            )
        # Global logger'ı kullanarak örnek seviyesinde logger oluştur
        self.logger = logger  # Global logger'ın tanımlı olduğu varsayılıyor
        self.logger.debug("MultiHeadAttention __init__ çağrıldı.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.debug = debug

        # **Dropout**
        self.dropout_rate = dropout  # Float değerini kaydediyoruz
        self.dropout = nn.Dropout(self.dropout_rate)  # Dropout modülü

        # **Projeksiyon katmanları**
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # **Normalizasyon katmanı**
        if normalization_type == "batch_norm":
            self.norm = nn.BatchNorm1d(embed_dim)
        elif normalization_type == "layer_norm":
            self.norm = nn.LayerNorm(embed_dim)
        elif normalization_type == "instance_norm":
            self.norm = nn.InstanceNorm1d(embed_dim)
        elif normalization_type == "group_norm":
            num_groups = max(1, min(embed_dim // self.head_dim, 32))  # Dinamik grup sayısı hesaplama
            self.norm = nn.GroupNorm(num_groups, embed_dim)
        else:
            raise ValueError(
                f"Desteklenmeyen normalizasyon türü: {normalization_type}. "
                f"Geçerli seçenekler: 'layer_norm', 'batch_norm', 'instance_norm', 'group_norm'."
            )

        # **Hata ayıklama (Debug Modu)**
        if self.debug:
            print(f"MultiHeadAttention Başlatıldı: embed_dim={embed_dim}, num_heads={num_heads}, head_dim={self.head_dim}")
            print(f"Normalizasyon türü: {normalization_type}")
            print(f"Dropout oranı: {self.dropout_rate}")


    def scaled_dot_product_attention(self, query, key, value, mask=None, temperature=1.0, apply_dropout=True):
        """
        Ölçeklenmiş nokta çarpımı dikkat mekanizmasını uygular.
        
        Args:
            query (torch.Tensor): Sorgu tensörü (B, H, T, D).
            key (torch.Tensor): Anahtar tensörü (B, H, T, D).
            value (torch.Tensor): Değer tensörü (B, H, T, D).
            mask (torch.Tensor, optional): Maske tensörü (broadcast edilebilir boyutlarda).
            temperature (float): Softmax sıcaklık faktörü.
            apply_dropout (bool): Dropout'un uygulanıp uygulanmayacağını belirler (varsayılan: True).
        
        Returns:
            tuple: (Çıkış, Dikkat ağırlıkları)
        """
        start_time = time.time()
        
        # Giriş istatistikleri (debug modunda)
        if self.debug:
            print(f"[DEBUG] Query: min={query.min().item():.6f}, max={query.max().item():.6f}, mean={query.mean().item():.6f}")
            print(f"[DEBUG] Key:   min={key.min().item():.6f}, max={key.max().item():.6f}, mean={key.mean().item():.6f}")
            print(f"[DEBUG] Value: min={value.min().item():.6f}, max={value.max().item():.6f}, mean={value.mean().item():.6f}")
        
        # Ölçekleme faktörü (sqrt(D))
        d_k = query.size(-1)
        scaling_factor = math.sqrt(d_k)
        
        # Dot-product hesaplama ve ölçekleme
        scores = torch.matmul(query, key.transpose(-2, -1)) / (scaling_factor * temperature)
        if self.debug:
            print(f"[DEBUG] Scaling factor (with temperature): {scaling_factor:.6f}")
        
        # Maske uygulanıyorsa
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            if self.debug:
                print(f"[DEBUG] After mask: min={scores.min().item():.6f}, max={scores.max().item():.6f}")
                print(f"[DEBUG] Mask-applied count: {(scores == -1e9).sum().item()}")
        
        # Log-Sum-Exp stabilizasyonu: Her satırdaki maksimum değeri çıkararak
        max_scores, _ = scores.max(dim=-1, keepdim=True)
        scores = scores - max_scores
        
        # Düşük sıcaklık durumunda (temperature < 0.01) tie-breaker uygulaması:
        if temperature < 0.01:
            # Tüm satırlarda maksimum değere eşit olmayan skorları çok düşük (−1e9) yap.
            scores = torch.where(scores < 0, -1e9 * torch.ones_like(scores), scores)
        
        # NaN/Inf temizliği
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
        if self.debug:
            print(f"[DEBUG] After nan/inf cleanup: min={scores.min().item():.6f}, max={scores.max().item():.6f}")
        
        # Softmax uygulaması
        attn_weights = F.softmax(scores, dim=-1)
        if self.debug:
            print(f"[DEBUG] After softmax (pre-dropout): min={attn_weights.min().item():.6f}, max={attn_weights.max().item():.6f}")
        
        # Opsiyonel olarak dropout (sadece training modunda ve isteniyorsa uygulanır)
        if apply_dropout and self.training:
            attn_weights = self.dropout(attn_weights)
            if self.debug:
                print(f"[DEBUG] After dropout: min={attn_weights.min().item():.6f}, max={attn_weights.max().item():.6f}")
        
        # Debug: final attention weights
        if self.debug:
            print(f"[DEBUG] Final attention weights (post-dropout): min={attn_weights.min().item():.6f}, max={attn_weights.max().item():.6f}")
            if attn_weights.max().item() > 0.9:
                print("[WARNING] Belirli bir token'e aşırı odaklanma tespit edildi!")
            if attn_weights.min().item() < 1e-5:
                print("[WARNING] Dikkat ağırlıklarının bir kısmı tamamen sıfır!")
        
        # Çıkış tensörünü hesapla
        output = torch.matmul(attn_weights, value)
        if self.debug:
            print(f"[DEBUG] Attention output: min={output.min().item():.6f}, max={output.max().item():.6f}, mean={output.mean().item():.6f}")
            print(f"[DEBUG] Zero count in output: {(output == 0).sum().item()}")
            print(f"[DEBUG] Negative count in output: {(output < 0).sum().item()}")
            print(f"[DEBUG] NaN count in output: {(torch.isnan(output)).sum().item()}, Inf count: {(torch.isinf(output)).sum().item()}")
            hist_out = torch.histc(output, bins=10, min=output.min().item(), max=output.max().item())
            print(f"[DEBUG] Output histogram: {hist_out.tolist()}")
            elapsed = time.time() - start_time
            print(f"[DEBUG] Scaled dot-product attention computation time: {elapsed:.6f} seconds")
        
        return output, attn_weights


    def forward(self, query, key, value, mask=None, return_attention_weights=False, apply_dropout=True):
            """
            Uygulanmış Çok Başlıklı Dikkat (MultiHeadAttention) işlemi.

            Args:
                query (torch.Tensor): Giriş tensörü, shape: (batch_size, seq_len, embed_dim).
                key (torch.Tensor): Giriş tensörü, shape: (batch_size, seq_len, embed_dim).
                value (torch.Tensor): Giriş tensörü, shape: (batch_size, seq_len, embed_dim).
                mask (torch.Tensor, optional): Maske tensörü (broadcast edilebilir boyutlarda).
                return_attention_weights (bool): True ise, attention ağırlıkları da döndürülür.
                apply_dropout (bool): Dropout'un uygulanıp uygulanmayacağını belirler (varsayılan: True).

            Returns:
                torch.Tensor veya tuple: Eğer return_attention_weights False ise, sadece çıktı;
                                        aksi halde (çıktı, attention_weights) tuple'ı.
            """
            import time
            t_start = time.time()
            
            # 1. Girdi doğrulaması
            if query is None or key is None or value is None:
                raise ValueError("[ERROR] Giriş tensörleri (query, key, value) None olamaz!")
            if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
                raise ValueError(f"[ERROR] Giriş tensörleri 3 boyutlu olmalıdır! Query: {query.shape}, Key: {key.shape}, Value: {value.shape}")
            if query.size(-1) != self.embed_dim or key.size(-1) != self.embed_dim or value.size(-1) != self.embed_dim:
                raise ValueError(f"[ERROR] Giriş tensörlerinin son boyutu embed_dim ({self.embed_dim}) ile eşleşmiyor! Got: Query {query.size(-1)}, Key {key.size(-1)}, Value {value.size(-1)}")
            
            self.logger.debug(f"[MultiHeadAttention FORWARD] Input shapes: Query: {query.shape}, Key: {key.shape}, Value: {value.shape}")
            if self.debug:
                self.logger.debug(f"[MultiHeadAttention FORWARD] Query stats: min={query.min().item():.6f}, max={query.max().item():.6f}, mean={query.mean().item():.6f}")
                self.logger.debug(f"[MultiHeadAttention FORWARD] Key stats: min={key.min().item():.6f}, max={key.max().item():.6f}, mean={key.mean().item():.6f}")
                self.logger.debug(f"[MultiHeadAttention FORWARD] Value stats: min={value.min().item():.6f}, max={value.max().item():.6f}, mean={value.mean().item():.6f}")

            # 2. Maske kontrolü (varsa)
            if mask is not None:
                if mask.dim() not in [3, 4]:
                    raise ValueError(f"[ERROR] Mask tensörü hatalı boyut: {mask.dim()} (Beklenen: 3 veya 4).")
                self.logger.debug(f"[MultiHeadAttention FORWARD] Mask shape: {mask.shape}")
                if self.debug:
                    self.logger.debug(f"[MultiHeadAttention FORWARD] Mask'deki sıfır eleman sayısı: {(mask == 0).sum().item()}")

            # 3. Projeksiyon işlemleri
            batch_size, seq_len, _ = query.size()

            def apply_projection(tensor, projection_layer, name):
                proj = projection_layer(tensor)
                proj = proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                if self.debug:
                    self.logger.debug(f"[MultiHeadAttention FORWARD] {name} Projection output shape: {proj.shape}")
                    self.logger.debug(f"[MultiHeadAttention FORWARD] {name} stats: min={proj.min().item():.6f}, max={proj.max().item():.6f}, mean={proj.mean().item():.6f}")
                if torch.isnan(proj).any() or torch.isinf(proj).any():
                    raise ValueError(f"[ERROR] {name} projection produced NaN or Inf values!")
                return proj

            t_proj_start = time.time()
            query_proj = apply_projection(query, self.query_proj, "Query")
            key_proj = apply_projection(key, self.key_proj, "Key")
            value_proj = apply_projection(value, self.value_proj, "Value")
            t_proj_end = time.time()
            self.logger.debug(f"[MultiHeadAttention FORWARD] Projection time: {t_proj_end - t_proj_start:.6f}s")

            # 4. Dropout uygulaması (apply_dropout kontrolüne bağlı)
            if apply_dropout:
                query_proj = self.dropout(query_proj)
                key_proj = self.dropout(key_proj)
                value_proj = self.dropout(value_proj)
            if self.debug:
                self.logger.debug(f"[MultiHeadAttention FORWARD] After Dropout Query: min={query_proj.min().item():.6f}, max={query_proj.max().item():.6f}")

            # 5. Ölçeklenmiş Nokta Çarpımı Dikkat
            t_attention_start = time.time()
            # scaled_dot_product_attention metoduna apply_dropout parametresi aktarılıyor.
            attn_output, attn_weights = self.scaled_dot_product_attention(query_proj, key_proj, value_proj, mask, temperature=1.0, apply_dropout=apply_dropout)
            t_attention_end = time.time()
            self.logger.debug(f"[MultiHeadAttention FORWARD] Scaled Dot-Product Attention time: {t_attention_end - t_attention_start:.6f}s")
            if self.debug:
                self.logger.debug(f"[MultiHeadAttention FORWARD] Attention Output shape: {attn_output.shape}")
                self.logger.debug(f"[MultiHeadAttention FORWARD] Attention Weights shape: {attn_weights.shape}")
                self.logger.debug(f"[MultiHeadAttention FORWARD] Attention Weights stats: min={attn_weights.min().item():.6f}, max={attn_weights.max().item():.6f}, mean={attn_weights.mean().item():.6f}")

            # 6. Çıkış Projeksiyonu ve Normalizasyon
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
            final_projection = self.out_proj(attn_output) * (1.0 / (self.embed_dim ** 0.5))
            self.logger.debug(f"[MultiHeadAttention FORWARD] Final Projection stats: min={final_projection.min().item():.6f}, max={final_projection.max().item():.6f}, mean={final_projection.mean().item():.6f}")
            output = self.norm(final_projection + query)
            if self.debug:
                self.logger.debug(f"[MultiHeadAttention FORWARD] Final Output shape: {output.shape}")
                self.logger.debug(f"[MultiHeadAttention FORWARD] Final Output stats: min={output.min().item():.6f}, max={output.max().item():.6f}, mean={output.mean().item():.6f}")

            t_end = time.time()
            self.logger.info(f"[MultiHeadAttention FORWARD] Forward pass completed in {t_end - t_start:.6f} seconds.")

            if not return_attention_weights:
                return output
            else:
                return output, attn_weights