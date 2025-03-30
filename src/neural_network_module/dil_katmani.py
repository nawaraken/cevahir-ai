import sys
import os
import torch
import torch.nn as nn
import logging
import math

#  Proje kök dizinini modül yoluna ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

#  src dizinini ekle
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

print(f" PYTHONPATH Güncellendi: {sys.path}")  # Hangi yolların eklendiğini görmek için

from neural_network_module.dil_katmani_module.language_embedding import LanguageEmbedding
from neural_network_module.dil_katmani_module.seq_projection import SeqProjection

class PositionalEncoding(nn.Module):
    """
    Transformer tabanlı modeller için pozisyonel kodlama.
    """
    def __init__(self, embed_dim, max_len=2048):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Batch boyutuna uygun hale getir

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class DilKatmani(nn.Module):
    """
    DilKatmani sınıfı, metin verilerini işlemek için kullanılan dil katmanını tanımlar.
    """
    def __init__(self, vocab_size, embed_dim, seq_proj_dim, embed_init_method="xavier", seq_init_method="xavier", log_level=logging.INFO, dropout=0.1):
        """
        DilKatmani sınıfını başlatır.

        Args:
            vocab_size (int): Kelime dağarcığı boyutu.
            embed_dim (int): Gömme boyutu.
            seq_proj_dim (int): Sekans projeksiyon boyutu.
            embed_init_method (str): Gömme başlatma yöntemi (varsayılan: "xavier").
            seq_init_method (str): Sekans projeksiyon başlatma yöntemi (varsayılan: "xavier").
            log_level (int): Log seviyesi.
            dropout (float): Dropout oranı.
        """
        super(DilKatmani, self).__init__()

        # Logger yapılandırması
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        self.logger.info(f"DilKatmani initializing with vocab_size={vocab_size}, embed_dim={embed_dim}, seq_proj_dim={seq_proj_dim}")

        # Embedding katmanı
        self.language_embedding = LanguageEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            init_method=embed_init_method,
            log_level=log_level
        )

        # Positional Encoding ekleme
        self.positional_encoding = PositionalEncoding(embed_dim)

        # Sekans projeksiyon katmanı
        self.seq_projection = SeqProjection(
            input_dim=embed_dim,
            proj_dim=seq_proj_dim,
            init_method=seq_init_method,
            log_level=log_level
        )

        # Layer Normalization ve Dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _log_tensor_stats(self, tensor, stage_name):
        """Tensor istatistiklerini loglar."""
        if tensor is None:
            self.logger.warning(f"{stage_name}: Tensor is None!")
            return
        try:
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            mean_val = tensor.mean().item()
            std_val = tensor.std().item()
            self.logger.debug(f"{stage_name} Stats -> shape: {tensor.shape}, min: {min_val:.4f}, max: {max_val:.4f}, mean: {mean_val:.4f}, std: {std_val:.4f}")
        except Exception as e:
            self.logger.error(f"Error logging stats for {stage_name}: {e}", exc_info=True)

    def forward(self, x):
        """
        DilKatmani'nin ileri yönlü hesaplamasını gerçekleştirir.
        Adım adım süre ölçümleri, giriş doğrulaması ve detaylı istatistik loglaması yapılır.
        
        Args:
            x (torch.Tensor): Giriş tensörü.
        
        Returns:
            torch.Tensor: Sekans projeksiyon sonrası işlenmiş tensör.
        
        Raises:
            TypeError, ValueError, RuntimeError: Giriş doğrulaması veya hesaplama sırasında oluşan hatalar.
        """
        import time
        t_start = time.time()
        try:
            # Giriş doğrulaması
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Giriş tensörü torch.Tensor olmalıdır, alınan: {type(x)}")
            self.logger.debug(f"[DilKatmani FORWARD] Forward pass başlatıldı. Input shape: {x.shape}")
            
            # 1. Language Embedding (Gömme işlemi)
            t0 = time.time()
            embedded = self.language_embedding(x)
            self._log_tensor_stats(embedded, "After LanguageEmbedding")
            t1 = time.time()

            # 2. Positional Encoding (Pozisyonel kodlama)
            encoded = self.positional_encoding(embedded)
            self._log_tensor_stats(encoded, "After PositionalEncoding")
            t2 = time.time()

            # 3. Normalizasyon ve Dropout
            normalized = self.layer_norm(encoded)
            self._log_tensor_stats(normalized, "After LayerNorm")
            dropped_out = self.dropout(normalized)
            self._log_tensor_stats(dropped_out, "After Dropout")
            t3 = time.time()

            # 4. Sequence Projection (Sekans projeksiyonu)
            projected = self.seq_projection(dropped_out)
            self._log_tensor_stats(projected, "After SeqProjection")
            t4 = time.time()

            total_time = t4 - t_start
            self.logger.debug(
                f"[DilKatmani FORWARD] Zaman Ölçümleri: "
                f"Embedding: {t1 - t0:.4f}s, Positional: {t2 - t1:.4f}s, Norm/Dropout: {t3 - t2:.4f}s, "
                f"Projection: {t4 - t3:.4f}s, Toplam: {total_time:.4f}s"
            )
            return projected

        except Exception as e:
            self.logger.error(f"[DilKatmani FORWARD] Hata oluştu: {e}", exc_info=True)
            raise


