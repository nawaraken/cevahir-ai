import sys
import os
import torch
import torch.nn as nn
import logging

#  Proje kök dizinini modül yoluna ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

#  src dizinini ekle
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

print(f" PYTHONPATH Güncellendi: {sys.path}")  # Hangi yolların eklendiğini görmek için

from neural_network_module.dil_katmani import DilKatmani
from neural_network_module.ortak_katman_module.neural_layer_processor import NeuralLayerProcessor
from neural_network_module.ortak_katman_module.memory_manager import MemoryManager
from neural_network_module.ortak_katman_module.tensor_processing_manager import TensorProcessingManager


class CevahirNeuralNetwork(nn.Module):
    def __init__(self, learning_rate,dropout, vocab_size, embed_dim, seq_proj_dim, num_heads,
                 attention_type="multi_head", normalization_type="layer_norm",
                  log_level=logging.INFO):
        """
        Cevahir sinir ağını başlatır.

        Args:
            vocab_size (int): Kelime hazinesi boyutu.
            embed_dim (int): Gömme boyutu.
            seq_proj_dim (int): Projeksiyon boyutu.
            num_heads (int): Çoklu başlık sayısı.
            attention_type (str): Kullanılan dikkat mekanizması türü.
            normalization_type (str): Kullanılan normalizasyon yöntemi.
            dropout (float): Dropout oranı.
            log_level (int): Log seviyesi.
        """
        super(CevahirNeuralNetwork, self).__init__()

        self.learning_rate=learning_rate
        self.dropout=dropout
        # **Logger yapılandırması**
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # **Modüllerin Tanımlanması**
        self.dil_katmani = DilKatmani(vocab_size, embed_dim, seq_proj_dim)
        self.layer_processor = NeuralLayerProcessor(
            embed_dim=seq_proj_dim,
            num_heads=num_heads,
            attention_type=attention_type,
            normalization_type=normalization_type,
            dropout=self.dropout
        )
        self.memory_manager = MemoryManager()
        self.tensor_processing_manager = TensorProcessingManager(input_dim=seq_proj_dim, output_dim=seq_proj_dim,learning_rate=self.learning_rate)

        # **Çıktıyı vocab_size'a dönüştüren Linear katman**
        self.output_layer = nn.Linear(seq_proj_dim, vocab_size)

        self.logger.info(f"[INIT] Cevahir sinir ağı başlatıldı: vocab_size={vocab_size}, embed_dim={embed_dim}, seq_proj_dim={seq_proj_dim}, num_heads={num_heads}")

    def forward(self, x):
        """
        Cevahir sinir ağının ileri yönlü hesaplama sürecini gerçekleştirir.
        Adım adım süre ölçümleri, giriş doğrulaması ve detaylı loglama ile hata ayıklamayı kolaylaştırır.
        
        Args:
            x (torch.Tensor): Giriş tensörü.
        
        Returns:
            tuple: (final_output, attn_weights) – Çıktı tensörü ve (varsa) dikkat ağırlıkları.
        
        Raises:
            TypeError, ValueError, RuntimeError: Giriş doğrulaması veya hesaplama sırasında oluşan hatalar.
        """
        import time
        t_start = time.time()
        try:
            # 1. Giriş doğrulaması
            if not isinstance(x, torch.Tensor):
                self.logger.error(f"[FORWARD] Hatalı giriş türü: {type(x)}. torch.Tensor bekleniyordu.")
                raise TypeError(f"Beklenen giriş türü torch.Tensor, ancak {type(x)} alındı.")
            self.logger.debug(f"[FORWARD] Giriş Verisi -> shape={x.shape}, dtype={x.dtype}, device={x.device}")
            t_input = time.time()
            
            # 2. Embedding İşlemi (Dil Katmanı)
            try:
                embedded = self.dil_katmani(x)
                if not isinstance(embedded, torch.Tensor):
                    raise TypeError("DilKatmani çıktısı geçerli bir tensör değil!")
                self.logger.debug(f"[FORWARD] Embedding Sonrası -> shape={embedded.shape}, dtype={embedded.dtype}")
                self.logger.debug(
                    f"[FORWARD] Embedding Stats -> min: {embedded.min().item():.4f}, max: {embedded.max().item():.4f}, "
                    f"mean: {embedded.mean().item():.4f}, std: {embedded.std().item():.4f}"
                )
            except Exception as e:
                self.logger.error(f"[FORWARD] Hata, DilKatmani çalıştırılırken: {e}", exc_info=True)
                raise
            t_embedding = time.time()
            
            # 3. Attention & Layer Processing
            try:
                attention_output, attn_weights = self.layer_processor(embedded, key=embedded, value=embedded)
                if not isinstance(attention_output, torch.Tensor):
                    raise TypeError("NeuralLayerProcessor çıktısı geçerli bir tensör değil!")
                self.logger.debug(f"[FORWARD] Attention Sonrası -> shape={attention_output.shape}, dtype={attention_output.dtype}")
                self.logger.debug(
                    f"[FORWARD] Attention Stats -> min: {attention_output.min().item():.4f}, max: {attention_output.max().item():.4f}, "
                    f"mean: {attention_output.mean().item():.4f}, std: {attention_output.std().item():.4f}"
                )
                if attn_weights is not None:
                    self.logger.debug(
                        f"[FORWARD] Attention Weights -> shape={attn_weights.shape}, min: {attn_weights.min().item():.4f}, "
                        f"max: {attn_weights.max().item():.4f}, mean: {attn_weights.mean().item():.4f}, "
                        f"std: {attn_weights.std().item():.4f}"
                    )
                else:
                    self.logger.warning("[FORWARD] Attention Weights is None!")
            except Exception as e:
                self.logger.error(f"[FORWARD] Hata, Attention/Layers çalıştırılırken: {e}", exc_info=True)
                raise
            t_attention = time.time()
            
            # 4. Feed-Forward İşlemi (Projeksiyon)
            try:
                projected_output = self.tensor_processing_manager.project(attention_output)
                if not isinstance(projected_output, torch.Tensor):
                    raise TypeError("TensorProcessingManager çıktısı geçerli bir tensör değil!")
                self.logger.debug(f"[FORWARD] Projeksiyon Sonrası -> shape={projected_output.shape}, dtype={projected_output.dtype}")
                self.logger.debug(
                    f"[FORWARD] Projeksiyon Stats -> min: {projected_output.min().item():.4f}, "
                    f"max: {projected_output.max().item():.4f}, mean: {projected_output.mean().item():.4f}, "
                    f"std: {projected_output.std().item():.4f}"
                )
            except Exception as e:
                self.logger.error(f"[FORWARD] Hata, Projeksiyon işlemi sırasında: {e}", exc_info=True)
                raise
            t_projection = time.time()
            
            # 5. Çıktı Katmanı
            try:
                final_output = self.output_layer(projected_output)
                if not isinstance(final_output, torch.Tensor):
                    raise TypeError("Çıktı katmanı geçerli bir tensör döndürmüyor!")
                self.logger.debug(f"[FORWARD] Çıktı Katmanı Sonrası -> shape={final_output.shape}, dtype={final_output.dtype}")
                self.logger.debug(
                    f"[FORWARD] Final Output Stats -> min: {final_output.min().item():.4f}, "
                    f"max: {final_output.max().item():.4f}, mean: {final_output.mean().item():.4f}, "
                    f"std: {final_output.std().item():.4f}"
                )
            except Exception as e:
                self.logger.error(f"[FORWARD] Hata, Output Layer çalıştırılırken: {e}", exc_info=True)
                raise
            t_output = time.time()
            
            # 6. Bellek Yönetimi (Memory Manager) ile entegrasyon
            try:
                self.logger.debug("[FORWARD] Bellek yönetimine veri kaydediliyor...")
                self.memory_manager.store("final_output", final_output)
                self.memory_manager.store("attention_output", attention_output)
                stored_output = self.memory_manager.retrieve("final_output")
                stored_attention = self.memory_manager.retrieve("attention_output")
                if stored_output is None:
                    raise RuntimeError("MemoryManager final_output'u saklamadı!")
                if stored_attention is None:
                    raise RuntimeError("MemoryManager attention_output'u saklamadı!")
                self.logger.debug(
                    f"[FORWARD] MemoryManager Çıkışları -> final_output shape: {stored_output.shape}, "
                    f"attention_output shape: {stored_attention.shape}"
                )
            except Exception as e:
                self.logger.error(f"[FORWARD] Hata, MemoryManager entegrasyonu sırasında: {e}", exc_info=True)
                raise
            t_memory = time.time()
            
            # 7. Bellekten tekrar çağırma (Reuse Check)
            try:
                reuse_check = self.memory_manager.retrieve("final_output")
                if reuse_check is not None:
                    self.logger.debug("[FORWARD] Bellekten çağrılan veri kullanılıyor.")
                else:
                    self.logger.warning("[FORWARD] Bellekten çağrılan veri eksik! MemoryManager bağlantısı kontrol edilmeli.")
            except Exception as e:
                self.logger.error(f"[FORWARD] Hata, Bellek reuse kontrolü sırasında: {e}", exc_info=True)
                raise
            t_reuse = time.time()
            
            # Loglama: Zaman Ölçümleri
            self.logger.debug(
                f"[FORWARD] Zaman Ölçümleri: Input Validation: {t_input - t_start:.4f}s, "
                f"Embedding: {t_embedding - t_input:.4f}s, Attention: {t_attention - t_embedding:.4f}s, "
                f"Projection: {t_projection - t_attention:.4f}s, Output Layer: {t_output - t_projection:.4f}s, "
                f"Memory Ops: {t_memory - t_output:.4f}s, Reuse Check: {t_reuse - t_memory:.4f}s, "
                f"Toplam: {t_reuse - t_start:.4f}s"
            )
            return final_output, attn_weights

        except Exception as e:
            self.logger.error(f"[FORWARD] Hata oluştu: {e}", exc_info=True)
            raise



