import torch
import logging

class MemoryAttentionBridge:
    """
    MemoryAttentionBridge sınıfı, bellek ve dikkat mekanizmaları arasında optimize edilmiş bir köprü kurar.
    Dinamik tensör hizalama, maskelerle entegrasyon ve ileri seviye hata yönetimi içerir.
    """
    
    _logger_initialized = False  # Log sistemi sadece bir kez başlatılacak

    def __init__(self, normalize=True, scale_factor=None, log_level=logging.INFO):
        """
        MemoryAttentionBridge sınıfını başlatır.

        Args:
            normalize (bool): Tensörleri normalizasyon işlemine tabi tutar.
            scale_factor (float, optional): Tensörleri ölçeklendirmek için bir faktör belirler.
            log_level (int): Log seviyesi (örn. logging.DEBUG, logging.INFO).
        """
        self.normalize = normalize
        self.scale_factor = scale_factor
        self.logger = logging.getLogger(self.__class__.__name__)

        if not MemoryAttentionBridge._logger_initialized:
            self.logger.setLevel(log_level)
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

            MemoryAttentionBridge._logger_initialized = True  # Log başlatma yalnızca bir kez yapılır.

        self.logger.info(f"MemoryAttentionBridge initialized with normalize={self.normalize}, scale_factor={self.scale_factor}")

    def bridge_attention(self, memory_tensor, attention_tensor, memory_mask=None, attention_mask=None):
        """
        Bellek ve dikkat mekanizmaları arasında dinamik bir köprü kurar.
        Tensörleri ölçeklendirir, hizalar ve maskelerle entegre eder.

        Args:
            memory_tensor (torch.Tensor): Bellek tensörü.
            attention_tensor (torch.Tensor): Dikkat tensörü.
            memory_mask (torch.Tensor, optional): Bellek maskesi tensörü.
            attention_mask (torch.Tensor, optional): Dikkat maskesi tensörü.

        Returns:
            torch.Tensor: Optimize edilmiş köprülenmiş tensör.

        Raises:
            ValueError: Eğer tensör boyutları uyuşmuyorsa.
            RuntimeError: Köprü işlemi sırasında hata oluşursa.
        """
        import time
        start_time = time.time()
        try:
            # 1. Giriş Tensörlerinin Doğrulanması
            self.logger.debug("[bridge_attention] Validating input tensors...")
            self._validate_tensors(memory_tensor, attention_tensor)
            self.logger.debug("[bridge_attention] Input tensors validated successfully.")
            
            # 2. Opsiyonel: Maskelerin doğrulanması
            if memory_mask is not None:
                self.logger.debug("[bridge_attention] Validating memory_mask...")
                self._validate_mask(memory_mask, memory_tensor)
                self.logger.debug("[bridge_attention] memory_mask validated.")
            if attention_mask is not None:
                self.logger.debug("[bridge_attention] Validating attention_mask...")
                self._validate_mask(attention_mask, attention_tensor)
                self.logger.debug("[bridge_attention] attention_mask validated.")

            # 3. Tensörlerin hizalanması
            self.logger.debug("[bridge_attention] Aligning memory and attention tensors...")
            aligned_memory, aligned_attention = self._align_tensors(memory_tensor, attention_tensor)
            self.logger.debug(f"[bridge_attention] Tensors aligned: memory shape={aligned_memory.shape}, attention shape={aligned_attention.shape}")

            # 4. Ölçeklendirme (varsa)
            if self.scale_factor is not None:
                self.logger.debug(f"[bridge_attention] Scaling tensors by factor {self.scale_factor}...")
                aligned_memory, aligned_attention = self._scale_tensors(aligned_memory, aligned_attention)
                self.logger.debug("[bridge_attention] Tensors scaled successfully.")

            # 5. Normalizasyon (varsa)
            if self.normalize:
                self.logger.debug("[bridge_attention] Normalizing tensors...")
                normalized_memory, normalized_attention = self._normalize_tensors(aligned_memory, aligned_attention)
                self.logger.debug("[bridge_attention] Tensors normalized successfully.")
            else:
                normalized_memory, normalized_attention = aligned_memory, aligned_attention

            # 6. Tensörlerin birleştirilmesi (katsayılar/attention köprüsü)
            self.logger.debug("[bridge_attention] Bridging memory and attention tensors...")
            bridged_tensor = torch.cat((normalized_memory, normalized_attention), dim=-1)
            self.logger.debug(f"[bridge_attention] Bridged tensor shape: {bridged_tensor.shape}")

            # 7. Sonuç tensörünün istatistiklerini loglama
            self.logger.info("[bridge_attention] Attention bridge created successfully.")
            self.logger.debug(f"[bridge_attention] Memory tensor shape: {normalized_memory.shape}")
            self.logger.debug(f"[bridge_attention] Attention tensor shape: {normalized_attention.shape}")
            self.logger.debug(f"[bridge_attention] Bridged tensor stats - min: {bridged_tensor.min().item():.6f}, max: {bridged_tensor.max().item():.6f}, mean: {bridged_tensor.mean().item():.6f}, std: {bridged_tensor.std().item():.6f}")

            total_time = time.time() - start_time
            self.logger.info(f"[bridge_attention] Total execution time: {total_time:.6f} seconds.")
            return bridged_tensor

        except ValueError as ve:
            self.logger.error(f"[bridge_attention] ValueError: {ve}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"[bridge_attention] Unexpected error: {e}", exc_info=True)
            raise RuntimeError(f"Failed to bridge attention: {e}")


    def _validate_tensors(self, *tensors):
        """
        Girdi tensörlerini doğrular ve hataları loglar.

        Args:
            *tensors (torch.Tensor): Doğrulanacak tensörler.

        Raises:
            ValueError: Eğer tensörler geçersizse.
        """
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("[ERROR] Input must be a torch.Tensor.")
            if tensor.dim() < 2:
                raise ValueError("[ERROR] Tensor must have at least 2 dimensions.")
            if tensor.isnan().any():
                raise ValueError("[ERROR] Tensor contains NaN values.")
            if tensor.isinf().any():
                raise ValueError("[ERROR] Tensor contains infinite values.")

    def _validate_mask(self, mask, tensor):
        """
        Maskelerin geçerliliğini kontrol eder.

        Args:
            mask (torch.Tensor): Maske tensörü.
            tensor (torch.Tensor): Maske ile aynı boyutta olması gereken tensör.

        Raises:
            ValueError: Eğer maskeler geçersizse.
        """
        if mask.shape != tensor.shape[:-1]:
            raise ValueError(f"[ERROR] Mask shape {mask.shape} does not match tensor shape {tensor.shape[:-1]}")

    def _align_tensors(self, tensor1, tensor2):
        """
        İki tensörü aynı boyuta hizalar.

        Args:
            tensor1 (torch.Tensor): Birinci tensör.
            tensor2 (torch.Tensor): İkinci tensör.

        Returns:
            tuple: (hizalanmış tensor1, hizalanmış tensor2)
        """
        max_dim = max(tensor1.shape[-1], tensor2.shape[-1])

        if tensor1.shape[-1] < max_dim:
            pad_size = max_dim - tensor1.shape[-1]
            tensor1 = torch.nn.functional.pad(tensor1, (0, pad_size))

        if tensor2.shape[-1] < max_dim:
            pad_size = max_dim - tensor2.shape[-1]
            tensor2 = torch.nn.functional.pad(tensor2, (0, pad_size))

        return tensor1, tensor2

    def _scale_tensors(self, tensor1, tensor2):
        """
        Tensörleri ölçeklendirir.

        Args:
            tensor1 (torch.Tensor): Birinci tensör.
            tensor2 (torch.Tensor): İkinci tensör.

        Returns:
            tuple: (ölçeklenmiş tensor1, ölçeklenmiş tensor2)
        """
        return tensor1 * self.scale_factor, tensor2 * self.scale_factor

    def _normalize_tensors(self, tensor1, tensor2):
        """
        Tensörleri normalizasyon işlemine tabi tutar.

        Args:
            tensor1 (torch.Tensor): Birinci tensör.
            tensor2 (torch.Tensor): İkinci tensör.

        Returns:
            tuple: (normalize edilmiş tensor1, normalize edilmiş tensor2)
        """
        return torch.nn.functional.normalize(tensor1, dim=-1), torch.nn.functional.normalize(tensor2, dim=-1)

    def _log_attention_bridge(self, memory_tensor, attention_tensor, bridged_tensor):
        """
        Dikkat köprüsü işlemi sırasında loglama yapar.

        Args:
            memory_tensor (torch.Tensor): Bellek tensörü.
            attention_tensor (torch.Tensor): Dikkat tensörü.
            bridged_tensor (torch.Tensor): Köprülenmiş tensör.
        """
        self.logger.info("[INFO] Attention bridge created.")
        self.logger.debug(f"[DEBUG] Memory tensor shape: {memory_tensor.shape}")
        self.logger.debug(f"[DEBUG] Attention tensor shape: {attention_tensor.shape}")
        self.logger.debug(f"[DEBUG] Bridged tensor shape: {bridged_tensor.shape}")
