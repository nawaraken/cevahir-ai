"""
training_tensorizer.py

Bu modül, eğitim verisi üzerinde postprocessing işleminden geçmiş metinleri (örneğin,
token ID dizileri veya post-processed metinler) PyTorch tensörlerine dönüştürür.
Bu örnekte, basit bir yaklaşım olarak her metindeki karakterlerin ASCII kodlarına dayalı
bir tensör oluşturulmaktadır. Gerçek uygulamalarda, genellikle token ID'leri doğrudan tensöre
dönüştürülür veya embedding işlemi uygulanır.
"""

import logging
from typing import List, Tuple
import torch

logger = logging.getLogger(__name__)


class TrainingTensorizer:
    """
    TrainingTensorizer:
    Token ID dizilerini (input_ids, target_ids) içeren eğitim verisini PyTorch tensörlerine dönüştürür.
    Her iki diziyi de pad'leyerek sabit boyutlu tensörler oluşturur.
    """

    def __init__(self, pad_token_id: int = 0):
        """
        Args:
            pad_token_id (int): Pad için kullanılacak token ID. Varsayılan: 0 (<PAD>)
        """
        self.pad_token_id = pad_token_id
        logger.info("TrainingTensorizer başarıyla başlatıldı.")

    def tensorize_pairs(
        self,
        data: List[Tuple[List[int], List[int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Eğitim verisinde yer alan (input_ids, target_ids) çiftlerini sabit boyutlu tensörlere dönüştürür.

        Args:
            data (List[Tuple[List[int], List[int]]]): Her öğesi (input_ids, target_ids) olan tuple'lardan oluşan liste.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                input_tensor: (batch_size x max_input_len)
                target_tensor: (batch_size x max_target_len)
        """
        try:
            if not data or not isinstance(data, list):
                raise ValueError("Geçersiz eğitim verisi: boş veya uygun formatta değil.")

            input_ids_list, target_ids_list = zip(*data)

            max_input_len = max(len(ids) for ids in input_ids_list)
            max_target_len = max(len(ids) for ids in target_ids_list)

            input_tensor = torch.full((len(data), max_input_len), self.pad_token_id, dtype=torch.long)
            target_tensor = torch.full((len(data), max_target_len), self.pad_token_id, dtype=torch.long)

            for i, (input_ids, target_ids) in enumerate(data):
                input_tensor[i, :len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
                target_tensor[i, :len(target_ids)] = torch.tensor(target_ids, dtype=torch.long)

            logger.info(f"[✓] Tensorize işlemi tamamlandı. Input Tensor: {input_tensor.shape}, Target Tensor: {target_tensor.shape}")
            return input_tensor, target_tensor

        except Exception as e:
            logger.error(f"[X] Tensorize işleminde hata: {e}")
            raise
