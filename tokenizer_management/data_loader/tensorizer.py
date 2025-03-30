import torch
import logging
from typing import List, Optional
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ============================
#  Sabitler ve Cihaz Kontrolü
# ============================
PAD_TOKEN = 0

# Cihazı otomatik algıla (CUDA veya CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
#  Hata Sınıfları
# ============================
class TensorizerError(Exception):
    """Genel tensorizasyon hatası"""
    pass

class InvalidInputError(TensorizerError):
    """Geçersiz giriş hatası"""
    pass

class DimensionMismatchError(TensorizerError):
    """Boyut uyuşmazlığı hatası"""
    pass

class MemoryError(TensorizerError):
    """CUDA belleği yetersiz hatası"""
    pass


# ============================
#  Tensor Dönüşümü (Tekil)
# ============================
def convert_to_tensor(token_ids: List[int], max_length: int = 128) -> torch.Tensor:
    """
    Bir token ID listesini belirlenen maksimum uzunluğa göre tensor formatına dönüştürür.
    
    Args:
        token_ids (List[int]): Dönüştürülecek token ID listesi.
        max_length (int): Maksimum uzunluk.
    
    Returns:
        torch.Tensor: (max_length,) boyutunda tensor.
    """
    if not isinstance(token_ids, list):
        raise InvalidInputError(f"token_ids tipi 'list' olmalıdır, alınan: {type(token_ids)}")

    if not all(isinstance(t, int) for t in token_ids):
        raise InvalidInputError("Tüm token elemanları integer olmalıdır.")

    if not isinstance(max_length, int) or max_length <= 0:
        raise InvalidInputError(f"max_length pozitif bir tamsayı olmalıdır, alınan: {max_length}")

    try:
        # Liste uzunluğu fazla ise kesiyoruz.
        token_ids = token_ids[:max_length]

        # PyTorch `full` fonksiyonunu kullanarak direkt padding ekleyelim
        tensor = torch.full((max_length,), PAD_TOKEN, dtype=torch.long, device=DEVICE)
        tensor[:len(token_ids)] = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)

        logger.debug(f" Tensorize edilmiş veri: {tensor.tolist()[:10]}...")
        return tensor

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error(f" CUDA bellek hatası: {e}")
            raise MemoryError(f"CUDA bellek hatası: {e}")
        else:
            logger.error(f" Tensorize hatası: {e}", exc_info=True)
            raise TensorizerError(f"Tensorize sırasında hata oluştu: {e}")

    except Exception as e:
        logger.error(f" Bilinmeyen tensorize hatası: {e}", exc_info=True)
        raise TensorizerError(f"Tensorize sırasında bilinmeyen hata oluştu: {e}")


# ============================
#  Tensor Dönüşümü (Batch)
# ============================
def tensorize_batch(batch: List[List[int]], max_length: int = 128) -> torch.Tensor:
    """
    Bir batch içindeki token ID listelerini tensorize eder.
    
    Args:
        batch (List[List[int]]): Token ID listelerinin bulunduğu batch.
        max_length (int): Maksimum uzunluk.

    Returns:
        torch.Tensor: (batch_size, max_length) boyutunda tensor.
    """
    if not isinstance(batch, list):
        raise InvalidInputError(f"batch tipi 'list' olmalıdır, alınan: {type(batch)}")

    if len(batch) == 0:
        raise InvalidInputError("Batch içeriği boş olamaz.")

    try:
        tensor_list = [convert_to_tensor(seq, max_length) for seq in batch]
        batch_tensor = torch.stack(tensor_list, dim=0).to(DEVICE)

        logger.debug(f" Batch tensorize edildi. Boyut: {batch_tensor.size()}")
        return batch_tensor

    except RuntimeError as e:
        if "size mismatch" in str(e):
            logger.error(f" Boyut uyuşmazlığı: {e}")
            raise DimensionMismatchError(f"Boyut uyuşmazlığı: {e}")
        if "CUDA out of memory" in str(e):
            logger.error(f" CUDA bellek hatası: {e}")
            raise MemoryError(f"CUDA bellek hatası: {e}")
        raise

    except Exception as e:
        logger.error(f" Batch tensorize hatası: {e}", exc_info=True)
        raise TensorizerError(f"Batch tensorize sırasında bilinmeyen hata oluştu: {e}")


# ============================
#  Tensorizer Sınıfı
# ============================
class Tensorizer:
    """
    Tensorizer, token ID listelerini tensor formatına dönüştürür.
    
    - Tekli tensor dönüşümü için: `tensorize_text`
    - Batch tensor dönüşümü için: `tensorize_batch_text`
    """

    def __init__(self, max_length: int = 128):
        if not isinstance(max_length, int) or max_length <= 0:
            raise InvalidInputError(f"max_length pozitif bir tamsayı olmalıdır, alınan: {max_length}")
        self.max_length = max_length

    def tensorize_text(self, token_ids: List[int], max_length: Optional[int] = None) -> torch.Tensor:
        """
        Tek bir token ID listesini tensorize eder.
        """
        length = max_length if max_length is not None else self.max_length
        return convert_to_tensor(token_ids, length)

    def tensorize_batch_text(self, batch: List[List[int]], max_length: Optional[int] = None) -> torch.Tensor:
        """
        Bir batch içindeki token ID listelerini tensorize eder.
        """
        length = max_length if max_length is not None else self.max_length
        return tensorize_batch(batch, length)

