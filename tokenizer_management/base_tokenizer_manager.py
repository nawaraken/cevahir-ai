from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseTokenizerManager(ABC):
    """
    Tüm tokenizer manager sınıflarının uyması gereken temel arayüzdür.
    """

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Girdi metnini token ID'lerine dönüştürür.
        """
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """
        Token ID listesini orijinal metne dönüştürür.
        """
        pass

    @abstractmethod
    def train(self, corpus: List[str], vocab_size: Optional[int] = None) -> None:
        """
        Girdi metinlerinden bir tokenizer modeli veya vocab oluşturur.
        """
        pass

    @abstractmethod
    def get_vocab(self) -> Dict[str, Dict]:
        """
        Mevcut sözlüğü döndürür.
        """
        pass

    @abstractmethod
    def set_vocab(self, vocab: Dict[str, Dict]) -> None:
        """
        Harici bir sözlüğü yükler.
        """
        pass

    @abstractmethod
    def update_reverse_vocab(self) -> None:
        """
        ID'den kelimeye çözümleme için ters sözlük oluşturur/günceller.
        """
        pass
