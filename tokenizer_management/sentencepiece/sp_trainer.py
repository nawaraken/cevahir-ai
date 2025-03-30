"""
sp_trainer.py

Bu modül, SentencePiece algoritmasını eğitmek için kullanılan SPTrainer sınıfını içerir.
SPTrainer, verilen bir corpus üzerinden basitleştirilmiş bir yöntemle (örneğin,
kelime frekanslarına dayalı) alt sözcük vocab'unu oluşturur veya günceller.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class SPTrainer:
    """
    SPTrainer, SentencePiece tarzı alt sözcük (subword) modelini eğitir.
    Bu örnekte, eğitim algoritması basit bir kelime frekansı hesabına dayanmaktadır.
    Gerçek dünyada, SentencePiece algoritması daha karmaşık optimizasyon yöntemleri içerir.
    """
    def __init__(self, vocab: Dict[str, dict]):
        """
        SPTrainer sınıfı başlatılırken mevcut vocab (varsayılan veya önceden oluşturulmuş)
        sözlüğü ile başlar. Bu vocab, eğitim sürecinde güncellenecek ve genişletilecektir.

        Args:
            vocab (Dict[str, dict]): Başlangıç vocab sözlüğü.
        """
        if not isinstance(vocab, dict):
            raise ValueError("Vocab bir sözlük olmalıdır.")
        self.vocab = vocab.copy()
        logger.info("SPTrainer başarıyla başlatıldı. Başlangıç vocab boyutu: %d", len(self.vocab))

    def train(self, corpus: List[str], target_vocab_size: int):
        """
        Basitleştirilmiş bir SentencePiece eğitimi gerçekleştirir.
        Bu metod, corpus içindeki kelimeleri (whitespace tabanlı tokenize edilerek)
        frekanslarına göre sıralar ve mevcut vocab'a ekler. Hedef vocab boyutuna
        ulaşılıncaya kadar yeni kelimeler eklenir.

        Args:
            corpus (List[str]): Eğitim verisi, metin cümleleri listesi.
            target_vocab_size (int): Oluşturulması istenen toplam vocab boyutu.
        """
        if not corpus or target_vocab_size <= 0:
            logger.warning("Corpus boş veya hedef vocab boyutu geçersiz; eğitim iptal ediliyor.")
            return

        # Basitçe kelime frekansı hesaplaması yapılıyor.
        word_freq = {}
        for sentence in corpus:
            words = sentence.strip().split()
            for word in words:
                word_lower = word.lower()
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1

        logger.debug("Corpus içerisindeki kelime frekansları hesaplandı: %s", word_freq)

        # Vocab'da hali hazırda bulunan tokenler hariç, frekansa göre sıralanmış kelimeler eklenir.
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        current_vocab_size = len(self.vocab)
        for word, freq in sorted_words:
            if current_vocab_size >= target_vocab_size:
                break
            if word not in self.vocab:
                self.vocab[word] = {"id": current_vocab_size, "total_freq": freq, "positions": []}
                current_vocab_size += 1
                logger.debug("Yeni token eklendi: %s (Frekans: %d)", word, freq)

        logger.info("Eğitim tamamlandı. Son vocab boyutu: %d", len(self.vocab))

    def get_vocab(self) -> Dict[str, dict]:
        """
        Mevcut vocab yapısını döner.

        Returns:
            Dict[str, dict]: Eğitim sonrası güncellenmiş vocab sözlüğü.
        """
        return self.vocab

    def set_vocab(self, new_vocab: Dict[str, dict]) -> None:
        """
        Dışarıdan yeni bir vocab sözlüğü alır, mevcut vocab'ı günceller.
        Bu metod, SPTrainer'ın vocab yapısını dışarıdan güncelleme
        işlemlerinde kullanılmak üzere eklenmiştir.

        Args:
            new_vocab (Dict[str, dict]): Güncellenecek yeni vocab sözlüğü.
        """
        if not isinstance(new_vocab, dict):
            raise ValueError("Vocab bir sözlük olmalıdır.")
        self.vocab = new_vocab.copy()
        logger.info("[+] SPTrainer vocab başarıyla güncellendi. Yeni vocab boyutu: %d", len(self.vocab))
