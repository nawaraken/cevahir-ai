import logging
from typing import List, Dict, Tuple
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)

class BPETrainingError(Exception):
    pass

class BPETrainer:
    def __init__(self, vocab: Dict[str, dict]):
        """
        BPETrainer sınıfı, Byte Pair Encoding algoritmasını kullanarak eğitim gerçekleştirir.
        
        Args:
            vocab (Dict[str, dict]): Başlangıç vocab yapısı dışarıdan verilir.
                                     (Dosya işlemleri dışarıda yönetilmelidir.)
        """
        if not isinstance(vocab, dict):
            logger.error("[X] Başlangıç vocab'ı sözlük tipinde değil.")
            raise TypeError("Vocab bir sözlük olmalıdır.")
        
        # Dışarıdaki sözlüğü değiştirmemek için kopyalayarak kullanıyoruz.
        self.vocab = copy.deepcopy(vocab)

        if not self.vocab:
            logger.warning("[!] Başlangıç vocab boş. Varsayılan vocab yükleniyor.")
            self.vocab = self._default_vocab()

        logger.info("[+] BPETrainer başarıyla başlatıldı.")

    def _default_vocab(self) -> Dict[str, dict]:
        """Varsayılan (özel) tokenları içeren vocab'u döner."""
        default_vocab = {
            "<PAD>": {"id": 0, "total_freq": 1, "positions": []},
            "<UNK>": {"id": 1, "total_freq": 1, "positions": []},
            "<BOS>": {"id": 2, "total_freq": 1, "positions": []},
            "<EOS>": {"id": 3, "total_freq": 1, "positions": []},
        }
        logger.info("[+] Varsayılan vocab yüklendi.")
        return default_vocab

    def train(self, corpus: List[str], target_merges: int, max_iter: int = 1000):
        """
        BPE algoritması ile eğitim gerçekleştirir ve vocab'a yeni tokenlar ekler.

        Args:
            corpus (List[str]): Eğitim verisi (metin listesi)
            target_merges (int): Hedeflenen birleştirme sayısı
            max_iter (int): Maksimum iterasyon sayısı
        """
        if not corpus or not isinstance(corpus, list) or not all(isinstance(text, str) for text in corpus):
            logger.error("[X] Corpus boş veya geçersiz formatta.")
            raise ValueError("Corpus boş veya geçersiz formatta.")

        if not isinstance(target_merges, int) or target_merges <= 0:
            logger.error("[X] target_merges değeri geçersiz.")
            raise ValueError("target_merges pozitif bir tam sayı olmalıdır.")

        if not isinstance(max_iter, int) or max_iter <= 0:
            logger.error("[X] max_iter değeri geçersiz.")
            raise ValueError("max_iter pozitif bir tam sayı olmalıdır.")

        sequence = self._build_sequence(corpus)
        if not sequence:
            logger.warning("[!] Oluşturulan dizi boş, işlem yapılmayacak.")
            return

        merge_count = 0
        logger.info(f"[+] Eğitim başlatıldı. Hedef merge sayısı: {target_merges}")

        while merge_count < target_merges and merge_count < max_iter:
            pairs = self._get_stats(sequence)
            if not pairs:
                logger.warning("[!] Birleştirilecek çift bulunamadı. Eğitim durduruluyor.")
                break

            best_pair = max(pairs, key=pairs.get)
            new_token = "".join(best_pair)

            if new_token not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[new_token] = {
                    "id": new_id,
                    "total_freq": pairs[best_pair],
                    "positions": [idx for idx in range(len(sequence)-1) if (sequence[idx], sequence[idx+1]) == best_pair]
                }

            sequence = self._merge_pair(sequence, best_pair)
            merge_count += 1

            logger.debug(f"[Iter {merge_count}] Merged {best_pair} -> {new_token} (freq: {pairs[best_pair]})")

        logger.info(f"[+] Eğitim tamamlandı. Toplam {merge_count} merge gerçekleştirildi.")

    def _build_sequence(self, corpus: List[str]) -> List[str]:
        sequence = []
        for text in corpus:
            sequence.extend(list(text))
            sequence.append(" ")
        if sequence and sequence[-1] == " ":
            sequence.pop()
        return sequence

    def _get_stats(self, sequence: List[str]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for i in range(len(sequence) - 1):
            pairs[(sequence[i], sequence[i + 1])] += 1
        return pairs

    def _merge_pair(self, sequence: List[str], pair: Tuple[str, str]) -> List[str]:
        new_sequence = []
        i = 0
        while i < len(sequence):
            if i < len(sequence)-1 and sequence[i] == pair[0] and sequence[i+1] == pair[1]:
                new_sequence.append("".join(pair))
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1
        return new_sequence

    def reset(self):
        """Vocab'ı sıfırlar ve varsayılan değerlerle başlatır."""
        self.vocab = self._default_vocab()
        logger.info("[+] BPETrainer sıfırlandı.")

    def get_vocab(self) -> Dict[str, dict]:
        """Mevcut vocab yapısını döner."""
        if not self.vocab:
            logger.error("[X] Vocab boş, lütfen reset veya train işlemi yapın.")
            raise BPETrainingError("Vocab boş, lütfen reset veya train işlemi yapın.")

        if any(not isinstance(v, dict) for v in self.vocab.values()):
            logger.error("[X] Vocab içindeki token yapısı geçersiz.")
            raise BPETrainingError("Vocab içindeki token yapısı geçersiz.")

        return copy.deepcopy(self.vocab)

    def update_vocab(self, new_tokens: List[str]):
        """Yeni tokenları mevcut vocab'a ekler."""
        for token in new_tokens:
            if token not in self.vocab:
                self.vocab[token] = {"id": len(self.vocab), "total_freq": 1, "positions": []}
        logger.info("[+] Vocab güncellendi.")
