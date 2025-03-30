import logging
from typing import List, Dict
from collections import OrderedDict

logger = logging.getLogger(__name__)

class BPEEncodingError(Exception):
    pass

class BPEEncoder:
    def __init__(self, vocab: Dict[str, dict]):
        """
        BPEEncoder sınıfı, BPE kodlaması yapar.
        
        Args:
            vocab (Dict[str, dict]): Doğrudan vocab sözlüğü alınır.
        """
        if not isinstance(vocab, dict):
            raise TypeError("Vocab bir sözlük olmalıdır.")

        self.vocab = vocab
        if not self.vocab:
            raise ValueError("Vocab yüklenemedi veya boş.")

        # Özel token ID'lerini kontrol edelim
        self._validate_special_tokens()

        logger.info("[+] BPEEncoder başlatıldı.")


    def set_vocab(self, new_vocab: Dict[str, dict]) -> None:
        if not isinstance(new_vocab, dict):
            raise TypeError("new_vocab bir sözlük olmalıdır.")
        self.vocab = new_vocab
        logger.info("[+] BPEEncoder vocab güncellendi.")

        
    def _validate_special_tokens(self):
        """
        Vocab içinde özel token ID'lerini kontrol eder.
        """
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        missing_tokens = []
        for token in special_tokens:
            if token not in self.vocab:
                missing_tokens.append(token)
            elif 'id' not in self.vocab[token]:
                raise ValueError(f"{token} için ID bulunamadı.")

        if missing_tokens:
            raise ValueError(f"Vocab içinde eksik özel tokenlar: {missing_tokens}")

        logger.info("[+] Özel token ID'leri doğrulandı.")

    def encode(self, tokens: List[str]) -> List[int]:
        if not isinstance(tokens, list):
            raise TypeError("[X] Giriş tipi 'List[str]' olmalıdır.")
        
        if not tokens:
            raise ValueError("[X] Kodlama işlemi için boş giriş kabul edilemez.")

        token_ids = []

        for idx, token in enumerate(tokens):
            token = token.strip().lower()
            token_id = self.vocab.get(token, {}).get('id')

            if token_id is None:
                # BPE merge kurallarını uygula
                token_id = self._apply_bpe_merge(token)
                if token_id is None:
                    # Bilinmeyen tokenler için UNK ID belirle
                    token_id = self._handle_unknown_token(token)

            token_ids.append(token_id)

            #  Pozisyonları doğrudan güncelleyelim
            if token in self.vocab:
                self.vocab[token]["total_freq"] += 1
                self.vocab[token]["positions"].append(idx)

        # Reverse vocab güncellemesi
        self._update_reverse_vocab()

        # ID çakışmasını çözelim
        token_ids = self._resolve_token_id_conflict(token_ids)

        logger.debug(f"[+] Kodlanmış Token ID'leri: {token_ids}")
        return token_ids



    def _apply_bpe_merge(self, token: str) -> int:
        if len(token) <= 1:
            return None

        pairs = [(token[:i], token[i:]) for i in range(1, len(token))]

        for pair in pairs:
            merged_token = "".join(pair)
            if merged_token in self.vocab:
                #  Pozisyon bilgisini koruyalım
                if "positions" not in self.vocab[merged_token]:
                    self.vocab[merged_token]["positions"] = []

                if len(self.vocab[merged_token]["positions"]) < 100:  # Çok büyük olmaması için limit
                    current_pos = len(self.vocab[merged_token]["positions"])
                    self.vocab[merged_token]["positions"].append(current_pos)

                return self.vocab[merged_token]["id"]

        return None



    def _resolve_token_id_conflict(self, token_ids: List[int]) -> List[int]:
        """
        Token ID'lerindeki çakışmaları çözerek sıralamayı korur.

        Args:
            token_ids (List[int]): Kodlanmış token ID listesi.

        Returns:
            List[int]: Çakışma çözülmüş ID listesi.
        """
        seen = set()
        resolved_token_ids = []

        for token_id in token_ids:
            if token_id not in seen:
                seen.add(token_id)
                resolved_token_ids.append(token_id)
            else:
                # Çakışma varsa yeni bir ID belirle
                new_id = max(seen) + 1
                resolved_token_ids.append(new_id)
                seen.add(new_id)
                logger.warning(f"[!] ID çakışması tespit edildi. Yeni ID atandı: {new_id}")

        return resolved_token_ids




    def _handle_unknown_token(self, token: str) -> int:
        """
        Bilinmeyen token'ı UNK token ile eşleştirir. 
        Eğer UNK token vocab içinde yoksa ekler ve pozisyon bilgisini günceller.

        Args:
            token (str): Bilinmeyen token.

        Returns:
            int: UNK token ID'si.
        """
        # Giriş tipini kontrol et
        if not isinstance(token, str):
            raise TypeError("[X] Bilinmeyen token tipi geçersiz. 'str' türünde olmalıdır.")
        
        # Boş string kontrolü
        if not token.strip():
            raise ValueError("[X] Bilinmeyen token boş olamaz.")
        
        try:
            # 🔎 UNK token ID'sini al
            unk_id = self.vocab.get("<UNK>", {}).get("id")

            #  Eğer UNK token vocab içinde yoksa ekleyelim
            if unk_id is None:
                logger.warning("[!] UNK token tanımlı değil, ekleniyor...")

                # Kullanılan ID'leri kontrol et ve çakışmayı önlemek için yeni ID belirle
                existing_ids = {info["id"] for info in self.vocab.values()}
                unk_id = max(existing_ids, default=3) + 1

                # Vocab içine UNK token'ı ekle
                self.vocab["<UNK>"] = {
                    "id": unk_id,
                    "total_freq": 0,
                    "positions": []
                }

                #  Reverse vocab güncellemesi yap
                self._update_reverse_vocab()

                logger.info(f"[+] UNK token '{unk_id}' olarak vocab içine eklendi.")

            #  Pozisyon bilgisini güncelleyelim
            if "<UNK>" in self.vocab:
                if "positions" not in self.vocab["<UNK>"]:
                    self.vocab["<UNK>"]["positions"] = []

                # Pozisyon bilgisi 100 kayıtla sınırlandırılabilir (aşırı büyümeyi önlemek için)
                if len(self.vocab["<UNK>"]["positions"]) < 100:
                    current_pos = len(self.vocab["<UNK>"]["positions"])
                    self.vocab["<UNK>"]["positions"].append(current_pos)
                    logger.debug(f"[+] UNK token pozisyonu eklendi: {current_pos}")

                #  Frekans bilgisini de artır
                self.vocab["<UNK>"]["total_freq"] += 1

            #  Reverse vocab eş zamanlı güncellemesi
            self._update_reverse_vocab()

            logger.debug(f"[!] Bilinmeyen token '{token}' -> ID: {unk_id}")

            return unk_id

        except Exception as e:
            logger.error(f"[X] Bilinmeyen token işleme hatası: {e}")
            raise BPEEncodingError(f"UNK token handling error: {e}")




    def _update_reverse_vocab(self):
        """
        Vocab güncellendiğinde reverse vocab'i de günceller.
        """
        try:
            if not self.vocab:
                raise ValueError("Vocab boş. Reverse vocab güncellenemedi.")
            
            self.reverse_vocab = {info["id"]: token for token, info in self.vocab.items()}
            logger.info(f"[+] Reverse vocab başarıyla güncellendi. Toplam {len(self.reverse_vocab)} token yüklendi.")

        except Exception as e:
            logger.error(f"[X] Reverse vocab güncellenirken hata: {e}")
            raise BPEEncodingError(f"Reverse vocab güncelleme hatası: {e}")



    def update_vocab(self, new_vocab: Dict[str, dict]):
        """
        Yeni bir vocab yükleyip mevcut vocab'ı günceller.

        Args:
            new_vocab (Dict[str, dict]): Güncellenmiş vocab sözlüğü.
        """
        if not isinstance(new_vocab, dict):
            raise TypeError("[X] Vocab bir sözlük olmalıdır.")

        logger.info("[+] Vocab güncelleniyor...")

        # Özel tokenları koru
        preserved_special_tokens = {
            "<PAD>": self.vocab.get("<PAD>", {"id": 0}),
            "<UNK>": self.vocab.get("<UNK>", {"id": 1}),
            "<BOS>": self.vocab.get("<BOS>", {"id": 2}),
            "<EOS>": self.vocab.get("<EOS>", {"id": 3}),
        }

        # Geçerli format kontrolü
        invalid_tokens = []
        for token, info in new_vocab.items():
            if not isinstance(info, dict) or "id" not in info or "total_freq" not in info or "positions" not in info:
                invalid_tokens.append(token)
        if invalid_tokens:
            raise ValueError(f"[X] Geçersiz formatta tokenlar tespit edildi: {invalid_tokens}")

        # ID çakışması kontrolü ve çözümü
        existing_ids = {info["id"] for info in self.vocab.values()}
        updated_vocab = {}
        for token, info in new_vocab.items():
            token_id = info["id"]

            # Çakışma varsa yeni ID belirle
            if token_id in existing_ids:
                logger.warning(f"[!] ID çakışması tespit edildi: '{token}' için ID '{token_id}' zaten mevcut.")
                token_id = max(existing_ids) + 1
                logger.info(f"[+] Yeni ID atandı: {token} -> {token_id}")

            updated_vocab[token] = {
                "id": token_id,
                "total_freq": info.get("total_freq", 0),
                "positions": info.get("positions", [])
            }
            existing_ids.add(token_id)

        # Vocab güncellemesi (sıralamayı koruyarak)
        self.vocab = OrderedDict({**preserved_special_tokens, **updated_vocab})

        # Reverse vocab güncellemesi
        self._update_reverse_vocab()

        logger.info(f"[+] Vocab başarıyla güncellendi. Toplam token sayısı: {len(self.vocab)}")


    def reset(self):
        """
        Encoder yapılandırmasını sıfırlar.
        Özel tokenlar (<PAD>, <UNK>, <BOS>, <EOS>) korunur.
        """
        logger.warning("[!] BPEEncoder sıfırlanıyor...")

        # Özel tokenları koruyarak sıfırlama
        self.vocab = {
            "<PAD>": self.vocab.get("<PAD>", {"id": 0}),
            "<UNK>": self.vocab.get("<UNK>", {"id": 1}),
            "<BOS>": self.vocab.get("<BOS>", {"id": 2}),
            "<EOS>": self.vocab.get("<EOS>", {"id": 3}),
        }
        logger.info("[+] Encoder sıfırlandı.")

