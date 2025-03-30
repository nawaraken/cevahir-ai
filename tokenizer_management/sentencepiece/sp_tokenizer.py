import logging

logger = logging.getLogger(__name__)

class SentencePieceTokenizer:
    """
    SentencePieceTokenizer, SentencePiece tokenizasyon işlemlerini dışarıya sunan yüksek seviyeli arayüzdür.
    Tüm işlemler (kodlama, çözme, eğitim, vocabulary güncelleme, sıfırlama vb.) kendisine dışarıdan 
    enjekte edilen bir SentencePieceManager örneğine delege edilir.
    """

    def __init__(self, manager):
        """
        SentencePieceTokenizer oluşturulurken bir manager örneği verilmelidir.
        Eğer manager None ise, hata fırlatılır.
        
        Args:
            manager: Dışarıdan sağlanacak SentencePieceManager örneği.
        """
        if manager is None:
            raise ValueError("Bir SentencePieceManager örneği sağlanmalıdır.")
        self.manager = manager
        logger.info("SentencePieceTokenizer başlatıldı.")

    def encode(self, text: str) -> list:
        """
        Verilen metni token ID'lerine dönüştürür.
        
        Args:
            text (str): Kodlanacak metin.
        
        Returns:
            list: Token ID listesini döner.
        """
        return self.manager.encode(text)

    def decode(self, token_ids: list) -> str:
        """
        Verilen token ID listesini orijinal metne dönüştürür.
        
        Args:
            token_ids (list): Kodlanmış token ID'leri.
        
        Returns:
            str: Çözülmüş metin.
        """
        return self.manager.decode(token_ids)

    def train(self, corpus: list, vocab_size: int):
        """
        Verilen corpus üzerinde SentencePiece eğitimini gerçekleştirir ve vocabulary'i günceller.
        
        Args:
            corpus (list): Eğitim verisi (metin listesi).
            vocab_size (int): Hedef vocabulary boyutu.
        """
        self.manager.train(corpus, vocab_size)

    def update_vocab(self, new_tokens: list):
        """
        Yeni token'ler ekleyerek vocabulary'i günceller.
        
        Args:
            new_tokens (list): Eklenmek istenen token listesi.
        """
        self.manager.update_vocab(new_tokens)

    def reset(self):
        """
        Vocabulary'i sıfırlar.
        """
        self.manager.reset()

    def get_vocab(self) -> dict:
        """
        Mevcut vocabulary yapısını döner.
        
        Returns:
            dict: Vocabulary yapısı.
        """
        return self.manager.get_vocab()

    def get_token_ids(self, text: str) -> list:
        """
        Verilen metni token ID'lerine dönüştürür.
        
        Args:
            text (str): Kodlanacak metin.
        
        Returns:
            list: Token ID listesi.
        """
        return self.manager.get_token_ids(text)

    def get_text(self, token_ids: list) -> str:
        """
        Verilen token ID listesini metne dönüştürür.
        
        Args:
            token_ids (list): Token ID listesi.
        
        Returns:
            str: Çözülmüş metin.
        """
        return self.manager.get_text(token_ids)
