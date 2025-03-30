import unittest
import os
import json
import logging
from modules.json_tokenizer import JsonTokenizer
from modules.json_data_loader import JsonDataLoader
from modules.json_vocab_manager import VocabManager
from utils.json_vocab_manager_utils import JsonVocabUtils

from config.parameters import PROCESSED_DATA_DIR, JSON_VOCAB_SAVE_PATH, MIN_FREQ, MAX_VOCAB_SIZE, MAX_TOKENS, STOPWORDS, SPECIAL_TOKENS, NUM_RANDOM_JSONS

class TestVocabManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Tokenizer ve VocabManager'ı oluştur
        cls.tokenizer = JsonTokenizer(max_tokens=MAX_TOKENS, stopwords=STOPWORDS, special_tokens=SPECIAL_TOKENS)
        cls.vocab_manager = VocabManager()
        cls.vocab_manager_utils = JsonVocabUtils()

        # Processed data directory yoksa oluştur
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        cls.processed_file_path = os.path.join(PROCESSED_DATA_DIR, "test_tokenized.json")

        # Rastgele JSON dosyalarını yükle ve tokenize et
        cls.loaded_data = JsonDataLoader.load_random_json_files()
        if not cls.loaded_data or len(cls.loaded_data) < NUM_RANDOM_JSONS:
            raise FileNotFoundError("Yeterli sayıda rastgele JSON dosyası yüklenemedi.")
        cls.processed_data = cls.tokenizer.tokenize(cls.loaded_data)

    def test_build_vocab_min_freq(self):
        """MIN_FREQ değerine göre düşük frekanslı kelimeleri filtreler."""
        self.vocab_manager.build_vocab(self.processed_data)
        vocab = self.vocab_manager.vocab
        # Yeni yapıya uygun kontrol: vocab değerlerinin total_freq alanı kontrol edilir
        self.assertTrue(all(item['total_freq'] >= MIN_FREQ for item in vocab.values()), "Tüm kelimelerin frekansı MIN_FREQ değerine eşit veya daha yüksek olmalıdır.")

    def test_build_vocab_max_size(self):
        """MAX_VOCAB_SIZE sınırına göre kelime hazinesi boyutunu sınırlandırır."""
        self.vocab_manager.build_vocab(self.processed_data)
        vocab = self.vocab_manager.vocab
        self.assertLessEqual(len(vocab), MAX_VOCAB_SIZE, "Kelime hazinesi boyutu MAX_VOCAB_SIZE değerini aşmamalıdır.")

    def test_save_vocab(self):
        """Kelime hazinesinin belirtilen dosya yoluna doğru kaydedildiğini test eder."""
        self.vocab_manager.build_vocab(self.processed_data)
        self.vocab_manager.save_vocab(filepath=JSON_VOCAB_SAVE_PATH)
        self.assertTrue(os.path.exists(JSON_VOCAB_SAVE_PATH), "JSON_VOCAB_SAVE_PATH dosyası oluşturulmuş olmalıdır.")

        # Dosyayı açıp içeriğini kontrol et
        with open(JSON_VOCAB_SAVE_PATH, 'r', encoding='utf-8') as f:
            saved_vocab = json.load(f)
        self.assertEqual(saved_vocab, self.vocab_manager.vocab, "Kaydedilen kelime hazinesi, oluşturulanla aynı olmalıdır.")

    def test_load_vocab(self):
        """Kaydedilen kelime hazinesinin doğru bir şekilde yüklendiğini test eder."""
        self.vocab_manager.build_vocab(self.processed_data)
        self.vocab_manager.save_vocab(filepath=JSON_VOCAB_SAVE_PATH)

        # Yeni bir VocabManager örneği oluşturup kelime hazinesini yükle
        new_vocab_manager = VocabManager()
        new_vocab_manager.load_vocab(filepath=JSON_VOCAB_SAVE_PATH)

        # Yüklenen kelime hazinesi, kaydedilenle aynı olmalı
        self.assertEqual(new_vocab_manager.vocab, self.vocab_manager.vocab, "Yüklenen kelime hazinesi, kaydedilenle aynı olmalıdır.")

    def test_limit_vocab_size(self):
        """Kelime hazinesinin boyutunu MAX_VOCAB_SIZE ile sınırlandırır."""
        sample_vocab = {f"token_{i}": {"total_freq": i, "source": {"Soru": i, "Cevap": i}, "files": ["test.json"]} for i in range(1, MAX_VOCAB_SIZE + 50)}
        limited_vocab = self.vocab_manager_utils.limit_vocab_size(sample_vocab, MAX_VOCAB_SIZE)
        self.assertEqual(len(limited_vocab), MAX_VOCAB_SIZE, "Vocab boyutu MAX_VOCAB_SIZE değerine eşit olmalıdır.")
        self.assertTrue(all(limited_vocab[token]["total_freq"] >= limited_vocab.get(next_token, {}).get("total_freq", 0) for token, next_token in zip(list(limited_vocab)[:-1], list(limited_vocab)[1:])), "Token frekansları sıralı olmalıdır.")



if __name__ == "__main__":
    unittest.main()