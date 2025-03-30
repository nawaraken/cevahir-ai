import unittest
import os
import json
import logging
from modules.docx_tokenizer import DocxTokenizer
from modules.docx_data_loader import DocxDataLoader
from utils.docx_vocab_manager_utils import DocxVocabUtils
from modules.docx_vocab_manager import DocxVocabManager
from config.parameters import PROCESSED_DATA_DIR, DOCX_VOCAB_SAVE_PATH, MIN_FREQ, MAX_VOCAB_SIZE, NUM_RANDOM_DOCX, MAX_TOKENS, STOPWORDS, SPECIAL_TOKENS

class TestDocxVocabManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Tokenizer ve VocabManager'ı oluştur
        cls.tokenizer = DocxTokenizer(max_tokens=MAX_TOKENS, stopwords=STOPWORDS, special_tokens=SPECIAL_TOKENS)
        cls.vocab_manager = DocxVocabManager()
        cls.vocab_utils = DocxVocabUtils()

        # Processed data directory yoksa oluştur
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        # Rastgele DOCX dosyalarını yükle ve tokenize et
        cls.loaded_data = DocxDataLoader.load_random_docx_files(NUM_RANDOM_DOCX)
        if not cls.loaded_data or len(cls.loaded_data) < NUM_RANDOM_DOCX:
            raise FileNotFoundError("Yeterli sayıda rastgele DOCX dosyası yüklenemedi.")
        
        # DOCX verilerini tokenize et
        cls.processed_data = cls.tokenizer.tokenize(cls.loaded_data)

    def test_build_vocab_min_freq(self):
        """MIN_FREQ değerine göre düşük frekanslı kelimeleri filtreler."""
        self.vocab_manager.process_docx_data(self.processed_data)
        vocab = self.vocab_manager.load_existing_vocab()
        self.assertTrue(all(item['total_freq'] >= MIN_FREQ for item in vocab.values()), "Tüm kelimelerin frekansı MIN_FREQ değerine eşit veya daha yüksek olmalıdır.")

    def test_build_vocab_max_size(self):
        """MAX_VOCAB_SIZE sınırına göre kelime hazinesi boyutunu sınırlandırır."""
        self.vocab_manager.process_docx_data(self.processed_data)
        vocab = self.vocab_manager.load_existing_vocab()
        self.assertLessEqual(len(vocab), MAX_VOCAB_SIZE, "Kelime hazinesi boyutu MAX_VOCAB_SIZE değerini aşmamalıdır.")

    def test_save_vocab_to_file(self):
        """Kelime hazinesinin belirtilen dosya yoluna doğru kaydedildiğini test eder."""
        self.vocab_manager.process_docx_data(self.processed_data)
        self.vocab_utils.save_vocab_to_file(self.vocab_manager.load_existing_vocab(), DOCX_VOCAB_SAVE_PATH)
        self.assertTrue(os.path.exists(DOCX_VOCAB_SAVE_PATH), "DOCX_VOCAB_SAVE_PATH dosyası oluşturulmuş olmalıdır.")

        # Dosyayı açıp içeriğini kontrol et
        with open(DOCX_VOCAB_SAVE_PATH, 'r', encoding='utf-8') as f:
            saved_vocab = json.load(f)
        self.assertEqual(saved_vocab, self.vocab_manager.load_existing_vocab(), "Kaydedilen kelime hazinesi, oluşturulanla aynı olmalıdır.")

    def test_load_vocab_from_file(self):
        """Kaydedilen kelime hazinesinin doğru bir şekilde yüklendiğini test eder."""
        self.vocab_manager.process_docx_data(self.processed_data)
        self.vocab_utils.save_vocab_to_file(self.vocab_manager.load_existing_vocab(), DOCX_VOCAB_SAVE_PATH)

        # Yüklenen kelime hazinesini test et
        vocab = self.vocab_manager.load_existing_vocab()
        self.assertIsInstance(vocab, dict, "Vocab dosyadan yüklenirken sözlük formatında döndürülmedi.")
        self.assertGreater(len(vocab), 0, "Yüklenen vocab beklenenden farklı.")
        logging.info("Vocab'ı dosyadan yükleme metodu başarıyla test edildi.")

    def test_update_vocab_with_new_data(self):
        """Yeni verilerin mevcut kelime hazinesine doğru eklendiğini test eder."""
        initial_vocab = self.vocab_manager.load_existing_vocab()
        initial_vocab_size = len(initial_vocab)

        # Yeni veriyi ekle
        self.vocab_manager.update_vocab_with_new_data(self.processed_data)
        updated_vocab = self.vocab_manager.load_existing_vocab()

        # Yeni tokenler ve içeriklerinin kelime hazinesine eklendiğinden emin ol
        for item in self.processed_data:
            if isinstance(item, dict) and 'token' in item:
                token = item['token']
                self.assertIn(token, updated_vocab, f"Token '{token}' kelime hazinesine eklenmelidir.")
                
                # design_info ve metadata'nın da doğru şekilde eklendiğinden emin ol
                self.assertEqual(
                    item.get('design_info', {}),
                    updated_vocab[token].get('design_info', {}),
                    f"Token '{token}' için 'design_info' kelime hazinesine doğru kaydedilmiş olmalıdır."
                )
                self.assertEqual(
                    item.get('metadata', {}),
                    updated_vocab[token].get('metadata', {}),
                    f"Token '{token}' için 'metadata' kelime hazinesine doğru kaydedilmiş olmalıdır."
                )

        self.assertGreater(len(updated_vocab), initial_vocab_size, "Güncellenmiş kelime hazinesi, başlangıçtaki boyuttan büyük olmalıdır.")
        logging.info("Vocab güncelleme metodu başarıyla test edildi.")



    def test_vocab_metadata_and_design_info(self):
        """Vocab içindeki metadata ve design_info alanlarının doğru biçimde eklendiğini test eder."""
        self.vocab_manager.process_docx_data(self.processed_data)
        vocab = self.vocab_manager.load_existing_vocab()

        for token, data in vocab.items():
            self.assertIn('metadata', data, f"{token} için 'metadata' alanı eksik.")
            self.assertIsInstance(data['metadata'], dict, f"{token} için 'metadata' alanı dict olmalıdır.")
            self.assertIn('design_info', data, f"{token} için 'design_info' alanı eksik.")
            self.assertIsInstance(data['design_info'], dict, f"{token} için 'design_info' alanı dict olmalıdır.")

if __name__ == "__main__":
    unittest.main()
