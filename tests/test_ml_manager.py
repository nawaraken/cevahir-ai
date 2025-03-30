import unittest
import os
import numpy as np
import json
from modules.machine_learning_manager import MachineLearningManager
from utils.machine_learning_manager_utils import MachineLearningUtils
from config.parameters import VOCAB_SAVE_PATH, MODEL_TRAINING_DATA_DIR, INPUT_DIM

class TestMachineLearningManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Testler için gerekli dosyaların ve klasörlerin hazırlanması.
        """
        cls.ml_manager = MachineLearningManager()
        cls.sample_vocab_path = VOCAB_SAVE_PATH
        cls.training_data_dir = MODEL_TRAINING_DATA_DIR
        
        # Örnek vocab dosyası oluştur
        cls.sample_vocab = {
            "kelime1": {"total_freq": 10, "source": {"Soru": 5, "Cevap": 5}, "files": ["test.json"]},
            "kelime2": {"total_freq": 8, "source": {"Soru": 3, "Cevap": 5}, "files": ["test2.json"]},
            "kelime3": {"total_freq": 15, "source": {"Soru": 10, "Cevap": 5}, "files": ["test3.json"]}
        }
        
        os.makedirs(os.path.dirname(cls.sample_vocab_path), exist_ok=True)
        with open(cls.sample_vocab_path, 'w', encoding='utf-8') as f:
            json.dump(cls.sample_vocab, f, ensure_ascii=False, indent=4)

        os.makedirs(cls.training_data_dir, exist_ok=True)

    def test_load_vocab_data(self):
        """Vocab dosyasının doğru şekilde yüklendiğini test eder."""
        vocab_data = MachineLearningUtils.load_vocab_data(self.sample_vocab_path)
        self.assertEqual(vocab_data, self.sample_vocab, "Yüklenen vocab verisi, örnek vocab ile aynı olmalıdır.")

    def test_prepare_data_for_model(self):
        """Vocab verilerinin model girişine uygun hale getirildiğini test eder."""
        vocab_data = MachineLearningUtils.load_vocab_data(self.sample_vocab_path)
        processed_data = MachineLearningUtils.prepare_data_for_model(vocab_data, INPUT_DIM)
        
        # Giriş verisinin uygun boyutta olup olmadığını kontrol et
        self.assertIsInstance(processed_data, list, "İşlenmiş veri bir liste olmalıdır.")
        self.assertTrue(all(isinstance(vec, np.ndarray) and vec.shape == (INPUT_DIM,) for vec in processed_data),
                        "Tüm giriş vektörlerinin boyutu INPUT_DIM olmalıdır.")

    def test_save_training_data(self):
        """Eğitim verisinin doğru şekilde kaydedildiğini test eder."""
        sample_data = [np.random.rand(INPUT_DIM) for _ in range(10)]
        MachineLearningUtils.save_training_data(sample_data, self.training_data_dir, INPUT_DIM)

        
        # Kaydedilen dosyanın varlığını kontrol et
        saved_files = [f for f in os.listdir(self.training_data_dir) if f.endswith('_ready_data.npy')]
        self.assertTrue(len(saved_files) > 0, "Eğitim verisi dosyası kaydedilmiş olmalıdır.")
        
        # Dosyanın içeriğini kontrol et
        if saved_files:
            loaded_data = np.load(os.path.join(self.training_data_dir, saved_files[0]), allow_pickle=True)
            self.assertTrue(np.array_equal(loaded_data, sample_data), "Kaydedilen eğitim verisi, orijinal veri ile aynı olmalıdır.")

    @classmethod
    def tearDownClass(cls):
        """
        Test için oluşturulan dosyaları temizler.
        """
        if os.path.exists(cls.sample_vocab_path):
            os.remove(cls.sample_vocab_path)
        saved_files = [f for f in os.listdir(cls.training_data_dir) if f.endswith('_ready_data.npy')]
        for file in saved_files:
            os.remove(os.path.join(cls.training_data_dir, file))

if __name__ == "__main__":
    unittest.main()
