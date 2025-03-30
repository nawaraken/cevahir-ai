import unittest
import os
import json
import logging
from modules.json_tokenizer import JsonTokenizer
from modules.json_data_loader import JsonDataLoader
from config.parameters import PROCESSED_DATA_DIR, JSON_DIR, MAX_TOKENS, STOPWORDS, SPECIAL_TOKENS, NUM_RANDOM_JSONS

class TestJsonTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Tokenizer'ı oluştur
        cls.tokenizer = JsonTokenizer(max_tokens=MAX_TOKENS, stopwords=STOPWORDS, special_tokens=SPECIAL_TOKENS)
        
        # Processed data directory yoksa oluştur
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        cls.processed_file_path = os.path.join(PROCESSED_DATA_DIR, "test_tokenized.json")

        # Rastgele JSON dosyalarını yükle
        cls.loaded_data = JsonDataLoader.load_random_json_files()
        if not cls.loaded_data or len(cls.loaded_data) < NUM_RANDOM_JSONS:
            raise FileNotFoundError("Yeterli sayıda rastgele JSON dosyası yüklenemedi.")
        
    def test_load_json_valid(self):
        """Geçerli bir JSON dosyasını yüklemeyi test eder."""
        data = self.loaded_data[0] if self.loaded_data else None
        self.assertIsNotNone(data, "JSON verisi yüklenemedi, None döndü.")
        self.assertIsInstance(data, dict, "Yüklenen JSON verisi dict formatında değil.")  # Değişiklik burada yapıldı

        # 'data' anahtarını kontrol et ve içindeki veriyi kontrol et
        self.assertIn("data", data, "'data' anahtarı eksik.")
        json_content = data["data"]

        # 'data' içindeki öğelerin liste olduğunu ve gerekli anahtarları içerdiğini kontrol et
        self.assertIsInstance(json_content, list, "'data' anahtarının içeriği liste formatında değil.")
        if json_content:
            self.assertIn("Soru", json_content[0], "'Soru' anahtarı eksik.")
            self.assertIn("Cevap", json_content[0], "'Cevap' anahtarı eksik.")
        logging.info("Geçerli JSON verisi başarıyla yüklendi.")




    def test_process_text_normal(self):
        """Normal bir metin ile tokenizasyon işlemini test eder."""
        text = "Bu bir örnek metin cümlesidir."
        tokens = self.tokenizer.process_text(text)
        self.assertIsInstance(tokens, list, "Tokenizasyon sonucunda liste döndürülmedi.")
        self.assertNotIn("bir", tokens, "'bir' stopword olarak çıkarılmadı.")
        self.assertLessEqual(len(tokens), MAX_TOKENS, "Token sayısı MAX_TOKENS limitini aştı.")
        logging.info("Normal metin başarılı bir şekilde işlendi ve tokenlara ayrıldı.")

    def test_tokenize_json_file_valid(self):
            """Rastgele yüklenmiş JSON verisini tokenlere ayırma işlemini test eder."""
            # İlk JSON verilerini yükleyip tokenize edelim
            json_data_list = self.loaded_data if isinstance(self.loaded_data, list) else [self.loaded_data]
            tokenized_data = self.tokenizer.tokenize(json_data_list)

            # Tokenizasyon doğrulama
            self.assertIsInstance(tokenized_data, list, "Tokenize edilmiş veri bir liste formatında değil.")
            self.assertGreater(len(tokenized_data), 0, "Tokenize edilmiş veri listesi boş.")

            # İlk öğeyi almak için ilk indeksin değerini al
            first_tokenized_entry = tokenized_data[0] if tokenized_data else {}

            self.assertIsInstance(first_tokenized_entry, dict, "Tokenize edilmiş verinin her öğesi bir dict formatında olmalı.")
            self.assertIn("filename", first_tokenized_entry, "'filename' anahtarı tokenize edilmiş veride bulunamadı.")
            self.assertIn("data", first_tokenized_entry, "'data' anahtarı tokenize edilmiş veride bulunamadı.")
            self.assertIsInstance(first_tokenized_entry["data"], list, "'data' anahtarının değeri liste formatında değil.")

            # İlk 'data' öğesini kontrol et
            if first_tokenized_entry["data"]:
                first_data_entry = first_tokenized_entry["data"][0]
                self.assertIn("Soru", first_data_entry, "'Soru' anahtarı tokenize edilmiş veride bulunamadı.")
                self.assertIn("Cevap", first_data_entry, "'Cevap' anahtarı tokenize edilmiş veride bulunamadı.")
                self.assertIsInstance(first_data_entry["Soru"], list, "Tokenize edilmiş 'Soru' liste formatında değil.")
                self.assertIsInstance(first_data_entry["Cevap"], list, "Tokenize edilmiş 'Cevap' liste formatında değil.")

            logging.info("Rastgele yüklenmiş JSON verisi başarıyla tokenleştirildi.")




    def test_save_processed_data(self):
        """Tokenize edilmiş veriyi doğru dizinde JSON olarak kaydeder."""
        processed_data = [
            {
                "file_name": "temp_test_file_1.json",
                "data": [
                    {"index": 0, "Soru": ["örnek", "soru"], "Cevap": ["örnek", "cevap"]},
                    {"index": 1, "Soru": ["ikinci", "soru"], "Cevap": ["ikinci", "cevap"]}
                ]
            },
            {
                "file_name": "temp_test_file_2.json",
                "data": [
                    {"index": 0, "Soru": ["başka", "örnek"], "Cevap": ["başka", "cevap"]},
                    {"index": 1, "Soru": ["sonraki", "soru"], "Cevap": ["sonraki", "cevap"]}
                ]
            }
        ]
        
        self.tokenizer.save_processed_data(processed_data)

        # Kayıtlı dosyaların doğruluğunu kontrol et
        for entry in processed_data:
            tokenized_filename = entry["file_name"].replace(".json", "_tokenized_data.json")
            save_path = os.path.join(PROCESSED_DATA_DIR, tokenized_filename)

            self.assertTrue(os.path.exists(save_path), f"Kaydedilen dosya bulunamadı: {save_path}")
            with open(save_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                self.assertEqual(saved_data, entry["data"], f"Kaydedilen veri beklenenden farklı: {save_path}")
                logging.info(f"Tokenize edilmiş veriler başarıyla kaydedildi: {save_path}")

            # Geçici dosyayı temizleyin
            os.remove(save_path)




if __name__ == "__main__":
    unittest.main()
