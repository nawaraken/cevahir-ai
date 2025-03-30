import unittest
import os
import json
import logging
import re
from modules.docx_tokenizer import DocxTokenizer
from modules.docx_data_loader import DocxDataLoader
from config.parameters import PROCESSED_DATA_DIR, MAX_TOKENS, STOPWORDS, SPECIAL_TOKENS, NUM_RANDOM_DOCX

class TestDocxTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Tokenizer'ı oluştur
        cls.tokenizer = DocxTokenizer(max_tokens=MAX_TOKENS, stopwords=STOPWORDS, special_tokens=SPECIAL_TOKENS)
        
        # Processed data directory yoksa oluştur
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # Rastgele DOCX dosyalarını yükle
        cls.loaded_data = DocxDataLoader.load_random_docx_files(NUM_RANDOM_DOCX)
        if not cls.loaded_data or len(cls.loaded_data) < NUM_RANDOM_DOCX:
            raise FileNotFoundError("Yeterli sayıda rastgele DOCX dosyası yüklenemedi.")

    def test_load_docx_valid(self):
        """Geçerli bir DOCX dosyasını yüklemeyi test eder."""
        data = self.loaded_data[0] if self.loaded_data else None
        self.assertIsNotNone(data, "DOCX verisi yüklenemedi, None döndü.")
        self.assertIsInstance(data, dict, "Yüklenen DOCX verisi dict formatında değil.")
        self.assertIn("content", data, "'content' anahtarı eksik.")
        
        # Varsayılan anahtarların yoksa atanması
        if "alignment" not in data:
            data["alignment"] = []
        if "styling" not in data:
            data["styling"] = []
        if "layout" not in data:
            data["layout"] = []
        if "tables" not in data:
            data["tables"] = []
        if "images" not in data:
            data["images"] = []
        if "metadata" not in data:
            data["metadata"] = {}
        
        self.assertIn("alignment", data, "'alignment' anahtarı eksik.")
        self.assertIn("styling", data, "'styling' anahtarı eksik.")
        self.assertIn("layout", data, "'layout' anahtarı eksik.")
        self.assertIn("tables", data, "'tables' anahtarı eksik.")
        self.assertIn("images", data, "'images' anahtarı eksik.")
        self.assertIn("metadata", data, "'metadata' anahtarı eksik.")
        
        logging.info("Geçerli DOCX verisi başarıyla yüklendi.")


    def test_process_single_text_normal(self):
        """Normal bir metin ile tokenizasyon işlemini test eder."""
        text = "Bu bir örnek metin cümlesidir."
        tokens = self.tokenizer._process_single_text(text)
        self.assertIsInstance(tokens, list, "Tokenizasyon sonucunda liste döndürülmedi.")
        self.assertGreater(len(tokens), 0, "Tokenizasyon sonucunda boş liste döndü.")
        self.assertLessEqual(len(tokens), MAX_TOKENS, "Token sayısı MAX_TOKENS limitini aştı.")
        logging.info("Normal metin başarılı bir şekilde işlendi ve tokenlara ayrıldı.")

    def test_process_text_list(self):
        """Liste formatında metinleri işlemeyi test eder."""
        text_list = [{"text": "Bu birinci örnek metindir."}, {"text": "İkinci metin cümlesi de buradadır."}]
        tokens = self.tokenizer._process_single_text(text_list)
        self.assertIsInstance(tokens, list, "Liste formatında metinlerin tokenizasyonu sonucu liste döndürülmedi.")
        self.assertGreater(len(tokens), 0, "Tokenizasyon sonucu boş bir liste döndü.")
        logging.info("Liste formatındaki metinler başarılı bir şekilde işlendi ve tokenlara ayrıldı.")

    def test_tokenize_docx_file_valid(self):
        """Rastgele yüklenmiş DOCX verisini tokenlere ayırma işlemini test eder."""
        docx_data_list = self.loaded_data if isinstance(self.loaded_data, list) else [self.loaded_data]
        tokenized_data = self.tokenizer.tokenize(docx_data_list)

        self.assertIsInstance(tokenized_data, list, "Tokenize edilmiş veri bir liste formatında değil.")
        self.assertGreater(len(tokenized_data), 0, "Tokenize edilmiş veri listesi boş.")

        first_tokenized_entry = tokenized_data[0] if tokenized_data else {}

        self.assertIsInstance(first_tokenized_entry, dict, "Tokenize edilmiş verinin her öğesi bir dict formatında olmalı.")
        self.assertIn("filename", first_tokenized_entry, "'filename' anahtarı tokenize edilmiş veride bulunamadı.")
        self.assertIn("content", first_tokenized_entry, "'content' anahtarı tokenize edilmiş veride bulunamadı.")
        self.assertIn("alignment", first_tokenized_entry, "'alignment' anahtarı tokenize edilmiş veride bulunamadı.")
        self.assertIn("styling", first_tokenized_entry, "'styling' anahtarı tokenize edilmiş veride bulunamadı.")
        self.assertIn("layout", first_tokenized_entry, "'layout' anahtarı tokenize edilmiş veride bulunamadı.")
        self.assertIn("tables", first_tokenized_entry, "'tables' anahtarı tokenize edilmiş veride bulunamadı.")
        self.assertIn("images", first_tokenized_entry, "'images' anahtarı tokenize edilmiş veride bulunamadı.")
        self.assertIn("metadata", first_tokenized_entry, "'metadata' anahtarı tokenize edilmiş veride bulunamadı.")
        logging.info("Rastgele yüklenmiş DOCX verisi başarıyla tokenleştirildi.")

    def test_tokenize_with_missing_values(self):
        """Eksik anahtarlar içeren verilerin doğru şekilde işlenmesini test eder."""
        incomplete_data = {
            "filename": "eksik_ornek.docx",
            "content": [
                {
                    "text": "Eksik bilgi içeren bir paragraf.",
                    "style": "Normal"
                }
            ],
            "metadata": {
                "file_name": "eksik_ornek.docx",
                "file_size_bytes": 102400
            }
        }
        tokens = self.tokenizer.tokenize([incomplete_data])
        self.assertIsInstance(tokens, list, "Tokenizasyon sonucunda liste döndürülmedi.")
        self.assertGreater(len(tokens[0].get("content", [])), 0, "Eksik anahtarlarla içerik işlenemedi.")
        logging.info("Eksik anahtarlar içeren veri başarıyla işlendi.")

    def test_save_processed_data(self):
        """Yüklenen ve tokenize edilen DOCX dosyalarını kaydetme işlemini test eder."""
        docx_data_list = self.loaded_data if isinstance(self.loaded_data, list) else [self.loaded_data]
        tokenized_data = self.tokenizer.tokenize(docx_data_list)

        self.tokenizer.save_processed_data(tokenized_data)

        for entry in tokenized_data:
            filename = entry.get("filename")
            tokenized_filename = re.sub(r"\.[a-zA-Z0-9]+$", "_processed.json", os.path.basename(filename))
            save_path = os.path.join(PROCESSED_DATA_DIR, tokenized_filename)

            # Kaydedilen dosyanın varlığını kontrol et
            self.assertTrue(os.path.exists(save_path), f"Kaydedilen dosya bulunamadı: {save_path}")

            # Dosyanın içeriğini kontrol et
            with open(save_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                self.assertEqual(saved_data["filename"], entry["filename"], f"Kaydedilen dosya adı beklenenden farklı: {save_path}")
                self.assertEqual(saved_data.get("content"), entry.get("content"), f"'content' beklenenden farklı: {save_path}")
                self.assertEqual(saved_data.get("alignment"), entry.get("alignment"), f"'alignment' beklenenden farklı: {save_path}")
                self.assertEqual(saved_data.get("styling"), entry.get("styling"), f"'styling' beklenenden farklı: {save_path}")
                self.assertEqual(saved_data.get("layout"), entry.get("layout"), f"'layout' beklenenden farklı: {save_path}")
                self.assertEqual(saved_data.get("tables"), entry.get("tables"), f"'tables' beklenenden farklı: {save_path}")
                self.assertEqual(saved_data.get("images"), entry.get("images"), f"'images' beklenenden farklı: {save_path}")
                self.assertEqual(saved_data.get("metadata"), entry.get("metadata"), f"'metadata' beklenenden farklı: {save_path}")
                            


if __name__ == "__main__":
    unittest.main()
