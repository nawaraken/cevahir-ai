import unittest
import os
from modules.docx_data_loader import DocxDataLoader
from utils.docx_data_loader_utils import DocxDataLoaderUtils
from config.parameters import DOCX_DIR, NUM_RANDOM_DOCX
import logging

# Log yapılandırması
process_logger = logging.getLogger("process_logger")
error_logger = logging.getLogger("error_logger")

class TestDocxDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Test için örnek DOCX dosyasının konumunu belirleyin
        cls.test_docx_file = os.path.join('data', 'test_docx', 'test.docx')
        process_logger.info(f"Test dosyası yükleniyor: {cls.test_docx_file}")

        # Dosyanın varlığını ve erişilebilirliğini kontrol et
        if not os.path.exists(cls.test_docx_file):
            error_logger.error(f"Test DOCX dosyası bulunamadı: {cls.test_docx_file}")
            raise FileNotFoundError(f"Test DOCX dosyası bulunamadı: {cls.test_docx_file}")
        if not os.path.isfile(cls.test_docx_file):
            error_logger.error(f"Test DOCX dosya yolu bir dosya değil: {cls.test_docx_file}")
            raise ValueError(f"Test DOCX dosya yolu bir dosya değil: {cls.test_docx_file}")
        if not os.access(cls.test_docx_file, os.R_OK):
            error_logger.error(f"Test DOCX dosyasına okuma izni yok: {cls.test_docx_file}")
            raise PermissionError(f"Test DOCX dosyasına okuma izni yok: {cls.test_docx_file}")
        
        process_logger.info("Test dosyası başarıyla bulundu ve erişilebilir durumda.")

    def test_load_docx_valid(self):
        """Geçerli bir DOCX dosyasını yüklemeyi ve doğrulamayı test eder."""
        try:
            data = DocxDataLoader.load_docx([self.test_docx_file])
            process_logger.debug(f"{self.test_docx_file} DOCX verisi başarıyla yüklendi.")
            self.assertIsNotNone(data, "DOCX dosyası yüklenemedi.")
            self.assertIsInstance(data, list, "Yüklenen veri list formatında değil.")
            process_logger.info("DOCX dosyası başarıyla yüklendi ve doğrulandı.")
        except Exception as e:
            error_logger.error(f"{self.test_docx_file} dosyasının doğrulaması sırasında beklenmeyen hata: {e}")
            self.fail(f"{self.test_docx_file} dosyasının doğrulaması sırasında beklenmeyen hata: {e}")

    def test_validate_docx_format_valid(self):
        """Geçerli DOCX dosya formatını doğrulayın."""
        try:
            is_valid = DocxDataLoaderUtils.validate_docx_format(self.test_docx_file)
            self.assertTrue(is_valid, "DOCX dosya formatı geçersiz.")
            process_logger.info(f"{self.test_docx_file} dosya formatı başarıyla doğrulandı.")
        except Exception as e:
            error_logger.error(f"{self.test_docx_file} format doğrulaması sırasında beklenmeyen hata: {e}")
            self.fail(f"{self.test_docx_file} format doğrulaması sırasında beklenmeyen hata: {e}")

    def test_extract_headings_and_styles(self):
        """DOCX dosyasındaki başlıkları ve stilleri çıkarma testini yapar."""
        try:
            headings = DocxDataLoaderUtils.extract_headings_and_styles(self.test_docx_file)
            self.assertIsInstance(headings, list, "Başlıklar list formatında değil.")
            process_logger.info(f"{self.test_docx_file} dosyasından başlıklar başarıyla çıkarıldı.")
        except Exception as e:
            error_logger.error(f"{self.test_docx_file} başlık çıkarma sırasında beklenmeyen hata: {e}")
            self.fail(f"{self.test_docx_file} başlık çıkarma sırasında beklenmeyen hata: {e}")

    def test_extract_tables(self):
        """DOCX dosyasındaki tabloları çıkarma testini yapar."""
        try:
            tables = DocxDataLoaderUtils.extract_tables(self.test_docx_file)
            self.assertIsInstance(tables, list, "Tablolar list formatında değil.")
            process_logger.info(f"{self.test_docx_file} dosyasından tablolar başarıyla çıkarıldı.")
        except Exception as e:
            error_logger.error(f"{self.test_docx_file} tablo çıkarma sırasında beklenmeyen hata: {e}")
            self.fail(f"{self.test_docx_file} tablo çıkarma sırasında beklenmeyen hata: {e}")

    def test_extract_paragraph_alignment_and_indentation(self):
        """Paragrafların hizalama ve girinti bilgilerini çıkarma testini yapar."""
        try:
            alignment_data = DocxDataLoaderUtils.extract_paragraph_alignment_and_indentation(self.test_docx_file)
            self.assertIsInstance(alignment_data, list, "Hizalama verisi list formatında değil.")
            process_logger.info(f"{self.test_docx_file} dosyasından hizalama bilgileri başarıyla çıkarıldı.")
        except Exception as e:
            error_logger.error(f"{self.test_docx_file} hizalama çıkarma sırasında beklenmeyen hata: {e}")
            self.fail(f"{self.test_docx_file} hizalama çıkarma sırasında beklenmeyen hata: {e}")

    def test_extract_text_styling(self):
        """Metinlerin yazı tipi renkleri ve süsleme özelliklerini çıkarma testini yapar."""
        try:
            styling_data = DocxDataLoaderUtils.extract_text_styling(self.test_docx_file)
            self.assertIsInstance(styling_data, list, "Stil verisi list formatında değil.")
            process_logger.info(f"{self.test_docx_file} dosyasından metin süsleme bilgileri başarıyla çıkarıldı.")
        except Exception as e:
            error_logger.error(f"{self.test_docx_file} metin süsleme çıkarma sırasında beklenmeyen hata: {e}")
            self.fail(f"{self.test_docx_file} metin süsleme çıkarma sırasında beklenmeyen hata: {e}")

    def test_extract_page_layout(self):
        """Sayfa düzeni ve kenar boşluklarını çıkarma testini yapar."""
        try:
            layout_data = DocxDataLoaderUtils.extract_page_layout(self.test_docx_file)
            self.assertIsInstance(layout_data, list, "Sayfa düzeni verisi list formatında değil.")
            process_logger.info(f"{self.test_docx_file} dosyasından sayfa düzeni bilgileri başarıyla çıkarıldı.")
        except Exception as e:
            error_logger.error(f"{self.test_docx_file} sayfa düzeni çıkarma sırasında beklenmeyen hata: {e}")
            self.fail(f"{self.test_docx_file} sayfa düzeni çıkarma sırasında beklenmeyen hata: {e}")

    def test_load_random_docx_files(self):
        """Geçerli DOCX dosyalarını rastgele seçme ve yükleme testini yapar."""
        try:
            loaded_data = DocxDataLoader.load_random_docx_files(NUM_RANDOM_DOCX)
            self.assertIsInstance(loaded_data, list, "Yüklenen veri list formatında değil.")
            for data in loaded_data:
                self.assertIn("content", data, "Yüklenen veri 'content' anahtarını içermiyor.")
                self.assertIn("alignment", data, "Yüklenen veri 'alignment' anahtarını içermiyor.")
                self.assertIn("styling", data, "Yüklenen veri 'styling' anahtarını içermiyor.")
                self.assertIn("layout", data, "Yüklenen veri 'layout' anahtarını içermiyor.")
            process_logger.info(f"Rastgele DOCX dosyaları başarıyla yüklendi: {loaded_data}")
        except Exception as e:
            error_logger.error(f"Rastgele DOCX dosya yükleme sırasında beklenmeyen hata: {e}")
            self.fail(f"Rastgele DOCX dosya yükleme sırasında beklenmeyen hata: {e}")

if __name__ == "__main__":
    unittest.main()
