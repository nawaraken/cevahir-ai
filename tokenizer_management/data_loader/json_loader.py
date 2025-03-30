import os
import json
import logging
import traceback
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class JSONLoaderError(Exception):
    pass

class JSONFileNotFoundError(JSONLoaderError):
    pass

class JSONFormatError(JSONLoaderError):
    pass

class JSONLoader:
    """
    Sabit yapılı Soru-Cevap JSON dosyalarını yükler.
    Beklenen format:
    [
        {"Soru": "...", "Cevap": "..."},
        ...
    ]
    """

    def load_file(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise JSONFileNotFoundError(f"Dosya bulunamadı: {file_path}")

        if not file_path.endswith(".json"):
            raise JSONFormatError(f"Geçersiz dosya türü: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                try:
                    content = f.read()
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    raise JSONFormatError(f"JSON hatası: Satır {e.lineno}, Sütun {e.colno} → {e.msg}") from e

            if not isinstance(data, list):
                raise JSONFormatError("Veri listesi bekleniyordu ancak farklı bir yapı bulundu.")

            joined_texts: List[str] = []

            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    raise JSONFormatError(f"{idx}. öğe dict değil: {item}")

                soru = item.get("Soru")
                cevap = item.get("Cevap")

                if not isinstance(soru, str) or not isinstance(cevap, str):
                    raise JSONFormatError(f"{idx}. öğede 'Soru' veya 'Cevap' string değil.")

                soru = soru.strip()
                cevap = cevap.strip()

                if not soru or not cevap:
                    logger.warning(f"{idx}. öğede boş 'Soru' veya 'Cevap' değeri atlandı.")
                    continue

                joined_texts.append(f"__tag__soru {soru}")
                joined_texts.append(f"__tag__cevap {cevap}")

            if not joined_texts:
                raise JSONFormatError(f"{file_path} içinde işlenebilir içerik bulunamadı.")

            combined_text = " ".join(joined_texts)

            return {
                "data": combined_text,
                "modality": "text",
                "source_file": os.path.basename(file_path)
            }

        except Exception as e:
            logger.error(f"{file_path} işlenirken hata: {e}\n{traceback.format_exc()}")
            raise JSONLoaderError(f"{file_path} işlenirken hata oluştu: {e}") from e
