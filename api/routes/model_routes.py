"""
api/routes/model_routes.py
=================================
Bu dosya, Flask tabanlı bir RESTful API ile ModelManager sınıfını entegre eder. 
Modelle ilgili işlemleri (başlatma, yükleme, kaydetme, ileri yayılım vb.) API endpointleri olarak sunar.

Endpointler:
-----------
1. /initialize [POST]:
   Model ve bileşenlerin başlatılmasını sağlar.
2. /forward [POST]:
   Modelin ileri yayılım işlemini gerçekleştirir.
3. /save [POST]:
   Modeli belirtilen bir dosyaya kaydeder.
4. /load [POST]:
   Daha önce kaydedilen bir modeli yükler.
5. /update [POST]:
   Model parametrelerini günceller.

Notlar:
------
- Tüm endpointler JSON formatında veri alır ve döner.
- Hatalar durumunda uygun HTTP durum kodları ve hata mesajları döner.
"""

from flask import Blueprint, request, jsonify
from model_management.model_manager import ModelManager
from config.parameters import DEFAULT_LAYER_CONFIG, MODEL_SAVE_PATH, MODEL_DIR
import logging
import torch

# Blueprint tanımı
model_routes = Blueprint("model_routes", __name__)

# Logger yapılandırması
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
api_logger = logging.getLogger("model_routes")

# ModelManager örneği
model_manager = ModelManager(config=DEFAULT_LAYER_CONFIG)

# API Endpointleri

@model_routes.route("/initialize", methods=["POST"])
def initialize_model():
    """
    Model ve bileşenlerin başlatılmasını sağlar.
    """
    try:
        model_manager.initialize()

        # Modeli kaydet
        model_manager.save(save_dir=MODEL_DIR, model_name='cevahir_model.pth')
        return jsonify({"message": "Model ve bileşenler başarıyla başlatıldı ve kaydedildi."}), 200
    except Exception as e:
        api_logger.error(f"Model başlatılamadı: {str(e)}")
        return jsonify({"error": "Model başlatılamadı.", "details": str(e)}), 500




@model_routes.route("/forward", methods=["POST"])
def forward():
    """
    Modelin ileri yayılım işlemini gerçekleştirir.
    """
    try:
        # JSON veriyi al
        data = request.get_json()
        inputs = data.get("inputs")
        context = data.get("context", None)
        environment_data = data.get("environment_data", None)

        # Giriş verilerini tensöre dönüştür
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

        context_tensor = torch.tensor(context, dtype=torch.float32) if context else None
        environment_tensor = torch.tensor(environment_data, dtype=torch.float32) if environment_data else None

        # İleri yayılım işlemi
        outputs = model_manager.forward(
            inputs=inputs_tensor, context=context_tensor, environment_data=environment_tensor
        )
        return jsonify({"outputs": {k: v.tolist() for k, v in outputs.items()}}), 200

    except ValueError as ve:
        api_logger.error(f"Şekil doğrulama hatası: {str(ve)}")
        return jsonify({"error": "Şekil doğrulama hatası.", "details": str(ve)}), 400

    except Exception as e:
        api_logger.error(f"İleri yayılım hatası: {str(e)}")
        return jsonify({"error": "İleri yayılım hatası.", "details": str(e)}), 500


@model_routes.route("/save", methods=["POST"])
def save_model():
    """
    Modeli belirtilen bir dosyaya kaydeder.
    """
    try:
        # Modeli kaydet
        model_manager.save(save_dir=MODEL_DIR, model_name='cevahir_model.pth')
        return jsonify({"message": f"Model başarıyla kaydedildi: {MODEL_SAVE_PATH}"}), 200

    except Exception as e:
        api_logger.error(f"Model kaydedilemedi: {str(e)}")
        return jsonify({"error": "Model kaydedilemedi.", "details": str(e)}), 500


@model_routes.route("/load", methods=["POST"])
def load_model():
    """
    Daha önce kaydedilen bir modeli yükler.
    """
    try:
        # Modeli yükle
        model_manager.load(model_path=MODEL_SAVE_PATH)
        return jsonify({"message": f"Model başarıyla yüklendi: {MODEL_SAVE_PATH}"}), 200

    except Exception as e:
        api_logger.error(f"Model yüklenemedi: {str(e)}")
        return jsonify({"error": "Model yüklenemedi.", "details": str(e)}), 500


@model_routes.route("/update", methods=["POST"])
def update_model():
    """
    Model parametrelerini günceller.
    """
    try:
        # JSON veriyi al
        data = request.get_json()
        update_params = data["update_params"]

        # Modeli güncelle
        model_manager.update(update_params)
        return jsonify({"message": "Model parametreleri başarıyla güncellendi."}), 200

    except Exception as e:
        api_logger.error(f"Model parametreleri güncellenemedi: {str(e)}")
        return jsonify({"error": "Model parametreleri güncellenemedi.", "details": str(e)}), 500
