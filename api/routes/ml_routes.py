# api/routes/ml_routes.py
from flask import Blueprint, request, jsonify, current_app
from modules.machine_learning_manager import MachineLearningManager  # MachineLearningManager sınıfını import ediyoruz
from config.parameters import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, GRADIENT_CLIP, LOG_INTERVAL, 
    VALIDATION_SPLIT, MODEL_SAVE_PATH, MAX_DEVICE_USAGE
)
import logging
import os

ml_routes = Blueprint('ml_routes', __name__)

# Model Eğitim Endpoint'i
@ml_routes.route('/train-model', methods=['POST'])
def train_model_route():
    try:
        # Eğitim verisi alımı
        data = request.get_json()
        if not data or 'training_data' not in data:
            current_app.logger.error("Eğitim verisi eksik veya hatalı.")
            return jsonify({"error": "Eğitim verisi eksik veya hatalı."}), 400

        training_data = data['training_data']

        # MachineLearningManager örneğini oluştur
        model = ...  # Burada modelinizi tanımlayın
        vocab = ...  # Burada vocab'inizi tanımlayın
        ml_manager = MachineLearningManager(model, vocab)

        # Model eğitimi başlat
        results = ml_manager.train_model(training_data)

        # Eğitim sonucu yanıtı
        current_app.logger.info("Model eğitimi başarıyla tamamlandı.")
        return jsonify({"message": "Model eğitimi başarıyla tamamlandı.", "results": results}), 200

    except Exception as e:
        # Hata loglama
        current_app.logger.error(f"Model eğitimi başarısız oldu: {str(e)}")
        return jsonify({"error": "Model eğitimi başarısız oldu"}), 500

# Model Doğrulama Endpoint'i
@ml_routes.route('/validate-model', methods=['POST'])
def validate_model_route():
    try:
        # Doğrulama verisi alımı
        data = request.get_json()
        if not data or 'validation_data' not in data:
            current_app.logger.error("Doğrulama verisi eksik veya hatalı.")
            return jsonify({"error": "Doğrulama verisi eksik veya hatalı."}), 400

        validation_data = data['validation_data']

        # MachineLearningManager örneğini oluştur
        model = ...  # Burada modelinizi tanımlayın
        vocab = ...  # Burada vocab'inizi tanımlayın
        ml_manager = MachineLearningManager(model, vocab)

        # Model doğrulama işlemi
        results = ml_manager.validate_model(validation_data)

        # Doğrulama sonucu yanıtı
        current_app.logger.info("Model doğrulaması başarıyla tamamlandı.")
        return jsonify({"message": "Model doğrulaması başarıyla tamamlandı.", "results": results}), 200

    except Exception as e:
        # Hata loglama
        current_app.logger.error(f"Model doğrulama başarısız oldu: {str(e)}")
        return jsonify({"error": "Model doğrulama başarısız oldu"}), 500

# Model Test Endpoint'i
@ml_routes.route('/test-model', methods=['POST'])
def test_model_route():
    try:
        # Test verisi alımı
        data = request.get_json()
        if not data or 'test_data' not in data:
            current_app.logger.error("Test verisi eksik veya hatalı.")
            return jsonify({"error": "Test verisi eksik veya hatalı."}), 400

        test_data = data['test_data']

        # MachineLearningManager örneğini oluştur
        model = ...  # Burada modelinizi tanımlayın
        vocab = ...  # Burada vocab'inizi tanımlayın
        ml_manager = MachineLearningManager(model, vocab)

        # Model test işlemi
        results = ml_manager.test_model(test_data)

        # Test sonucu yanıtı
        current_app.logger.info("Model test süreci başarıyla tamamlandı.")
        return jsonify({"message": "Model test süreci başarıyla tamamlandı.", "results": results}), 200

    except Exception as e:
        # Hata loglama
        current_app.logger.error(f"Model test işlemi başarısız oldu: {str(e)}")
        return jsonify({"error": "Model test işlemi başarısız oldu"}), 500
