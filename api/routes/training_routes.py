import os
import logging
import numpy as np
import requests
import torch
from flask import Blueprint, jsonify, request
from torch.utils.data import DataLoader, TensorDataset
from training_management.training_manager import TrainingManager
from config.parameters import (
    DEVICE,
    INPUT_DIM,
    MODEL_TRAINING_DATA_DIR,
    TRAIN_TIMEOUT,
    DEFAULT_LAYER_CONFIG,
    BATCH_SIZE,
    EPOCHS,
    MODEL_ROUTE_BASE_URL,
    LEARNING_RATE,
    SEQ_LEN,
    OUTPUT_DIM,
)

# Blueprint tanımlaması
training_routes = Blueprint('training_routes', __name__)

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
training_logger = logging.getLogger("training_routes")


def create_data_loaders(training_data, training_labels, batch_size=BATCH_SIZE):
    """
    Verilen eğitim verisi ve etiketlerden DataLoader nesneleri oluşturur.

    Args:
        training_data (numpy.ndarray): Eğitim verisi.
        training_labels (numpy.ndarray): Eğitim etiketleri.
        batch_size (int): Batch boyutu.

    Returns:
        DataLoader: Eğitim ve doğrulama DataLoader nesneleri.
    """
    # PyTorch tensörlerine dönüştür
    train_data_tensor = torch.tensor(training_data, dtype=torch.float32)
    train_labels_tensor = torch.tensor(training_labels, dtype=torch.long)

    # Eğitim verisi için DataLoader oluştur
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Doğrulama için aynı verileri kullanıyoruz
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


@training_routes.route('/train_model', methods=['POST'])
def train_model():
    """
    Modeli eğitmek için bir endpoint.
    Eğitim verisi ve etiketleri dosya sisteminden yüklenir ve model eğitimi başlatılır.
    """
    try:
        # Modeli initialize et
        init_response = requests.post(f"{MODEL_ROUTE_BASE_URL}/initialize")
        if init_response.status_code != 200:
            error_message = init_response.json().get("details", "Bilinmeyen hata")
            training_logger.error(f"Model initialize edilemedi: {error_message}")
            return jsonify({"status": "error", "message": f"Model initialize edilemedi: {error_message}"}), 500

        # Tensor verilerini yükleme
        training_data_path = os.path.join(
            MODEL_TRAINING_DATA_DIR, 
            f"training_data_dim{INPUT_DIM}_seq{SEQ_LEN}_out{OUTPUT_DIM}.npy"
        )
        labels_path = os.path.join(
            MODEL_TRAINING_DATA_DIR, 
            f"training_labels_dim{INPUT_DIM}_seq{SEQ_LEN}_out{OUTPUT_DIM}.npy"
        )

        if not os.path.exists(training_data_path):
            error_message = f"Eğitim verisi dosyası bulunamadı: {training_data_path}"
            training_logger.error(error_message)
            return jsonify({"message": error_message, "status": "error"}), 404

        if not os.path.exists(labels_path):
            error_message = f"Etiket dosyası bulunamadı: {labels_path}"
            training_logger.error(error_message)
            return jsonify({"message": error_message, "status": "error"}), 404

        # Eğitim verilerini ve etiketleri yükleme
        training_data = np.load(training_data_path)
        training_labels = np.load(labels_path)

        training_logger.info(f"Eğitim verisi başarıyla yüklendi. Boyutlar: {training_data.shape}")
        training_logger.info(f"Etiketler başarıyla yüklendi. Boyutlar: {training_labels.shape}")

        # DataLoader'ları oluştur
        train_loader, val_loader = create_data_loaders(training_data, training_labels)

        # TrainingManager başlat
        from src.neural_network import CevahirNeuralNetwork
        model = CevahirNeuralNetwork(config=DEFAULT_LAYER_CONFIG)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.CrossEntropyLoss()

        training_manager = TrainingManager(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion
        )

        # Modeli eğit
        training_logger.info("Model eğitimi başlatılıyor...")
        training_manager.train()

        return jsonify({"status": "success", "message": "Model başarıyla eğitildi."}), 200

    except Exception as e:
        error_message = f"Eğitim sırasında bir hata oluştu: {str(e)}"
        training_logger.error(error_message)
        return jsonify({"status": "error", "message": error_message}), 500






@training_routes.route('/load_model', methods=['POST'])
def load_model():
    """
    Kaydedilmiş modeli yüklemek için bir endpoint.
    """
    try:
        load_response = requests.post(f"{MODEL_ROUTE_BASE_URL}/load")
        if load_response.status_code != 200:
            error_message = load_response.json().get("details", "Bilinmeyen hata")
            training_logger.error(f"Model yüklenemedi: {error_message}")
            return jsonify({"status": "error", "message": f"Model yüklenemedi: {error_message}"}), 500

        return jsonify({"status": "success", "message": "Model başarıyla yüklendi."}), 200

    except requests.exceptions.RequestException as e:
        training_logger.error(f"Model yükleme sırasında bağlantı hatası: {str(e)}")
        return jsonify({"status": "error", "message": f"Model yükleme sırasında bağlantı hatası: {str(e)}"}), 500
    except Exception as e:
        error_message = f"Model yüklenirken bir hata oluştu: {str(e)}"
        training_logger.error(error_message)
        return jsonify({"status": "error", "message": error_message}), 500


@training_routes.route('/status', methods=['GET'])
def training_status():
    """
    Eğitim servisinin durumunu kontrol etmek için bir endpoint.
    """
    try:
        response = requests.get(f"{MODEL_ROUTE_BASE_URL}/status")
        if response.status_code != 200:
            return jsonify({"status": "error", "message": "Model durumu alınamadı."}), 500

        status_details = response.json()
        return jsonify({"status": "success", "details": status_details}), 200
    except Exception as e:
        error_message = f"Durum kontrolü sırasında hata oluştu: {str(e)}"
        training_logger.error(error_message)
        return jsonify({"status": "error", "message": error_message}), 500
