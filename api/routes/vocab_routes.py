from flask import Blueprint, request, jsonify, current_app
import os
from modules.json_vocab_manager import VocabManager  # VocabManager sınıfını içe aktarıyoruz
from config.parameters import MIN_FREQ, MAX_VOCAB_SIZE, JSON_VOCAB_SAVE_PATH # Parametreleri import ediyoruz
import logging
import json

vocab_routes = Blueprint('vocab_routes', __name__)

# Kelime haznesi oluşturma endpoint'i
@vocab_routes.route('/vocab-create', methods=['POST'])
def create_vocab():
    try:
        # Tokenize edilmiş veri alımı
        data = request.get_json()

        # Veri doğrulama: Gelen verinin tokenize edilmiş bir formata sahip olup olmadığı kontrol ediliyor
        if not data or 'tokens' not in data:
            current_app.logger.error("Hatalı veri formatı: 'tokens' anahtarı eksik.")
            return jsonify({"error": "Tokenize edilmiş verinin 'tokens' anahtarı eksik."}), 400

        tokens = data['tokens']

        # Vocab oluşturma işlemi
        vocab_manager = VocabManager()  # VocabManager sınıfından bir örnek oluştur
        vocab_manager.build_vocab(tokens)  # build_vocab metodunu çağır

        # Vocab kaydedilmesi
        os.makedirs(os.path.dirname(JSON_VOCAB_SAVE_PATH), exist_ok=True)
        with open(JSON_VOCAB_SAVE_PATH, 'w', encoding='utf-8') as vocab_file:
            json.dump(vocab_manager.vocab, vocab_file, ensure_ascii=False, indent=4)  # Kelime haznesini JSON formatında kaydet

        # Başarı yanıtı ve loglama
        current_app.logger.info(f"Vocab başarıyla oluşturuldu ve {JSON_VOCAB_SAVE_PATH} yoluna kaydedildi.")
        return jsonify({"vocab": vocab_manager.vocab, "message": "Kelime haznesi başarıyla oluşturuldu."}), 200

    except Exception as e:
        # Hata loglama ve kullanıcıya hata mesajı döndürme
        current_app.logger.error(f"Vocab oluşturma başarısız oldu: {str(e)}")
        return jsonify({"error": "Vocab oluşturma başarısız oldu"}), 500
