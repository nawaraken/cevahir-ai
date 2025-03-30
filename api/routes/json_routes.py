from flask import Blueprint, request, jsonify
import os
import logging
from modules.json_data_loader import JsonDataLoader
from modules.json_tokenizer import JsonTokenizer
from config.parameters import JSON_DIR, PROCESSED_DATA_DIR, MAX_LENGTH, STOPWORDS, SPECIAL_TOKENS

json_routes = Blueprint('json_routes', __name__)

# Log ayarları
logging.basicConfig(filename='logs/errors.log', level=logging.ERROR)
process_log = logging.getLogger('process_log')
process_log.addHandler(logging.FileHandler('logs/process.log'))
process_log.setLevel(logging.INFO)

# POST /json-tokenize endpointi
@json_routes.route('/json-tokenize', methods=['POST'])
def json_tokenize():
    try:
        # JSON verisinin alınması
        data = request.get_json()
        
        # JSON doğrulama
        if not data or 'Soru' not in data or 'Cevap' not in data:
            return jsonify({"error": "Eksik veri. JSON verisi 'Soru' ve 'Cevap' alanlarını içermelidir."}), 400
        
        # Veriyi data_loader ile işleme uygunluğunu doğrula
        if not JsonDataLoader.is_valid_format(data):
            return jsonify({"error": "JSON formatı geçersiz veya işlenemiyor."}), 400
        
        # Tokenizasyon işlemini başlatma
        tokenizer = JsonTokenizer(max_length=MAX_LENGTH, stopwords=STOPWORDS, special_tokens=SPECIAL_TOKENS)
        tokenized_data = tokenizer.tokenize_text(data['Soru'], data['Cevap'])
        
        # Sonuçların kaydedilmesi
        output_file = os.path.join(PROCESSED_DATA_DIR, 'tokenized_data.json')
        with open(output_file, 'w') as f:
            f.write(tokenized_data)
        
        # İşlem başarılı, yanıt döndür
        process_log.info("Tokenizasyon başarılı: Dosya kaydedildi - " + output_file)
        return jsonify({"message": "Tokenizasyon başarılı", "tokenized_data": tokenized_data}), 200
    
    except Exception as e:
        logging.error(f"Tokenizasyon işlemi sırasında hata oluştu: {str(e)}")
        return jsonify({"error": "Tokenizasyon başarısız oldu"}), 500
