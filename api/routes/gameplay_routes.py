from flask import Blueprint, jsonify, request
import requests
from config.parameters import MAX_GAME_STEPS, MODEL_ROUTE_BASE_URL

# Blueprint oluşturma
gameplay_routes = Blueprint('gameplay', __name__)

# Endpointler

@gameplay_routes.route('/play_game', methods=['POST'])
def play_game():
    """
    Oyunu başlatır ve modeli kullanarak bir oyun oynar.
    """
    try:
        # Modelin başlatılması veya yüklenmesi
        init_response = requests.post(f"{MODEL_ROUTE_BASE_URL}/initialize")
        if init_response.status_code != 200:
            return jsonify({
                "status": "error",
                "message": "Model başlatılamadı.",
                "details": init_response.json().get("details", "Bilinmiyor")
            }), init_response.status_code

        # Girdi verisinin alınması
        data = request.get_json()
        if not data or "inputs" not in data:
            return jsonify({
                "status": "error",
                "message": "Girdi verisi eksik. 'inputs' parametresi gereklidir."
            }), 400

        inputs = data.get("inputs")
        if not isinstance(inputs, list) or len(inputs) == 0:
            return jsonify({
                "status": "error",
                "message": "'inputs' bir liste olmalı ve boş olmamalıdır."
            }), 400

        # Modelin ileri yayılım işlemi
        forward_response = requests.post(f"{MODEL_ROUTE_BASE_URL}/forward", json={"inputs": inputs})
        if forward_response.status_code != 200:
            return jsonify({
                "status": "error",
                "message": "İleri yayılım işlemi başarısız.",
                "details": forward_response.json().get("details", "Bilinmiyor")
            }), forward_response.status_code

        # Model çıktılarının işlenmesi
        outputs = forward_response.json().get("outputs", {})
        if not outputs:
            return jsonify({
                "status": "error",
                "message": "Modelden çıktı alınamadı."
            }), 500

        # Oyun mantığını işleme
        game_result = process_game_logic(inputs, outputs)

        return jsonify({
            "status": "success",
            "message": "Oyun başarıyla tamamlandı.",
            "game_result": game_result
        }), 200

    except requests.exceptions.RequestException as req_err:
        return jsonify({
            "status": "error",
            "message": f"Model API'ye bağlanılamadı: {str(req_err)}"
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Oyun sırasında beklenmeyen bir hata oluştu: {str(e)}"
        }), 500


def process_game_logic(inputs, outputs):
    """
    Oyun mantığını işleyen yardımcı fonksiyon.
    Örneğin, model tahminlerini kullanarak oyunu oynatır.
    """
    # Burada örnek bir oyun mantığı işlenebilir.
    # Model tahminlerini değerlendirerek bir sonuç döndürün.
    steps_taken = len(inputs)
    game_status = "devam ediyor" if steps_taken < MAX_GAME_STEPS else "tamamlandı"

    return {
        "inputs": inputs,
        "outputs": outputs,
        "steps_taken": steps_taken,
        "game_status": game_status,
        "max_steps_allowed": MAX_GAME_STEPS
    }
