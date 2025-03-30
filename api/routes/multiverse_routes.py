from flask import Blueprint, jsonify, request
from threading import Thread
from config.parameters import MULTIVERSE_LOG_PATH
from api.service.multiverse_service import MultiverseService

# Blueprint oluşturma
multiverse_bp = Blueprint('multiverse', __name__)

# MultiverseService örneği
multiverse_service = MultiverseService(log_file=MULTIVERSE_LOG_PATH)


def generate_endpoints():
    """
    Dinamik olarak endpointler oluşturur ve multiverse_bp'ye ekler.
    Endpointlerin yapısı, tanımlanan yapılandırmalara veya meta-verilere bağlıdır.
    """
    # Dinamik olarak oluşturulacak endpointlerin yapılandırması
    dynamic_routes = [
        {
            "endpoint": "/dynamic/<string:name>",
            "methods": ["GET"],
            "handler": dynamic_handler,
            "description": "Dinamik bir endpoint örneği."
        },
        {
            "endpoint": "/dynamic/log/<int:log_id>",
            "methods": ["GET"],
            "handler": get_log_by_id,
            "description": "Belirli bir log kaydını döndürür."
        }
    ]

    # Dinamik endpointleri Blueprint'e ekleme
    for route in dynamic_routes:
        multiverse_bp.add_url_rule(
            route["endpoint"],
            view_func=route["handler"],
            methods=route["methods"]
        )


def dynamic_handler(name):
    """
    Dinamik bir endpoint için handler.
    Kullanıcı tarafından sağlanan bir parametreyi işler.
    """
    message = f"Hello, {name}! This is a dynamic endpoint."
    return jsonify({"message": message, "timestamp": multiverse_service.virtual_time.strftime("%Y-%m-%d %H:%M:%S")})


def get_log_by_id(log_id):
    """
    Belirli bir log kaydını döndürür.
    """
    try:
        with open(MULTIVERSE_LOG_PATH, "r") as log_file:
            logs = log_file.readlines()

        if log_id < 0 or log_id >= len(logs):
            return jsonify({"error": "Log ID out of range"}), 404

        return jsonify({"log_id": log_id, "log": logs[log_id]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@multiverse_bp.route('/start_lifecycle', methods=['POST'])
def start_lifecycle():
    """
    Multiverse yaşam döngüsünü başlatır.
    """
    if multiverse_service.is_running:
        return jsonify({"message": "Lifecycle is already running"}), 400

    thread = Thread(target=multiverse_service.life_cycle)
    multiverse_service.is_running = True
    thread.start()
    return jsonify({"message": "Lifecycle started"}), 200


@multiverse_bp.route('/stop_lifecycle', methods=['POST'])
def stop_lifecycle():
    """
    Multiverse yaşam döngüsünü durdurur.
    """
    if not multiverse_service.is_running:
        return jsonify({"message": "Lifecycle is not running"}), 400

    multiverse_service.stop_life_cycle()
    return jsonify({"message": "Lifecycle stopped"}), 200


@multiverse_bp.route('/status', methods=['GET'])
def status():
    """
    MultiverseService'in mevcut durumunu döndürür.
    """
    status = {
        "is_running": multiverse_service.is_running,
        "thought_depth": multiverse_service.internal_state["thought_depth"],
        "tasks_completed": multiverse_service.internal_state["tasks_completed"],
        "active_thoughts": len(multiverse_service.internal_state["active_thoughts"]),
        "universal_knowledge_processed": multiverse_service.internal_state["universal_knowledge_processed"],
    }
    return jsonify(status), 200


@multiverse_bp.route('/philosophical_inquiry', methods=['POST'])
def philosophical_inquiry():
    """
    Felsefi bir sorgulama başlatır ve sonucu döndürür.
    """
    multiverse_service.felsefi_sorgulama()
    return jsonify({"message": "Philosophical inquiry initiated"}), 200


@multiverse_bp.route('/start_game', methods=['POST'])
def start_game():
    """
    Multiverse evrenindeki oyunu başlatır.
    """
    try:
        multiverse_service.start_game_in_multiverse()
        return jsonify({"message": "Game started successfully in the multiverse"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@multiverse_bp.route('/log', methods=['GET'])
def get_logs():
    """
    MultiverseService loglarını döndürür.
    """
    try:
        with open(MULTIVERSE_LOG_PATH, "r") as log_file:
            logs = log_file.readlines()
        return jsonify({"logs": logs}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Dinamik endpointleri oluşturma
generate_endpoints()
