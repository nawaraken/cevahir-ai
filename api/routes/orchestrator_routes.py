from flask import Blueprint, jsonify
from orchestrator.json_orchestrator import JSONOrchestrator
from config.parameters import PROCESSED_DATA_DIR

# Blueprint tanımlaması
orchestrator_routes = Blueprint('orchestrator_routes', __name__)

@orchestrator_routes.route('/run_orchestrator', methods=['POST'])
def run_orchestrator():
    """
    JSON Orchestrator'ı çalıştıran endpoint.
    """
    try:
        orchestrator = JSONOrchestrator()
        orchestrator.run()
        return jsonify({
            "message": "Orchestrator çalıştırıldı ve işlemler başarıyla tamamlandı.",
            "processed_data_dir": PROCESSED_DATA_DIR,
            "status": "success"
        }), 200
    except Exception as e:
        return jsonify({
            "message": f"Orchestrator çalıştırılırken hata oluştu: {str(e)}",
            "status": "error"
        }), 500
