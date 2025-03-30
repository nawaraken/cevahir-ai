# api/routes/test_routes.py

from flask import Blueprint, jsonify
import unittest
import os

# Blueprint tanımla
test_routes = Blueprint("test_routes", __name__)

# Test dosyaları ve yollar
TEST_FILES = {
    "data_loader": "tests/test_data_loader.py",
    "json_tokenizer": "tests/test_json_tokenizer.py",
    "vocab_manager": "tests/test_vocab_manager.py",
    "machine_learning_manager": "tests/test_ml_manager.py"
}

def run_unittest(test_file):
    """Belirtilen test dosyasını çalıştırır ve sonucu JSON uyumlu şekilde döndürür."""
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(test_file), pattern=os.path.basename(test_file))
    result = unittest.TestResult()
    suite.run(result)
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        "details": {
            "failures": [str(failure[1]) for failure in result.failures],
            "errors": [str(error[1]) for error in result.errors]
        }
    }

@test_routes.route("/test/data_loader", methods=["POST"])
def test_data_loader():
    """data_loader modülünü test eder."""
    result = run_unittest(TEST_FILES["data_loader"])
    return jsonify({"data_loader_test_results": result})

@test_routes.route("/test/json_tokenizer", methods=["POST"])
def test_json_tokenizer():
    """json_tokenizer modülünü test eder."""
    result = run_unittest(TEST_FILES["json_tokenizer"])
    return jsonify({"json_tokenizer_test_results": result})

@test_routes.route("/test/vocab_manager", methods=["POST"])
def test_vocab_manager():
    """vocab_manager modülünü test eder."""
    result = run_unittest(TEST_FILES["vocab_manager"])
    return jsonify({"vocab_manager_test_results": result})

@test_routes.route("/test/machine_learning_manager", methods=["POST"])
def test_machine_learning_manager():
    """machine_learning_manager modülünü test eder."""
    result = run_unittest(TEST_FILES["machine_learning_manager"])
    return jsonify({"machine_learning_manager_test_results": result})
