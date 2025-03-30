import sys
import os
import pytest
import torch
from unittest.mock import patch, MagicMock

# Proje dizinini sys.path'e ekleme
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.service.chat_service import ChatService
from modules.json_tokenizer import JsonTokenizer
from modules.json_vocab_manager import VocabManager
from api.service.channel_service import ChannelService


@pytest.fixture
def mock_chat_service():
    """ChatService için bir mock nesnesi oluşturur."""
    with patch("api.service.chat_service.ChannelService") as MockChannelService, \
         patch("modules.json_vocab_manager.VocabManager") as MockVocabManager, \
         patch("modules.json_tokenizer.JsonTokenizer") as MockJsonTokenizer, \
         patch("src.neural_network.CevahirNeuralNetwork") as MockNeuralNetwork:

        # Mock nesneleri oluştur
        mock_channel_service = MockChannelService.return_value
        mock_vocab_manager = MockVocabManager.return_value
        mock_tokenizer = MockJsonTokenizer.return_value
        mock_neural_network = MockNeuralNetwork.return_value

        # ChatService başlat
        chat_service = ChatService()

        # Mock bileşenlerini döndür
        chat_service.channel_service = mock_channel_service
        chat_service.vocab_manager = mock_vocab_manager
        chat_service.tokenizer = mock_tokenizer
        chat_service.model = mock_neural_network

        yield chat_service


def test_create_session(mock_chat_service):
    """Oturum oluşturma işlemini test eder."""
    mock_chat_service.channel_service.create_session.return_value = {
        "session_id": "test_session_id"
    }

    session_id = mock_chat_service.create_session(user_id="test_user")

    assert session_id == "test_session_id"
    mock_chat_service.channel_service.create_session.assert_called_once_with(user_id="test_user")


def test_send_message_success(mock_chat_service):
    """Mesaj gönderme işleminin başarılı bir şekilde çalıştığını test eder."""
    mock_chat_service.channel_service.validate_session.return_value = {"status": "success"}
    mock_chat_service.tokenizer.tokenize.return_value = ["hello", "world"]
    mock_chat_service.vocab_manager.word_to_id.side_effect = [1, 2]
    mock_chat_service.model.return_value = torch.tensor([[0.1, 0.9]])
    mock_chat_service.vocab_manager.id_to_word.side_effect = {1: "Hi", 2: "There"}.get

    response = mock_chat_service.send_message(session_id="test_session", message="Hello World")

    assert response["status"] == "success"
    assert "response" in response
    mock_chat_service.channel_service.validate_session.assert_called_once_with("test_session")


def test_send_message_invalid_session(mock_chat_service):
    """Geçersiz oturum kimliği ile mesaj gönderme işlemini test eder."""
    mock_chat_service.channel_service.validate_session.return_value = {"status": "error", "message": "Invalid session"}

    response = mock_chat_service.send_message(session_id="invalid_session", message="Hello World")

    assert response["status"] == "error"
    assert response["message"] == "Invalid session"
    mock_chat_service.channel_service.validate_session.assert_called_once_with("invalid_session")


def test_prepare_model_input(mock_chat_service):
    """Model girdisi hazırlama işlemini test eder."""
    mock_chat_service.tokenizer.tokenize.return_value = ["test", "input"]
    mock_chat_service.vocab_manager.word_to_id.side_effect = [5, 10]

    input_tensor = mock_chat_service._prepare_model_input("Test input")

    assert input_tensor.shape[1] == 2  # İki token olmalı
    assert input_tensor.dtype == torch.long
    mock_chat_service.tokenizer.tokenize.assert_called_once_with("Test input")


def test_decode_output(mock_chat_service):
    """Model çıktısını çözümleme işlemini test eder."""
    mock_chat_service.vocab_manager.id_to_word.side_effect = {1: "Hello", 2: "World"}.get

    output_tensor = torch.tensor([[1, 2]])
    decoded_output = mock_chat_service._decode_output(output_tensor)

    assert decoded_output == "Hello World"


def test_get_session_history(mock_chat_service):
    """Oturum geçmişi alma işlemini test eder."""
    mock_chat_service.channel_service.load_session_data.return_value = [
        {"Soru": "Test?", "Cevap": "This is a test response."}
    ]

    history = mock_chat_service.get_session_history("test_session_id")

    assert len(history) == 1
    assert history[0]["Soru"] == "Test?"
    mock_chat_service.channel_service.load_session_data.assert_called_once_with("test_session_id")


def test_save_session_data(mock_chat_service):
    """Oturum verisini kaydetme işlemini test eder."""
    mock_chat_service.save_session_data("test_session_id")

    mock_chat_service.channel_service.save_session_data.assert_called_once_with("test_session_id")


def test_send_message_tokenize_failure(mock_chat_service):
    """Mesaj tokenize edilirken oluşan hata durumunu test eder."""
    mock_chat_service.tokenizer.tokenize.side_effect = RuntimeError("Tokenization failed")

    with pytest.raises(RuntimeError) as excinfo:
        mock_chat_service._prepare_model_input("Test input")

    assert "Tokenization failed" in str(excinfo.value)


def test_decode_output_invalid_token(mock_chat_service):
    """Geçersiz bir token çıktısı çözümleme işlemini test eder."""
    mock_chat_service.vocab_manager.id_to_word.side_effect = {1: "Valid"}.get

    output_tensor = torch.tensor([[1, 999]])  # 999 tanımsız bir token
    decoded_output = mock_chat_service._decode_output(output_tensor)

    assert decoded_output == "Valid anlaşılamayan kelime"


def test_send_message_with_context(mock_chat_service):
    """Bağlam verisiyle mesaj gönderme işlemini test eder."""
    mock_chat_service.channel_service.validate_session.return_value = {"status": "success"}
    mock_chat_service.tokenizer.tokenize.return_value = ["contextual", "message"]
    mock_chat_service.vocab_manager.word_to_id.side_effect = [3, 4]
    context_tensor = torch.tensor([[10, 20]])

    mock_chat_service.model.return_value = torch.tensor([[0.5, 0.5]])
    response = mock_chat_service.send_message(session_id="context_session", message="Contextual message", context=context_tensor)

    assert response["status"] == "success"
    assert "response" in response
    mock_chat_service.channel_service.validate_session.assert_called_once_with("context_session")
