import pytest
from tokenizer_management.chatting.chatting_manager import ChattingManager, ChattingManagerError

# Dummy implementations for testing purposes.
class DummyPreprocessor:
    def preprocess(self, text: str) -> str:
        # For testing, simply lowercases and strips the text.
        return text.lower().strip()

class DummyTokenizer:
    def tokenize(self, text: str) -> list:
        # Splits the text by whitespace.
        return text.split()

class DummyEncoder:
    def encode(self, tokens: list) -> list:
        # Identity mapping (tokens remain as-is).
        return tokens

class DummyDecoder:
    def decode(self, token_ids: list) -> list:
        # Identity mapping (token IDs remain as tokens).
        return token_ids

class DummyPostprocessor:
    def process(self, tokens: list) -> str:
        # Joins tokens with a single space.
        return " ".join(tokens)

@pytest.fixture
def dummy_chatting_manager():
    # Create an instance of ChattingManager.
    manager = ChattingManager()
    # Override its submodules with the dummy implementations.
    manager.preprocessor = DummyPreprocessor()
    manager.tokenizer = DummyTokenizer()
    manager.encoder = DummyEncoder()
    manager.decoder = DummyDecoder()
    manager.postprocessor = DummyPostprocessor()
    return manager

def test_process_chat(dummy_chatting_manager):
    """
    Test that processing a chat message returns the expected final text.
    For the dummy implementations:
      - preprocess: lowercases and strips,
      - tokenize: splits by whitespace,
      - encode: identity,
      - decode: identity,
      - postprocess: joins tokens with a space.
    So input "Hello World" should yield "hello world".
    """
    input_text = "Hello World"
    result = dummy_chatting_manager.process_chat(input_text)
    assert result == "hello world"

def test_encode_chat(dummy_chatting_manager):
    """
    Test that encoding a chat message returns the expected token list.
    """
    input_text = "Test Chat Message"
    token_ids = dummy_chatting_manager.encode_chat(input_text)
    # Expecting tokens to be lowercased and split
    assert token_ids == ["test", "chat", "message"]

def test_decode_chat(dummy_chatting_manager):
    """
    Test that decoding a given token ID list returns the properly postprocessed chat message.
    """
    token_ids = ["this", "is", "a", "test"]
    result = dummy_chatting_manager.decode_chat(token_ids)
    # Postprocessor joins tokens with a space.
    assert result == "this is a test"

def test_get_chat_tokens(dummy_chatting_manager):
    """
    Test that the get_chat_tokens method returns the expected list of tokens.
    """
    input_text = "Another test message"
    tokens = dummy_chatting_manager.get_chat_tokens(input_text)
    assert tokens == ["another", "test", "message"]

def test_empty_chat(dummy_chatting_manager):
    """
    Test the behavior when an empty chat message is provided.
    Depending on the design, an empty string may result in an empty string output.
    """
    input_text = "   "
    result = dummy_chatting_manager.process_chat(input_text)
    # With the dummy preprocessor (which lowercases and strips), empty input becomes empty.
    assert result == ""
