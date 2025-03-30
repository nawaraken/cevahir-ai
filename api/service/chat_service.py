import logging
import torch
import traceback
from chatting_organizator.chatting_organization import ChattingOrganization
from src.neural_network import CevahirNeuralNetwork
from detokenization_system.detokenization_system_modules.detokenization_pipeline import DetokenizationPipeline
from config.parameters import MAX_INPUT_LENGTH, LOGGING_PATH, MODEL_SAVE_PATH, DEVICE, INPUT_DIM, JSON_VOCAB_SAVE_PATH, HIDDEN_SIZE, OUTPUT_SIZE

class ChatService:
    """
    ChatService, ChattingOrganization ve Cevahir Neural Network ile sohbet süreçlerini yönetir.
    """

    def __init__(self):
        """
        ChatService başlatılır. ChattingOrganization, model ve detokenization pipeline entegre edilir.
        """
        # Logger başlatılır
        self.logger = self._initialize_logger()
        self.logger.info("ChatService başlatılıyor...")

        # ChattingOrganization başlatılır
        self.chat_organization = self._initialize_chat_organization()

        # Maksimum giriş uzunluğu kontrolü
        self.max_input_length = self._initialize_max_input_length()

        # Model yükleme
        self.model = self._initialize_model()

        # DetokenizationPipeline başlatma
        self.detokenization_pipeline = self._initialize_detokenization_pipeline()

    def _initialize_logger(self):
        """
        Logger yapılandırmasını başlatır.
        """
        logger = logging.getLogger("ChatService")
        logger.setLevel(logging.DEBUG)

        # Dosya loglama
        file_handler = logging.FileHandler(f"{LOGGING_PATH}/chat_service.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        # Konsol loglama
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)

        return logger

    def _initialize_chat_organization(self):
        try:
            chat_organization = ChattingOrganization()
            self.logger.info("ChattingOrganization başarıyla yüklendi.")
            return chat_organization
        except Exception:
            self.logger.error(f"ChattingOrganization yüklenirken hata oluştu: {traceback.format_exc()}")
            raise RuntimeError("ChattingOrganization başlatılamadı.")

    def _initialize_max_input_length(self):
        if MAX_INPUT_LENGTH <= 0:
            self.logger.error("MAX_INPUT_LENGTH yapılandırma hatası. Pozitif bir değer bekleniyor.")
            raise ValueError("MAX_INPUT_LENGTH geçerli bir değer değil.")
        self.logger.info(f"Maksimum giriş uzunluğu: {MAX_INPUT_LENGTH}")
        return MAX_INPUT_LENGTH

    def _initialize_model(self):
        try:
            self.logger.info("Model yükleniyor...")
            model = CevahirNeuralNetwork(
                input_size=INPUT_DIM,
                hidden_layers=[HIDDEN_SIZE],
                output_size=OUTPUT_SIZE
            )
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state'])
            model.to(DEVICE)
            model.eval()  # Model değerlendirme modunda çalıştırılıyor
            self.logger.info("Model başarıyla yüklendi.")
            return model
        except Exception:
            self.logger.error(f"Model yüklenirken hata oluştu: {traceback.format_exc()}")
            raise RuntimeError("Model yüklenemedi.")

    def _initialize_detokenization_pipeline(self):
        try:
            detokenization_pipeline = DetokenizationPipeline(JSON_VOCAB_SAVE_PATH)
            self.logger.info("DetokenizationPipeline başarıyla başlatıldı.")
            return detokenization_pipeline
        except Exception:
            self.logger.error(f"DetokenizationPipeline başlatılırken hata oluştu: {traceback.format_exc()}")
            raise RuntimeError("DetokenizationPipeline başlatılamadı.")

    def send_message(self, session_id: str, message: str):
        """
        Kullanıcı mesajını işler ve model yanıtını döner.

        Args:
            session_id (str): Oturum kimliği.
            message (str): Kullanıcı mesajı.

        Returns:
            dict: Model yanıtı.
        """
        try:
            if len(message) > self.max_input_length:
                self.logger.warning("Mesaj uzunluğu sınırını aşıyor. Kesiliyor.")
                message = message[:self.max_input_length]

            # Mesajı işleme
            processed_message = self.chat_organization.process_message(session_id, message)
            if processed_message["status"] != "success":
                return processed_message

            # Model girdisi hazırlama
            tokenized_input = self._prepare_model_input(processed_message["data"]["preprocessed_message"])

            # Model yanıtını üretme
            with torch.no_grad():
                output = self.model(tokenized_input)
            response = self._decode_model_output(output)

            # Yanıtı bağlama ekleme
            self.chat_organization.chat_manager.context_manager.set_context(session_id, "last_response", response)

            self.logger.info(f"Model yanıtı başarıyla üretildi: {response}")
            return {"status": "success", "response": response}
        except Exception:
            self.logger.error(f"Mesaj işlenirken hata oluştu: {traceback.format_exc()}")
            return {"status": "error", "message": "Mesaj işlenirken bir hata oluştu."}

    def _prepare_model_input(self, preprocessed_message: str) -> torch.Tensor:
        """
        Model için uygun formatta giriş hazırlar.

        Args:
            preprocessed_message (str): İşlenmiş mesaj.

        Returns:
            torch.Tensor: Model girdisi.

        Raises:
            ValueError: Geçersiz veya boş tokenizasyon sonucu.
            RuntimeError: Model girdisi hazırlanırken hata oluşursa.
        """
        try:
            if not isinstance(preprocessed_message, str) or not preprocessed_message.strip():
                self.logger.error("Geçersiz mesaj formatı. Boş veya eksik giriş.")
                raise ValueError("Geçersiz mesaj. Giriş bir dize olmalıdır.")

            tokenized_message = self.chat_organization.chat_manager.message_tokenizer.tokenize_message(
                preprocessed_message
            )
            if not tokenized_message:
                self.logger.error("Tokenizasyon başarısız oldu.")
                raise ValueError("Boş veya hatalı tokenizasyon sonucu alındı.")

            padded_message = self._pad_or_truncate(tokenized_message)
            tokenized_input = torch.tensor(padded_message, dtype=torch.long)
            return tokenized_input.unsqueeze(0).to(DEVICE)
        except Exception:
            self.logger.error(f"Model girdisi hazırlanırken hata oluştu: {traceback.format_exc()}")
            raise RuntimeError("Model girdisi hazırlanamadı.")

    def _pad_or_truncate(self, tokenized_message: list):
        """
        Token listesini modelin beklediği INPUT_DIM uzunluğuna pad veya truncate eder.

        Args:
            tokenized_message (list): Token listesi.

        Returns:
            list: Pad edilmiş veya kısaltılmış token listesi.
        """
        if len(tokenized_message) > INPUT_DIM:
            return tokenized_message[:INPUT_DIM]
        return tokenized_message + [0] * (INPUT_DIM - len(tokenized_message))

    def _decode_model_output(self, output):
        """
        Model çıktısını anlamlı bir yanıta dönüştürür.

        Args:
            output (torch.Tensor): Model çıktısı.

        Returns:
            str: Modelin insan okunabilir yanıtı.
        """
        try:
            decoded_output = self.detokenization_pipeline.run_pipeline(output.tolist())
            return decoded_output.strip()
        except Exception:
            self.logger.error(f"Model çıktısı çözülürken hata oluştu: {traceback.format_exc()}")
            raise RuntimeError("Model çıktısı çözülürken hata oluştu.")
