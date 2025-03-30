import torch
import logging

class ResidualConnection:
    """
    ResidualConnection sınıfı, tensörler üzerinde artık bağlantı işlemlerini gerçekleştirir.
    """

    def __init__(self, log_level=logging.INFO, method="add"):
        """
        ResidualConnection sınıfını başlatır.

        Args:
            log_level (int): Log seviyesi (örn. logging.DEBUG, logging.INFO).
            method (str): Artık bağlantı yöntemi. "add" -> toplama, "mean" -> ortalama.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Tekrarlanan logları engellemek için bir set kullan
        if not hasattr(self.logger, "seen_messages"):
            self.logger.seen_messages = set()

        if method not in ["add", "mean"]:
            self.logger.error("Invalid method. Use 'add' or 'mean'.")
            raise ValueError("Invalid method. Use 'add' or 'mean'.")
        
        self.method = method
        self._log_once(f"ResidualConnection initialized with method={method}.")

    def _log_once(self, message):
        """
        Aynı log mesajlarının tekrar tekrar yazılmasını önler.

        Args:
            message (str): Log mesajı.
        """
        if message not in self.logger.seen_messages:
            self.logger.info(message)
            self.logger.seen_messages.add(message)

    def apply(self, tensor1, tensor2):
        """
        Artık bağlantı işlemini gerçekleştirir.

        Args:
            tensor1 (torch.Tensor): Birinci tensör.
            tensor2 (torch.Tensor): İkinci tensör.

        Returns:
            torch.Tensor: Artık bağlantı sonrası tensör.
        """
        if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
            self.logger.error("Both inputs must be torch.Tensors.")
            raise TypeError("Both inputs must be torch.Tensors.")

        self.logger.debug(f"Tensor1 shape: {tensor1.shape}, dtype: {tensor1.dtype}, device: {tensor1.device}")
        self.logger.debug(f"Tensor2 shape: {tensor2.shape}, dtype: {tensor2.dtype}, device: {tensor2.device}")

        # Tensorlerin aynı boyutta olup olmadığını kontrol et
        if tensor1.shape != tensor2.shape:
            self._log_once(f"Shape mismatch detected: {tensor1.shape} vs {tensor2.shape}. Adjusting shapes.")

            try:
                # En büyük boyuta göre genişletme yap
                max_shape = [max(tensor1.shape[i], tensor2.shape[i]) for i in range(len(tensor1.shape))]
                tensor1 = tensor1.expand(max_shape) if tensor1.shape != max_shape else tensor1
                tensor2 = tensor2.expand(max_shape) if tensor2.shape != max_shape else tensor2

                # Eğer hala eşit değilse, boyutu kırp
                min_shape = [min(tensor1.shape[i], tensor2.shape[i]) for i in range(len(tensor1.shape))]
                tensor1 = tensor1[:, :min_shape[1], :min_shape[2]]
                tensor2 = tensor2[:, :min_shape[1], :min_shape[2]]
            except Exception as e:
                self.logger.error(f"Error while adjusting tensor shapes: {e}")
                raise ValueError(f"Error in shape adjustment: {e}")

        # Artık bağlantı işlemi (toplama veya ortalama)
        if self.method == "add":
            residual_tensor = tensor1 + tensor2
            connection_type = "Addition Connection"
        else:
            residual_tensor = (tensor1 + tensor2) / 2
            connection_type = "Mean Connection"

        self._log_connection(tensor1, tensor2, residual_tensor, connection_type)
        return residual_tensor

    def _log_connection(self, tensor1, tensor2, result, connection_type):
        """
        Artık bağlantı işlemleri sırasında loglama yapar.

        Args:
            tensor1 (torch.Tensor): Birinci tensör.
            tensor2 (torch.Tensor): İkinci tensör.
            result (torch.Tensor): Artık bağlantı sonrası tensör.
            connection_type (str): Kullanılan artık bağlantı türü.
        """
        self._log_once(f"{connection_type} applied successfully.")
        self.logger.debug(f"Result tensor shape: {result.shape}, dtype: {result.dtype}, device: {result.device}")
