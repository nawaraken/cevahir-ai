import torch
import torch.nn as nn
import logging

class LanguageEmbedding(nn.Module):
    """
    LanguageEmbedding sınıfı, kelimeleri vektörlere dönüştürmek için kullanılır.
    """
    def __init__(self, vocab_size, embed_dim, init_method="xavier", log_level=logging.INFO):
        """
        LanguageEmbedding sınıfını başlatır.

        Args:
            vocab_size (int): Kelime dağarcığı boyutu.
            embed_dim (int): Gömme boyutu.
            init_method (str): Başlatma yöntemi (varsayılan: "xavier").
            log_level (int): Log seviyesi (örn. logging.DEBUG, logging.INFO).
        """
        super(LanguageEmbedding, self).__init__()

        # Logger yapılandırması
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Gömme katmanı (Embedding Layer)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Ağırlık başlatma
        self._initialize_weights(init_method)

        self.logger.info(f"LanguageEmbedding initialized with vocab_size={vocab_size}, embed_dim={embed_dim}, init_method={init_method}")

    def _initialize_weights(self, method):
        """
        Gömme katmanındaki ağırlıkları başlatır.
        """
        if method == "xavier":
            nn.init.xavier_uniform_(self.embedding.weight)
        elif method == "kaiming":
            nn.init.kaiming_uniform_(self.embedding.weight, nonlinearity='relu')
        elif method == "normal":
            nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        elif method == "uniform":
            nn.init.uniform_(self.embedding.weight, a=-0.1, b=0.1)
        else:
            raise ValueError(f"Invalid init_method: {method}. Choose from ['xavier', 'kaiming', 'normal', 'uniform']")

    def _log_tensor_stats(self, tensor, stage):
        """
        Belirtilen aşamadaki tensor istatistiklerini loglar.
        
        Args:
            tensor (torch.Tensor): Loglanacak tensör.
            stage (str): Aşama adı (örn. "After Embedding").
        """
        try:
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            mean_val = tensor.mean().item()
            std_val = tensor.std().item()
            self.logger.debug(f"{stage} Stats -> shape: {tensor.shape}, min: {min_val:.4f}, max: {max_val:.4f}, mean: {mean_val:.4f}, std: {std_val:.4f}")
        except Exception as e:
            self.logger.error(f"Error logging stats at {stage}: {e}", exc_info=True)

    def forward(self, x):
        """
        İleri adım hesaplamasını gerçekleştirir.

        Args:
            x (torch.Tensor): Giriş tensörü.

        Returns:
            torch.Tensor: Gömülü tensör.
        """
        self.logger.debug(f"Forward pass with input shape: {x.shape}")
        output = self.embedding(x)
        self._log_tensor_stats(output, "After LanguageEmbedding")
        return output
