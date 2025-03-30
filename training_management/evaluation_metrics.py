"""
evaluation_metrics.py
======================

Bu dosya, Cevahir Sinir Sistemi projesi kapsamında eğitim sürecinde performansı değerlendirmek için metrik hesaplama işlevlerini içerir. 
Bu metrikler, modelin doğruluğunu, kaybını ve diğer performans ölçütlerini değerlendirmek için kullanılır.

Dosya İçeriği:
--------------
1. `EvaluationMetrics` Sınıfı:
   - Doğruluk (Accuracy) hesaplaması.
   - Kesinlik (Precision), Duyarlılık (Recall) ve F1 Skoru hesaplaması.
   - Kayıp (Loss) hesaplaması.
   - Hesaplanan metriklerin loglanması.

Notlar:
------
- Tüm metrikler `training_logger.py` üzerinden loglanır.
- Hesaplamalar için PyTorch ve NumPy kütüphaneleri kullanılmıştır.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import torch
from training_management.training_logger import TrainingLogger

class EvaluationMetrics:
    """
    Eğitim sürecinde performans metriklerini hesaplamak için kullanılan sınıf.
    """
    def __init__(self):
        """
        EvaluationMetrics sınıfını başlatır.
        """
        self.logger = TrainingLogger()

    def calculate_accuracy(self, predictions, targets):
        """
        Doğruluk (Accuracy) hesaplar.

        Args:
            predictions (torch.Tensor): Modelin tahmin ettiği değerler.
            targets (torch.Tensor): Gerçek değerler.

        Returns:
            float: Doğruluk değeri (%).
        """
        correct_predictions = (predictions.argmax(dim=1) == targets).sum().item()
        total_samples = targets.size(0)
        accuracy = (correct_predictions / total_samples) * 100
        self.logger.log_info(f"Accuracy: {accuracy:.2f}%")
        return accuracy

    def calculate_precision_recall_f1(self, predictions, targets, num_classes):
        """
        Kesinlik (Precision), Duyarlılık (Recall) ve F1 Skoru hesaplar.

        Args:
            predictions (torch.Tensor): Modelin tahmin ettiği değerler.
            targets (torch.Tensor): Gerçek değerler.
            num_classes (int): Sınıf sayısı.

        Returns:
            dict: Her bir metrik için skorları içeren sözlük.
        """
        preds = predictions.argmax(dim=1).cpu().numpy()
        targets = targets.cpu().numpy()

        # Confusion matrix hesaplama
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(targets, preds):
            confusion_matrix[t, p] += 1

        # Her sınıf için metrikler
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1_score = np.zeros(num_classes)

        for i in range(num_classes):
            true_positives = confusion_matrix[i, i]
            false_positives = confusion_matrix[:, i].sum() - true_positives
            false_negatives = confusion_matrix[i, :].sum() - true_positives

            precision[i] = true_positives / (true_positives + false_positives + 1e-10)
            recall[i] = true_positives / (true_positives + false_negatives + 1e-10)
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-10)

        metrics = {
            "precision": np.mean(precision) * 100,
            "recall": np.mean(recall) * 100,
            "f1_score": np.mean(f1_score) * 100,
        }

        self.logger.log_info(f"Precision: {metrics['precision']:.2f}%")
        self.logger.log_info(f"Recall: {metrics['recall']:.2f}%")
        self.logger.log_info(f"F1 Score: {metrics['f1_score']:.2f}%")
        return metrics

    def calculate_loss(self, loss_function, predictions, targets):
        """
        Toplam kaybı (Loss) hesaplar.

        Args:
            loss_function (callable): Kayıp fonksiyonu (örneğin, CrossEntropyLoss).
            predictions (torch.Tensor): Modelin tahmin ettiği değerler.
            targets (torch.Tensor): Gerçek değerler.

        Returns:
            float: Toplam kayıp değeri.
        """
        loss = loss_function(predictions, targets).item()
        self.logger.log_info(f"Loss: {loss:.4f}")
        return loss

    def log_metrics(self, epoch, accuracy, metrics, loss):
        """
        Hesaplanan metrikleri loglar.

        Args:
            epoch (int): Epoch sayısı.
            accuracy (float): Doğruluk değeri.
            metrics (dict): Kesinlik, Duyarlılık ve F1 Skoru değerleri.
            loss (float): Toplam kayıp değeri.
        """
        self.logger.log_metrics(
            epoch=epoch,
            training_loss=loss,
            validation_loss=None,  # Doğrulama kaybı isteğe bağlıdır.
            accuracy=accuracy
        )
        self.logger.log_info(f"Epoch {epoch} Metrics - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%, "
                             f"Precision: {metrics['precision']:.2f}%, Recall: {metrics['recall']:.2f}%, "
                             f"F1 Score: {metrics['f1_score']:.2f}%")
