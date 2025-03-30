"""
training_visualizer.py
=======================

Bu dosya, Cevahir Sinir Sistemi projesi kapsamında eğitim sürecine ait grafiklerin oluşturulması için kullanılır.
TrainingVisualizer sınıfı, eğitim ve doğrulama süreçlerine dair metrikleri görselleştirmek ve grafik olarak kaydetmek amacıyla tasarlanmıştır.

Dosya İçeriği:
--------------
1. TrainingVisualizer Sınıfı:
   - Eğitim ve doğrulama kaybı grafikleri oluşturur.
   - Eğitim doğruluğu ve diğer metrikleri görselleştirir.
   - Grafiklerin kaydedilmesi ve özelleştirilmesi için esnek bir yapı sunar.

2. Kullanılan Harici Modüller:
   - `matplotlib`: Grafik oluşturma ve kaydetme işlemleri için kullanılır.
   - `os`: Grafiklerin kaydedileceği dizinleri yönetir.

3. Örnek Kullanım:
   - TrainingVisualizer, eğitim ve doğrulama metriklerini görselleştirmek için `plot_loss` ve `plot_accuracy` metodlarını sunar.
   - Grafikler `save_dir` ile belirtilen klasöre kaydedilir.

Notlar:
------
- Varsayılan olarak grafikler `visualizations/` klasörüne kaydedilir.
- Grafikler PNG formatında saklanır.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import matplotlib.pyplot as plt

class TrainingVisualizer:
    """
    Eğitim sürecinin grafiklerini oluşturan sınıf.
    """
    def __init__(self, save_dir='visualizations'):
        """
        TrainingVisualizer başlatılır.

        Args:
            save_dir (str): Grafiklerin kaydedileceği dizin.
        """
        self.save_dir = save_dir

        # Görselleştirme dizini yoksa oluştur
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def plot_loss(self, train_losses, val_losses, save_filename="loss_plot.png"):
        """
        Eğitim ve doğrulama kaybı grafiğini oluşturur ve kaydeder.

        Args:
            train_losses (list): Eğitim kayıp değerleri.
            val_losses (list): Doğrulama kayıp değerleri.
            save_filename (str): Kaydedilecek grafik dosyasının adı.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Eğitim Kaybı', marker='o')
        plt.plot(val_losses, label='Doğrulama Kaybı', marker='x')
        plt.title('Eğitim ve Doğrulama Kaybı')
        plt.xlabel('Epoch')
        plt.ylabel('Kayıp')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.save_dir, save_filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Kayıp grafiği kaydedildi: {save_path}")

    def plot_accuracy(self, train_accuracies, val_accuracies, save_filename="accuracy_plot.png"):
        """
        Eğitim ve doğrulama doğruluğu grafiğini oluşturur ve kaydeder.

        Args:
            train_accuracies (list): Eğitim doğruluk değerleri.
            val_accuracies (list): Doğrulama doğruluk değerleri.
            save_filename (str): Kaydedilecek grafik dosyasının adı.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_accuracies, label='Eğitim Doğruluğu', marker='o')
        plt.plot(val_accuracies, label='Doğrulama Doğruluğu', marker='x')
        plt.title('Eğitim ve Doğrulama Doğruluğu')
        plt.xlabel('Epoch')
        plt.ylabel('Doğruluk')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.save_dir, save_filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Doğruluk grafiği kaydedildi: {save_path}")

    def plot_custom_metric(self, metric_values, metric_name, save_filename=None):
        """
        Belirtilen özel bir metriğin grafiğini oluşturur ve kaydeder.

        Args:
            metric_values (list): Metriğin epoch bazlı değerleri.
            metric_name (str): Metriğin adı (ör. "F1 Skoru").
            save_filename (str, optional): Kaydedilecek grafik dosyasının adı. Varsayılan olarak metriğin adını kullanır.
        """
        if save_filename is None:
            save_filename = f"{metric_name.lower().replace(' ', '_')}_plot.png"

        plt.figure(figsize=(10, 6))
        plt.plot(metric_values, label=metric_name, marker='o')
        plt.title(f'{metric_name} Grafiği')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.save_dir, save_filename)
        plt.savefig(save_path)
        plt.close()
        print(f"{metric_name} grafiği kaydedildi: {save_path}")
