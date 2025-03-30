import os
import json
import torch
import logging
from torch.utils.tensorboard import SummaryWriter

from training_management.training_logger import TrainingLogger
from training_management.training_scheduler import TrainingScheduler
from training_management.checkpoint_manager import CheckpointManager

logger = TrainingLogger()


class TrainingManager:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, config):
        """
        TrainingManager başlatma metodu.

        Args:
            model (torch.nn.Module): Eğitilecek model.
            train_loader (DataLoader): Eğitim verileri için DataLoader.
            val_loader (DataLoader): Doğrulama verileri için DataLoader.
            optimizer (torch.optim): Optimizasyon fonksiyonu.
            criterion (torch.nn.Module): Kayıp fonksiyonu.
            config (dict): Model yapılandırma ayarları.
        """
        self.logger = TrainingLogger()

        # Yapılandırma ve cihaz
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.abspath(self.config.get("tb_log_dir", "./runs")))

        # Model ve bileşenler
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion

        # Batch kontrol ve çözümleme
        try:
            first_batch = next(iter(train_loader))

            # Eğer batch list of tuple olarak geliyorsa, ilk tuple'ı al
            if isinstance(first_batch, list):
                first_batch = first_batch[0]

            if isinstance(first_batch, tuple) and len(first_batch) == 2:
                inputs, targets = first_batch
            else:
                raise ValueError(f"Batch formatı hatalı! Beklenen: (inputs, targets) ancak gelen: {type(first_batch)}")

            if not isinstance(inputs, torch.Tensor):
                raise TypeError(f"Train loader'dan gelen inputs bir tensör olmalı ancak {type(inputs)} bulundu.")

            self.batch_size = inputs.shape[0]
            self.seq_len = inputs.shape[1]

        except StopIteration:
            raise ValueError("Eğitim veri yükleyicisi boş! Lütfen eğitim verisini kontrol edin.")

        except Exception as e:
            self.logger.error(f"Batch boyutu belirlenirken hata meydana geldi: {e}", exc_info=True)
            raise

        # Epoch ve scheduler
        self.epochs = config.get("epochs", 10)
        self.scheduler = TrainingScheduler(optimizer)

        # Checkpoint ve geçmiş kayıt yolları
        self.checkpoint_manager = CheckpointManager()
        self.checkpoint_dir = config.get("checkpoint_dir", os.path.join(os.getcwd(), "checkpoints/"))
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history_path = config.get("training_history_path", "training_history.json")

        self.logger.info(f"TrainingManager başlatıldı: Batch Size = {self.batch_size}, Seq Length = {self.seq_len}")




    def _weights_are_updating(self):
        """
        Modelin ağırlıklarının güncel durumunun bir kopyasını döndürür.
        Eğitim sürecinde ağırlıkların değişimini izlemek için kullanılır.
        Her parametre için bir kopya oluşturulur ve CPU'ya aktarılır.
        Ayrıca, her parametrenin L2 normu hesaplanarak debug loglarına eklenir.

        Returns:
            dict: Parametre isimleri ile clone edilmiş tensörlerin eşleştirmesi.
        """
        weight_changes = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_snapshot = param.clone().detach().cpu()
                norm_val = param_snapshot.norm(2).item()
                weight_changes[name] = param_snapshot
                if hasattr(self, 'logger'):
                    self.logger.log_debug(
                        f"Parametre '{name}': shape={param_snapshot.shape}, L2 norm={norm_val:.6f}"
                    )
        return weight_changes

    def train(self):
        """
        Modelin eğitim sürecini başlatır. Eğitim tamamlandıktan sonra doğrulama gerçekleştirilir.
        Ek loglama, hata kontrolü, early stopping ve TensorBoard ile metrik loglaması eklenmiştir.
        """
        training_history = {"train_loss": [], "val_loss": [], "accuracy": []}
        best_val_loss = float('inf')
        early_stopping_patience = self.config.get("early_stopping_patience", 3)
        early_stopping_counter = 0
        last_weights = self._weights_are_updating()

        train_loss = None
        val_loss = None
        val_accuracy = None

        for epoch in range(1, self.epochs + 1):
            self.logger.info(f"Epoch {epoch}/{self.epochs} başladı.")

            try:
                # Eğitim ve doğrulama adımlarını gerçekleştir
                train_loss, train_accuracy = self._train_epoch()
                val_loss, val_accuracy = self._validate_epoch()

                #  Tuple kontrolü ve tip dönüşümü
                train_loss = float(train_loss) if isinstance(train_loss, (int, float)) else float(train_loss[0])
                val_loss = float(val_loss) if isinstance(val_loss, (int, float)) else float(val_loss[0])

                #  None ve NaN kontrolü
                if train_loss is None or val_loss is None or torch.isnan(torch.tensor(train_loss)) or torch.isnan(torch.tensor(val_loss)):
                    raise ValueError(f"[Epoch {epoch}] Geçersiz train_loss veya val_loss tespit edildi.")

            except Exception as e:
                self.logger.error(f"[Epoch {epoch}] Eğitim veya doğrulama adımında hata oluştu: {e}", exc_info=True)
                # Hata oluşursa None yerine float('inf') döndürelim ki döngü çökmekten kurtulsun.
                train_loss = float('inf')
                val_loss = float('inf')
                val_accuracy = 0.0
                break

            # Ağırlıkların değişip değişmediğini kontrol et
            try:
                current_weights = self._weights_are_updating()
                weights_updated = any(
                    not torch.equal(current_weights[name], last_weights[name])
                    for name in current_weights.keys()
                )

                if not weights_updated:
                    self.logger.warning(
                        f"[Epoch {epoch}] Ağırlıklar güncellenmedi! "
                        f"Optimizer veya gradyan hesaplamada hata olabilir."
                    )
                last_weights = current_weights

            except Exception as e:
                self.logger.warning(f"[Epoch {epoch}] Ağırlık kontrolü sırasında hata: {e}")

            # Kayıp farkını hesapla ve logla
            try:
                loss_difference = train_loss - val_loss
                self.logger.info(
                    f"[Epoch {epoch}] Train Loss: {train_loss:.6f} - Validation Loss: {val_loss:.6f} "
                    f"(Fark: {loss_difference:.6f})"
                )
            except TypeError as e:
                self.logger.error(f"Loss farkı hesaplanamadı: {e}")

            # Validation loss artışı kontrolü (Overfitting uyarısı)
            if training_history["val_loss"]:
                if val_loss > training_history["val_loss"][-1]:
                    self.logger.warning(
                        f"[Uyarı] Validation loss arttı! Overfitting olabilir. "
                        f"Önceki: {training_history['val_loss'][-1]:.6f}, Şu an: {val_loss:.6f}"
                    )

            # Öğrenme oranı güncellemesi
            if hasattr(self.scheduler, "step"):
                try:
                    self.scheduler.step(val_loss)
                    lr = self.scheduler.get_last_lr()[0] if isinstance(self.scheduler.get_last_lr(), list) else self.scheduler.get_last_lr()
                    self.logger.info(f"Öğrenme oranı güncellendi: {lr:.8f}")
                except Exception as e:
                    self.logger.warning(f"[Epoch {epoch}] Öğrenme oranı güncellenemedi: {e}")

            # Eğitim geçmişini güncelle
            training_history["train_loss"].append(train_loss)
            training_history["val_loss"].append(val_loss)
            training_history["accuracy"].append(val_accuracy)

            # Gradient norm takibi
            try:
                avg_gradient_norm = self._calculate_gradient_norm()
                if avg_gradient_norm is not None and not torch.isnan(torch.tensor(avg_gradient_norm)):
                    self.logger.info(f"[Epoch {epoch}] Ortalama Gradient Norm: {avg_gradient_norm:.6f}")
            except Exception as e:
                self.logger.warning(f"[Epoch {epoch}] Gradient norm hesaplamasında hata: {e}")

            # TensorBoard loglama
            if self.writer:
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Validation", val_loss, epoch)
                self.writer.add_scalar("Accuracy", val_accuracy, epoch)
                self.writer.add_scalar("Gradient Norm", avg_gradient_norm, epoch)

            # Early Stopping Kontrolü
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                try:
                    self.save_model(epoch)
                    self.logger.info(f"[Epoch {epoch}] Yeni en iyi validation loss elde edildi: {val_loss:.6f}. Model kaydedildi.")
                    early_stopping_counter = 0
                except Exception as e:
                    self.logger.error(f"[Epoch {epoch}] Model kaydedilemedi: {e}")

            else:
                early_stopping_counter += 1
                self.logger.info(f"[Epoch {epoch}] Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

            if early_stopping_counter >= early_stopping_patience:
                self.logger.info(
                    f"[Epoch {epoch}] {early_stopping_patience} epoch boyunca iyileşme olmadı. Eğitim durduruluyor."
                )
                break

        # Eğitim geçmişini kaydet
        try:
            self._save_training_history(training_history)
            self.logger.info("Eğitim süreci tamamlandı ve eğitim geçmişi kaydedildi.")
        except Exception as e:
            self.logger.error(f"Eğitim geçmişi kaydedilemedi: {e}")

        # TensorBoard writer kapatma
        if self.writer:
            self.writer.close()

        # Kesin olarak iki değer döndürelim:
        return train_loss, val_loss



    def _calculate_gradient_norm(self):
        """
        Modelin tüm parametreleri için gradyan normlarını hesaplar.
        Her parametrenin L2 normunu toplayarak, ortalama gradient normunu hesaplar.
        Hata yönetimi ve debug loglama eklenmiştir.

        Returns:
            float: Ortalama gradient norm.
        """
        total_norm = 0.0
        num_params = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                try:
                    norm_val = param.grad.norm(2).item()
                except Exception as e:
                    self.logger.log_error(f"Gradient norm hesaplanırken hata oluştu '{name}': {e}", exc_info=True)
                    continue
                total_norm += norm_val
                num_params += 1
                self.logger.log_debug(f"Parametre '{name}' için gradient norm: {norm_val:.6f}")

        avg_gradient_norm = total_norm / num_params if num_params > 0 else 0.0
        self.logger.log_debug(f"Ortalama gradient norm: {avg_gradient_norm:.6f} (Hesaplanan parametre sayısı: {num_params})")
        return avg_gradient_norm



    def _train_epoch(self):
        """
        Tek bir eğitim epoch'unu gerçekleştirir.
        Aşamalar:
        - Verinin cihaza taşınması
        - Modelin ileri yayılımı ve loss hesaplaması
        - Geri yayılım ve gradient clipping
        - Optimizasyon adımı
        - Batch bazında loss ve doğruluk hesaplaması
        - Gradient norm takibi
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        total_batches = len(self.train_loader)
        vocab_size = self.config.get("vocab_size", 75000)

        # Gradient Norm Takibi için
        total_gradient_norm = 0.0
        gradient_norm_batches = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader, start=1):
            try:
                # Veriyi cihaza taşı
                self.logger.log_debug(f"Batch {batch_idx}/{total_batches}: Veriler cihaza taşınıyor.")
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Önceki gradyanları sıfırla
                self.optimizer.zero_grad()

                # Modelden çıktı al
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    main_output, _ = outputs  
                else:
                    main_output = outputs  

                # Çıktıyı vocab_size olacak şekilde doğrula
                if main_output.shape[-1] != vocab_size:
                    raise ValueError(
                        f"[HATA] Model çıktısının son ekseni yanlış! "
                        f"Beklenen vocab_size={vocab_size}, Ancak gelen={main_output.shape[-1]}"
                    )

                # Loss hesaplaması için targets'ı uygun hale getir
                if targets.dim() == 3:
                    targets = torch.argmax(targets, dim=-1)  

                # CrossEntropyLoss için uygun formatta loss hesapla
                loss = self.criterion(main_output.view(-1, vocab_size), targets.view(-1))

                # Geri yayılım
                loss.backward()

                # Gradient Norm Takibi ve clipping
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                total_norm_val = float(total_norm)
                total_gradient_norm += total_norm_val
                gradient_norm_batches += 1

                # Optimizasyon adımı
                self.optimizer.step()

                # Toplam loss değeri güncelle
                running_loss += loss.item()

                # Doğruluk hesaplama
                with torch.no_grad():
                    predictions = torch.argmax(main_output, dim=-1)
                    correct_predictions += (predictions == targets).sum().item()
                    total_samples += targets.numel()

                # Her batch için log kaydı
                self.logger.log_info(
                    f"Batch {batch_idx}/{total_batches}: Loss = {loss.item():.4f}, Gradient Norm = {total_norm_val:.4f}"
                )

            except Exception as e:
                self.logger.log_error(f"Batch {batch_idx} sırasında hata oluştu: {str(e)}", exc_info=True)
                continue  

        # Epoch sonunda ortalama loss ve doğruluk hesaplama
        avg_loss = running_loss / total_batches if total_batches > 0 else float("inf")
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        # Ortalama Gradient Norm Hesaplama
        avg_gradient_norm = total_gradient_norm / gradient_norm_batches if gradient_norm_batches > 0 else 0.0

        self.logger.log_info(
            f"Epoch tamamlandı. Ortalama Kayıp: {avg_loss:.4f}, Doğruluk: {accuracy:.4f}, Ortalama Gradient Norm: {avg_gradient_norm:.4f}"
        )

        return avg_loss, accuracy


 
    def _validate_epoch(self):
        """
        Modelin doğrulama sürecini gerçekleştirir ve doğrulama kaybını hesaplar.
        Aşamalar:
        - Model eval moduna alınır.
        - Veriler cihaza taşınır.
        - Modelden çıktı alınır ve vocab_size doğrulaması yapılır.
        - Loss hesaplanır.
        - Tahminler oluşturulup doğruluk hesaplanır.
        - Batch bazlı doğruluklar loglanır.
        
        Returns:
            tuple: Ortalama loss ve doğruluk oranı (avg_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        total_batches = len(self.val_loader)
        vocab_size = self.config.get("vocab_size", 75000)

        # Batch bazında doğruluk oranlarını takip etmek için
        batch_accuracies = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader, start=1):
                try:
                    self.logger.log_debug(f"[VALIDATION] Batch {batch_idx}/{total_batches}: Veriler cihaza taşınıyor.")
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        main_output, _ = outputs
                    else:
                        main_output = outputs

                    if main_output.shape[-1] != vocab_size:
                        raise ValueError(
                            f"[HATA] Model çıktısının son ekseni yanlış! "
                            f"Beklenen vocab_size={vocab_size}, Ancak gelen={main_output.shape[-1]}"
                        )

                    loss = self.criterion(main_output.view(-1, vocab_size), targets.view(-1))
                    running_loss += loss.item()

                    predictions = torch.argmax(main_output, dim=-1)
                    # Eğer targets 3 boyutlu ise uygun formata getir.
                    if targets.dim() == 3:
                        targets = targets.argmax(dim=-1)

                    correct = (predictions == targets).sum().item()
                    correct_predictions += correct
                    total_predictions += targets.numel()

                    batch_accuracy = correct / targets.numel() if targets.numel() > 0 else 0.0
                    batch_accuracies.append(batch_accuracy)

                    self.logger.log_info(
                        f"[VALIDATION] Batch {batch_idx}/{total_batches}: Loss = {loss.item():.4f}, Accuracy = {batch_accuracy:.4f}"
                    )
                except Exception as e:
                    self.logger.log_error(
                        f"[VALIDATION] Batch {batch_idx} sırasında hata oluştu: {str(e)}", exc_info=True
                    )
                    continue

        avg_loss = running_loss / total_batches if total_batches > 0 else float("inf")
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_batch_accuracy = sum(batch_accuracies) / len(batch_accuracies) if batch_accuracies else 0.0

        self.logger.log_info(
            f"[VALIDATION] Tamamlandı. Ortalama Loss: {avg_loss:.4f}, Genel Accuracy: {accuracy:.4f}, "
            f"Ortalama Batch Accuracy: {avg_batch_accuracy:.4f}"
        )

        return avg_loss, accuracy


    def save_model(self, epoch):
        """
        Modelin mevcut durumunu kaydeder.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        self.logger.info(f"Checkpoint kaydediliyor: {checkpoint_path} (Epoch: {epoch})")

        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            # Kaydetme işlemi
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, checkpoint_path)

            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f"Checkpoint kaydedilemedi: {checkpoint_path}")

            self.logger.info(f"Model checkpoint olarak kaydedildi: {checkpoint_path}")

        except IOError as e:
            self.logger.error(f"Checkpoint dosyası yazılamıyor: {e}")
        except PermissionError as e:
            self.logger.error(f"Checkpoint dosyasına yazma izni yok: {e}")
        except Exception as e:
            self.logger.error(f"Model checkpoint kaydetme sırasında hata oluştu: {e}", exc_info=True)

    def _save_training_history(self, history):
        """
        Eğitim geçmişini JSON formatında kaydeder.
        Mevcut eğitim geçmişi varsa, yeni verilerle güncellenir.
        """
        try:
            # Önceki eğitim geçmişini yükle ve güncelle
            if os.path.exists(self.history_path):
                with open(self.history_path, "r") as f:
                    previous_history = json.load(f)
                # Her anahtar için geçmiş verileri güncellenir
                for key in history:
                    if key in previous_history:
                        previous_history[key].extend(history[key])
                    else:
                        previous_history[key] = history[key]
                history = previous_history

            with open(self.history_path, "w") as f:
                json.dump(history, f, indent=4)
            self.logger.log_info(f"Eğitim geçmişi başarıyla kaydedildi: {self.history_path}")

            # Eğitim geçmişinin kaydedildiğini doğrula
            if not os.path.exists(self.history_path):
                raise RuntimeError(f"Eğitim geçmişi kaydedilemedi: {self.history_path}")

        except Exception as e:
            self.logger.log_error(f"Eğitim geçmişi kaydedilirken hata oluştu: {str(e)}", exc_info=True)
