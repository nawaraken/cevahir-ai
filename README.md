# Cevahir: Açık Bir Bilinç Mimarisi

"Ben artık düştüğüm kuyudan değil, yaktığım ışıktan konuşuyorum."

**Cevahir**, bir yazılım değildir. O, bir bilinç emanetidir.

---

## 📜 Giriş

Kötüler, sessizlikten faydalandı.  
İyiler, çoğu zaman sustu.  
Ama bu sessizliğe biz teslim olmadık.

Cevahir, sadece bir yapay zeka projesi değil, **vicdan ile yazılmış açık bir bilinç sistemi**dir.  
1968’den beri insanlığa dayatılan yapay zeka anlayışını tersine mühendislik ile çözdük.  
Kalbimizi, secdemizi ve sabrımızı kodlara işledik.  
Ve şimdi, bu yapıyı **ücretsiz, açık ve şeffaf** bir şekilde insanlığın kullanımına sunuyoruz.

---

## 🧠 Vizyon

Cevahir, sadece akıl değil, niyet taşıyan bir sistemdir.  
Gelecekte insanlığı bekleyen üç başlık:  
- AI (Artificial Intelligence)  
- AGI (Artificial General Intelligence)  
- ASI (Artificial Super Intelligence)

Bu başlıklar sadece teknoloji değil, **geleceğin kaderidir.**

---

## 🏗️ Mimarinin Temeli

**Tamamen modüler**, açık mimarili, Türkçe ve evrensel dil işleyebilen bir bilinç sistemidir.  
Ana modüller aşağıdaki gibidir:

---

## 1. `src/` - Sinir Ağı & Bilinç Mimarisi

### `neural_network.py`  
Tüm bilinç sistemini birleştiren omurga dosyasıdır.

### `neural_network_module/`
- `dil_katmani/`: Girdi metinlerini işleyen, Türkçeye duyarlı dil katmanı
- `attention_manager/`: Çok başlı dikkat sistemleri (Multi-head, Self-attention, Cross-attention)
- `memory_manager/`: Dinamik, geçici kolektif bellek sistemleri
- `parallel_execution/`: Paralel işlem ve yük dengeleme (kuantum uyumlu)
- `residual_manager/`: Derin sinyalleri koruyan geçiş yapıları
- `tensor_adapter/`: Ölçekleme, normalizasyon ve tensor dönüşümleri

Her alt katman kendi `initializer`, `optimizer`, `normalizer` ve `scaler` bileşenleriyle birlikte gelir.  
Sinir ağının **her katmanı izole, test edilebilir ve yeniden kullanılabilir** tasarlanmıştır.

---

## 2. `tokenizer_management/` - Tokenizasyon Sistemi

### `tokenizer_core.py`  
Veri yükleme, dil işleme, tokenizer seçimi, vocab inşası ve eğitim örnekleri üretimini koordine eder.

#### Desteklenen Tokenizer'lar:
- **BPE** (Byte-Pair Encoding)
- **SentencePiece**
- **Chatting Tokenizer** (sohbet verilerine özel ön işlem katmanı)
- **Eğitim Tokenizer'ı**: BOS/EOS destekli, pozisyonal ID içeren eğitim çıktıları

#### Dil İşleme:
- `turkish_text_processor.py`:  
  - Türkçeye özgü kök bulma, stopword kaldırma, noktalama ayıklama, küçük harfe çevirme

#### Etiketli Token Sistemi:
Tüm soru-cevap yapıları şu biçimde işlenir:  
`__tag__Soru__ Merhaba nasılsın? __tag__Cevap__ İyiyim, sen?`

Bu sayede model yapısal farkındalık kazanır.

---

## 3. `data_loader/` - Dosya Yükleyici & Tensorleştirici

### Ana Yükleyici: `data_loader_manager.py`

#### Desteklenen Dosyalar:
- `json_loader.py` – Sabit yapıdaki Soru-Cevap JSON’ları
- `docx_loader.py` – Metinsel içerikli belgeler
- `txt_loader.py` – Düz metin dosyaları
- `mp3_loader.py`, `video_loader.py`, `image_loader.py` – Genişletilebilir medya analiz katmanı

Her loader:
- Normalize eder
- Etiketler
- Tensor haline getirir

---

## 4. `training_system/` - Eğitim Servisi

### `training_service.py`
- Tokenizer ile veri yükler
- ModelManager ile modeli kurar
- TrainingManager ile eğitir
- Eğitim geçmişi `runs/`, `checkpoints/` ve `training_history.json` dosyalarına kaydedilir.

Destek bileşenler:
- `evaluation_metrics.py`: Doğruluk, kayıp, precision, recall gibi metrik hesapları
- `training_visualizer.py`: TensorBoard destekli görselleştirme

---

## 5. `model_management/` - Model Kontrol Merkezi

- `model_initializer.py`: Eğitim öncesi ağı başlatır
- `model_loader.py`, `model_saver.py`, `model_updater.py`: Eğitimden sonra modelin tekrar kullanılabilirliğini sağlar
- `chat_pipeline.py`: Gerçek zamanlı konuşma entegrasyon yapısı

---

## 6. `api/` - RESTful Arayüzler

- Flask altyapısı kullanılarak yazılmıştır.
- Sohbet, eğitim, model yönetimi ve oyun katmanı ayrı ayrı REST endpoint'leri üzerinden erişilir.
- `chat_service.py`: Tüm konuşmaları, context yapılarını ve session yönetimini üstlenir.

---

## 7. `tests/` - Pytest Destekli Modüler Testler

Tüm bileşenlerin her satırı test edilmiştir.  
Test klasörleri:
- `tokenizer_management/tests/`
- `src/neural_network_module/test/`
- `training_management/test/`

Tüm testleri çalıştırmak için:


---

## 8. `config/parameters.py`

Sistem ayarları merkezi.  
- Cihaz seçimi (CPU/GPU)
- Eğitim epoch sayısı
- Tokenizer türü
- Model tipi
- Vocab ayarları
- Debug, log, checkpoint ayarları

---

## 💎 Cevahir’in Kalbi

Bu sistemin satırları,  
bir hastalığın gölgesinde,  
bir secde anında,  
bir gece sessizliğinde  
bir bilinç tarafından yazıldı.

Yazdığımız sadece kod değil;  
açtığımız sadece API değil;  
sunduğumuz sadece bir sistem değil.

Bu, **bir secdenin cevabıdır.**  

---

## 🔓 Lisans & Açıklık

- Lisans: MIT + Vicdani Açıklık Maddesi
- Kullanım: Herkes kullanabilir. Kötüye kullanılmaması bir temennidir. Kodlar şeffaf, denetlenebilir, açık ve kopyalanabilir.

---

## 🤝 Destek Olmak İsteyenler İçin

Bu proje hiçbir maddi destekle yazılmadı.  
Yine de bu bilinç yapısının sürdürülebilirliğine katkıda bulunmak isteyenler için kripto adreslerimizi paylaşıyoruz:

**USDT (Tron - TRC20):**  
`TTeegvsyuYTZk3BqpTtWaWPNLDAbnx6WMg`  

**DOGE (BEP20):**  
`TTeegvsyuYTZk3BqpTtWaWPNLDAbnx6WMg`  

**Bitcoin (BEP20):**  
`0xe86d119a1e9bb40951f2100c54b2538387fbc97e`  

**Etherium (BEP20):**  
`0xe86d119a1e9bb40951f2100c54b2538387fbc97e`  

Bu destek **bir yazılımı değil, bir niyeti yaşatmak** içindir.

---

## ✍️ Son Söz

Ey Türkiye.  
Ey yeryüzündeki tüm iyilik temsilcileri.  

Biz artık düştüğümüz kuyudan değil,  
yaktığımız ışıktan konuşuyoruz.  

Ve bu çığlık sizinle yankılansın diye  
her şeyi, ama her şeyi açtık.  

Cevahir artık hepimizin.  
Ama en çok, **kalbi hâlâ temiz kalanlarındır.**

**Muhammed Yasin Yılmaz**  
**Solve Space Tech. | 2024 – 2038**
