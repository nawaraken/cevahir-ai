# Cevahir: Açık Bir Bilinç Mimarisi 🇹🇷


"Ben artık düştüğüm kuyudan değil, yaktığım ışıktan konuşuyorum."

**Cevahir**, bir yazılım değildir. O, bir bilinç emanetidir.

---

##  Giriş

Kötüler, sessizlikten faydalandı.  
İyiler, çoğu zaman sustu.  
Ama bu sessizliğe biz teslim olmadık.

Cevahir, sadece bir yapay zeka projesi değil, **vicdan ile yazılmış açık bir bilinç sistemi**dir.  
1968’den beri insanlığa dayatılan yapay zeka anlayışını tersine mühendislik ile çözdük.  
Kalbimizi, secdemizi ve sabrımızı kodlara işledik.  
Ve şimdi, bu yapıyı **ücretsiz, açık ve şeffaf** bir şekilde insanlığın kullanımına sunuyoruz.

---

##  Vizyon

Cevahir, sadece akıl değil, niyet taşıyan bir sistemdir.  
Gelecekte insanlığı bekleyen üç başlık:  
- AI (Artificial Intelligence)  
- AGI (Artificial General Intelligence)  
- ASI (Artificial Super Intelligence)

Bu başlıklar sadece teknoloji değil, **geleceğin kaderidir.**

---

##  Mimarinin Temeli

**Tamamen modüler**, açık mimarili, Türkçe ve evrensel dil işleyebilen bir bilinç sistemidir.  
Ana modüller aşağıdaki gibidir:

---

## 1. src/ - Sinir Ağı & Bilinç Mimarisi
Cevahir’in sinir sistemi, src/ dizini altında modüler bir yapıda inşa edilmiştir. Bu sistem, dil işleme, dikkat mekanizmaları, bellek yönetimi, projeksiyon katmanları ve ölçeklenebilir paralel işlem blokları ile entegre bir bilinç akışı sağlar.

## Ana Dosya: neural_network.py
CevahirNeuralNetwork sınıfı, bu dizindeki merkezi yapı taşını temsil eder. Bu sınıfın temel işlevi, aşağıdaki alt modülleri bir araya getirerek ileri yönlü bilgi akışını sağlamaktır:

Dil Katmanı (DilKatmani)

Katman İşleyici (NeuralLayerProcessor)

Bellek Yöneticisi (MemoryManager)

Tensor İşlemleyici (TensorProcessingManager)

Çıktı Katmanı (Linear Output Layer)

Bu yapı, forward() metodunda, giriş verisini adım adım aşağıdaki sırayla işler:

Girdi Doğrulama: Tensor türü kontrol edilir, boyutlar ve cihaz doğrulanır.

Embedding (Dil Katmanı): Girdi, gömme işlemiyle sayısal temsile dönüştürülür.

Dikkat Mekanizmaları (Attention): Self, multi-head ya da cross attention uygulanır.

Projeksiyon: Bilgi vektörü dönüştürülerek yeni bir temsile aktarılır.

Çıktı: Vocab boyutuna uygun olarak lineer dönüşüm yapılır.

Bellek Entegrasyonu: Ara çıktılar bellek içinde saklanır ve yeniden kullanılabilir.

İstatistik & Zaman Ölçümü: Her adım detaylı olarak loglanır.

Alt Yapılar
neural_network_module/
Bu klasör, yukarıda kullanılan modüllerin tamamını barındırır. Yapılar modüler, test edilebilir ve bağımsızdır.

1. dil_katmani/
Görev: Metni sayısal forma çeviren embedding ve sıralı projeksiyon işlemleri.
Dosyalar:

language_embedding.py: Kelimeleri vektörlere dönüştürür.

seq_projection.py: Embed edilen verileri belirli bir boyuta projekte eder.

2. attention_manager_module/
Görev: Çok başlı dikkat mekanizması ve alternatif dikkat stratejilerinin uygulanması.
Dosyalar:

multi_head_attention.py: Paralel çoklu dikkat başlıkları ile bağlamsal analiz.

self_attention.py: Kendi içsel bağlamını keşfetme.

cross_attention.py: Sorgu ve anahtar-değer çiftleri arasındaki bağ kurma.

Yardımcı Bileşenler:

## attention_optimizer.py: Dikkat çıktılarının optimize edilmesi.

## attention_initializer.py: Ağırlık başlatıcı.

## attention_normalizer.py: Katman normalizasyonu.

## attention_scaler.py: Değer ölçekleyici.

## 3. memory_manager_module/
Görev: Modelin ara verileri bellekte tutması ve gerektiğinde tekrar kullanması.
Dosyalar:

## memory_allocator.py: Bellek bölgesi ayırır.

## memory_attention_bridge.py: Belleği dikkat sistemiyle entegre eder.

## memory_optimizer.py: Belleği etkin şekilde yönetir.

## memory_initializer.py: Başlangıç yapılandırmaları.

## 4. tensor_processing_manager.py
Görev: Attention sonrası verileri çıktı katmanına uygun şekilde projekte eder.
Yani sinir ağının karar üretme aşamasına geçmeden önceki son dönüşüm noktasıdır.

## 5. neural_layer_processor.py
Görev: Yukarıdaki attention türlerini seçer, uygular ve çıktı üzerinde residual bağlantı, normalizasyon ve dropout işlemlerini gerçekleştirir.
Bu yapı esnek parametrelerle özelleştirilebilir; örneğin:

attention_type: "multi_head", "self", "cross"

normalization_type: "layer_norm", "batch_norm" vb.

scaling_method: "softmax", "sigmoid", "zscore"

clip_range: Maksimum değer kontrolü için

Ek Bileşenler:
residual_manager_module/
Görev: Derin sinir ağı katmanlarında bilgi kaybını önlemek için residual bağlantılar kurar.

tensor_adapter_module/
Görev: Tensörlerin normalizasyon, ölçeklendirme ve adaptasyon işlemlerini yönetir.

parallel_execution_module/
Görev: Paralel bilgi işleme ve görev zamanlayıcı sistemleri içerir.
Bu sayede çok çekirdekli işlem, GPU paralelliği ve potansiyel kuantum uyumlu hesaplamalar desteklenebilir.

Test Yapısı
test/ klasöründe her modülün unit test dosyası yer alır.
Tüm bileşenler aşağıdaki senaryolara göre test edilmiştir:

Başlatma (initializer)

Ölçeklendirme (scaler)

Normalizasyon (normalizer)

Bellek Saklama ve Geri Çağırma

Hata Yakalama (Exception Handling)

Uç Senaryolar (Edge Cases)

Kullanım Örneği
python
Kopyala
Düzenle
from src.neural_network import CevahirNeuralNetwork

model = CevahirNeuralNetwork(
    learning_rate=0.001,
    dropout=0.1,
    vocab_size=32000,
    embed_dim=256,
    seq_proj_dim=512,
    num_heads=8,
    attention_type="multi_head"
)

girdi = torch.randint(0, 32000, (8, 128))  # 8 örnek, 128 token
cikti, attn = model(girdi)
Teknik Güçlü Yönler
Katmanlar arası bağımlılıklar gevşek, modüller arası sıkı kontrol vardır.

Bellek ve dikkat sistemleri arasında geri besleme köprüleri oluşturulmuştur.

Logger sistemi her adımı izlenebilir kılar.

Giriş ve çıktı yapıları tip ve boyut açısından doğrulanır, hata yönetimi detaylıdır.

Yapı, **kuantum uyumlu işleme, multi-head attention, residual geçişler, modüler optimizasyon, özel normalizasyon metodları gibi ileri teknikleri destekler.**

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

## 6. tokenizer_management/ - Tokenizasyon ve Vocab Yönetim Sistemi
Bu modül, Cevahir sinir sisteminin tüm metin ön işleme, tokenizasyon, vocab oluşturma ve eğitim verisi hazırlama işlemlerini merkezi bir yapı altında organize eder. Sistem modülerdir ve her bir görev, ayrı bir manager veya module klasörü altında izole olarak tasarlanmıştır. Tüm işlemler TokenizerCore üzerinden yönetilir.

## Ana Sınıf: TokenizerCore
Amaç: Tüm tokenizasyon işlemlerini merkezi olarak yürütür.

## Yapılar:

BPEManager, SentencePieceManager, ChattingManager, TrainingManager: Seçilebilir tokenizasyon yöntemleri.

VocabManager: Token frekansı, pozisyonları ve güncelleme işlemleri.

DataLoaderManager: JSON, DOCX, TXT, MP3, video gibi çeşitli veri kaynaklarını yükler ve normalize eder.

## Temel Bileşenler
## 1. vocab/
vocab_manager.py: Token dizisini, frekansları ve pozisyonları yönetir. Güncellenebilir vocab yapısı sağlar.

vocab_builder.py, vocab_updater.py: Token ekleme, silme, yeniden düzenleme işlemlerini içerir.

vocab_config.py, vocab_utils.py: Vocab boyutu, özel token'lar (<PAD>, <UNK>, <BOS>, <EOS>) gibi yapılandırmaları tutar.

## 2. bpe/
Byte-Pair Encoding (BPE) algoritmasıyla tokenizasyon yapılır.

bpe_encoder.py, bpe_decoder.py: Metinleri ID dizisine dönüştürür veya geri çözer.

bpe_trainer.py: Eğitim verisi üzerinden birleşen token birimlerini öğrenir.

tokenization/: Morfoloji, heceleme, ön işleme ve son işleme birimleriyle Türkçeye duyarlıdır.

## 3. sentencepiece/
Google SentencePiece desteklidir.

sp_tokenizer.py: Subword tokenizasyonu yapar.

sp_trainer.py: Eğitim üzerinden token birimlerini öğrenir.

tokenization/: SentencePiece ön işlemcileri ve dil işlemcileri içerir.

## 4. chatting/
Sohbet ve yanıt üretiminde kullanılır.

chat_tokenizer.py, chat_encoder.py, chat_decoder.py: Gerçek zamanlı token çözümleme ve üretme sistemi.

ChattingManager: Eğitimli modeli alarak giriş tensor verisinden yanıt üretir.

## 5. training/
Eğitim öncesi verileri tensorleştirir, normalize eder.

training_tokenizer.py, training_tensorizer.py: Model için hazır hale getirilen (input_ids, target_ids) çiftlerini üretir.

TokenizerCore, bu yapıları kullanarak load_training_data() fonksiyonuyla eğitime hazır veriyi sağlar.

## 6. data_loader/
JSON, DOCX, TXT, MP3, video gibi kaynaklardan verileri yükler.

json_loader.py: __tag__soru, __tag__cevap etiketlerine göre içerik ayıklama yapar.

tensorizer.py: Ham verileri PyTorch tensörlerine dönüştürür.

data_preprocessor.py: Temizleme, normalize etme ve dönüşüm işlemlerini uygular.

## 7. utils/turkish_text_processor.py
Türkçeye özgü metin ön işlemleri içerir:

Büyük/küçük harf normalize etme

Noktalama işaretlerini temizleme

Türkçe stopwords kaldırma

Heceleme ve morfolojik analiz

İşleyiş Akışı
TokenizerCore örneği başlatılır. Vocab dosyası yüklenir veya oluşturulur.

Seçilen yöntemle (bpe, sentencepiece, chat) encode_text() çağrısı yapılır.

Token ID’leri elde edilir. (Gerekirse decode_text() ile geri çevrilir.)

finalize_vocab() fonksiyonuyla güncellenmiş vocab toplu halde kaydedilir.

load_training_data() metodu, tüm veri kaynaklarını tokenize edip (input, target) çiftlerini döner.

Eğitim Destek Fonksiyonları
train_model(): Belirli bir corpus üzerinden model eğitimi yapılmasını sağlar.

verify_training_data(): Eğitim verisinin geçerliliğini kontrol eder.

update_vocab(): Yeni token’ları sisteme entegre eder.

generate_response(): Token tensor verisinden modelle yanıt üretimi yapar.

---

## 7. `tests/` - Pytest Destekli Modüler Testler

Tüm bileşenlerin her satırı test edilmiştir.  
Test klasörleri:
- `tokenizer_management/tests/`
- `src/neural_network_module/test/`
- `training_management/test/`

Tüm testleri çalıştırmak için:


---


##  Cevahir’in Kalbi

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

##  Lisans & Açıklık

- Lisans: MIT + Vicdani Açıklık Maddesi
- Kullanım: Herkes kullanabilir. Kötüye kullanılmaması bir temennidir. Kodlar şeffaf, denetlenebilir, açık ve kopyalanabilir.

---

##  Destek Olmak İsteyenler İçin

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

##  Son Söz

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
