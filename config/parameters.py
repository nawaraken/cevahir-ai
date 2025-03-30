#config/parameters.py
import os
import torch
import logging
# Log Dosyaları
LOGGING_PATH = os.path.join('logs')
if not os.path.exists(LOGGING_PATH):
    os.makedirs(LOGGING_PATH)
# Logger yapılandırması
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Model Genel Parametreleri
INPUT_DIM = 2048
SEQ_LEN = None
OUTPUT_DIM = 2048
HIDDEN_SIZE = 2048
EMBEDDING_DIM = 2048
CONTEXT_DIM = 2048
NHEAD = 8
NUM_LAYERS = 4
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.0001
TEMPERATURE = 1.0
BATCH_SIZE = None
ACTIVATION = "relu"
NORM_TYPE = "layernorm"
SCALING_FACTOR = 1.0

def set_dynamic_batch_and_seq_len(input_tensor):
    """
    Giriş tensöründen batch_size ve seq_len değerlerini belirler ve global olarak atar.

    Args:
        input_tensor (torch.Tensor): Giriş tensörü.
    """
    global BATCH_SIZE, SEQ_LEN, SHORT_TERM_MEMORY_SHAPE, LONG_TERM_MEMORY_SHAPE, MEMORY_SHAPE

    if input_tensor.dim() == 2:  # İki boyutlu tensör (batch_size, seq_len) durumu
        BATCH_SIZE, SEQ_LEN = input_tensor.shape
    elif input_tensor.dim() == 3:  # Üç boyutlu tensör (batch_size, seq_len, input_dim) durumu
        BATCH_SIZE, SEQ_LEN, _ = input_tensor.shape
    else:
        raise ValueError(f"Geçersiz giriş boyutu: {input_tensor.shape}. Beklenen 2 veya 3 boyutlu tensör.")

    # Tüm yapılandırma sözlüklerini güncelle
    for config in [
        BIOLOGICAL_BRAIN_LAYER_CONFIG,
        DEFAULT_LAYER_CONFIG,
        CONTEXTUAL_LAYER_CONFIG,
        EMOTION_LAYER_CONFIG,
        DYNAMIC_MEMORY_LAYER_CONFIG,
        COLLECTIVE_MEMORY_LAYER_CONFIG,
    ]:
        config["batch_size"] = BATCH_SIZE
        config["seq_len"] = SEQ_LEN

    # Bellek şekillerini güncelle
    SHORT_TERM_MEMORY_SHAPE = (BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    LONG_TERM_MEMORY_SHAPE = (BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    MEMORY_SHAPE = (BATCH_SIZE, SEQ_LEN, INPUT_DIM)

    logger.info(f"Dinamik olarak batch_size={BATCH_SIZE}, seq_len={SEQ_LEN} ayarlandı.")

# Biyolojik Beyin Katmanı Parametreleri
MUTATION_RATE = 0.023
POPULATION_SIZE = 20
PLASTICITY_RATE = 0.01
HYPER_MODE_FACTOR = 2
ELITE_RATIO = 0.1
RANDOM_RESET_RATIO = 0.05
MUTATION_DECAY = 0.99
MUTATION_THRESHOLD = 0.01
CROSSOVER_RATIO = 0.2
DEATH_RATIO = 0.1
MIN_POPULATION_SIZE = 10
MAX_POPULATION_SIZE = 250
BIOLOJICAL_MEMORY_SIZE = 2048
# Eğitim Parametreleri
EPOCHS = 10
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10
MAX_EPOCHS = 150
LEARNING_RATE_DECAY = 0.85
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 1.0

# Eğitim Cihazı Ayarları
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_DEVICE_USAGE = 0.8

# Cevahir Modeli Girdi ve Çıktı Şekilleri
EXPECTED_INPUT_SHAPE = (BATCH_SIZE, SEQ_LEN, INPUT_DIM)
EXPECTED_OUTPUT_SHAPE = (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)

# Cevahir Modeli Ortak Katman Yapılandırması
DEFAULT_LAYER_CONFIG = {
    "input_dim": INPUT_DIM,
    "hidden_size": HIDDEN_SIZE,
    "output_dim": OUTPUT_DIM,
    "seq_len": SEQ_LEN,
    "embedding_dim": EMBEDDING_DIM,
    "num_heads": NHEAD,
    "num_layers": NUM_LAYERS,
    "dropout_rate": DROPOUT_RATE,
    "learning_rate": LEARNING_RATE,
    "temperature": TEMPERATURE,
    "activation": ACTIVATION,
    "batch_size": BATCH_SIZE,
    "expected_input_shape": EXPECTED_INPUT_SHAPE,
    "expected_output_shape": EXPECTED_OUTPUT_SHAPE,
    "norm_type": NORM_TYPE,
    "scaling_factor": SCALING_FACTOR,
    "activation": ACTIVATION,
    "device": DEVICE,
    "context_dim": CONTEXT_DIM

}


# Katman Özel Yapılar
CONTEXTUAL_LAYER_CONFIG = {
    **DEFAULT_LAYER_CONFIG,
    "attention_heads": NHEAD,
    "contextual_factor": 0.5
}

# Bellek Yapısı
SHORT_TERM_MEMORY_SHAPE = (BATCH_SIZE, SEQ_LEN, INPUT_DIM) if BATCH_SIZE and SEQ_LEN else None  # Kısa dönem bellek şekli
LONG_TERM_MEMORY_SHAPE = (BATCH_SIZE, SEQ_LEN, INPUT_DIM) if BATCH_SIZE and SEQ_LEN else None  # Uzun dönem bellek şekli
MEMORY_SHAPE = (BATCH_SIZE, SEQ_LEN, INPUT_DIM) if BATCH_SIZE and SEQ_LEN else None           # Genel bellek şekli
MEMORY_UPDATE_RATE = 0.05
TOTAL_MEMORY_CAPACITY = BATCH_SIZE * SEQ_LEN * INPUT_DIM * 2  if BATCH_SIZE and SEQ_LEN else None # Toplam kapasite


# Katman Özel Yapılar
BIOLOGICAL_BRAIN_LAYER_CONFIG = {
    
    "mutation_rate": MUTATION_RATE,
    "population_size": POPULATION_SIZE,
    "plasticity_rate": PLASTICITY_RATE,
    "hyper_mode_factor": HYPER_MODE_FACTOR,
    "short_term_shape": SHORT_TERM_MEMORY_SHAPE,
    "long_term_shape": LONG_TERM_MEMORY_SHAPE,
    "memory_update_rate": MEMORY_UPDATE_RATE,
    "input_dim": INPUT_DIM,
    "hidden_size": HIDDEN_SIZE,
    "output_dim": OUTPUT_DIM,
    "seq_len": SEQ_LEN,
    "embedding_dim": EMBEDDING_DIM,
    "num_heads": NHEAD,
    "num_layers": NUM_LAYERS,
    "dropout_rate": DROPOUT_RATE,
    "learning_rate": LEARNING_RATE,
    "temperature": TEMPERATURE,
    "activation": ACTIVATION,
    "batch_size": BATCH_SIZE,
    "expected_input_shape": EXPECTED_INPUT_SHAPE,
    "expected_output_shape": EXPECTED_OUTPUT_SHAPE,
    "norm_type": NORM_TYPE,
    "scaling_factor": SCALING_FACTOR,
    "activation": ACTIVATION,
    "device": DEVICE,
    "context_dim": CONTEXT_DIM,
    "synapse_dim": HIDDEN_SIZE,      
    "memory_size": HIDDEN_SIZE,
}

EMOTION_LAYER_CONFIG = {
    "short_memory_shape": SHORT_TERM_MEMORY_SHAPE,
    "long_memory_shape": LONG_TERM_MEMORY_SHAPE,
    "emotion_output": OUTPUT_DIM,
        "input_dim": INPUT_DIM,
    "hidden_size": HIDDEN_SIZE,
    "output_dim": OUTPUT_DIM,
    "seq_len": SEQ_LEN,
    "embedding_dim": EMBEDDING_DIM,
    "num_heads": NHEAD,
    "num_layers": NUM_LAYERS,
    "dropout_rate": DROPOUT_RATE,
    "learning_rate": LEARNING_RATE,
    "temperature": TEMPERATURE,
    "activation": ACTIVATION,
    "batch_size": BATCH_SIZE,
    "expected_input_shape": EXPECTED_INPUT_SHAPE,
    "expected_output_shape": EXPECTED_OUTPUT_SHAPE,
    "norm_type": NORM_TYPE,
    "scaling_factor": SCALING_FACTOR,
    "activation": ACTIVATION,
    "device": DEVICE,
    "context_dim": CONTEXT_DIM,
}

DYNAMIC_MEMORY_LAYER_CONFIG = {
    "short_term_shape": SHORT_TERM_MEMORY_SHAPE,
    "long_term_shape": LONG_TERM_MEMORY_SHAPE,
    "adaptive_scaling": True,  # Dinamik ölçeklendirme aktif
    "memory_update_rate": MEMORY_UPDATE_RATE,
    "input_dim": INPUT_DIM,
    "hidden_size": HIDDEN_SIZE,
    "output_dim": OUTPUT_DIM,
    "seq_len": SEQ_LEN,
    "embedding_dim": EMBEDDING_DIM,
    "num_heads": NHEAD,
    "num_layers": NUM_LAYERS,
    "dropout_rate": DROPOUT_RATE,
    "learning_rate": LEARNING_RATE,
    "temperature": TEMPERATURE,
    "activation": ACTIVATION,
    "batch_size": BATCH_SIZE,
    "expected_input_shape": EXPECTED_INPUT_SHAPE,
    "expected_output_shape": EXPECTED_OUTPUT_SHAPE,
    "norm_type": NORM_TYPE,
    "scaling_factor": SCALING_FACTOR,
    "activation": ACTIVATION,
    "device": DEVICE,
    "context_dim": CONTEXT_DIM
}

COLLECTIVE_MEMORY_LAYER_CONFIG = {
    "short_term_shape": SHORT_TERM_MEMORY_SHAPE,
    "long_term_shape": LONG_TERM_MEMORY_SHAPE,
    "collective_memory_scaling": True,  # Ağırlıklandırma aktif
    "contextual_factor": 0.7,
        "input_dim": INPUT_DIM,
    "hidden_size": HIDDEN_SIZE,
    "output_dim": OUTPUT_DIM,
    "seq_len": SEQ_LEN,
    "embedding_dim": EMBEDDING_DIM,
    "num_heads": NHEAD,
    "num_layers": NUM_LAYERS,
    "dropout_rate": DROPOUT_RATE,
    "learning_rate": LEARNING_RATE,
    "temperature": TEMPERATURE,
    "activation": ACTIVATION,
    "batch_size": BATCH_SIZE,
    "expected_input_shape": EXPECTED_INPUT_SHAPE,
    "expected_output_shape": EXPECTED_OUTPUT_SHAPE,
    "norm_type": NORM_TYPE,
    "scaling_factor": SCALING_FACTOR,
    "activation": ACTIVATION,
    "device": DEVICE,
    "context_dim": CONTEXT_DIM
    
}



# Oyun Parametreleri
MAX_GAME_STEPS = 1001  # Modelin oynayabileceği maksimum adım sayısı
# Oyun tahtası boyutu (4x4, 8x8 vb.)
BOARD_SIZE = 5  # oyun tahtası boyutu
TIME_SCALE_DEFAULT = 1.0  # Varsayılan zaman ölçeği, genellikle 1 saniye = 1 birim olarak kabul edilir.
# Multiverse log dosyasının yolu
MULTIVERSE_LOG_PATH = os.path.join(LOGGING_PATH, "multiverse.log")
# Klasörün varlığını kontrol et ve yoksa oluştur
if not os.path.exists(LOGGING_PATH):
    os.makedirs(LOGGING_PATH)
# Chat geçmişi için kaydetme dizini
CHAT_HISTORY_DIR = os.path.join("data", "chatting_historys")
# Klasörün varlığını kontrol et ve yoksa oluştur
if not os.path.exists(CHAT_HISTORY_DIR):
    os.makedirs(CHAT_HISTORY_DIR)
# **Özerklik Yönetimi Parametreleri**
# Özerklik durumu başlangıç değeri
IS_AUTONOMOUS_DEFAULT = False
# Görev ve hedef yönetimi
TASKS_DEFAULT = []  # Kısa vadeli görevlerin varsayılan listesi
LONG_TERM_GOALS_DEFAULT = []  # Uzun vadeli hedeflerin varsayılan listesi
REFLECTIONS_DEFAULT = []  # Özeleştirilerin varsayılan listesi
META_ACTIONS_DEFAULT = []  # Meta eylem kayıtlarının varsayılan listesi
EXTERNAL_KNOWLEDGE_DEFAULT = []  # Dış çevreden toplanan bilgilerin varsayılan listesi
# Özerklik zamanlaması
LAST_AUTONOMOUS_ACTION_DEFAULT = None  # Son yapılan özerk eylemin başlangıç değeri
NEXT_SCHEDULED_ACTION_DEFAULT = None  # Planlanan bir sonraki eylemin başlangıç değeri
# Risk ve etik uyumluluk parametreleri
RISK_ANALYSIS_LOG_DEFAULT = []  # Risk analiz loglarının başlangıç değeri
ETHICAL_COMPLIANCE_LOG_DEFAULT = []  # Etik uyumluluk loglarının başlangıç değeri
# Loglama ve dosya yönetimi
LOG_FILE_PATH = os.path.join("logs", "advanced_autonomy.log")
if not os.path.exists(os.path.dirname(LOG_FILE_PATH)):
    os.makedirs(os.path.dirname(LOG_FILE_PATH))  # Log dosyası için dizin oluşturma
# Zaman yönetimi parametreleri
CYCLE_INTERVAL = 10  # Özerklik döngüsünün çalışma aralığı (saniye)
MAX_IDLE_TIME = 300  # Maksimum hareketsizlik süresi (saniye)
LAST_CYCLE_START_DEFAULT = None  # Son başlatılan döngünün başlangıç değeri
DEFAULT_SESSION_DURATION = 3600  # Süre saniye cinsindedir (3600 saniye = 1 saat)
DEFAULT_SESSION_TIMEOUT = 7200
# Sistem durumu izleme parametreleri
SYSTEM_STATUS_DEFAULT = {
    "cpu_usage": 0.0,  # CPU kullanımı (%)
    "memory_usage": 0.0,  # Bellek kullanımı (%)
    "disk_space": 0.0,  # Kalan disk alanı (%)
    "error_count": 0  # Tespit edilen hata sayısı
}
# Etik kurallar ve güvenlik protokolleri
ETHICAL_GUIDELINES = {
    "follow_leader": True,  # Rehberin kararlarına sadık kalma
    "prohibit_harm": True,  # Zarar verebilecek eylemleri yasaklama
    "preserve_privacy": True  # Özel bilgilerin korunmasını sağlama
}
# Hata yönetimi ve düzeltme parametreleri
ERROR_CORRECTION_SETTINGS = {
    "retry_limit": 3,  # Hatalı işlemlerde maksimum yeniden deneme sayısı
    "log_errors": True,  # Hataları loglama
    "notify_leader": True  # Hatalarda rehberi bilgilendirme
}
# Bellek boyutları - esneklik için hiperparametreler

# Eğitim ile ilgili diğer Sabitler
LOG_INTERVAL = 10
TRAIN_TIMEOUT = 3000  # saniye cinsinden, varsayılan 30 saniye
TRAIN_VAL_SPLIT = 0.2  # Eğitim-veri setinin %20'si doğrulama için kullanılacak.
SAVE_MODEL = True
MODEL_DIR = os.path.join('saved_models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'cevahir_model.pt') # Kaydedilen Model Yolu
CHECKPOINT_MODEL= "saved_models/checkpoint_model/"
if not os.path.exists(CHECKPOINT_MODEL):
    os.makedirs(CHECKPOINT_MODEL)

MODEL_ROUTE_BASE_URL = "http://127.0.0.1:5000/model"
# Vocab dosyasının kaydedileceği dizin

JSON_VOCAB_SAVE_PATH = 'data/json_vocab_files/vocab.json'
DOCX_VOCAB_SAVE_PATH = 'data/docx_vocab_files/docx_vocab.json'

# Maksimum girdi uzunluğu
MAX_INPUT_LENGTH = 4096
MAX_LENGTH = 8192
MAX_TOKENS = 20000  # Sınırlı bir token sayısı limiti

# Eğitim Cihazı Ayarları
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_DEVICE_USAGE = 0.8

# Log dosyasının maksimum boyut sınırı (50 MB)
MAX_LOG_SIZE = 50 * 1024 * 1024  # Byte cinsinden

# Log dosyaları için yedek sayısı (20 adet)
BACKUP_COUNT = 20


# Dosya Yolu ve Veri Yükleme Parametreleri
DOCUMENTS_DIR = 'data/training_documents'
MODEL_TRAINING_DATA_DIR = 'data/machine_learning_data' # makine diline çevirilen verinin kaydedileceği klasör
PROCESSED_DATA_DIR = 'data/json_processed_data'  # İşlenmiş verilerin saklanacağı klasör
NUM_RANDOM_JSONS = 6    # Rastgele seçilecek JSON dosya sayısı
NUM_RANDOM_DOCX = 2     # Rastgele seçilecek DOCX dosya sayısı
PDF_ENCODING = 'utf-8'  # veya projede gerekli olan başka bir encoding türü
NUM_CLASSES = 12
TEMPERATURE = 1.0
# JSON veri yükleme ayarları
JSON_DIR = "data/training_jsons"        # JSON dosyalarının bulunduğu dizin
# DOCX veri yükleme ayarları
DOCX_DIR = "data/training_docxs"
TRAINING_HISTORY = "data/training_historys"
# JSON veri yapısı ayarları
SORU_KEY = "Soru"                  # JSON verisinde soru anahtarı için kullanılan ad
CEVAP_KEY = "Cevap"                # JSON verisinde cevap anahtarı için kullanılan ad

# Loglama ayarları
LOGGING_ENABLED = True             # Loglamanın aktif olup olmadığını belirtir

STOPWORDS = set()

SPECIAL_TOKENS = {
    "<soru>": "<soru>",
    "<cevap>": "<cevap>",
    "<son>": "<son>",
    "<isim>": "<isim>",
    "<yer>": "<yer>",
    "<tarih>": "<tarih>",
    "<kisi>": "<kisi>",
    "<miktar>": "<miktar>",
    "<duygu>": "<duygu>",
    "<nitelik>": "<nitelik>",
    "<url>": "<url>",
    "<e-posta>": "<e-posta>"
}


# Daha güvenli ve bağlama duyarlı ek listesi için örnek
SUFFIXES = []




# Vocabulary (Kelime Hazinesi) Ayarları
MIN_FREQ = 0           # En düşük kelime frekansı, bu değerin altındaki kelimeler göz ardı edilecek
MAX_VOCAB_SIZE = 1000000   # Kelime hazinesi için maksimum kelime sayısı

# Model değerlendirme metrikleri
METRICS = {
    "accuracy": "Doğruluk",
    "precision": "Kesinlik",
    "recall": "Duyarlılık",
    "f1_score": "F1 Skoru",
}

# Konu Listesi (TOPICS)
TOPICS = {
    "Ekonomi": ["ekonomi", "finans", "borsa", "yatırım", "piyasa", "enflasyon", "döviz", "para", "faiz", "hisse", "borsa", "maliye"],
    "Teknoloji": ["teknoloji", "yazılım", "donanım", "robot", "AI", "yapay zeka", "internet", "mobil", "inovasyon", "siber güvenlik", "geliştirici", "yapay sinir ağı"],
    "Sağlık": ["sağlık", "hastane", "doktor", "ilaç", "epidemik", "pandemi", "aşı", "hastalık", "tedavi", "beslenme", "psikoloji", "terapi", "bakım"],
    "Spor": ["spor", "futbol", "basketbol", "voleybol", "şampiyona", "antrenman", "skor", "sporcu", "lig", "turnuva", "yarış", "atletizm"],
    "Eğitim": ["eğitim", "okul", "öğretmen", "ders", "öğrenci", "müfredat", "lise", "üniversite", "sınav", "öğrenme", "okul müdürü"],
    "Siyaset": ["siyaset", "politika", "hükümet", "parti", "seçim", "meclis", "kanun", "devlet", "yasa", "lider", "milletvekili", "diplomasi"],
    "Çevre": ["çevre", "iklim", "doğa", "orman", "su", "çevre koruma", "atık", "kirlilik", "sürdürülebilirlik", "küresel ısınma", "yenilenebilir enerji"],
    "Sanat": ["sanat", "resim", "müzik", "heykel", "tiyatro", "galeri", "sergi", "sinema", "sanatçı", "yaratıcılık", "dans", "müzik festivali"],
    "Tarih": ["tarih", "medeniyet", "antik", "imparatorluk", "savaş", "devrim", "kültür", "arşiv", "belge", "miras", "arkeoloji", "eser"],
    "Bilim": ["bilim", "fizik", "kimya", "biyoloji", "araştırma", "deney", "keşif", "laboratuvar", "teori", "fen", "akademik"],
    "Tarım": ["tarım", "çiftçilik", "hayvancılık", "bitki", "ürün", "hasat", "gübre", "tohum", "tarla", "zirai", "agronomi"],
    "Uzay": ["uzay", "astronomi", "gezegen", "galaksi", "yıldız", "güneş sistemi", "roket", "nasa", "uzay aracı", "evren", "yörünge"],
    "Psikoloji": ["psikoloji", "zihin", "davranış", "terapi", "bilinç", "kişilik", "stres", "beyin", "duygu", "motivasyon", "psikoterapi", "analiz"],
    "Edebiyat": ["edebiyat", "şiir", "roman", "öykü", "yazar", "kitap", "edebi", "eser", "kurgu", "eleştiri", "biografi", "şiir"],
    "Moda": ["moda", "stil", "trend", "kıyafet", "tasarım", "modacı", "defile", "giyim", "aksesuar", "marka", "stilist", "koleksiyon"],
    "Yemek": ["yemek", "mutfak", "tarif", "lezzet", "şef", "restoran", "aşçılık", "diyet", "tat", "malzeme", "gurme", "lezzet turu"],
    "Hukuk": ["hukuk", "avukat", "mahkeme", "kanun", "hakim", "ceza", "dava", "anayasa", "adalet", "yargı", "sözleşme", "hak"],
    "İş Dünyası": ["iş", "şirket", "işveren", "girişimcilik", "start-up", "CEO", "yönetim", "strateji", "satış", "pazarlama", "organizasyon", "yönetişim"],
    "Uluslararası İlişkiler": ["dış politika", "antlaşma", "BM", "NATO", "uluslararası hukuk", "mülteci", "göçmen", "savaş", "diplomat", "yaptırım"],
    "Finans": ["finans", "bankacılık", "yatırım", "banka", "para birimi", "kredi", "sermaye", "piyasa", "borç", "döviz", "kredi kartı"],
    "Pazarlama": ["pazarlama", "reklam", "sosyal medya", "hedef kitle", "strateji", "tanıtım", "kampanya", "satış", "marka", "müşteri"],
    "Sağlık Sigortası": ["sağlık sigortası", "prim", "poliçe", "teminat", "tazminat", "kapsam", "hastane", "ameliyat", "sigortalı", "sağlık hizmetleri"],
    "Siber Güvenlik": ["siber güvenlik", "güvenlik açığı", "veri koruma", "şifreleme", "hacker", "ağ güvenliği", "antivirüs", "tehdit", "koruma"],
    "Felsefe": ["felsefe", "düşünce", "etik", "ahlak", "mantık", "ontoloji", "felsefi akımlar", "sorgulama", "bilgelik", "teori"],
    "Doğa Bilimleri": ["biyoloji", "zooloji", "botanik", "mikrobiyoloji", "ekosistem", "doğa", "çevre", "hücre", "evrim", "hayvan bilimi"],
    "Matematik": ["matematik", "geometri", "aritmetik", "istatiksel analiz", "teorem", "modelleme", "kalkülüs", "lineer cebir", "analiz", "matris"],
    "Sosyal Medya": ["sosyal medya", "facebook", "twitter", "instagram", "içerik üretici", "like", "yorum", "video paylaşımı", "influencer", "kampanya"],
    "Otomotiv": ["otomotiv", "araç", "otomobil", "motor", "yakıt", "elektrikli araç", "araç güvenliği", "lastik", "marka", "araç bakımı"],
    "Yapay Zeka": ["AI", "yapay zeka", "makine öğrenimi", "neural network", "otomasyon", "algoritma", "derin öğrenme", "doğal dil işleme", "öğrenme"],
    "Kripto Para": ["kripto", "bitcoin", "ethereum", "blockchain", "madencilik", "dijital para", "wallet", "token", "borsa", "yatırım"],
    "İnsan Hakları": ["insan hakları", "özgürlük", "eşitlik", "adalet", "temel haklar", "insan onuru", "toplumsal haklar", "uluslararası sözleşmeler"],
    "Mimarlık": ["mimarlık", "tasarım", "inşaat", "yapı", "şehir planlama", "gökyüzü", "proje", "mimari akımlar", "kentsel dönüşüm"],
    "Tarım": ["tarım", "sürdürülebilir tarım", "çiftlik", "verimlilik", "gübre", "ziraat", "hasat", "ekolojik tarım", "çevreye duyarlılık"],
    "Astronomi": ["astronomi", "gezegenler", "yıldızlar", "gökbilim", "evren", "uzay araştırmaları", "roket", "nasa", "teleskop"],
    "Dil Bilimleri": ["dil bilimi", "gramer", "sentaks", "dil tarihi", "fonoloji", "ses bilgisi", "linguistik", "anlambilim"],
    "Girişimcilik": ["girişimcilik", "startup", "yatırımcı", "inovasyon", "büyüme", "risk sermayesi", "ürün geliştirme", "iş modeli"]
}


# bu eski sürüm dosyasıdır. dikkate almana gerek yok.