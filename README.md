# 🔥 Türkiye Orman Yangını Erken Tespit Sistemi

NASA FIRMS uydu verisiyle beslenen, Türkiye genelinde orman yangını riskini
**dört farklı makine öğrenmesi yaklaşımıyla** (yangın olasılığı, sınıflandırma,
zaman serisi tahmini, anomali tespiti) analiz eden uçtan uca bir sistem.
Sonuçlar, bir LLM katmanıyla doğal dilde de açıklanıyor.

## Proje Hikayesi

Bu proje, tek bir modelin ötesine geçip **ortak bir veri işleme hattından
beslenen dört bağımsız ML problemi** kurmayı hedefliyor. Amaç, gerçek zamanlı
uydu verisiyle çalışan, otomatik veri toplayan ve sürekli güncellenen bir
sistem inşa etmek — statik bir Kaggle veri setinden değil, canlı bir API'den.

📍 Detaylı yol haritası ve ilerleme takibi için: [Turkey Wildfire Detection Roadmap](https://github.com/users/pelsingunduz/projects/2/views/1)

## Mimari
```
NASA FIRMS API (VIIRS_SNPP_NRT canlı + VIIRS_SNPP_SP geçmiş, Türkiye bounding box)
↓
INGESTION — her 4 saatte bir otomatik veri çekme (GitHub Actions) + 3 yıllık geçmiş veri backfill
↓
FEATURE ENGINEERING
├── Duplicate temizleme
├── Türkiye sınır filtresi (point-in-polygon)
├── Zaman dilimi düzeltmesi (UTC → Türkiye saati)
├── 0.25° grid ataması
├── Günlük, bölge bazlı özet tablo (tespit edilen günler)
└── Tam panel: grid × takvim günü (yangınsız günler dahil, olasılık modeli için)
↓
┌───────────────┬─────────────────┬────────────────────┬──────────────────────┐
YANGIN OLASILIĞI SINIFLANDIRMA ZAMAN SERİSİ ANOMALİ TESPİTİ
(yarın olur mu?) (risk seviyesi) (24s tahmin) (z-score tabanlı
sapma tespiti)
└───────────────┴─────────────────┴────────────────────┴──────────────────────┘
↓
DASHBOARD (Streamlit) — harita üzerinde bölge seçimi, 4 bağımsız panel
+ LLM (OpenAI) ile doğal dilde açıklama
```

## Otomasyon

Veri toplama ve işleme, GitHub Actions üzerinde **her 4 saatte bir** otomatik
çalışır (`.github/workflows/data_pipeline.yml`). Bu, projenin bilgisayarın
açık/kapalı olma durumundan tamamen bağımsız, sürekli veri biriktirmesini
sağlar:

1. FIRMS API'den yeni veri çekilir
2. Veri temizlenir, işlenir, günlük özet tablo ve olasılık paneli güncellenir
3. Güncellenen veri otomatik olarak repoya commit'lenir

FIRMS API key'i, GitHub Secrets üzerinden güvenli şekilde workflow'a
aktarılır — kod içinde hiçbir yerde açık şekilde bulunmaz.

Model eğitimi de otomasyona dahildir: veri güncellendikten sonra **dört
model** (yangın olasılığı, sınıflandırma, zaman serisi, anomali tespiti)
`src/models/` altındaki script'lerle otomatik olarak yeniden eğitilir ve
güncellenen model dosyaları (`outputs/models/*.joblib`) da veriyle birlikte
repoya commit'lenir. Bu sayede dashboard'u ne zaman açsan, en güncel veriyle
eğitilmiş modelleri kullanmış olursun (yalnızca `git pull` yeterlidir).

## Tasarım Kararları ve Gerekçeleri

Bu proje, her adımda bilinçli mühendislik kararları içeriyor:

- **Tek sensör (VIIRS_SNPP_NRT):** Çoklu sensör kullanmak duplicate detection
  karmaşıklığı getirirdi; VIIRS'in ~375m çözünürlüğü küçük yangınları da
  yakalayabiliyor.
- **0.25° grid boyutu:** 0.5° bölgesel farkları (örn. Akdeniz kıyı şeridinin
  yüksek riski) eritiyor, 0.1° ise çoğu hücreyi anlamlı veri olmadan boş
  bırakıyor. 0.25°, ilçe ölçeğine yakın bir çözünürlükle dengeyi tutturuyor.
- **Günlük + haftalık baseline:** Uydu geçişleri düzensiz olduğu için saatlik
  dilimler güvenilir değil; günlük özet her zaman dolu veri sağlıyor.
- **Risk etiketleri veri dağılımından türetildi:** Keyfi eşikler yerine,
  `fire_count`'un yüzdelik dilimlerine (%50, %80, %90) bakılarak
  düşük/orta/yüksek sınırları belirlendi.
- **Kronolojik train/test split (zaman serisi ve olasılık modeli):** Rastgele
  split, modelin "geleceği bilerek" eğitilmesine (data leakage) yol açardı.
- **Anomali baseline'ı bugünü hariç tutuyor:** `shift(1)` ile hesaplanan
  baseline, bir anomalinin kendi ortalamasını yukarı çekip kendini
  normalleştirmesini engelliyor.
- **Ham veri (`data/raw`) ile işlenmiş veri (`data/processed`) ayrımı:**
  Ham veri asla değiştirilmiyor; işlenmiş veri istenildiği an yeniden
  üretilebilir.
- **Duplicate ingestion, tasarımın bilinçli bir sonucu:** FIRMS her çekimde
  son 24 saatin tamamını döndürdüğü için, 4 saatlik aralıklı çekim
  güvenilirlik sağlıyor (bir çekim başarısız olsa bile veri kaybolmuyor),
  ama `drop_duplicates()` adımını pipeline'ın kalıcı bir parçası haline
  getiriyor.
- **Yerel cron yerine GitHub Actions:** İlk versiyonda yerel `cron` kullanıldı,
  ama bu bilgisayar kapalı/uykudayken veri kaybına yol açıyordu. Pipeline,
  bilgisayardan bağımsız çalışabilmesi için GitHub Actions'a taşındı — artık
  veri toplama, laptop kapalı olsa bile kesintisiz devam ediyor.
- **Sınıflandırmada beş algoritma karşılaştırıldı:** Random Forest, XGBoost,
  Logistic Regression, SVM ve KNN aynı veri ve test seti üzerinde eğitilip
  karşılaştırıldı. Ağaç tabanlı olmayan üç model (Logistic Regression, SVM,
  KNN) için `StandardScaler` ile ölçeklendirme uygulandı, çünkü bu
  algoritmalar feature büyüklüğünden etkileniyor. Dashboard'un resmi
  modeli, karşılaştırma sonuçlarına göre **Logistic Regression** olarak
  seçildi (bkz. Model Karşılaştırmaları — bu karşılaştırma küçük bir veri
  setiyle yapıldı, geçmiş veri backfill'i sonrası yeniden değerlendirilmeyi
  bekliyor).
- **Anomali tespitinde sıfır-varyans düzeltmesi:** Bir grid hücresinin
  geçmişinde hiç varyans yoksa (örn. hep aynı sayıda tespit), z-score
  hesaplaması sıfıra bölme nedeniyle yapay olarak "sonsuz anomali"
  üretiyordu. Bu durumlar artık `NaN` (hesaplanamaz) olarak işaretleniyor,
  yanlış pozitif anomali sayısını önlüyor.
- **3 yıllık geçmiş veri backfill'i (SP + NRT hibrit kaynak):** FIRMS'in
  `VIIRS_SNPP_SP` (bilimsel kalite, ~5 ay gecikmeli) kaynağı geçmiş veri
  için tercih edildi; SP boş/geçersiz dönerse otomatik olarak
  `VIIRS_SNPP_NRT`'ye düşülüyor (son birkaç ay henüz SP'ye işlenmemiş
  olabileceği için). Bu, veri setini ~260 satırdan ~46.000 satıra
  (bölge × gün) çıkardı.
- **Yangın olasılığı modeli — target leakage önleme:** Bu modelin
  feature'ları arasında `avg_brightness`/`avg_frp`/`max_frp` YOK, çünkü
  bunlar ancak bir yangın zaten tespit edildiğinde ölçülebilir; bir
  "olacak mı" tahmininde kullanmak dolambaçlı bir tautoloji olurdu.
  Bunun yerine sadece coğrafi konum, takvimsel (ay, yılın günü) ve
  geçmişe dayalı (lag/rolling occurrence rate) feature'lar kullanılıyor.
- **Aşırı sınıf dengesizliği (yangın oranı ~%3.3):** `class_weight='balanced'`
  (Random Forest, Logistic Regression) ve `scale_pos_weight` (XGBoost) ile
  ele alındı; başarı ölçütü olarak accuracy yerine **ROC-AUC** ve azınlık
  sınıfının (yangın=evet) precision/recall'u kullanıldı.
- **Zaman serisi tahmininde negatif değer kırpması:** `LinearRegression`
  çıktısını 0'da sınırlamıyor, matematiksel olarak negatif bir "sıcak nokta
  sayısı" üretebiliyordu (anlamsız). Her tahmin artık `np.clip(tahmin, 0, None)`
  ile sıfırın altına düşürülmüyor.
- **Büyük model dosyaları git'e commit'lenmiyor:** Karşılaştırma amaçlı
  eğitilen ara modeller (örn. Random Forest varyantları, 100MB+ olabiliyor)
  `.gitignore`'da; sadece dashboard'un gerçekten kullandığı, küçük resmi
  modeller (`classifier.joblib`, `occurrence.joblib` vb.) repoya commit'leniyor.
- **LLM açıklama katmanı, güvenli fallback ile:** Dashboard, sayısal
  sonuçları OpenAI (`gpt-4.1-nano`) ile doğal dile çeviriyor. API
  hatası/kota/bağlantı sorunu olursa, elle yazılmış bir şablon özete
  otomatik düşülüyor — kullanıcı hiçbir zaman boş/bozuk bir ekran görmüyor.
  Aynı bölge + aynı gün için tekrar tıklamalarda API'ye tekrar gidilmemesi
  için sonuçlar cache'leniyor.

## Model Karşılaştırmaları

### Yangın Olasılığı (occurrence) — ~1.4M satır, kronolojik split

| Model | ROC-AUC | Precision (yangın) | Recall (yangın) | F1 (yangın) |
|---|---|---|---|---|
| XGBoost | 0.91 | 0.20 | 0.76 | 0.32 |
| Logistic Regression | 0.90 | 0.28 | 0.68 | 0.40 |
| Random Forest | 0.86 | 0.33 | 0.27 | 0.30 |

Resmi model olarak **XGBoost** seçildi — en yüksek ROC-AUC ve en yüksek
recall'a sahip. Erken uyarı sisteminde kaçırılan bir yangının maliyeti
yanlış alarmdan çok daha yüksek olduğu için düşük precision kasıtlı bir
tercih.

### Sınıflandırma (risk seviyesi) — ~260 satır (ilk veri seti), test seti: 52 satır

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.69 | 0.58 |
| SVM | 0.65 | 0.53 |
| Random Forest | 0.62 | 0.52 |
| XGBoost | 0.60 | 0.51 |
| KNN | 0.56 | 0.48 |

**Not:** Bu karşılaştırma, geçmiş veri backfill'inden ÖNCEKİ küçük veri
setine dayanıyor. Artık aynı büyük veri setiyle (~46.000 satır) yeniden
çalıştırılıp güncellenmesi gerekiyor — bkz. Gelecek Geliştirmeler.

## Bilinen Sınırlamalar

- **Grid-il/ilçe eşleştirmesi yaklaşıktır.** 0.25° bir hücre birden fazla
  idari bölgeyi kapsayabilir; dashboard'daki isimler hücre merkezinin en
  yakın idari birimini gösterir, kesin sınır değildir.
- **Bounding box, komşu ülkelerin sınır bölgelerini de içeriyordu**
  (Irak, Suriye, Yunanistan vb.); bu, `geopandas` ile point-in-polygon
  filtrelemesiyle giderildi.
- **Sınıflandırma, zaman serisi ve anomali modelleri, geçmiş veri
  backfill'i sonrası henüz yeniden eğitilip karşılaştırılmadı.** Bu üç
  model teknik olarak artık büyük veri setiyle (GitHub Actions her 4
  saatte bir yeniden eğitiyor) çalışıyor, ama algoritma seçimi kararları
  hâlâ küçük veri setindeki karşılaştırmaya dayanıyor.
- **Yangın olasılığı modelinin precision'ı düşük (~%20-33).** Bu kasıtlı
  bir tercih (yüksek recall'u önceliklendirme) ama pratikte "gerçek"
  alarmların çoğu yanlış alarm anlamına geliyor — dashboard kullanıcısı
  bunu bilerek yorumlamalı.

## Teknoloji Yığını

- **Veri:** NASA FIRMS API (VIIRS_SNPP_NRT canlı, VIIRS_SNPP_SP geçmiş)
- **İşleme:** pandas, geopandas, shapely, numpy
- **Modelleme:** scikit-learn (RandomForestClassifier, LogisticRegression,
  LinearRegression, SVC, KNeighborsClassifier), XGBoost
- **LLM:** OpenAI API (`gpt-4.1-nano`) — sonuçların doğal dilde açıklanması
- **Otomasyon:** GitHub Actions (bulut tabanlı, sürekli çalışan CI/CD pipeline'ı)
- **Dashboard:** Streamlit, folium, streamlit-folium
- **Coğrafi isimlendirme:** geopy (Nominatim/OpenStreetMap)

## Proje Yapısı
```
turkey-wildfire-detection/
├── data/
│ ├── raw/ # FIRMS'ten çekilen ham CSV'ler (canlı + geçmiş, zaman damgalı)
│ └── processed/ # Günlük özet, tam panel, grid-il/ilçe eşleşmesi
├── src/
│ ├── ingestion.py # FIRMS API'den canlı veri çekme
│ ├── historical_backfill.py # 3 yıllık geçmiş veri (SP+NRT hibrit)
│ ├── geocode_grid.py # Grid hücreleri için il/ilçe isimleri
│ ├── features.py # Temizlik, sınır filtresi, grid, günlük özet, tam panel
│ └── models/
│ ├── occurrence.py # Yangın olasılığı (XGBoost)
│ ├── classifier.py
│ ├── forecaster.py
│ └── anomaly.py
├── notebooks/
│ └── eda.ipynb # Keşifsel analiz
├── dashboard/
│ └── app.py # Streamlit dashboard + LLM açıklama katmanı
├── outputs/
│ └── models/ # Eğitilmiş modeller (joblib, sadece resmi olanlar commit'li)
└── requirements.txt
```

## Kurulum ve Çalıştırma

```bash
git clone <repo-url>
cd turkey-wildfire-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`.env` dosyasına kendi API key'lerinizi ekleyin:

```
FIRMS_MAP_KEY=your_firms_key_here
OPENAI_API_KEY=your_openai_key_here
```

FIRMS key için [ücretsiz kayıt](https://firms.modaps.eosdis.nasa.gov/api/map_key/),
OpenAI key için [platform.openai.com](https://platform.openai.com/api-keys)
(LLM açıklama katmanı opsiyoneldir — key yoksa dashboard otomatik olarak
şablon özete düşer).

Veri toplama ve işleme:

```bash
python src/ingestion.py
python src/features.py
```

Geçmiş veri backfill'i (tek seferlik, opsiyonel — repo zaten 3 yıllık veri içeriyor):

```bash
python src/historical_backfill.py
```

Grid hücreleri için il/ilçe isimleri (tek seferlik, opsiyonel):

```bash
python src/geocode_grid.py
```

Modelleri eğitme:

```bash
python src/models/occurrence.py
python src/models/classifier.py
python src/models/forecaster.py
python src/models/anomaly.py
```

Dashboard'u başlatma:

```bash
streamlit run dashboard/app.py
```

## Gelecek Geliştirmeler

- Sınıflandırma, zaman serisi ve anomali modellerinin büyük veri setiyle
  (~46.000 satır) yeniden karşılaştırılıp algoritma seçiminin gözden
  geçirilmesi
- Yangın olasılığı modelinin eşik değerinin (şu an varsayılan 0.5)
  precision/recall dengesi için ayarlanması
- İnsan kaynaklı risk faktörlerinin (nüfus/atık yoğunluğu gibi) modele
  eklenmesi
- Çoklu sensör desteği (MODIS ile karşılaştırmalı analiz)
- Zaman serisi ve anomali tespiti için de çoklu algoritma karşılaştırması
  (Random Forest Regressor, Isolation Forest gibi)

<img width="1130" height="589" alt="Ekran Resmi 2026-09-12 15 24 31" src="https://github.com/user-attachments/assets/fdce683d-66c0-4cbd-8653-ec7f03f27eea" />
<img width="1139" height="601" alt="Ekran Resmi 2026-09-12 15 24 19" src="https://github.com/user-attachments/assets/1cba0552-e300-4b42-913a-456dec82be83" />
<img width="1145" height="603" alt="Ekran Resmi 2026-09-12 15 23 41" src="https://github.com/user-attachments/assets/7af7bda4-44df-4723-94ba-472e7ad9063d" />
