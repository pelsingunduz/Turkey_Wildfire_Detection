# 🔥 Türkiye Orman Yangını Erken Tespit Sistemi

NASA FIRMS uydu verisiyle beslenen, Türkiye genelinde orman yangını riskini
üç farklı makine öğrenmesi yaklaşımıyla (sınıflandırma, zaman serisi tahmini,
anomali tespiti) analiz eden uçtan uca bir sistem.

## Proje Hikayesi

Bu proje, tek bir modelin ötesine geçip **ortak bir veri işleme hattından
beslenen üç bağımsız ML problemi** kurmayı hedefliyor. Amaç, gerçek zamanlı
uydu verisiyle çalışan, otomatik veri toplayan ve sürekli güncellenen bir
sistem inşa etmek — statik bir Kaggle veri setinden değil, canlı bir API'den.

## Mimari

\`\`\`

NASA FIRMS API (VIIRS_SNPP_NRT, Türkiye bounding box)
↓
INGESTION — her 4 saatte bir otomatik veri çekme (cron)
↓
FEATURE ENGINEERING
├── Duplicate temizleme
├── Türkiye sınır filtresi (point-in-polygon)
├── Zaman dilimi düzeltmesi (UTC → Türkiye saati)
├── 0.25° grid ataması
└── Günlük, bölge bazlı özet tablo
↓
┌─────────────────┬────────────────────┬──────────────────────┐
SINIFLANDIRMA ZAMAN SERİSİ ANOMALİ TESPİTİ
(risk seviyesi) (24s tahmin) (z-score tabanlı
sapma tespiti)
└─────────────────┴────────────────────┴──────────────────────┘
↓
DASHBOARD (Streamlit) — harita üzerinde bölge seçimi, 3 bağımsız panel

\`\`\`

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
- **Kronolojik train/test split (zaman serisi):** Rastgele split, modelin
  "geleceği bilerek" eğitilmesine (data leakage) yol açardı.
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

## Bilinen Sınırlamalar

- **Zaman serisi ve anomali tespiti, henüz istatistiksel olarak anlamlı
  değil.** Proje erken aşamada olduğu için (birkaç günlük veri), bu iki
  model az sayıda örnekle çalışıyor. Sınıflandırma dışındaki metrikler,
  veri biriktikçe (birkaç hafta) yeniden değerlendirilmelidir.
- **Grid-il/ilçe eşleştirmesi yaklaşıktır.** 0.25° bir hücre birden fazla
  idari bölgeyi kapsayabilir; dashboard'daki isimler hücre merkezinin en
  yakın idari birimini gösterir, kesin sınır değildir.
- **Bounding box, komşu ülkelerin sınır bölgelerini de içeriyordu**
  (Irak, Suriye, Yunanistan vb.); bu, `geopandas` ile point-in-polygon
  filtrelemesiyle giderildi.

## Teknoloji Yığını

- **Veri:** NASA FIRMS API (VIIRS_SNPP_NRT)
- **İşleme:** pandas, geopandas, shapely
- **Modelleme:** scikit-learn (RandomForestClassifier, LinearRegression)
- **Otomasyon:** cron
- **Dashboard:** Streamlit, folium
- **Coğrafi isimlendirme:** geopy (Nominatim/OpenStreetMap)

## Proje Yapısı

\`\`\`

turkey-wildfire-detection/
├── data/
│ ├── raw/ # FIRMS'ten çekilen ham CSV'ler (zaman damgalı)
│ └── processed/ # Temizlenmiş, grid'e oturtulmuş özet tablolar
├── src/
│ ├── ingestion.py # FIRMS API'den veri çekme
│ ├── features.py # Temizlik, sınır filtresi, grid, günlük özet
│ └── models/
│ ├── classifier.py
│ ├── forecaster.py
│ └── anomaly.py
├── notebooks/
│ └── eda.ipynb # Keşifsel analiz
├── dashboard/
│ └── app.py # Streamlit dashboard
├── outputs/
│ └── models/ # Eğitilmiş modeller (joblib)
└── requirements.txt

\`\`\`

## Kurulum ve Çalıştırma

```bash
git clone <repo-url>
cd turkey-wildfire-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`.env` dosyasına kendi FIRMS MAP_KEY'inizi ekleyin ([ücretsiz kayıt](https://firms.modaps.eosdis.nasa.gov/api/map_key/)):

FIRMS_MAP_KEY=your_key_here


Veri toplama ve işleme:
```bash
python src/ingestion.py
python src/features.py
```

Modelleri eğitme:
```bash
python src/models/classifier.py
python src/models/forecaster.py
python src/models/anomaly.py
```

Dashboard'u başlatma:
```bash
streamlit run dashboard/app.py
```

## Gelecek Geliştirmeler

- Veri biriktikçe (4-8 hafta) anomali tespitinin gerçek baseline ile
  çalışması
- Model eğitiminin otomatikleştirilmesi (şu an elle tetikleniyor)
- Çoklu sensör desteği (MODIS ile karşılaştırmalı analiz)
- Bulut tabanlı sürekli çalışma (yerel cron yerine, laptop kapalıyken de
  veri toplanabilmesi)
- Konum bilgisinin (grid koordinatları veya bölge kategorileri) veri
  arttıkça modele feature olarak eklenmesi

