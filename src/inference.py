"""
Tahmin (inference) katmanı -- dört modelin çıktısını Streamlit'ten bağımsız,
düz Python fonksiyonları olarak sunar.

Neden ayrı bir katman: Daha önce bu mantığın tamamı dashboard/app.py içinde,
Streamlit'in cache dekoratörlerine ve harita tıklamasına bağlıydı. Aynı
tahminlere artık başka tüketiciler de ihtiyaç duyuyor:
    - LLM agent'ı (tool calling: "Muğla'da yarın risk ne?")
    - n8n / zamanlanmış uyarılar ("yarın en riskli 10 bölge hangisi?")
    - ileride FastAPI endpoint'leri ve değerlendirme (eval) script'leri

Bu yüzden tahmin mantığı tek bir yerde duruyor; dashboard da dahil herkes
bu fonksiyonları çağırıyor. Fonksiyonlar JSON'a çevrilebilir sözlükler
döndürüyor (numpy tipleri değil), böylece doğrudan bir LLM'e ya da HTTP
cevabına verilebiliyor.

Kullanım (repo kök dizininden, diğer script'lerle aynı şekilde):
    from src.inference import find_regions, get_region_report, get_top_risk_regions

    find_regions("muğla")                 # -> [{'grid_id': ..., 'display_name': 'Muğla - Marmaris', ...}, ...]
    get_region_report("36.75_28.25")      # -> 4 modelin çıktısı tek sözlükte
    get_top_risk_regions(n=10)            # -> yarın için en yüksek yangın olasılıklı bölgeler

Veri ve modeller süreç boyunca bir kez yüklenip önbellekte tutulur. Uzun
süre çalışan bir süreçte (ör. API sunucusu) yeni veriyi almak için
clear_cache() çağrılmalı.
"""

import os
import sys
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

# Proje script'leri (dashboard dahil) modülleri src/ ve src/models/ altından
# doğrudan isimleriyle import ediyor; aynı düzeni koruyoruz ki bu dosya hem
# `from src.inference import ...` hem de `from inference import ...` ile
# kullanılabilsin.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (_SRC_DIR, os.path.join(_SRC_DIR, "models")):
    if _path not in sys.path:
        sys.path.append(_path)

from features import (  # noqa: E402
    WEATHER_FEATURE_COLUMNS,
    build_latest_occurrence_features,
    fetch_forecast_weather_batch,
)
from anomaly import calculate_zscore_features, flag_anomalies  # noqa: E402
from forecaster import add_lag_features, predict_clipped, FEATURE_COLUMNS as FORECASTER_FEATURES  # noqa: E402
from classifier import REVERSE_LABEL_MAP, FEATURE_COLUMNS as CLASSIFIER_FEATURES  # noqa: E402
from occurrence import FEATURE_COLUMNS as OCCURRENCE_FEATURES  # noqa: E402

DAILY_SUMMARY_PATH = "data/processed/daily_grid_summary.csv"
LOCATION_NAMES_PATH = "data/processed/grid_location_names.csv"
MODEL_DIR = "outputs/models"
UNKNOWN = "Bilinmiyor"

_TR_ASCII = str.maketrans("çğıİöşüÇĞÖŞÜâîû", "cgiiosucgosuaiu")


def normalize_text(text):
    """Türkçe karakterleri sadeleştirip küçük harfe çevirir: 'MUĞLA', 'Mugla',
    'muğla' hepsi 'mugla' olur. Kullanıcı (ya da LLM) bölge adını farklı
    yazsa da eşleşme bulunabilsin diye."""
    return str(text).translate(_TR_ASCII).lower().strip()


def clear_cache():
    """Önbelleğe alınmış veri ve modelleri temizler (yeni veri commit'lendikten
    sonra uzun süre çalışan bir süreçte çağrılır)."""
    for loader in (_load_daily_summary, _load_lag_frame, _load_anomaly_frame,
                   load_grid_catalog, _load_models):
        loader.cache_clear()


# ---------------------------------------------------------------------------
# Veri ve model yükleme (süreç başına bir kez)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_daily_summary():
    df = pd.read_csv(DAILY_SUMMARY_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df


@lru_cache(maxsize=1)
def _load_lag_frame():
    """Zaman serisi modeli için lag feature'ları (forecaster.py ile aynı mantık)."""
    return add_lag_features(_load_daily_summary())


@lru_cache(maxsize=1)
def _load_anomaly_frame():
    """z-score ve anomali bayrağı (anomaly.py ile aynı mantık)."""
    df = _load_daily_summary().sort_values(['grid_id', 'date']).reset_index(drop=True)
    return flag_anomalies(calculate_zscore_features(df))


@lru_cache(maxsize=1)
def _load_models():
    return {
        'occurrence': joblib.load(os.path.join(MODEL_DIR, "occurrence.joblib")),  # XGBoost
        'classifier': joblib.load(os.path.join(MODEL_DIR, "classifier.joblib")),  # XGBoost, ölçeklendirme gerektirmez
        'forecaster': joblib.load(os.path.join(MODEL_DIR, "forecaster.joblib")),  # Random Forest Regressor
    }


@lru_cache(maxsize=1)
def load_grid_catalog():
    """İzlenen tüm grid hücreleri, il/ilçe isimleriyle birlikte.

    Dashboard'daki gibi hem ili hem ilçesi bilinmeyen (deniz/sınır üzerindeki)
    hücreler çıkarılır. Dashboard'un harita için yaptığı "her isimden tek
    nokta" tekilleştirmesi burada YAPILMAZ -- o bir görünüm tercihi; agent ve
    otomasyonların tüm hücreleri görebilmesi gerekiyor."""
    grids = _load_daily_summary()[['grid_id', 'grid_lat', 'grid_lon']].drop_duplicates('grid_id')
    names = pd.read_csv(LOCATION_NAMES_PATH)
    grids = grids.merge(names, on='grid_id', how='left')
    grids[['province', 'district']] = grids[['province', 'district']].fillna(UNKNOWN)
    grids = grids[~((grids['province'] == UNKNOWN) & (grids['district'] == UNKNOWN))].copy()
    grids['display_name'] = np.where(
        grids['district'] == UNKNOWN,
        grids['province'],
        grids['province'] + " - " + grids['district'],
    )
    # Sıra, daily_grid_summary'deki ilk görülme sırası (dashboard'un harita
    # tekilleştirmesi eskisi gibi aynı hücreyi seçsin diye sıralama yapılmıyor).
    return grids.reset_index(drop=True)


def _grid_row(grid_id):
    catalog = load_grid_catalog()
    match = catalog[catalog['grid_id'] == grid_id]
    if match.empty:
        raise KeyError(f"Bilinmeyen grid_id: {grid_id!r} (find_regions ile arayın)")
    return match.iloc[0]


def _region_info(row):
    return {
        'grid_id': row['grid_id'],
        'display_name': row['display_name'],
        'province': row['province'],
        'district': row['district'],
        'grid_lat': float(row['grid_lat']),
        'grid_lon': float(row['grid_lon']),
    }


def _float_or_none(value):
    return None if value is None or pd.isna(value) else float(value)


# ---------------------------------------------------------------------------
# Herkese açık fonksiyonlar
# ---------------------------------------------------------------------------

def find_regions(query, limit=10):
    """İl veya ilçe adına göre grid hücresi arar (Türkçe karakter ve büyük/küçük
    harf duyarsız). Tam eşleşmeler (il ya da ilçe adının kendisi) önce gelir,
    sonra adın içinde geçenler.

    Örn. find_regions("mugla") -> Muğla'nın tüm ilçelerindeki hücreler."""
    q = normalize_text(query)
    if not q:
        return []
    catalog = load_grid_catalog()
    province = catalog['province'].map(normalize_text)
    district = catalog['district'].map(normalize_text)
    display = catalog['display_name'].map(normalize_text)

    exact = (province == q) | (district == q) | (display == q)
    partial = display.str.contains(q, regex=False) & ~exact
    ordered = pd.concat([catalog[exact], catalog[partial]])
    return [_region_info(row) for _, row in ordered.head(limit).iterrows()]


def get_region_report(grid_id, fetch_weather=True):
    """Bir grid hücresi için dört modelin çıktısını tek bir sözlükte döndürür.

    Dashboard'daki hesaplamanın birebir aynısıdır:
      - fire_probability: veri setindeki son günden bir sonraki gün (target_date)
        en az bir sıcak nokta tespit edilme olasılığı (0-1, XGBoost)
      - severity: bu hücredeki EN SON tespitin şiddet sınıfı (düşük/orta/yüksek)
      - next_day_hotspot_forecast: en son tespit gününe göre bir sonraki gün
        beklenen sıcak nokta sayısı (Random Forest)
      - anomaly: en son tespit gününün, hücrenin kendi geçmişine göre z-score'u

    Son üçü bu hücrede daha önce tespit olmuşsa doludur, yoksa None. Not:
    son üçü "bugün" değil, "en son tespit günü" hakkındadır -- tespit ne kadar
    eskiyse o kadar az güncel bilgi taşır; bunu yorumlayabilmek için
    last_detection.days_before_data_end alanı da döndürülür.

    fetch_weather=False: yarının hava durumu tahmini canlı API'den çekilmez
    (testler/çevrimdışı kullanım). Hava durumu alınamazsa model yine de
    çalışır; weather_used alanı bunu belirtir.
    """
    row = _grid_row(grid_id)
    models = _load_models()
    daily_summary = _load_daily_summary()
    data_end = daily_summary['date'].max()

    # 1) Yangın olasılığı -- her hücre için hesaplanabilir
    occ_features, target_date = build_latest_occurrence_features(
        daily_summary,
        grid_id=grid_id,
        grid_lat=row['grid_lat'],
        grid_lon=row['grid_lon'],
        fetch_weather=fetch_weather,
    )
    # Hava durumu alınamadığında sütunlar None (object tipi) oluyor; XGBoost
    # yalnızca sayısal tip kabul ettiği için float'a çeviriyoruz (None -> NaN).
    occ_X = occ_features[OCCURRENCE_FEATURES].astype(float)
    probability = float(models['occurrence'].predict_proba(occ_X)[0, 1])
    weather_used = not occ_X[WEATHER_FEATURE_COLUMNS].isna().all(axis=1).iloc[0]

    report = {
        **_region_info(row),
        'data_end_date': str(data_end.date()),
        'target_date': str(target_date.date()),
        'fire_probability': probability,
        'weather_used': bool(weather_used),
        'weather': {col: _float_or_none(occ_X[col].iloc[0]) for col in WEATHER_FEATURE_COLUMNS},
        'last_detection': None,
        'severity': None,
        'next_day_hotspot_forecast': None,
        'anomaly': None,
    }

    history = daily_summary[daily_summary['grid_id'] == grid_id]
    if history.empty:
        return report
    latest = history.sort_values('date').iloc[-1]
    report['last_detection'] = {
        'date': str(latest['date'].date()),
        'days_before_data_end': int((data_end - latest['date']).days),
        'fire_count': int(latest['fire_count']),
    }

    # 2) Şiddet sınıfı -- en son tespitin parlaklık/FRP değerleri
    severity_X = latest[CLASSIFIER_FEATURES].to_frame().T.astype(float)
    report['severity'] = REVERSE_LABEL_MAP[int(models['classifier'].predict(severity_X)[0])]

    # 3) Zaman serisi -- en az bir önceki tespit günü gerekiyor (lag_1)
    lag_rows = _load_lag_frame()
    lag_latest = lag_rows[lag_rows['grid_id'] == grid_id].sort_values('date').iloc[-1]
    if not pd.isna(lag_latest['lag_1_fire_count']):
        ts_X = lag_latest[FORECASTER_FEATURES].to_frame().T.astype(float)
        report['next_day_hotspot_forecast'] = float(predict_clipped(models['forecaster'], ts_X)[0])
        report['previous_fire_count'] = float(lag_latest['lag_1_fire_count'])

    # 4) Anomali -- baseline için yeterli geçmiş gerekiyor
    anomaly_rows = _load_anomaly_frame()
    anomaly_latest = anomaly_rows[anomaly_rows['grid_id'] == grid_id].sort_values('date').iloc[-1]
    if not pd.isna(anomaly_latest['z_score']):
        report['anomaly'] = {
            'z_score': float(anomaly_latest['z_score']),
            'is_anomaly': bool(anomaly_latest['is_anomaly']),
        }

    return report


def build_all_occurrence_features(as_of_date=None, window_days=30, fetch_weather=True):
    """build_latest_occurrence_features'ın tüm hücreler için vektörel versiyonu.

    ~1300 hücre için tek tek döngü kurmak yerine aynı feature'ları tek
    seferde hesaplar; hava durumu da toplu (batch) isteklerle çekilir. Aynı
    sonucu verdiği tests/test_inference.py'de doğrulanıyor."""
    daily_summary = _load_daily_summary()
    catalog = load_grid_catalog()

    as_of = pd.to_datetime(as_of_date) if as_of_date is not None else daily_summary['date'].max()
    target_date = as_of + pd.Timedelta(days=1)

    def occurrence_rate(days):
        start = as_of - pd.Timedelta(days=days - 1)
        in_window = daily_summary[
            (daily_summary['date'] >= start) & (daily_summary['date'] <= as_of)
            & (daily_summary['fire_count'] > 0)
        ]
        return in_window.groupby('grid_id')['date'].nunique() / days

    features = catalog[['grid_id', 'grid_lat', 'grid_lon']].copy()
    features['month'] = target_date.month
    features['day_of_year'] = target_date.dayofyear
    features['is_summer'] = int(target_date.month in [6, 7, 8])
    features['lag_1_occurred'] = features['grid_id'].map(occurrence_rate(1)).fillna(0.0)
    features['rolling_7_occurrence_rate'] = features['grid_id'].map(occurrence_rate(7)).fillna(0.0)
    features['rolling_30_occurrence_rate'] = features['grid_id'].map(occurrence_rate(window_days)).fillna(0.0)

    weather = (
        fetch_forecast_weather_batch(list(zip(features['grid_lat'], features['grid_lon'])), target_date)
        if fetch_weather else [None] * len(features)
    )
    for col in WEATHER_FEATURE_COLUMNS:
        features[col] = [w[col] if w else np.nan for w in weather]

    return features.reset_index(drop=True), target_date


def get_top_risk_regions(n=10, fetch_weather=True, province=None):
    """Bir sonraki gün için yangın olasılığı en yüksek n bölgeyi döndürür.

    province verilirse yalnızca o ilin hücreleri arasında sıralar.
    Zamanlanmış uyarılar (n8n) ve "en riskli yerler neresi?" gibi agent
    soruları için tasarlandı. Sadece olasılık modelini çalıştırır; bir
    bölgenin ayrıntısı için get_region_report kullanılmalı."""
    features, target_date = build_all_occurrence_features(fetch_weather=fetch_weather)
    X = features[OCCURRENCE_FEATURES].astype(float)
    features['fire_probability'] = _load_models()['occurrence'].predict_proba(X)[:, 1]
    features['weather_used'] = ~X[WEATHER_FEATURE_COLUMNS].isna().all(axis=1)

    ranked = features.merge(load_grid_catalog()[['grid_id', 'province', 'district', 'display_name']], on='grid_id')
    if province is not None:
        ranked = ranked[ranked['province'].map(normalize_text) == normalize_text(province)]
    ranked = ranked.sort_values('fire_probability', ascending=False).head(n)

    return [
        {
            **_region_info(row),
            'target_date': str(target_date.date()),
            'fire_probability': float(row['fire_probability']),
            'weather_used': bool(row['weather_used']),
        }
        for _, row in ranked.iterrows()
    ]
