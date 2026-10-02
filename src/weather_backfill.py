"""
İl bazlı hava durumu verisini Open-Meteo (archive-api) üzerinden çeker.

ARTIMLI (incremental) ÇALIŞIR: Her il için data/external/weather_by_province.csv'deki
EN SON kayıtlı tarihi bulur, sadece ondan sonraki günleri ister. Bu script hem
ilk büyük backfill (3 yıl) için hem de GitHub Actions'ta periyodik güncelleme
için (sadece dünün verisini ekler) AYNI KODLA çalışır -- ayrı bir "backfill"
ve "update" script'i tutmaya gerek yok.

NEDEN İL BAZINDA (grid hücresi değil): bkz. weather_backfill.py'nin önceki
sürümündeki açıklama -- hava durumu 0.25°'lik bir hücre çözünürlüğünde
anlamlı şekilde değişmiyor, il merkezi (centroid) yeterli bir yaklaşıklık.

SEÇİLEN DEĞİŞKENLER: Sıcaklık (max), yağış toplamı, rüzgar hızı (max) ve
rüzgar yönü -- klasik "yangın hava durumu" dörtlüsü.

ERA5 GECİKMESİ: Open-Meteo'nun archive API'si ~5 gün gecikmeli güncelleniyor
(resmi dokümantasyon). Yani "dün"ü istesek bile en güncel birkaç gün için
veri dönmeyebilir -- bu durumda features.py:add_weather_features'daki
ortalama ile doldurma mekanizması devreye girer, akış kesilmez.

RATE LIMIT VE DAYANIKLILIK: 429 alırsa 65 saniye bekleyip tekrar dener (en
fazla 3 kez); ağ hatalarında kısa bekleyişle tekrar dener; her il
tamamlandığında sonucu diske yazar (checkpoint).
"""
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = "temperature_2m_max,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant"
SLEEP_SECONDS = 2.0
RATE_LIMIT_WAIT = 65
MAX_RETRIES = 3
DEFAULT_START_DATE = "2023-09-11"  # FIRMS backfill'inin başladığı tarihle aynı


def compute_province_centroids(daily_summary_path="data/processed/daily_grid_summary.csv",
                                 grid_location_path="data/processed/grid_location_names.csv"):
    """Her il için, o ile ait grid hücrelerinin ortalama koordinatını hesaplar."""
    cells = pd.read_csv(daily_summary_path)[['grid_id', 'grid_lat', 'grid_lon']].drop_duplicates()
    locations = pd.read_csv(grid_location_path)[['grid_id', 'province']]
    merged = cells.merge(locations, on='grid_id', how='inner')

    centroids = merged.groupby('province').agg(
        lat=('grid_lat', lambda x: (x + 0.125).mean()),
        lon=('grid_lon', lambda x: (x + 0.125).mean()),
    ).reset_index()

    print(f"[compute_province_centroids] {len(centroids)} il için centroid hesaplandı")
    return centroids


def fetch_province_weather(province, lat, lon, start_date, end_date):
    """Tek bir il için, verilen tarih aralığında günlük hava durumu verisi çeker.
    429 (rate limit) ve bağlantı hatalarında otomatik olarak tekrar dener."""
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": start_date,
        "end_date": end_date,
        "daily": DAILY_VARS,
        "timezone": "Europe/Istanbul",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=45)
        except requests.exceptions.RequestException as e:
            print(f"  Bağlantı hatası ({attempt}/{MAX_RETRIES}): {e}")
            time.sleep(5 * attempt)
            continue

        if response.status_code == 429:
            print(f"  Rate limit ({attempt}/{MAX_RETRIES}), {RATE_LIMIT_WAIT} saniye bekleniyor...")
            time.sleep(RATE_LIMIT_WAIT)
            continue

        if response.status_code != 200:
            print(f"  UYARI: {province} için istek başarısız ({response.status_code}): {response.text[:200]}")
            return None

        data = response.json().get("daily", {})
        if not data or "time" not in data or len(data["time"]) == 0:
            return None

        return pd.DataFrame({
            "province": province,
            "date": data["time"],
            "temp_max": data.get("temperature_2m_max"),
            "precip_sum": data.get("precipitation_sum"),
            "wind_speed_max": data.get("wind_speed_10m_max"),
            "wind_direction_dominant": data.get("wind_direction_10m_dominant"),
        })

    print(f"  BAŞARISIZ: {province} için {MAX_RETRIES} denemeden sonra vazgeçildi.")
    return None


def run(end_date=None, output_path="data/external/weather_by_province.csv"):
    centroids = compute_province_centroids()

    if end_date is None:
        end_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        existing = pd.read_csv(output_path)
        last_dates = existing.groupby('province')['date'].max().to_dict()
        print(f"[run] Mevcut dosyada {existing['province'].nunique()} il var")
    except FileNotFoundError:
        existing = pd.DataFrame()
        last_dates = {}
        print("[run] Mevcut dosya yok, sıfırdan başlanıyor")

    all_new = []
    for i, row in centroids.iterrows():
        province = row['province']
        last_date = last_dates.get(province)

        if last_date is not None:
            next_start = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            next_start = DEFAULT_START_DATE

        if next_start > end_date:
            print(f"[{i + 1}/{len(centroids)}] {province}: zaten güncel, atlanıyor")
            continue

        print(f"[{i + 1}/{len(centroids)}] {province} ({row['lat']:.2f}, {row['lon']:.2f}): "
              f"{next_start} -> {end_date} isteniyor...")
        df = fetch_province_weather(province, row['lat'], row['lon'], next_start, end_date)
        if df is not None and len(df) > 0:
            all_new.append(df)
            # Checkpoint: her il tamamlandığında diske yaz
            combined = pd.concat([existing] + all_new, ignore_index=True) if len(existing) > 0 else pd.concat(all_new, ignore_index=True)
            combined = combined.drop_duplicates(subset=['province', 'date'], keep='last')
            combined.to_csv(output_path, index=False)
        time.sleep(SLEEP_SECONDS)

    final = pd.concat([existing] + all_new, ignore_index=True) if all_new else existing
    final = final.drop_duplicates(subset=['province', 'date'], keep='last')
    final.to_csv(output_path, index=False)
    print(f"[run] Tamamlandı: {output_path} ({len(final)} satır, {final['province'].nunique()} il)")
    return final


if __name__ == "__main__":
    run()