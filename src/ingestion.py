import os
import glob
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

MAP_KEY = os.getenv("FIRMS_MAP_KEY")
# Kasım 2026'ya kadar VIIRS_SNPP_NRT kullanılıyordu; Suomi NPP'nin NRT akışı
# 2026-11-02'de kesildi (FIRMS data_availability listesinde bitiş tarihi
# 2026-11-02) ve API o günden beri hata vermeden BOŞ CSV döndürüyordu.
# NOAA-20 aynı VIIRS cihazını taşıyor (aynı ~375m çözünürlük, aynı sütunlar,
# aynı confidence etiketleri), bu yüzden doğrudan yerine geçebiliyor.
SENSOR = "VIIRS_NOAA20_NRT"
BBOX = "25.5,35.5,44.5,42.5"  # Türkiye: west,south,east,north
DAY_RANGE = 1

EXPECTED_COLUMNS = {"latitude", "longitude", "acq_date", "acq_time", "confidence"}

# Art arda kaç çalıştırma boş dönerse pipeline hata versin. 6 çalıştırma x 4
# saat = 24 saat. Bounding box Irak/Suriye'deki sürekli yanan gaz yakma
# (flare) noktalarını da kapsadığı için, 24 saat boyunca HİÇ tespit gelmemesi
# gerçekte "yangın yok" değil, veri kaynağının bozulduğu anlamına gelir.
EMPTY_RUN_LIMIT = 6
LIVE_FILE_PATTERN = "data/raw/firms_[0-9]*.csv"  # firms_hist_* (backfill) hariç


def fetch_firms_data():
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SENSOR}/{BBOX}/{DAY_RANGE}"
    # timeout: FIRMS API bazen yanıt vermeyebilir -- timeout olmadan GitHub
    # Actions job'ı süresiz asılı kalabilirdi (bkz. historical_backfill.py'deki
    # aynı pattern). try/except, geçici ağ hatalarında (DNS, bağlantı kopması)
    # programın çökmek yerine anlamlı bir mesajla None dönmesini sağlar.
    try:
        response = requests.get(url, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"Ağ hatası: {e}")
        return None

    if response.status_code != 200:
        print(f"Hata: {response.status_code}")
        print(response.text)
        return None

    print("İstek başarılı!")
    return response.text


def is_valid_csv(text):
    """FIRMS bazı hatalarda (ör. geçersiz MAP_KEY) HTTP 200 dönüp gövdeye düz
    metin hata mesajı yazıyor; gerçek bir tespit CSV'si mi diye başlığa bakar."""
    if not text or not text.strip():
        return False
    header = {col.strip().lower() for col in text.splitlines()[0].split(",")}
    return EXPECTED_COLUMNS.issubset(header)


def parse_and_filter(csv_text):
    from io import StringIO
    df = pd.read_csv(StringIO(csv_text))

    print(f"Ham veri: {len(df)} satır")

    # Düşük güvenli tespitleri at
    df = df[df["confidence"] != "l"]

    print(f"Filtreleme sonrası: {len(df)} satır")
    return df


def save_raw(df):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    filepath = f"data/raw/firms_{now}.csv"
    df.to_csv(filepath, index=False)
    print(f"Kaydedildi: {filepath}")


def count_trailing_empty_runs(pattern=LIVE_FILE_PATTERN):
    """En yeni canlı ham dosyadan geriye doğru, art arda kaç dosyanın hiç veri
    satırı içermediğini sayar (dosya adları zaman damgalı, isim sırası = zaman sırası)."""
    count = 0
    for path in sorted(glob.glob(pattern), reverse=True):
        with open(path) as f:
            has_rows = sum(1 for _ in f) > 1  # başlık satırı hariç
        if has_rows:
            break
        count += 1
    return count


if __name__ == "__main__":
    if not MAP_KEY:
        raise SystemExit("FIRMS_MAP_KEY bulunamadı (.env ya da GitHub Secrets).")

    raw_text = fetch_firms_data()
    if raw_text:
        if not is_valid_csv(raw_text):
            raise SystemExit(f"FIRMS beklenmeyen bir cevap döndürdü: {raw_text[:200]!r}")
        df = parse_and_filter(raw_text)
        save_raw(df)

        empty_runs = count_trailing_empty_runs()
        if empty_runs >= EMPTY_RUN_LIMIT:
            # Exit kodu 1: GitHub Actions job'ı başarısız olur ve repo sahibine
            # e-posta gider. Sessizce boş veri biriktirmek yerine fark edilir.
            raise SystemExit(
                f"UYARI: Son {empty_runs} çalıştırmanın hepsi boş döndü (~{empty_runs * 4} saat). "
                f"{SENSOR} kaynağı kesilmiş olabilir; FIRMS data_availability listesini kontrol edin: "
                "https://firms.modaps.eosdis.nasa.gov/api/data_availability/"
            )
