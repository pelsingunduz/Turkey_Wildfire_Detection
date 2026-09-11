import os
import time
from io import StringIO

import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

MAP_KEY = os.getenv("FIRMS_MAP_KEY")
BBOX = "25.5,35.5,44.5,42.5"  # Türkiye: west,south,east,north
DAY_RANGE = 5  # Not: genel dokümantasyon 1-10 diyor ama bu MAP_KEY için API "Expects [1..5]" hatası veriyor

# Geçmiş veri için önce Standard Processing (bilimsel kalite, gecikmeli) denenir;
# henüz işlenmemiş yakın tarihlerde boş/geçersiz dönerse Near Real-Time'a düşülür.
SOURCES_TO_TRY = ["VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"]

EXPECTED_COLUMNS = {"latitude", "longitude", "acq_date"}


def fetch_chunk(source, date_str, day_range=DAY_RANGE):
    """Belirli bir kaynak, başlangıç tarihi ve gün aralığı için FIRMS'ten veri çeker.
    (metin, hata_mesajı) döner — başarılıysa hata_mesajı None olur."""
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{source}/{BBOX}/{day_range}/{date_str}"
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        return None, f"HTTP {response.status_code}: {response.text[:200]}"
    return response.text, None


def is_valid_csv(text):
    """Dönen metnin gerçek bir fire-detection CSV'si olup olmadığını kontrol eder
    (FIRMS geçersiz isteklerde 200 dönüp gövdede hata mesajı verebiliyor)."""
    if not text or not text.strip():
        return False
    first_line = text.splitlines()[0].lower()
    header_cols = set(col.strip() for col in first_line.split(","))
    return EXPECTED_COLUMNS.issubset(header_cols)


def fetch_historical_chunk(date_str, day_range=DAY_RANGE):
    """SP'yi dener; format geçersizse ya da 0 veri dönerse (henüz işlenmemiş olabilir)
    NRT'ye düşer. Hiçbirinde veri yoksa (gerçekten yangın yoktur diye) son geçerli
    boş sonucu kabul eder. (veri, kullanılan_kaynak, son_hata) döner."""
    last_error = None
    fallback_empty_result = None
    fallback_empty_source = None

    for source in SOURCES_TO_TRY:
        text, error = fetch_chunk(source, date_str, day_range)
        if error:
            last_error = f"{source}: {error}"
            continue
        if not is_valid_csv(text):
            last_error = f"{source}: beklenmeyen gövde formatı -> {text[:150]!r}"
            continue

        row_count = max(len(text.strip().splitlines()) - 1, 0)  # başlık satırı hariç
        if row_count > 0:
            return text, source, None

        # Format geçerli ama veri satırı yok — bir sonraki kaynağı da dene,
        # gerçekten veri yoksa diye bunu yedekte tutalım.
        fallback_empty_result, fallback_empty_source = text, source

    if fallback_empty_result is not None:
        return fallback_empty_result, fallback_empty_source, None

    return None, None, last_error


def parse_and_filter(csv_text):
    df = pd.read_csv(StringIO(csv_text))
    print(f"  Ham veri: {len(df)} satır")
    df = df[df["confidence"] != "l"]
    print(f"  Filtreleme sonrası: {len(df)} satır")
    return df


def save_raw(df, date_str, day_range):
    end_date = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=day_range - 1)).strftime("%Y-%m-%d")
    filepath = f"data/raw/firms_hist_{date_str}_{end_date}.csv"
    df.to_csv(filepath, index=False)
    print(f"  Kaydedildi: {filepath}")


def backfill(start_date, end_date, day_range=DAY_RANGE, sleep_seconds=1.5):
    current = start_date
    total_chunks = (end_date - start_date).days // day_range + 1
    chunk_no = 0

    while current <= end_date:
        chunk_no += 1
        date_str = current.strftime("%Y-%m-%d")
        print(f"[{chunk_no}/{total_chunks}] {date_str} ({day_range} günlük aralık) isteniyor...")

        text, source, error = fetch_historical_chunk(date_str, day_range)
        if text is None:
            print(f"  UYARI: {date_str} için veri alınamadı, atlanıyor. Sebep: {error}")
        else:
            df = parse_and_filter(text)
            if len(df) > 0:
                save_raw(df, date_str, day_range)
            else:
                print("  Bu aralıkta tespit yok, dosya oluşturulmadı.")
            print(f"  -> kaynak: {source}")

        current += timedelta(days=day_range)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    import sys

    if not MAP_KEY:
        raise SystemExit("FIRMS_MAP_KEY .env dosyasında bulunamadı.")

    if len(sys.argv) == 3:
        # Elle aralık: python historical_backfill.py 2026-04-28 2026-09-10
        start = datetime.strptime(sys.argv[1], "%Y-%m-%d")
        end = datetime.strptime(sys.argv[2], "%Y-%m-%d")
    else:
        end = datetime.now(timezone.utc) - timedelta(days=1)
        start = end - timedelta(days=365 * 3)

    print(f"Backfill aralığı: {start.date()} -> {end.date()}")
    backfill(start, end)
    print("Backfill tamamlandı.")