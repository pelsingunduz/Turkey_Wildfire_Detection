import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

MAP_KEY = os.getenv("FIRMS_MAP_KEY")
SENSOR = "VIIRS_SNPP_NRT"
BBOX = "25.5,35.5,44.5,42.5"  # Türkiye: west,south,east,north
DAY_RANGE = 1

def fetch_firms_data():
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SENSOR}/{BBOX}/{DAY_RANGE}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Hata: {response.status_code}")
        print(response.text)
        return None

    print("İstek başarılı!")
    return response.text

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

if __name__ == "__main__":
    raw_text = fetch_firms_data()
    if raw_text:
        df = parse_and_filter(raw_text)
        save_raw(df)