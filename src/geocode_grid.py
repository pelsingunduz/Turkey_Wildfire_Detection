"""
Grid hücreleri için il/ilçe isimlerini Nominatim (OpenStreetMap) ile bulur.

Sadece grid_location_names.csv'de EKSİK olan hücreleri sorgular -- zaten
bilinen hücreleri tekrar sormaz (hem hız hem Nominatim'in kullanım
politikasına saygı için). İleride yeni grid hücreleri eklendiğinde
(veri büyüdükçe) bu script tekrar çalıştırılabilir, sadece yeni
eklenenleri sorgular.

RATE LIMIT: Nominatim'in kullanım politikası saniyede en fazla 1 istek
öneriyor -- bu yüzden her sorgu arasında 1.1 saniye bekleniyor. Çok
sayıda eksik hücre varsa (örn. 1000+) bu dakikalar sürebilir.

CHECKPOINT: Her 50 hücrede bir sonuç diske kaydedilir -- script yarıda
kesilirse (ağ sorunu, Ctrl+C) o ana kadarki ilerleme kaybolmaz, script
tekrar çalıştırıldığında kaldığı yerden devam eder.
"""

import pandas as pd
import time
from geopy.geocoders import Nominatim

LOCATION_NAMES_PATH = "data/processed/grid_location_names.csv"
DAILY_SUMMARY_PATH = "data/processed/daily_grid_summary.csv"
CHECKPOINT_EVERY = 50
RATE_LIMIT_SECONDS = 1.1

geolocator = Nominatim(user_agent="turkey_wildfire_project")


def get_province_district(lat, lon):
    """Koordinattan (il, ilçe) çifti bulur. Bulunamazsa 'Bilinmiyor' döner."""
    try:
        location = geolocator.reverse((lat, lon), language='tr', timeout=10)
        if location is None:
            return 'Bilinmiyor', 'Bilinmiyor'
        address = location.raw.get('address', {})
        province = address.get('province') or address.get('state') or 'Bilinmiyor'
        district = (
            address.get('county') or address.get('district')
            or address.get('town') or address.get('municipality')
            or 'Bilinmiyor'
        )
        return province, district
    except Exception as e:
        print(f"  Hata ({lat}, {lon}): {e}")
        return 'Bilinmiyor', 'Bilinmiyor'


def load_existing():
    try:
        return pd.read_csv(LOCATION_NAMES_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=['grid_id', 'province', 'district'])


def find_missing_grid_cells(existing_df):
    """daily_grid_summary.csv'deki tüm hücrelerden, henüz geocode edilmemiş olanları bulur."""
    all_cells = pd.read_csv(DAILY_SUMMARY_PATH)[['grid_id', 'grid_lat', 'grid_lon']].drop_duplicates()
    known_ids = set(existing_df['grid_id'])
    missing = all_cells[~all_cells['grid_id'].isin(known_ids)].reset_index(drop=True)
    return missing


def run():
    existing = load_existing()
    missing = find_missing_grid_cells(existing)

    print(f"[run] Zaten bilinen: {len(existing)} hücre")
    print(f"[run] Eksik (sorgulanacak): {len(missing)} hücre")

    if len(missing) == 0:
        print("[run] Eksik hücre yok, yapılacak bir şey kalmadı.")
        return existing

    new_rows = []
    for i, row in missing.iterrows():
        center_lat = row['grid_lat'] + 0.125
        center_lon = row['grid_lon'] + 0.125
        province, district = get_province_district(center_lat, center_lon)
        new_rows.append({'grid_id': row['grid_id'], 'province': province, 'district': district})

        print(f"[{i + 1}/{len(missing)}] {row['grid_id']} -> {province} - {district}")
        time.sleep(RATE_LIMIT_SECONDS)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            checkpoint_df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            checkpoint_df.to_csv(LOCATION_NAMES_PATH, index=False)
            print(f"  [checkpoint] {len(checkpoint_df)} hücre kaydedildi")

    final_df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    final_df.to_csv(LOCATION_NAMES_PATH, index=False)
    print(f"[run] Tamamlandı. Toplam {len(final_df)} hücre kaydedildi -> {LOCATION_NAMES_PATH}")
    return final_df


if __name__ == "__main__":
    run()