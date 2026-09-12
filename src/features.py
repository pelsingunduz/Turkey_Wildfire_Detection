import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import glob


def load_raw_files(pattern="data/raw/firms_*.csv"):
    raw_files = glob.glob(pattern)
    dfs = []
    for file in raw_files:
        df = pd.read_csv(file)
        if len(df) > 0:
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"[load_raw_files] {len(raw_files)} dosya, {len(combined)} satır")
    return combined


def remove_duplicates(df):
    before = len(df)
    df_clean = df.drop_duplicates(subset=['latitude', 'longitude', 'acq_date', 'acq_time'], keep='first')
    print(f"[remove_duplicates] {before} -> {len(df_clean)} satır")
    return df_clean


def filter_turkey_boundary(df):
    turkey_boundary = gpd.read_file("https://raw.githubusercontent.com/johan/world.geo.json/master/countries/TUR.geo.json")
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    filtered = gpd.sjoin(points_gdf, turkey_boundary, how="inner", predicate="within")
    result = filtered[df.columns].reset_index(drop=True)
    print(f"[filter_turkey_boundary] {len(df)} -> {len(result)} satır")
    return result


def fix_datetime(df):
    df = df.copy()
    time_str = df['acq_time'].astype(str).str.zfill(4)
    hour = time_str.str[:2]
    minute = time_str.str[2:]
    dt_utc = pd.to_datetime(df['acq_date'] + ' ' + hour + ':' + minute, format='%Y-%m-%d %H:%M')
    df['acq_datetime_tr'] = dt_utc.dt.tz_localize('UTC').dt.tz_convert('Europe/Istanbul')
    print(f"[fix_datetime] tamamlandı")
    return df


def assign_grid(df, grid_size=0.25):
    df = df.copy()
    df['grid_lat'] = (df['latitude'] / grid_size).apply(lambda x: int(x)) * grid_size
    df['grid_lon'] = (df['longitude'] / grid_size).apply(lambda x: int(x)) * grid_size
    df['grid_id'] = df['grid_lat'].astype(str) + '_' + df['grid_lon'].astype(str)
    print(f"[assign_grid] tamamlandı, {df['grid_id'].nunique()} benzersiz hücre")
    return df


def build_daily_summary(df):
    df = df.copy()
    df['date'] = df['acq_datetime_tr'].dt.date
    summary = df.groupby(['grid_id', 'grid_lat', 'grid_lon', 'date']).agg(
        fire_count=('latitude', 'count'),
        avg_brightness=('bright_ti4', 'mean'),
        avg_frp=('frp', 'mean'),
        max_frp=('frp', 'max')
    ).reset_index()
    print(f"[build_daily_summary] {len(summary)} satır (bölge x gün kombinasyonu)")
    return summary


# ---------------------------------------------------------------------------
# Yangın OLASILIĞI modeli için: tam panel (grid x takvim günü)
#
# build_daily_summary() sadece en az 1 tespit olan grid-gün kombinasyonlarını
# üretir; yangınsız günler tabloda hiç yer almaz. "Yarın bu hücrede yangın
# çıkar mı?" sorusuna cevap verebilmek için, yangın OLMAYAN günlerin de
# (fire_count=0 ile) veri setinde bulunması gerekir.
#
# ÖNEMLİ: avg_brightness/avg_frp/max_frp gibi "ancak tespit olduğunda ölçülen"
# feature'lar bu panelde YOK -- bunları olasılık modelinde kullanmak target
# leakage olurdu (bir ölçüm varsa zaten yangın var demektir). Bunun yerine
# sadece geçmişe dayalı (lag/rolling) ve takvimsel feature'lar kullanılır.
# ---------------------------------------------------------------------------

def build_full_panel(with_grid_df, daily_summary_df):
    """Her grid hücresi x her takvim günü kombinasyonunu içeren tam panel oluşturur.
    Yangın olmayan gün/hücre çiftlerinde fire_count=0 olur.

    Not: Sadece 3 yıllık veri setinde EN AZ BİR KEZ tespit edilmiş grid hücreleri
    kullanılır (Türkiye'nin tamamı değil) -- bu, projenin basit tutulma
    prensibiyle uyumlu bir kapsam sınırlamasıdır."""
    grid_cells = with_grid_df[['grid_id', 'grid_lat', 'grid_lon']].drop_duplicates()
    all_dates = pd.date_range(
        start=with_grid_df['acq_datetime_tr'].dt.date.min(),
        end=with_grid_df['acq_datetime_tr'].dt.date.max(),
        freq='D'
    ).date

    # Cross join: her hücre x her gün
    grid_cells['_key'] = 1
    dates_df = pd.DataFrame({'date': all_dates, '_key': 1})
    panel = grid_cells.merge(dates_df, on='_key').drop(columns='_key')

    panel = panel.merge(
        daily_summary_df[['grid_id', 'date', 'fire_count']],
        on=['grid_id', 'date'],
        how='left'
    )
    panel['fire_count'] = panel['fire_count'].fillna(0).astype(int)
    panel['fire_occurred'] = (panel['fire_count'] > 0).astype(int)

    print(f"[build_full_panel] {len(grid_cells)} hücre x {len(all_dates)} gün = {len(panel)} satır")
    print(f"[build_full_panel] Yangın oranı: {panel['fire_occurred'].mean():.2%}")
    return panel


def add_calendar_features(panel):
    """Ay ve yılın günü gibi mevsimsel feature'lar ekler.
    Türkiye'de yangın mevsimi yaz aylarında yoğunlaştığı için bunlar güçlü
    bir sinyal olması beklenir."""
    panel = panel.copy()
    panel['date'] = pd.to_datetime(panel['date'])
    panel['month'] = panel['date'].dt.month
    panel['day_of_year'] = panel['date'].dt.dayofyear
    panel['is_summer'] = panel['month'].isin([6, 7, 8]).astype(int)
    print(f"[add_calendar_features] tamamlandı")
    return panel


def add_occurrence_lag_features(panel):
    """Geçmişe dayalı (leakage'sız) feature'lar ekler.
    shift(1) ile bugün HARİÇ tutulur -- rolling pencere de shift edilmiş
    seri üzerinden hesaplanır, yani bugünün kendisi asla feature'a sızmaz."""
    panel = panel.copy()
    panel = panel.sort_values(['grid_id', 'date']).reset_index(drop=True)

    shifted = panel.groupby('grid_id')['fire_occurred'].shift(1)
    panel['lag_1_occurred'] = shifted

    panel['rolling_7_occurrence_rate'] = (
        panel.groupby('grid_id')['fire_occurred']
        .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    )
    panel['rolling_30_occurrence_rate'] = (
        panel.groupby('grid_id')['fire_occurred']
        .transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean())
    )

    print(f"[add_occurrence_lag_features] tamamlandı")
    return panel


def run_occurrence_panel_pipeline():
    """Yangın olasılığı modeli için ayrı, tam panel pipeline'ı.
    Mevcut run_pipeline()'dan (şiddet/anomali modelleri için) bağımsızdır --
    daily_grid_summary.csv'ye dokunmaz."""
    raw = load_raw_files()
    clean = remove_duplicates(raw)
    turkey_only = filter_turkey_boundary(clean)
    with_time = fix_datetime(turkey_only)
    with_grid = assign_grid(with_time)
    daily_summary = build_daily_summary(with_grid)

    # daily_summary'nin 'date' sütunu datetime.date, panel de aynı tipte olmalı
    panel = build_full_panel(with_grid, daily_summary)
    panel = add_calendar_features(panel)
    panel = add_occurrence_lag_features(panel)

    # İlk satırlar (lag_1_occurred NaN) kullanılamaz -- her hücrenin ilk günü
    panel_clean = panel.dropna(subset=['lag_1_occurred']).reset_index(drop=True)
    print(f"[run_occurrence_panel_pipeline] Modelde kullanılabilir: {len(panel_clean)} / {len(panel)} satır")

    panel_clean.to_csv("data/processed/full_panel_daily.csv", index=False)
    print("[run_occurrence_panel_pipeline] Kaydedildi: data/processed/full_panel_daily.csv")
    return panel_clean


def run_pipeline():
    raw = load_raw_files()
    clean = remove_duplicates(raw)
    turkey_only = filter_turkey_boundary(clean)
    with_time = fix_datetime(turkey_only)
    with_grid = assign_grid(with_time)
    daily_summary = build_daily_summary(with_grid)

    daily_summary.to_csv("data/processed/daily_grid_summary.csv", index=False)
    print("[run_pipeline] Kaydedildi: data/processed/daily_grid_summary.csv")
    return daily_summary


if __name__ == "__main__":
    run_pipeline()
    run_occurrence_panel_pipeline()