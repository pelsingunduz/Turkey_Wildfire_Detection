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