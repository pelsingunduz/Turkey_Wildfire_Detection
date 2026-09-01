"""
Anomali tespiti: Her grid hücresi için, GEÇMİŞ (bugünü hariç tutarak)
ortalama ve standart sapmaya (baseline) göre bugünkü değerin z-score'unu
hesaplar. Yüksek |z_score|, o hücrenin normalden anormal şekilde saptığı
anlamına gelir.

Baseline hesaplamasında bugünün kendisi KASITLI OLARAK dışlanır (shift(1)
ile) -- aksi halde büyük bir anomali kendi baseline'ını yukarı çekip
kendini normalleştirir, bu da anomaliyi gizler.

VERİ GEREKSİNİMİ: Bu modül, diğer ikisinden (classifier, forecaster) daha
fazla veri gerektirir. Bir grid hücresinin z_score'unun hesaplanabilmesi
için o hücrenin EN AZ 3 kez (bugün + en az 2 geçmiş gün) görünmesi gerekir,
çünkü standart sapma tek bir noktadan hesaplanamaz. Proje başlangıcında
(ilk birkaç gün) bu koşulu sağlayan hücre sayısı 0 olabilir -- bu bir hata
değil, veri toplama süresinin henüz yetersiz olduğunun göstergesidir.

Şu an expanding() (şimdiye kadarki TÜM geçmiş) kullanılıyor. Yeterli veri
biriktiğinde (örn. 4-8 hafta), bunun rolling(28) gibi sabit pencereli,
mevsimsel bir baseline'a çevrilmesi daha doğru olur -- bkz. modül sonu notu.
"""

import pandas as pd


Z_SCORE_ANOMALY_THRESHOLD = 3  # kaç sigma üstü "anomali" sayılır (istatistikte yaygın eşik)


def calculate_zscore_features(df):
    """Her satır için baseline_mean, baseline_std ve z_score hesaplar.
    
    baseline_std == 0 olduğunda (çok az gözlemden hiç varyans çıkmadığında),
    z_score NaN olarak bırakılır -- bu matematiksel bir "sonsuz anomali" değil,
    örneklem yetersizliğinin işaretidir ve anomali olarak sayılmamalıdır.
    """
    df = df.copy()

    df['baseline_mean'] = (
        df.groupby('grid_id')['fire_count']
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df['baseline_std'] = (
        df.groupby('grid_id')['fire_count']
        .transform(lambda x: x.shift(1).expanding().std())
    )

    df['z_score'] = (df['fire_count'] - df['baseline_mean']) / df['baseline_std']
    # std == 0 durumunda z_score'u NaN yap (sıfıra bölme / sahte-sonsuz anomali önleme)
    df.loc[df['baseline_std'] == 0, 'z_score'] = float('nan')

    return df


def flag_anomalies(df, threshold=Z_SCORE_ANOMALY_THRESHOLD):
    """z_score'u eşik değerin üzerinde olan satırları anomali olarak işaretler."""
    df = df.copy()
    df['is_anomaly'] = df['z_score'].abs() >= threshold
    return df


def prepare_data(daily_summary_path="data/processed/daily_grid_summary.csv"):
    df = pd.read_csv(daily_summary_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['grid_id', 'date']).reset_index(drop=True)

    df = calculate_zscore_features(df)
    df = flag_anomalies(df)

    valid_count = df['z_score'].notna().sum()
    anomaly_count = df['is_anomaly'].sum()
    print(f"[prepare_data] z_score hesaplanabilen satır: {valid_count} / {len(df)}")
    print(f"[prepare_data] Tespit edilen anomali: {anomaly_count}")

    return df


def save_results(df, path="data/processed/anomaly_results.csv"):
    df.to_csv(path, index=False)
    print(f"[save_results] Kaydedildi: {path}")


def run():
    df = prepare_data()
    save_results(df)
    return df


if __name__ == "__main__":
    run()