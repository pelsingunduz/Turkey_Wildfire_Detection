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
çünkü standart sapma tek bir noktadan hesaplanamaz.

Şu an expanding() (şimdiye kadarki TÜM geçmiş) kullanılıyor. Yeterli veri
biriktiğinde (örn. 4-8 hafta), bunun rolling(28) gibi sabit pencereli,
mevsimsel bir baseline'a çevrilmesi daha doğru olur -- bkz. modül sonu notu.

GÜNCELLEME -- IKINCI BİR YÖNTEMLE KARŞILAŞTIRMA (Isolation Forest):
classifier.py ve occurrence.py'daki "algoritma seçimi veriye dayalı olmalı"
prensibini burada da uygulamak için, z-score yöntemine ek olarak Isolation
Forest ile de bir anomali tespiti yapılıyor (bkz. calculate_isolation_
forest_flags). ÖNEMLİ FARK: classifier/occurrence karşılaştırmalarının
aksine, burada "hangisi daha doğru" diye bir accuracy/MAE ölçümü YAPILAMAZ
-- çünkü anomali tespitinde gerçek etiket (ground truth) yok, hangi
satırların "gerçekten" anormal olduğunu bilmiyoruz. Bu yüzden karşılaştırma,
iki yöntemin kaç anomali işaretlediği ve ne kadar örtüştüğü üzerinden
yapılıyor (bkz. compare_anomaly_methods), kazanan/kaybeden belirlenmiyor.

Z-score YİNE DE resmi yöntem olarak kalıyor, iki metodolojik sebeple:
(1) yorumlanabilir -- bir z-score değeri "kaç standart sapma" demek,
    dashboard'da kullanıcıya doğrudan anlamlı şekilde gösterilebiliyor.
(2) MEKANA ÖZGÜ bağlam kullanıyor -- her grid hücresini KENDİ geçmişiyle
    karşılaştırıyor. Isolation Forest ise GLOBAL bir model (tüm hücreler
    birlikte), yani doğası gereği farklı bir şey ölçüyor: "bu satır genel
    örüntüye göre sıra dışı mı", "bu hücre kendi geçmişine göre sıra dışı
    mı" değil.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

Z_SCORE_ANOMALY_THRESHOLD = 3  # kaç sigma üstü "anomali" sayılır (istatistikte yaygın eşik)
ISOLATION_FOREST_FEATURES = ['fire_count', 'avg_brightness', 'avg_frp', 'max_frp']
ISOLATION_FOREST_CONTAMINATION = 0.03  # z-score'un |z|>=3 eşiğiyle kabaca uyumlu beklenen anomali payı


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


def calculate_isolation_forest_flags(df):
    """Global (tüm grid hücreleri birlikte) çok değişkenli bir anomali tespiti.
    z-score yönteminden FARKLI OLARAK, her hücrenin kendi geçmişiyle değil,
    tüm veri setindeki genel örüntüyle karşılaştırır -- bkz. modül docstring'i."""
    df = df.copy()
    model = IsolationForest(contamination=ISOLATION_FOREST_CONTAMINATION, random_state=42)
    predictions = model.fit_predict(df[ISOLATION_FOREST_FEATURES])
    df['is_anomaly_isolation_forest'] = predictions == -1
    return df


def compare_anomaly_methods(df):
    """İki yöntemin kaç anomali işaretlediğini ve ne kadar örtüştüğünü raporlar.
    NOT: 'Hangisi daha doğru' diye bir sonuç ÇIKARILAMAZ (gerçek etiket yok) --
    bkz. modül docstring'i. Bu sadece bilgilendirici bir karşılaştırma."""
    both_valid = df['is_anomaly'].notna()
    zscore_count = int(df.loc[both_valid, 'is_anomaly'].sum())
    iso_count = int(df.loc[both_valid, 'is_anomaly_isolation_forest'].sum())
    overlap = int((df.loc[both_valid, 'is_anomaly'] & df.loc[both_valid, 'is_anomaly_isolation_forest']).sum())
    print(f"[compare_anomaly_methods] z-score anomali: {zscore_count}")
    print(f"[compare_anomaly_methods] Isolation Forest anomali: {iso_count}")
    print(f"[compare_anomaly_methods] Örtüşen (iki yöntem de işaretledi): {overlap}")


def prepare_data(daily_summary_path="data/processed/daily_grid_summary.csv"):
    df = pd.read_csv(daily_summary_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['grid_id', 'date']).reset_index(drop=True)
    df = calculate_zscore_features(df)
    df = flag_anomalies(df)
    df = calculate_isolation_forest_flags(df)

    valid_count = df['z_score'].notna().sum()
    anomaly_count = df['is_anomaly'].sum()
    print(f"[prepare_data] z_score hesaplanabilen satır: {valid_count} / {len(df)}")
    print(f"[prepare_data] Tespit edilen anomali (z-score): {anomaly_count}")

    compare_anomaly_methods(df)
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