"""
Zaman serisi / regresyon modeli: Grid hücresi bazında, geçmiş günlerin
sıcak nokta sayısına bakarak bir sonraki günü tahmin eder.

Feature'lar:
- lag_1_fire_count: bir önceki günün fire_count değeri
- rolling_3_avg: son 3 günün (mevcut değilse eldeki kadarının) ortalaması

GÜNCELLEME (geçmiş veri backfill'i sonrası, ~46.000 satır): Artık yeterli
veri olduğu için ÜÇ algoritma karşılaştırılıyor (Linear Regression, Random
Forest Regressor, XGBoost Regressor) -- projenin diğer modellerindeki
(classifier.py, occurrence.py) "algoritma seçimi veriye dayalı, sistematik
bir süreç olmalı" prensibiyle tutarlı olması için. İlk sürümde (3 günlük
ham veri, ~14 satır) tek algoritma (Linear Regression) kullanılmıştı --
o kadar az veriyle karşılaştırma yapmanın bir anlamı yoktu.

Split KRONOLOJİK yapılır (rastgele değil) — modelin geleceği bilerek
eğitilmesini (data leakage) önlemek için.

NEGATİF TAHMİN UYARISI: Linear Regression, çıktısını 0'da sınırlamaz --
matematiksel olarak negatif bir "sıcak nokta sayısı" tahmini üretebilir,
ki bu anlamsızdır (sayım asla negatif olamaz). Ağaç tabanlı modeller
(Random Forest, XGBoost) bu sorunu yapısal olarak yaşamaz (yaprak
değerleri eğitim verisindeki gerçek değerlerden türediği için hep
pozitif), ama güvenlik için HER modelin tahmini np.clip(tahmin, 0, None)
ile sıfırın altına düşürülür -- bkz. predict_clipped() ve
dashboard/app.py'daki kullanım. Modelin kendisi (joblib'e kaydedilen
haliyle) bunu otomatik yapmaz, her çağıran kod predict_clipped()
kullanmalı.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib

FEATURE_COLUMNS = ['lag_1_fire_count', 'rolling_3_avg']


def add_lag_features(df):
    """Her grid hücresi için, kendi geçmiş günlerine dayalı lag feature'ları ekler."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['grid_id', 'date']).reset_index(drop=True)
    # shift(1): aynı grid_id içinde bir önceki satırın değerini getirir (dünkü değer)
    df['lag_1_fire_count'] = df.groupby('grid_id')['fire_count'].shift(1)
    # rolling(3, min_periods=1): son 3 günün ortalaması, veri azsa eldekiyle hesaplanır
    df['rolling_3_avg'] = (
        df.groupby('grid_id')['fire_count']
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )
    return df


def prepare_data(daily_summary_path="data/processed/daily_grid_summary.csv"):
    """Günlük özet tabloyu okur, lag feature ekler, geçmişi olmayan (NaN) satırları eler."""
    df = pd.read_csv(daily_summary_path)
    df = add_lag_features(df)
    # lag_1_fire_count NaN olan satırlar (geçmişi olmayan, tek-günlük hücreler) kullanılamaz
    ts_data = df.dropna(subset=['lag_1_fire_count']).copy()
    ts_data = ts_data.sort_values('date').reset_index(drop=True)
    print(f"[prepare_data] Zaman serisi için kullanılabilir satır: {len(ts_data)}")
    return ts_data


def chronological_split(ts_data, test_ratio=0.2):
    """Rastgele değil, tarihe göre böler: en eski satırlar train, en yeni satırlar test.
    Bu, modelin geleceği bilerek eğitilmesini (data leakage) engeller."""
    split_index = int(len(ts_data) * (1 - test_ratio))
    train = ts_data.iloc[:split_index]
    test = ts_data.iloc[split_index:]
    print(f"[chronological_split] Train: {len(train)}, Test: {len(test)}")
    return train, test


def predict_clipped(model, X):
    """Modelin tahminini alır ve negatif değerleri 0'a sabitler.
    Dashboard dahil, bu modeli kullanan HER yer bu fonksiyonu (ya da aynı
    np.clip mantığını) kullanmalı -- ham model.predict() (özellikle Linear
    Regression için) negatif değer üretebilir."""
    return np.clip(model.predict(X), 0, None)


def evaluate_model(name, y_test, y_pred, results):
    """MAE hesaplar, ekrana basar, karşılaştırma tablosu için kaydeder."""
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[{name}] MAE: {mae:.3f}")
    results.append({'model': name, 'mae': mae})


def save_model(model, path="outputs/models/forecaster.joblib"):
    joblib.dump(model, path)
    print(f"[save_model] Kaydedildi: {path}")


def run():
    ts_data = prepare_data()
    train, test = chronological_split(ts_data)

    X_train, y_train = train[FEATURE_COLUMNS], train['fire_count']
    X_test, y_test = test[FEATURE_COLUMNS], test['fire_count']

    results = []

    # 1. Linear Regression (doğrusal, negatif tahmin üretebilir -- clip gerekli)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    evaluate_model("LINEAR REGRESSION", y_test, predict_clipped(lr_model, X_test), results)
    joblib.dump(lr_model, "outputs/models/forecaster_lr.joblib")

    # 2. Random Forest Regressor (ağaç tabanlı, doğrusal olmayan ilişkileri yakalayabilir)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    evaluate_model("RANDOM FOREST", y_test, predict_clipped(rf_model, X_test), results)
    joblib.dump(rf_model, "outputs/models/forecaster_rf.joblib")

    # 3. XGBoost Regressor (ağaç tabanlı, boosting)
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    evaluate_model("XGBOOST", y_test, predict_clipped(xgb_model, X_test), results)
    joblib.dump(xgb_model, "outputs/models/forecaster_xgb.joblib")

    # Karşılaştırma tablosu
    print("\n" + "=" * 50)
    print("KARŞILAŞTIRMA TABLOSU (MAE'ye göre sıralı, düşük daha iyi)")
    print("=" * 50)
    comparison_df = pd.DataFrame(results).sort_values('mae')
    print(comparison_df.to_string(index=False))

    # Resmi model: en düşük MAE'ye sahip olan.
    best_name = comparison_df.iloc[0]['model']
    best_model = {"LINEAR REGRESSION": lr_model, "RANDOM FOREST": rf_model, "XGBOOST": xgb_model}[best_name]
    print(f"\n[run] Resmi model: {best_name}")
    save_model(best_model)

    return comparison_df


if __name__ == "__main__":
    run()