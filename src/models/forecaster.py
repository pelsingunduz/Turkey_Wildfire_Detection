"""
Zaman serisi / regresyon modeli: Grid hücresi bazında, geçmiş günlerin
sıcak nokta sayısına bakarak bir sonraki günü tahmin eder.

Feature'lar:
- lag_1_fire_count: bir önceki günün fire_count değeri
- rolling_3_avg: son 3 günün (mevcut değilse eldeki kadarının) ortalaması

ÖNEMLİ SINIRLAMA: Şu anki veri hacmiyle (3 günlük ham veri, ~14 kullanılabilir
satır), bu model istatistiksel olarak anlamlı bir performans göstermiyor.
MAE gibi metrikler bu aşamada güvenilir değildir — sadece pipeline'ın
(lag üretimi -> kronolojik split -> eğitim -> tahmin) uçtan uca çalıştığını
doğrulamak amacıyla kuruldu. Veri arttıkça (birkaç hafta sonra) yeniden
değerlendirilmelidir.

Split KRONOLOJİK yapılır (rastgele değil) — modelin geleceği bilerek
eğitilmesini (data leakage) önlemek için.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
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


def train_and_evaluate(train, test):
    """Linear Regression eğitir, MAE ile değerlendirir.
    NOT: Küçük veri setinde MAE metriği güvenilir değildir (bkz. modül docstring'i)."""
    X_train, y_train = train[FEATURE_COLUMNS], train['fire_count']
    X_test, y_test = test[FEATURE_COLUMNS], test['fire_count']

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[train_and_evaluate] MAE: {mae:.2f} (küçük veri setiyle güvenilir değil)")

    return model


def save_model(model, path="outputs/models/forecaster.joblib"):
    joblib.dump(model, path)
    print(f"[save_model] Kaydedildi: {path}")


def run():
    ts_data = prepare_data()
    train, test = chronological_split(ts_data)
    model = train_and_evaluate(train, test)
    save_model(model)
    return model


if __name__ == "__main__":
    run()