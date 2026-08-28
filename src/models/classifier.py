"""
Sınıflandırma modeli: Grid hücresi + gün bazlı özet veriden risk seviyesi tahmini.

Risk etiketi, fire_count sütununun dağılımından türetilir (keyfi eşik değil,
verinin kendi yüzdelik dilimlerine göre belirlenmiştir):
- düşük: fire_count == 1  (verinin ~%50'si, en yaygın durum)
- orta:  fire_count 2-3   (~%80'e kadar olan dilim)
- yüksek: fire_count >= 4 (üst %20'lik dilim, nadir/kritik durumlar)

Model girdisi olarak fire_count KULLANILMAZ, çünkü etiket zaten ondan türetildi
(target leakage riski). Bunun yerine, konumdan bağımsız şiddet/parlaklık
ölçümleri (avg_brightness, avg_frp, max_frp) kullanılır. Konum bilgisi
(grid_lat/grid_lon) da şimdilik dahil edilmiyor, çünkü veri seti küçük
(~100 satır, ~89 benzersiz hücre) ve model belirli koordinatları ezberleme
riski taşıyor. Veri arttıkça bu karar yeniden değerlendirilebilir.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


FEATURE_COLUMNS = ['avg_brightness', 'avg_frp', 'max_frp']


def assign_risk_label(fire_count):
    """fire_count'u risk kategorisine çevirir. Eşikler verinin yüzdelik
    dilimlerinden türetilmiştir (bkz. modül docstring'i)."""
    if fire_count == 1:
        return 'düşük'
    elif fire_count <= 3:
        return 'orta'
    else:
        return 'yüksek'


def prepare_data(daily_summary_path="data/processed/daily_grid_summary.csv"):
    """Günlük özet tabloyu okur, risk etiketini ekler, X/y olarak ayırır."""
    df = pd.read_csv(daily_summary_path)
    df['risk_label'] = df['fire_count'].apply(assign_risk_label)

    X = df[FEATURE_COLUMNS]
    y = df['risk_label']
    return X, y


def train_and_evaluate(X, y):
    """Train/test split yapar (stratify ile sınıf oranlarını korur),
    Random Forest eğitir, sınıf bazlı performans raporu üretir."""
    # stratify=y kritik: 'yüksek' sınıfı zaten az sayıda (~17 örnek),
    # stratify olmadan rastgele split bunların çoğunu tek bir sete
    # atayabilir ve değerlendirmeyi anlamsız kılabilir.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model


def save_model(model, path="outputs/models/classifier.joblib"):
    """Eğitilen modeli diske kaydeder (training ile serving'i ayırmak için)."""
    joblib.dump(model, path)
    print(f"[save_model] Kaydedildi: {path}")


def run():
    X, y = prepare_data()
    model = train_and_evaluate(X, y)
    save_model(model)
    return model


if __name__ == "__main__":
    run()