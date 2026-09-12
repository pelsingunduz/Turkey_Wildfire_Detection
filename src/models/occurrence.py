"""
Yangın OLASILIĞI modeli: Grid hücresi + gün bazlı tam panelden, o gün o
hücrede yangın çıkıp çıkmayacağının olasılığını tahmin eder.

classifier.py'dan farkı: classifier.py "yangın zaten varsa ne kadar ciddi"
sorusuna cevap verir (sadece tespit edilmiş günlerle eğitilir). Bu modül
"yangın çıkar mı" sorusuna cevap verir (tüm gün/hücre kombinasyonlarıyla
eğitilir, yangınsız günler dahil).

FEATURE SEÇİMİ -- TARGET LEAKAGE UYARISI: avg_brightness/avg_frp/max_frp gibi
"ancak tespit olduğunda ölçülen" değerler BURADA KULLANILMAZ -- bunlar bir
ölçüm varsa zaten yangın var demektir, dolambaçlı bir tautoloji olurdu.
Bunun yerine sadece BUGÜNDEN ÖNCE bilinen şeyler kullanılır:
- Coğrafi konum (grid_lat, grid_lon) -- her zaman bilinir
- Takvimsel (month, day_of_year, is_summer) -- her zaman bilinir
- Geçmişe dayalı (lag_1_occurred, rolling_7/30_occurrence_rate) -- shift(1)
  ile hesaplandığı için bugünü hiç görmez (bkz. features.py)

ÖLÇEK NEDENİYLE ALGORİTMA SEÇİMİ: Panel ~1.4 milyon satır içeriyor. SVM ve
KNN bu ölçekte pratik değil (SVM saatler sürebilir, KNN tahmin anında çok
yavaş). Sadece hızlı ölçeklenen algoritmalar karşılaştırılır: Random Forest,
XGBoost, Logistic Regression.

SINIF DENGESİZLİĞİ: Yangın oranı ~%3.26 -- yani "hep hayır de" bile %96+
accuracy verir ama işe yaramaz. Bu yüzden:
- class_weight='balanced' (XGBoost için scale_pos_weight) kullanılır
- Başarı ölçütü accuracy DEĞİL, ROC-AUC ve azınlık sınıfının (yangın=evet)
  precision/recall'udur

SPLIT: forecaster.py'daki prensiple aynı -- KRONOLOJİK (rastgele değil),
çünkü bu temelde bir zaman serisi problemi.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_fscore_support
import joblib


FEATURE_COLUMNS = [
    'grid_lat', 'grid_lon',
    'month', 'day_of_year', 'is_summer',
    'lag_1_occurred', 'rolling_7_occurrence_rate', 'rolling_30_occurrence_rate',
]
TARGET_COLUMN = 'fire_occurred'


def prepare_data(panel_path="data/processed/full_panel_daily.csv"):
    df = pd.read_csv(panel_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"[prepare_data] {len(df)} satır, yangın oranı: {df[TARGET_COLUMN].mean():.2%}")
    return df


def chronological_split(df, test_ratio=0.2):
    """Tarihe göre böler (rastgele değil) -- forecaster.py'daki prensiple aynı,
    modelin geleceği bilerek eğitilmesini önlemek için."""
    split_index = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    print(f"[chronological_split] Train: {len(train)} (yangın oranı {train[TARGET_COLUMN].mean():.2%}), "
          f"Test: {len(test)} (yangın oranı {test[TARGET_COLUMN].mean():.2%})")
    return train, test


def evaluate_model(name, y_test, y_pred, y_proba, results):
    """ROC-AUC ve azınlık sınıfı (yangın=evet) precision/recall'una odaklanır --
    accuracy bu dengesiz veri setinde yanıltıcı olur."""
    print("=" * 50)
    print(name)
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=['yangın_yok', 'yangın_var']))

    auc = roc_auc_score(y_test, y_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', pos_label=1
    )
    print(f"ROC-AUC: {auc:.4f}")

    results.append({
        'model': name,
        'roc_auc': auc,
        'precision_fire': precision,
        'recall_fire': recall,
        'f1_fire': f1,
    })


def run():
    df = prepare_data()
    train, test = chronological_split(df)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    results = []

    # 1. Random Forest (ağaç tabanlı, ölçeklendirme yok, class_weight ile dengesizlik ele alınır)
    rf_model = RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    evaluate_model("RANDOM FOREST", y_test, rf_model.predict(X_test), rf_proba, results)
    joblib.dump(rf_model, "outputs/models/occurrence_rf.joblib")

    # 2. XGBoost (ağaç tabanlı, scale_pos_weight ile dengesizlik ele alınır)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = XGBClassifier(
        n_estimators=100, scale_pos_weight=pos_weight, random_state=42,
        eval_metric='logloss', n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    evaluate_model("XGBOOST", y_test, xgb_model.predict(X_test), xgb_proba, results)
    joblib.dump(xgb_model, "outputs/models/occurrence_xgb.joblib")

    # 3. Logistic Regression (doğrusal, ölçeklendirilmiş veri, class_weight ile dengesizlik)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    evaluate_model("LOGISTIC REGRESSION", y_test, lr_model.predict(X_test_scaled), lr_proba, results)
    joblib.dump(lr_model, "outputs/models/occurrence_lr.joblib")
    joblib.dump(scaler, "outputs/models/occurrence_scaler.joblib")

    # Karşılaştırma tablosu
    print("\n" + "=" * 50)
    print("KARŞILAŞTIRMA TABLOSU (ROC-AUC'a göre sıralı)")
    print("=" * 50)
    comparison_df = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
    print(comparison_df.to_string(index=False))

    # Dashboard'un kullandığı ana model: XGBoost.
    # En yüksek ROC-AUC (~0.91) ve en yüksek recall (~%76) -- erken uyarı
    # sisteminde kaçırılan bir yangının maliyeti yanlış alarmdan çok daha
    # yüksek olduğu için düşük precision kasıtlı bir tercih. Eşik değeri
    # (şu an varsayılan 0.5) ileride precision/recall dengesini ayarlamak
    # için değiştirilebilir.
    joblib.dump(xgb_model, "outputs/models/occurrence.joblib")

    return comparison_df


if __name__ == "__main__":
    run()