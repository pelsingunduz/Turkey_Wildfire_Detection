"""
Sınıflandırma modeli: Grid hücresi + gün bazlı özet veriden risk seviyesi tahmini.

Risk etiketi, fire_count sütununun dağılımından türetilir (keyfi eşik değil,
verinin kendi yüzdelik dilimlerine göre belirlenmiştir):
- düşük: fire_count == 1  (verinin ~%50'si, en yaygın durum)
- orta:  fire_count 2-3   (~%80'e kadar olan dilim)
- yüksek: fire_count >= 4 (üst %20'lik dilim, nadir/kritik durumlar)

Model girdisi olarak fire_count KULLANILMAZ (target leakage riski).
Bunun yerine avg_brightness, avg_frp, max_frp kullanılır.

BEŞ ALGORİTMA KARŞILAŞTIRMASI:
- Random Forest, XGBoost: ağaç tabanlı, ölçeklendirme gerektirmez
- Logistic Regression, SVM, KNN: mesafe/büyüklük temelli, StandardScaler
  ile ölçeklendirme YAPILIR (aksi halde avg_brightness ~300 civarı, avg_frp
  ~1-80 civarı olduğu için büyük ölçekli feature yapay olarak baskın çıkar)

GÜNCELLEME (geçmiş veri backfill'i sonrası, ~46.000 satır): Resmi model
Logistic Regression'dan XGBoost'a değişti. İlk karşılaştırma küçük bir
veri setinde (~260 satır, test seti 52 satır) yapılmıştı ve o ölçekte
basit bir doğrusal model karmaşık ağaç modellerini geçmişti (az veri,
karmaşık modellerin overfitting'e daha yatkın olmasına yol açar). Veri
~46.000 satıra çıkınca bu dengeler tersine döndü: XGBoost açık ara önde
(accuracy 0.86, macro F1 0.78), Logistic Regression ise en düşük
performansı gösterdi (accuracy 0.76, macro F1 0.64). Bu, "az veri basit
modelleri, çok veri karmaşık modelleri kayırır" prensibinin ders kitabı
gibi bir örneği.

ÖNEMLİ: XGBoost ölçeklendirme gerektirmediği için, dashboard'da bu model
kullanılırken StandardScaler UYGULANMAMALI -- ham feature'lar doğrudan
verilmeli (bkz. dashboard/app.py).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib


FEATURE_COLUMNS = ['avg_brightness', 'avg_frp', 'max_frp']
LABEL_MAP = {'düşük': 0, 'orta': 1, 'yüksek': 2}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def assign_risk_label(fire_count):
    if fire_count == 1:
        return 'düşük'
    elif fire_count <= 3:
        return 'orta'
    else:
        return 'yüksek'


def prepare_data(daily_summary_path="data/processed/daily_grid_summary.csv"):
    df = pd.read_csv(daily_summary_path)
    df['risk_label'] = df['fire_count'].apply(assign_risk_label)
    X = df[FEATURE_COLUMNS]
    y = df['risk_label']
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate_model(name, y_test, y_pred, results):
    """Değerlendirir, sonucu ekrana basar, karşılaştırma tablosu için sonucu kaydeder."""
    print("=" * 50)
    print(name)
    print("=" * 50)
    print(classification_report(y_test, y_pred))

    results.append({
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro')
    })


def run():
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Ölçeklendirilmiş versiyonlar (Logistic Regression, SVM, KNN için)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_numeric = y_train.map(LABEL_MAP)

    results = []

    # 1. Random Forest (ağaç tabanlı, ölçeklendirme yok)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    evaluate_model("RANDOM FOREST", y_test, rf_model.predict(X_test), results)
    joblib.dump(rf_model, "outputs/models/classifier_rf.joblib")

    # 2. XGBoost (ağaç tabanlı, ölçeklendirme yok, sayısal etiket gerekir)
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train_numeric)
    xgb_pred = pd.Series(xgb_model.predict(X_test)).map(REVERSE_LABEL_MAP)
    evaluate_model("XGBOOST", y_test, xgb_pred, results)
    joblib.dump(xgb_model, "outputs/models/classifier_xgb.joblib")

    # 3. Logistic Regression (doğrusal, ölçeklendirilmiş veri)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    evaluate_model("LOGISTIC REGRESSION", y_test, lr_model.predict(X_test_scaled), results)
    joblib.dump(lr_model, "outputs/models/classifier_lr.joblib")

    # 4. SVM (sınır maksimizasyonu, ölçeklendirilmiş veri)
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    evaluate_model("SVM", y_test, svm_model.predict(X_test_scaled), results)
    joblib.dump(svm_model, "outputs/models/classifier_svm.joblib")

    # 5. KNN (en yakın komşular, ölçeklendirilmiş veri)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    evaluate_model("KNN", y_test, knn_model.predict(X_test_scaled), results)
    joblib.dump(knn_model, "outputs/models/classifier_knn.joblib")

    # Scaler'ı da kaydet -- Logistic Regression/SVM/KNN varyantlarını
    # denemek istersen hâlâ gerekecek (resmi model artık bunu kullanmıyor)
    joblib.dump(scaler, "outputs/models/scaler.joblib")

    # Karşılaştırma tablosu
    print("\n" + "=" * 50)
    print("KARŞILAŞTIRMA TABLOSU")
    print("=" * 50)
    comparison_df = pd.DataFrame(results).sort_values('macro_f1', ascending=False)
    print(comparison_df.to_string(index=False))

    # Dashboard'un kullandığı ana model: XGBoost (bkz. modül docstring'i).
    # ~46.000 satırlık veri setinde en yüksek accuracy (0.86) ve macro F1
    # (0.78) değerine sahip. Ölçeklendirme GEREKTİRMEZ -- dashboard'da bu
    # modelle tahmin yaparken scaler.joblib kullanılmamalı, ham feature'lar
    # doğrudan verilmeli.
    joblib.dump(xgb_model, "outputs/models/classifier.joblib")

    return comparison_df


if __name__ == "__main__":
    run()