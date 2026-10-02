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
- İNSAN KAYNAKLI: population_density (ilin km² başına nüfusu, TÜİK) --
  statik bir feature, her zaman bilinir (bkz. features.py:add_population_density)

İNSAN KAYNAKLI RİSK FAKTÖRÜ DEĞERLENDİRMESİ: population_density eklemeden
ÖNCE ve SONRA aynı algoritma (XGBoost) ile ayrı ayrı eğitilip ROC-AUC
karşılaştırılıyor (bkz. evaluate_population_density_impact) -- "bu feature
gerçekten modeli iyileştiriyor mu" sorusu varsayıma değil ölçüme dayanıyor.

SONUÇ (ölçüldü, varsayılmadı): population_density eklenince ROC-AUC
0.9135 -> 0.9129'a düştü (-0.0007, ihmal edilebilir/gürültü düzeyinde).
Yani bu feature modeli İYİLEŞTİRMEDİ. RESMİ MODELDE KULLANILMIYOR -- test
kodu (evaluate_population_density_impact), bu değerlendirmeyi
tekrarlanabilir kılmak için korunuyor.

SENSÖR FÜZYONU (hava durumu): temp_max/precip_sum/wind_speed_max/
wind_direction_dominant -- her ilin grid hücrelerinden hesaplanan merkez
koordinat için Open-Meteo'dan çekilen günlük sıcaklık, yağış, rüzgar hızı
ve yönü (bkz. features.py:add_weather_features, src/weather_backfill.py).
Klasik "sıcak + kuru + rüzgarlı = yüksek risk" yangın-hava ilişkisini
yakalaması bekleniyor.

SONUÇ (ölçüldü): hava durumu eklenince ROC-AUC 0.9135 -> 0.9170'e çıktı
(+0.0035, eşiğin üzerinde, ölçülebilir bir iyileşme). Bu yüzden RESMİ
MODELDE KULLANILIYOR -- population_density'nin aksine, bu feature gerçek
bir katkı sağladı.

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


# Taban feature seti -- insan kaynaklı/hava durumu gibi ek sinyaller olmadan.
BASE_FEATURE_COLUMNS = [
    'grid_lat', 'grid_lon',
    'month', 'day_of_year', 'is_summer',
    'lag_1_occurred', 'rolling_7_occurrence_rate', 'rolling_30_occurrence_rate',
]
# Sadece evaluate_weather_impact() içindeki ablation testi için (bkz.
# features.py:WEATHER_FEATURE_COLUMNS -- aynı liste, burada tekrar tanımlı
# çünkü bu script src/features.py'ı import etmeden bağımsız çalışabiliyor).
WEATHER_COLUMNS = ['temp_max', 'precip_sum', 'wind_speed_max', 'wind_direction_dominant']

# Resmi modelde kullanılan feature seti: taban + hava durumu (ölçülebilir
# iyileşme sağladığı için dahil). population_density İÇERMİYOR (ölçülen
# etkisi negatif/ihmal edilebilir çıktı -- bkz. modül docstring'i).
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + WEATHER_COLUMNS

# Sadece evaluate_population_density_impact() içindeki ablation testi için
# -- taban sete göre test edilir (hava durumu olmadan), orijinal ölçümle
# tutarlı kalması için.
WITH_POPULATION_DENSITY_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ['population_density']
TARGET_COLUMN = 'fire_occurred'


def prepare_data(panel_path="data/processed/full_panel_daily.csv"):
    df = pd.read_csv(panel_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # XGBoost/Random Forest NaN'ları kendiliğinden işleyebiliyor ama Logistic
    # Regression işleyemiyor -- hava durumu API'sinin dolduramadığı birkaç
    # eksik değeri (bkz. features.py:add_weather_features) sütun ortalamasıyla
    # dolduruyoruz. Çok küçük bir oran (1.4M satırda ~5 bin değer) olduğu için
    # bu basitleştirme kabul edilebilir.
    missing_before = df[WEATHER_COLUMNS].isna().sum().sum()
    if missing_before > 0:
        df[WEATHER_COLUMNS] = df[WEATHER_COLUMNS].fillna(df[WEATHER_COLUMNS].mean())
        print(f"[prepare_data] {missing_before} eksik hava durumu değeri sütun ortalamasıyla dolduruldu")

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
    return auc


def evaluate_population_density_impact(train, test):
    """population_density feature'ının gerçekten işe yarayıp yaramadığını
    ölçer: aynı algoritma (XGBoost), aynı split, tek fark bu feature'ın
    olup olmaması. Varsayıma değil ölçüme dayalı karar vermek için."""
    y_train, y_test = train[TARGET_COLUMN], test[TARGET_COLUMN]
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("\n" + "=" * 50)
    print("İNSAN KAYNAKLI RİSK FAKTÖRÜ ETKİSİ (population_density)")
    print("=" * 50)

    aucs = {}
    for label, cols in [("population_density OLMADAN", BASE_FEATURE_COLUMNS),
                         ("population_density İLE", WITH_POPULATION_DENSITY_FEATURE_COLUMNS)]:
        model = XGBClassifier(
            n_estimators=100, scale_pos_weight=pos_weight, random_state=42,
            eval_metric='logloss', n_jobs=-1
        )
        model.fit(train[cols], y_train)
        proba = model.predict_proba(test[cols])[:, 1]
        auc = roc_auc_score(y_test, proba)
        aucs[label] = auc
        print(f"[{label}] ROC-AUC: {auc:.4f}")

    diff = aucs["population_density İLE"] - aucs["population_density OLMADAN"]
    print(f"\nFark: {diff:+.4f} ROC-AUC puanı")
    if abs(diff) < 0.002:
        print("Sonuç: Fark ihmal edilebilir düzeyde -- feature gürültüden öteye geçmiyor gibi görünüyor.")
    elif diff > 0:
        print("Sonuç: population_density modeli ölçülebilir şekilde iyileştiriyor.")
    else:
        print("Sonuç: population_density modeli İYİLEŞTİRMİYOR, hatta hafifçe kötüleştiriyor.")


def evaluate_weather_impact(train, test):
    """Hava durumu feature'larının (sıcaklık, yağış, rüzgar hızı/yönü) gerçekten
    işe yarayıp yaramadığını ölçer -- aynı mantık, population_density ile
    yapılan değerlendirmenin aynısı (bkz. evaluate_population_density_impact)."""
    y_train, y_test = train[TARGET_COLUMN], test[TARGET_COLUMN]
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("\n" + "=" * 50)
    print("SENSÖR FÜZYONU ETKİSİ (hava durumu: sıcaklık, yağış, rüzgar)")
    print("=" * 50)

    aucs = {}
    for label, cols in [("hava durumu OLMADAN", BASE_FEATURE_COLUMNS),
                         ("hava durumu İLE", FEATURE_COLUMNS)]:
        model = XGBClassifier(
            n_estimators=100, scale_pos_weight=pos_weight, random_state=42,
            eval_metric='logloss', n_jobs=-1
        )
        model.fit(train[cols], y_train)
        proba = model.predict_proba(test[cols])[:, 1]
        auc = roc_auc_score(y_test, proba)
        aucs[label] = auc
        print(f"[{label}] ROC-AUC: {auc:.4f}")

    diff = aucs["hava durumu İLE"] - aucs["hava durumu OLMADAN"]
    print(f"\nFark: {diff:+.4f} ROC-AUC puanı")
    if abs(diff) < 0.002:
        print("Sonuç: Fark ihmal edilebilir düzeyde -- feature'lar gürültüden öteye geçmiyor gibi görünüyor.")
    elif diff > 0:
        print("Sonuç: Hava durumu feature'ları modeli ölçülebilir şekilde iyileştiriyor.")
    else:
        print("Sonuç: Hava durumu feature'ları İYİLEŞTİRMİYOR, hatta hafifçe kötüleştiriyor.")


def run():
    df = prepare_data()
    train, test = chronological_split(df)

    evaluate_population_density_impact(train, test)
    evaluate_weather_impact(train, test)

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
    # En yüksek ROC-AUC ve en yüksek recall -- erken uyarı sisteminde
    # kaçırılan bir yangının maliyeti yanlış alarmdan çok daha yüksek
    # olduğu için düşük precision kasıtlı bir tercih. Eşik değeri (şu an
    # varsayılan 0.5) ileride precision/recall dengesini ayarlamak için
    # değiştirilebilir.
    joblib.dump(xgb_model, "outputs/models/occurrence.joblib")

    return comparison_df


if __name__ == "__main__":
    run()