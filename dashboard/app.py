"""
Türkiye Orman Yangını Erken Tespit Dashboard'u

Harita üzerinde bir grid hücresine tıklayarak, o bölge için dört modelin
(yangın olasılığı, sınıflandırma, zaman serisi, anomali tespiti) çıktısını
görebilirsiniz.
"""

import streamlit as st
import pandas as pd
import folium
import joblib
import sys
import os
sys.path.append(os.path.abspath('src/models'))
sys.path.append(os.path.abspath('src'))
from anomaly import calculate_zscore_features, flag_anomalies # type: ignore
from forecaster import add_lag_features, FEATURE_COLUMNS as FORECASTER_FEATURES # type: ignore
from features import build_latest_occurrence_features # type: ignore
from streamlit_folium import st_folium

st.set_page_config(page_title="Türkiye Orman Yangını Tespit Sistemi", layout="wide")
st.title("🔥 Türkiye Orman Yangını Erken Tespit Sistemi")


@st.cache_data
def load_grid_points():
    """Günlük özet tablodan benzersiz grid hücrelerini, il/ilçe isimleriyle birlikte çıkarır."""
    df = pd.read_csv("data/processed/daily_grid_summary.csv")
    grid_points = df[['grid_id', 'grid_lat', 'grid_lon']].drop_duplicates()

    locations = pd.read_csv("data/processed/grid_location_names.csv")
    grid_points = grid_points.merge(locations, on='grid_id', how='left')

    # Hem il hem ilçe bilinmiyorsa (sınır/deniz üzerine düşen hücreler),
    # haritada anlamlı bir isim gösteremeyiz -- bu hücreleri haritadan
    # çıkarıyoruz (model/veri tarafında hiçbir şey değişmiyor, sadece görünürlük).
    unknown_both = (grid_points['province'] == 'Bilinmiyor') & (grid_points['district'] == 'Bilinmiyor')
    grid_points = grid_points[~unknown_both]

    grid_points['display_name'] = grid_points.apply(
        lambda row: row['province'] if row['district'] == 'Bilinmiyor' else f"{row['province']} - {row['district']}",
        axis=1
    )

    # Aynı il/ilçeye birden fazla grid hücresi düşebilir (0.25° hücre, idari
    # sınırdan küçük olabiliyor) -- haritada karışıklık/çakışma olmasın diye
    # her isimden sadece bir tanesini gösteriyoruz.
    grid_points = grid_points.drop_duplicates(subset='display_name', keep='first')

    return grid_points


@st.cache_data
def load_data_with_lag_features():
    """Günlük özet tabloyu okur, lag feature'ları hesaplar (forecaster.py'deki mantıkla aynı)."""
    df = pd.read_csv("data/processed/daily_grid_summary.csv")
    df = add_lag_features(df)
    return df


@st.cache_data
def load_data_with_anomaly_features():
    """Günlük özet tabloyu okur, z-score ve anomali bayrağını hesaplar."""
    df = pd.read_csv("data/processed/daily_grid_summary.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['grid_id', 'date']).reset_index(drop=True)
    df = calculate_zscore_features(df)
    df = flag_anomalies(df)
    return df


@st.cache_data
def load_daily_summary():
    """Ham günlük özet tabloyu okur (occurrence feature hesaplaması için)."""
    df = pd.read_csv("data/processed/daily_grid_summary.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_resource
def load_classifier():
    """Kayıtlı sınıflandırma modelini yükler (Logistic Regression, ölçeklendirilmiş veri bekler)."""
    return joblib.load("outputs/models/classifier.joblib")


@st.cache_resource
def load_scaler():
    """Sınıflandırma modeliyle birlikte eğitilen StandardScaler'ı yükler."""
    return joblib.load("outputs/models/scaler.joblib")


@st.cache_resource
def load_forecaster():
    """Kayıtlı zaman serisi modelini yükler."""
    return joblib.load("outputs/models/forecaster.joblib")


@st.cache_resource
def load_occurrence_model():
    """Kayıtlı yangın olasılığı modelini yükler (XGBoost, ölçeklendirme gerektirmez)."""
    return joblib.load("outputs/models/occurrence.joblib")


def get_latest_features(grid_id, df):
    """Belirli bir grid hücresi için en güncel (en son tarihli) satırı döndürür."""
    grid_rows = df[df['grid_id'] == grid_id].sort_values('date')
    if len(grid_rows) == 0:
        return None
    return grid_rows.iloc[-1]


grid_points = load_grid_points()

st.write(f"Toplam {len(grid_points)} grid hücresi izleniyor.")

m = folium.Map(location=[39.0, 35.0], zoom_start=6)

marker_names = []
for _, row in grid_points.iterrows():
    marker = folium.CircleMarker(
        location=[row['grid_lat'] + 0.125, row['grid_lon'] + 0.125],
        radius=6,
        popup=row['display_name'],
        tooltip=row['display_name'],
        color='red',
        fill=True,
        fill_color='red',
        opacity=0,
        fill_opacity=0,
    )
    marker.add_to(m)
    marker_names.append(marker.get_name())

# Marker'lar varsayılan olarak görünmez (opacity=0) -- estetik açıdan
# haritayı sade tutmak için. Fareyle üzerine gelindiğinde (mouseover)
# görünür hale gelir, uzaklaşınca (mouseout) tekrar kaybolur. Tıklama ve
# tooltip işlevselliği opaklıktan bağımsız çalışır, bu yüzden bu sadece
# görsel bir tercih, işlevsellik kaybı yok.
hover_js = "\n".join(
    f"{name}.on('mouseover', function(e) {{ this.setStyle({{opacity: 1, fillOpacity: 0.7}}); }});"
    f"{name}.on('mouseout', function(e) {{ this.setStyle({{opacity: 0, fillOpacity: 0}}); }});"
    for name in marker_names
)
m.get_root().html.add_child(folium.Element(f"<script>{hover_js}</script>"))

map_data = st_folium(m, width=900, height=500)

if map_data.get("last_object_clicked_tooltip"):
    selected_display_name = map_data['last_object_clicked_tooltip']
    matching_row = grid_points[grid_points['display_name'] == selected_display_name]

    if len(matching_row) > 0:
        selected_grid_row = matching_row.iloc[0]
        selected_grid = selected_grid_row['grid_id']
        st.success(f"Seçilen bölge: {selected_display_name}")

        full_data = pd.read_csv("data/processed/daily_grid_summary.csv")
        latest_row = get_latest_features(selected_grid, full_data)

        col0, col1, col2, col3 = st.columns(4)

        with col0:
            st.subheader("🔥 Yangın Olasılığı")
            daily_summary = load_daily_summary()
            occ_features, target_date = build_latest_occurrence_features(
                daily_summary,
                grid_id=selected_grid,
                grid_lat=selected_grid_row['grid_lat'],
                grid_lon=selected_grid_row['grid_lon'],
            )
            occurrence_model = load_occurrence_model()
            occurrence_columns = [
                'grid_lat', 'grid_lon', 'month', 'day_of_year', 'is_summer',
                'lag_1_occurred', 'rolling_7_occurrence_rate', 'rolling_30_occurrence_rate',
            ]
            probability = occurrence_model.predict_proba(occ_features[occurrence_columns])[0, 1]
            st.metric("Yarınki Tahmini Risk", f"%{probability * 100:.1f}")
            st.caption(f"Tahmin edilen gün: {target_date.date()}")

        if latest_row is not None:
            with col1:
                st.subheader("🎯 Risk Seviyesi")
                classifier = load_classifier()
                scaler = load_scaler()
                features = latest_row[['avg_brightness', 'avg_frp', 'max_frp']].values.reshape(1, -1)
                features_scaled = scaler.transform(features)
                prediction = classifier.predict(features_scaled)[0]
                st.metric("Tahmini Risk", prediction)
                st.caption(f"Bu bölgede son sıcak nokta tespiti: {latest_row['date']}")

            with col2:
                st.subheader("📈 Zaman Serisi Tahmini")
                data_with_lag = load_data_with_lag_features()
                grid_lag_data = data_with_lag[data_with_lag['grid_id'] == selected_grid].sort_values('date')

                if len(grid_lag_data) > 0:
                    latest_lag_row = grid_lag_data.iloc[-1]

                    if pd.isna(latest_lag_row['lag_1_fire_count']):
                        st.info("Bu bölge için henüz yeterli geçmiş veri yok (en az 2 gün gerekiyor).")
                    else:
                        forecaster = load_forecaster()
                        ts_features = latest_lag_row[FORECASTER_FEATURES].values.reshape(1, -1)
                        ts_prediction = forecaster.predict(ts_features)[0]
                        st.metric("Tahmini Sonraki Gün", f"{ts_prediction:.1f} sıcak nokta")
                        st.caption(f"Dünkü değer: {latest_lag_row['lag_1_fire_count']:.0f}")
                else:
                    st.info("Bu bölge için zaman serisi verisi bulunamadı.")

            with col3:
                st.subheader("🚨 Anomali Durumu")
                anomaly_data = load_data_with_anomaly_features()
                grid_anomaly_data = anomaly_data[anomaly_data['grid_id'] == selected_grid].sort_values('date')

                if len(grid_anomaly_data) > 0:
                    latest_anomaly_row = grid_anomaly_data.iloc[-1]

                    if pd.isna(latest_anomaly_row['z_score']):
                        st.info("Bu bölge için henüz yeterli geçmiş veri yok (en az 3 gün gerekiyor).")
                    else:
                        z_score = latest_anomaly_row['z_score']
                        is_anomaly = latest_anomaly_row['is_anomaly']

                        if is_anomaly:
                            st.error(f"⚠️ Anomali tespit edildi! (z-score: {z_score:.2f})")
                        else:
                            st.success(f"Normal aralıkta (z-score: {z_score:.2f})")
                else:
                    st.info("Bu bölge için anomali verisi bulunamadı.")
        else:
            st.info("Bu bölge için hiç sıcak nokta tespiti kaydı yok -- sadece yangın olasılığı paneli gösterilebiliyor.")