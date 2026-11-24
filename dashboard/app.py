"""
Türkiye Orman Yangını Erken Tespit Dashboard'u

Harita üzerinde bir grid hücresine tıklayarak, o bölge için dört modelin
(yangın olasılığı, sınıflandırma, zaman serisi, anomali tespiti) çıktısını
görebilirsiniz.
"""

import streamlit as st
import pandas as pd
import folium
import sys
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Streamlit Cloud'da .env dosyası yok -- key'ler yerine "Secrets" mekanizması
# kullanılıyor (st.secrets). Yerelde .env'den zaten okunuyor; bulutta da
# aynı os.environ üzerinden erişilebilmesi için buraya köprülüyoruz.
# (Yerel geliştirmede secrets.toml olmadığı için st.secrets erişimi hata
# verebilir -- bu yüzden try/except ile güvenli hale getiriyoruz.)
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

sys.path.append(os.path.abspath('src'))
from inference import load_grid_catalog, get_region_report # type: ignore
from streamlit_folium import st_folium

@st.cache_resource
def get_openai_client():
    """OpenAI istemcisini bir kere oluşturur (OPENAI_API_KEY ortam değişkeninden
    otomatik okunur)."""
    return OpenAI()


@st.cache_data
def generate_llm_explanation(region_name, target_date_str, probability_pct,
                              severity, ts_prediction, z_score, is_anomaly):
    """Sonuçları OpenAI ile doğal, sade bir Türkçe açıklamaya çevirir.

    @st.cache_data, aynı girdi kombinasyonu (aynı bölge + aynı gün + aynı
    sayılar) için sonucu önbellekte tutar -- aynı bölgeye aynı gün tekrar
    tıklanırsa API'ye tekrar istek gitmez, hem hızlı hem ucuz olur."""
    client = get_openai_client()

    facts = (
        f"- Bölge: {region_name}\n"
        f"- Tahmin edilen gün: {target_date_str}\n"
        f"- Yangın olasılığı: %{probability_pct:.1f}\n"
        f"- Şiddet sınıfı (varsa, en son tespite göre): {severity or 'veri yok'}\n"
        f"- Zaman serisi tahmini (varsa): "
        f"{f'{ts_prediction:.1f} sıcak nokta' if ts_prediction is not None else 'veri yok'}\n"
        f"- Anomali durumu (varsa): "
        f"{(('anomali' if is_anomaly else 'normal') + f' (z-score: {z_score:.2f})') if z_score is not None else 'veri yok'}"
    )

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": (
                "Sen bir orman yangını erken uyarı sisteminde, teknik bilgisi "
                "olmayan kullanıcılara sonuçları açıklayan bir asistansın. "
                "Sana verilen sayısal tahminleri 2-4 cümlelik sade, doğal "
                "Türkçe ile açıkla. Abartma, kesinlik iddia etme (bunlar "
                "tahmin modelleri, kesin bilgi değil). 'Veri yok' olan "
                "kısımlardan hiç bahsetme, sadece mevcut olanları anlat."
            )},
            {"role": "user", "content": facts},
        ],
        max_completion_tokens=200,
        temperature=0.4,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("LLM boş cevap döndürdü")
    return content


st.set_page_config(page_title="Türkiye Orman Yangını Tespit Sistemi", layout="wide")


def load_css(path="dashboard/style.css"):
    """Görsel stil dosyasını okuyup enjekte eder (bkz. dashboard/style.css)."""
    with open(path) as f:
        return f.read()


st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)
st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)
st.title("🔥 Türkiye Orman Yangını Erken Tespit Sistemi")

with st.expander("ℹ️ Bu sistem ne yapıyor? (açıklama için tıklayın)", expanded=True):
    st.markdown("""
    Bu sistem, NASA'nın **FIRMS** uydu verisini kullanarak Türkiye genelinde,
    ~27km × 21km'lik hücrelere (grid) bölünmüş bölgeler için orman yangını
    riskini **dört farklı açıdan** tahmin eder:

    - 🔥 **Yangın Olasılığı** — Seçtiğiniz bölgede *yarın* en az bir yangın
      tespit edilme ihtimali (yüzde olarak). Geçmiş aktiviteye ve mevsime
      (yaz ayları daha riskli) dayanır.
    - 🎯 **Risk Seviyesi** — Bölgede *daha önce* tespit edilmiş en güncel
      yangının ne kadar ciddi olduğu (düşük / orta / yüksek).
    - 📈 **Zaman Serisi Tahmini** — Geçmiş verilere bakarak, bir sonraki gün
      beklenen sıcak nokta (hotspot) sayısı.
    - 🚨 **Anomali Durumu** — Bölgenin bugünkü aktivitesinin, kendi geçmiş
      ortalamasından ne kadar saptığı (istatistiksel olarak sıra dışı mı).

    **Nasıl kullanılır:** Haritanın üzerinde gezinin — kırmızı noktalar
    sadece fareniz üzerlerine geldiğinde görünür. İlgilendiğiniz bölgeye
    tıklayın, sonuçlar altta görünecek.

    ⚠️ **Bilinmesi gerekenler:** Bu bir portföy/öğrenme projesidir, resmi
    bir afet yönetim aracı değildir. Bir grid hücresi bazen birden fazla
    il/ilçeyi kapsayabildiği için bölge isimleri yaklaşıktır.
    """)


@st.cache_data
def load_grid_points():
    """Haritada gösterilecek grid hücreleri (il/ilçe isimleriyle).

    Hücre listesi ve isimlendirme src/inference.py'den geliyor (agent ve
    otomasyonlar da aynı listeyi kullanıyor). Burada yalnızca haritaya özgü
    bir görünüm tercihi uygulanıyor: aynı il/ilçeye birden fazla grid hücresi
    düşebildiği için (0.25° hücre, idari sınırdan küçük olabiliyor) haritada
    çakışma olmasın diye her isimden sadece bir tanesi gösteriliyor."""
    return load_grid_catalog().drop_duplicates(subset='display_name', keep='first')


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

        # Dört modelin hesaplaması src/inference.py'de (agent/otomasyonlarla ortak).
        report = get_region_report(selected_grid)
        target_date = pd.to_datetime(report['target_date'])
        probability = report['fire_probability']
        severity_prediction = report['severity']
        ts_prediction = report['next_day_hotspot_forecast']
        z_score = report['anomaly']['z_score'] if report['anomaly'] else None
        is_anomaly = report['anomaly']['is_anomaly'] if report['anomaly'] else None
        latest_row = report['last_detection']  # None: bu hücrede hiç tespit yok
        # Hava durumu alınamadıysa olasılık panelinde uyarı gösterilir (bkz. aşağısı)
        weather_missing = not report['weather_used']

        # --- Doğal dilde özet ---
        summary = (
            f"**{selected_display_name}** bölgesinde yarın (**{target_date.date()}**) "
            f"yangın çıkma ihtimali **%{probability * 100:.1f}**. "
        )
        if latest_row is not None:
            summary += (
                f"Bu bölgede en son **{latest_row['date']}** tarihinde bir tespit oldu ve "
                f"şiddeti **{severity_prediction}** olarak sınıflandırıldı. "
            )
            if ts_prediction is not None:
                summary += f"Yarın için tahmini **{ts_prediction:.1f}** sıcak nokta bekleniyor. "
            if z_score is not None:
                if is_anomaly:
                    summary += f"Bu, bölgenin normal seviyesinin **belirgin şekilde üzerinde** (z-score: {z_score:.2f})."
                else:
                    summary += f"Bu, bölgenin normal aktivite aralığında (z-score: {z_score:.2f})."
        else:
            summary += "Bu bölgede daha önce hiç sıcak nokta tespiti kaydedilmemiş, bu yüzden diğer üç panel için yeterli veri yok."

        try:
            llm_explanation = generate_llm_explanation(
                region_name=selected_display_name,
                target_date_str=str(target_date.date()),
                probability_pct=probability * 100,
                severity=severity_prediction,
                ts_prediction=ts_prediction,
                z_score=z_score,
                is_anomaly=is_anomaly,
            )
            st.info(f"🤖 {llm_explanation}")
        except Exception as e:
            st.info(summary)
            st.caption(f"(LLM açıklaması şu an alınamıyor, yukarıda şablon özet gösteriliyor. Hata: {e})")

        # Panellerin üst şeritleri için renkler -- keyfi değil, zaten var olan
        # kategorilerden türetiliyor (classifier.py'daki düşük/orta/yüksek,
        # anomaly.py'daki is_anomaly eşiği).
        SEVERITY_COLORS = {'düşük': '#4C8C6B', 'orta': '#E8590C', 'yüksek': '#C7361B'}
        severity_color = SEVERITY_COLORS.get(severity_prediction, '#E8590C')
        if z_score is not None:
            anomaly_color = '#C7361B' if is_anomaly else '#4C8C6B'
        else:
            anomaly_color = '#5A6660'

        # --- Panel gösterimi ---
        col0, col1, col2, col3 = st.columns(4)

        with col0:
            st.markdown('<div class="panel-accent" style="border-top-color:#E8590C;"></div>', unsafe_allow_html=True)
            st.subheader("🔥 Yangın Olasılığı")
            st.metric(
                "Yarınki Tahmini Risk", f"%{probability * 100:.1f}",
                help="Bu bölgede yarın en az bir sıcak nokta tespit edilme ihtimali. "
                     "Geçmiş aktivite ve mevsimsel örüntülere dayanır (yaz ayları daha riskli). "
                     "%0 kesinlikle olmayacağı, %100 kesinlikle olacağı anlamına gelmez."
            )
            st.caption(f"Tahmin edilen gün: {target_date.date()}")
            if weather_missing:
                st.warning(
                    "Yarının hava durumu tahmini şu an alınamadı; bu olasılık hava durumu "
                    "olmadan hesaplandı ve gerçek riski olduğundan düşük gösterebilir."
                )

        if latest_row is not None:
            with col1:
                st.markdown(f'<div class="panel-accent" style="border-top-color:{severity_color};"></div>', unsafe_allow_html=True)
                st.subheader("🎯 Risk Seviyesi")
                st.metric(
                    "Tahmini Risk", severity_prediction,
                    help="Bu bölgede EN SON tespit edilen yangının şiddeti (düşük/orta/yüksek). "
                         "Yeni bir tahmin değil, geçmişteki en güncel tespitin sınıflandırmasıdır."
                )
                st.caption(f"Bu bölgede son sıcak nokta tespiti: {latest_row['date']}")

            with col2:
                st.markdown('<div class="panel-accent" style="border-top-color:#4C8C6B;"></div>', unsafe_allow_html=True)
                st.subheader("📈 Zaman Serisi Tahmini")
                if ts_prediction is not None:
                    st.metric(
                        "Tahmini Sonraki Gün", f"{ts_prediction:.1f} sıcak nokta",
                        help="Geçmiş sıcak nokta sayılarına bakarak, bir sonraki gün için "
                             "beklenen sıcak nokta sayısı (regresyon modeli)."
                    )
                    st.caption(f"Dünkü değer: {report['previous_fire_count']:.0f}")
                else:
                    st.info("Bu bölge için henüz yeterli geçmiş veri yok (en az 2 gün gerekiyor).")

            with col3:
                st.markdown(f'<div class="panel-accent" style="border-top-color:{anomaly_color};"></div>', unsafe_allow_html=True)
                st.subheader("🚨 Anomali Durumu")
                if z_score is not None:
                    if is_anomaly:
                        st.error(f"⚠️ Anomali tespit edildi! (z-score: {z_score:.2f})")
                    else:
                        st.success(f"Normal aralıkta (z-score: {z_score:.2f})")
                    st.caption(
                        "z-score, bugünkü aktivitenin bölgenin kendi geçmiş ortalamasından "
                        "kaç standart sapma uzakta olduğunu gösterir. |z| ≥ 3 anomali sayılır."
                    )
                else:
                    st.info("Bu bölge için henüz yeterli geçmiş veri yok (en az 3 gün gerekiyor).")
        else:
            st.info("Bu bölge için hiç sıcak nokta tespiti kaydı yok -- sadece yangın olasılığı paneli gösterilebiliyor.")