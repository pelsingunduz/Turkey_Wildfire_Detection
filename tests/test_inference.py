"""src/inference.py testleri.

Repoda commit'li gerçek veri ve modelleri kullanır; hava durumu için ağa
çıkmaz (fetch_weather=False), bu yüzden çevrimdışı da çalışır.

    pytest -q
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import inference
from src.features import build_latest_occurrence_features
from src.models.occurrence import FEATURE_COLUMNS as OCCURRENCE_FEATURES

SEVERITY_LABELS = {'düşük', 'orta', 'yüksek'}


@pytest.fixture(scope="module")
def catalog():
    return inference.load_grid_catalog()


@pytest.fixture(scope="module")
def active_grid_id():
    """En son tespitin yapıldığı hücrelerden biri (dört modelin hepsi dolu olmalı)."""
    summary = pd.read_csv(inference.DAILY_SUMMARY_PATH)
    counts = summary['grid_id'].value_counts()
    busiest = counts[counts >= 10].index
    catalog_ids = set(inference.load_grid_catalog()['grid_id'])
    return next(g for g in busiest if g in catalog_ids)


def test_normalize_text_handles_turkish_characters():
    assert inference.normalize_text("MUĞLA") == "mugla"
    assert inference.normalize_text("  İzmir ") == "izmir"
    assert inference.normalize_text("Çanakkale") == "canakkale"


def test_catalog_has_names_and_no_fully_unknown_cells(catalog):
    assert not catalog.empty
    assert catalog['grid_id'].is_unique
    fully_unknown = (catalog['province'] == inference.UNKNOWN) & (catalog['district'] == inference.UNKNOWN)
    assert not fully_unknown.any()


@pytest.mark.parametrize("query", ["Muğla", "mugla", "MUGLA"])
def test_find_regions_is_accent_and_case_insensitive(query):
    results = inference.find_regions(query)
    assert results, "Muğla için en az bir hücre bulunmalı"
    assert all(r['province'] == 'Muğla' for r in results)


def test_find_regions_matches_district_and_respects_limit():
    results = inference.find_regions("Marmaris", limit=2)
    assert 0 < len(results) <= 2
    assert all('Marmaris' in r['display_name'] for r in results)


def test_find_regions_unknown_returns_empty():
    assert inference.find_regions("Atlantis") == []
    assert inference.find_regions("   ") == []


def test_region_report_for_active_grid(active_grid_id):
    report = inference.get_region_report(active_grid_id, fetch_weather=False)

    assert 0.0 <= report['fire_probability'] <= 1.0
    assert report['weather_used'] is False
    assert report['severity'] in SEVERITY_LABELS
    assert report['last_detection'] is not None
    assert report['last_detection']['days_before_data_end'] >= 0
    assert report['next_day_hotspot_forecast'] is None or report['next_day_hotspot_forecast'] >= 0
    if report['anomaly'] is not None:
        assert isinstance(report['anomaly']['is_anomaly'], bool)
    assert pd.Timestamp(report['target_date']) == pd.Timestamp(report['data_end_date']) + pd.Timedelta(days=1)


def test_region_report_is_json_serializable(active_grid_id):
    """Rapor doğrudan bir LLM'e / HTTP cevabına verilebilmeli (numpy tipi kalmamalı)."""
    json.dumps(inference.get_region_report(active_grid_id, fetch_weather=False), ensure_ascii=False)


def test_region_report_unknown_grid_raises():
    with pytest.raises(KeyError):
        inference.get_region_report("0.0_0.0", fetch_weather=False)


def test_missing_weather_does_not_crash_occurrence_model(active_grid_id):
    """Hava durumu API'si cevap vermezse olasılık modeli yine çalışmalı
    (eskiden None değerli sütunlar XGBoost'ta ValueError'a yol açıyordu)."""
    report = inference.get_region_report(active_grid_id, fetch_weather=False)
    assert all(v is None for v in report['weather'].values())
    assert not np.isnan(report['fire_probability'])


def test_vectorized_features_match_single_grid_features(catalog):
    """Toplu (tüm hücreler) feature hesaplaması, dashboard'un kullandığı tek
    hücreli fonksiyonla birebir aynı sonucu vermeli."""
    all_features, target_date = inference.build_all_occurrence_features(fetch_weather=False)
    all_features = all_features.set_index('grid_id')
    summary = inference._load_daily_summary()

    sample = catalog.sample(40, random_state=0)
    for _, row in sample.iterrows():
        single, single_target = build_latest_occurrence_features(
            summary, row['grid_id'], row['grid_lat'], row['grid_lon'], fetch_weather=False
        )
        assert single_target == target_date
        expected = single[OCCURRENCE_FEATURES].astype(float).iloc[0]
        actual = all_features.loc[row['grid_id'], OCCURRENCE_FEATURES].astype(float)
        pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_top_risk_regions_sorted_and_limited():
    top = inference.get_top_risk_regions(n=5, fetch_weather=False)
    assert len(top) == 5
    probabilities = [r['fire_probability'] for r in top]
    assert probabilities == sorted(probabilities, reverse=True)


def test_top_risk_regions_matches_region_report():
    """Sıralamadaki olasılık, aynı hücrenin ayrıntılı raporundakiyle aynı olmalı."""
    top = inference.get_top_risk_regions(n=1, fetch_weather=False)[0]
    report = inference.get_region_report(top['grid_id'], fetch_weather=False)
    assert top['fire_probability'] == pytest.approx(report['fire_probability'])


def test_top_risk_regions_province_filter():
    top = inference.get_top_risk_regions(n=3, fetch_weather=False, province="antalya")
    assert top and all(r['province'] == 'Antalya' for r in top)
