"""src/ingestion.py'deki boş veri ve geçersiz cevap kontrollerinin testleri."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import ingestion  # noqa: E402

HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
ROW = "37.1,28.4,330.1,0.4,0.4,2026-11-23,1050,N20,VIIRS,n,2.0NRT,290.2,5.1,D\n"


def write(directory, name, content):
    (directory / name).write_text(content)


def test_trailing_empty_runs_counts_only_latest_streak(tmp_path):
    write(tmp_path, "firms_2026-11-01_00-00.csv", HEADER)          # eski boş (seriyi bozan doluya kadar sayılmaz)
    write(tmp_path, "firms_2026-11-01_04-00.csv", HEADER + ROW)    # dolu
    write(tmp_path, "firms_2026-11-01_08-00.csv", HEADER)
    write(tmp_path, "firms_2026-11-01_12-00.csv", HEADER)
    write(tmp_path, "firms_hist_2026-10-01_2026-10-05.csv", HEADER)  # backfill dosyası, sayılmamalı

    assert ingestion.count_trailing_empty_runs(str(tmp_path / "firms_[0-9]*.csv")) == 2


def test_trailing_empty_runs_zero_when_latest_has_data(tmp_path):
    write(tmp_path, "firms_2026-11-01_00-00.csv", HEADER)
    write(tmp_path, "firms_2026-11-01_04-00.csv", HEADER + ROW)
    assert ingestion.count_trailing_empty_runs(str(tmp_path / "firms_[0-9]*.csv")) == 0


def test_is_valid_csv_rejects_error_bodies():
    assert ingestion.is_valid_csv(HEADER + ROW)
    assert ingestion.is_valid_csv(HEADER)  # geçerli ama boş: ayrı kontrolle yakalanıyor
    assert not ingestion.is_valid_csv("Invalid MAP_KEY.")
    assert not ingestion.is_valid_csv("")
