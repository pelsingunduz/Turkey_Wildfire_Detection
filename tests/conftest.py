import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
    """Proje script'leri veri/model yollarını kök dizine göre okuyor
    (data/processed/..., outputs/models/...); testler de aynı yerden çalışsın."""
    monkeypatch.chdir(REPO_ROOT)
