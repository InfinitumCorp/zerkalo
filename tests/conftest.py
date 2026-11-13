"""Keep the suite on the default (geo-routing off) unless a test opts in."""

from __future__ import annotations

import pytest

from zerkalo.settings import set_cli_geo_override


@pytest.fixture(autouse=True)
def geo_routing_off(monkeypatch, tmp_path):
    monkeypatch.setenv("ZERKALO_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.delenv("INFINITUM_GEO_ENABLED", raising=False)
    monkeypatch.delenv("ZERKALO_GEO_DIR", raising=False)
    set_cli_geo_override(None)
    yield
    set_cli_geo_override(None)
