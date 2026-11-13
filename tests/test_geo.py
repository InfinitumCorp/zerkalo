"""Cyber geo-routing: synthetic .dat fixtures, the saved switch, and an unchanged forecast when off."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from zerkalo.cli import main
from zerkalo.config import Scenario, load_scenario
from zerkalo.geo import (
    announce,
    parse_geoip,
    parse_geosite,
    GeoDB,
)
from zerkalo.geo_fetch import ChecksumMismatch, install_datasets, load_lock, parse_sha256sum, sha256_bytes
from zerkalo.llm import MockLLM
from zerkalo.settings import Settings, geo_routing_enabled, save_settings, set_cli_geo_override
from zerkalo.simulation import Simulation

ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLE = ROOT / "examples" / "weekend-pedestrian" / "scenario.yaml"
LOCK = ROOT / "data" / "geo.lock.json"

SCENARIO_KEYS = [
    "name", "proposition", "goal", "context", "seed_files", "segments", "premise", "variants",
    "n_agents", "n_llm_agents", "rounds", "active_fraction", "language", "random_seed", "models",
]

GEOIP_US = bytes.fromhex("0a0e0a02555312080a040a0000001008")
GEOSITE_NEWS = bytes.fromhex("0a170a046e657773120f0802120b6578616d706c652e636f6d")
GEOSITE_NEWS_ATTR = bytes.fromhex("0a1c0a046e65777312140802120b6578616d706c652e636f6d1a030a0178")


def run(coro):
    return asyncio.run(coro)


def test_hardcoded_protobuf_samples():
    ips = parse_geoip(GEOIP_US)
    assert ips.country("10.1.2.3") == "US"
    assert ips.country("11.0.0.1") is None
    sites = parse_geosite(GEOSITE_NEWS)
    assert sites.categories("www.example.com") == ["news"]
    assert sites.categories("example.com") == ["news"]
    assert sites.categories("notexample.com") == []
    assert sites.categories("example.com.evil") == []
    # an ignored attribute does not change the category
    assert parse_geosite(GEOSITE_NEWS_ATTR).categories("a.example.com") == ["news"]


def test_truncated_protobuf_is_rejected():
    with pytest.raises(ValueError, match="truncated"):
        parse_geoip(bytes.fromhex("0a05ffff"))


def test_fixture_files_cover_prefix_domain_and_regex():
    db = GeoDB.load(FIXTURES)
    assert db.country("10.1.2.9") == "CN"  # /24 beats the /8
    assert db.country("10.2.0.1") == "US"
    assert db.country("10.9.0.1") == "US"  # reverse_match entry is not a country
    assert db.country("9.9.9.9") is None
    assert db.country("2001:db8::1") == "SE"
    assert db.country("2001:db9::1") is None
    assert db.categories("www.example.com") == ["news"]
    assert db.categories("EXACT.test") == ["exact"]
    assert db.categories("www.exact.test") == []
    assert db.categories("myads.example") == ["ads"]
    assert db.categories("ads.example.com") == ["ads", "news"]
    assert db.categories("x.test") == ["rx"]
    assert db.categories("xx.test") == []
    assert db.classify("10.1.2.9")["country"] == "CN"
    assert db.classify("www.example.com")["categories"] == ["news"]
    assert db.classify("not a host") is None


def test_lockfile_pins_v2fly_release_assets():
    datasets = {item["name"]: item for item in load_lock(LOCK)}
    geoip = datasets["geoip.dat"]
    geosite = datasets["geosite.dat"]
    assert geoip["tag"] == "202611080027"
    assert geoip["sha256"] == "eedb24b3b9fcba27e06b92c0428dab90d9496083091ae4cf39e11387fd99d0e2"
    assert geoip["url"] == "https://github.com/v2fly/geoip/releases/download/202611080027/geoip.dat"
    assert geoip["sha256sum_url"].endswith("/geoip.dat.sha256sum")
    assert geoip["license"] == "CC-BY-SA-4.0"
    assert geosite["tag"] == "20261108011013"
    assert geosite["sha256"] == "dc4192dd1b644a38276ed06136d26e729fa674d25785adb38c87e040e92376b2"
    assert geosite["source_asset"] == "dlc.dat"
    assert geosite["url"] == (
        "https://github.com/v2fly/domain-list-community/releases/download/20261108011013/dlc.dat"
    )
    assert geosite["license"] == "MIT"
    assert set(datasets) == {"geoip.dat", "geosite.dat"}
    assert "data/geo/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_official_checksum_text_parses():
    assert parse_sha256sum(
        "eedb24b3b9fcba27e06b92c0428dab90d9496083091ae4cf39e11387fd99d0e2  geoip.dat\n"
    ) == "eedb24b3b9fcba27e06b92c0428dab90d9496083091ae4cf39e11387fd99d0e2"


def _lock(tmp_path: Path, name: str, digest: str) -> Path:
    path = tmp_path / "geo.lock.json"
    path.write_text(json.dumps({
        "datasets": [{
            "name": name,
            "tag": "test",
            "url": f"https://example.test/{name}",
            "sha256sum_url": f"https://example.test/{name}.sha256sum",
            "sha256": digest,
        }]
    }), encoding="utf-8")
    return path


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self, _n=-1):
        data, self._data = self._data, b""
        return data

    def close(self):
        pass


def test_fetch_installs_when_checksums_match(tmp_path):
    payload = b"synthetic-geo"
    digest = sha256_bytes(payload)
    lock = _lock(tmp_path, "geoip.dat", digest)

    def opener(url: str):
        if url.endswith(".sha256sum"):
            return _Body(f"{digest}  geoip.dat\n".encode())
        return _Body(payload)

    installed = install_datasets(lock, tmp_path / "geo", opener)
    assert installed == [("geoip.dat", digest)]
    assert (tmp_path / "geo" / "geoip.dat").read_bytes() == payload
    assert not (tmp_path / "geo" / ".geoip.dat.partial").exists()


def test_fetch_stops_when_official_sum_differs(tmp_path):
    digest = "ab" * 32
    lock = _lock(tmp_path, "geoip.dat", "cd" * 32)
    calls = []

    def opener(url: str):
        calls.append(url)
        return _Body(f"{digest}  geoip.dat\n".encode())

    with pytest.raises(ChecksumMismatch, match="checksum mismatch") as caught:
        install_datasets(lock, tmp_path / "geo", opener)
    assert caught.value.official == digest
    assert caught.value.downloaded is None
    assert calls == ["https://example.test/geoip.dat.sha256sum"]
    assert not (tmp_path / "geo" / "geoip.dat").exists()


def test_fetch_stops_when_download_differs(tmp_path):
    digest = sha256_bytes(b"good")
    lock = _lock(tmp_path, "geosite.dat", digest)

    def opener(url: str):
        if url.endswith(".sha256sum"):
            return _Body(f"{digest}  dlc.dat\n".encode())
        return _Body(b"tampered")

    with pytest.raises(ChecksumMismatch, match="downloaded") as caught:
        install_datasets(lock, tmp_path / "geo", opener)
    assert caught.value.downloaded == sha256_bytes(b"tampered")
    assert not (tmp_path / "geo" / "geosite.dat").exists()


def test_switch_defaults_off_and_persists(monkeypatch):
    assert geo_routing_enabled() is False
    monkeypatch.setenv("INFINITUM_GEO_ENABLED", "0")
    save_settings(Settings(geo_routing=True))
    assert geo_routing_enabled() is False
    monkeypatch.setenv("INFINITUM_GEO_ENABLED", "1")
    assert geo_routing_enabled() is True
    monkeypatch.delenv("INFINITUM_GEO_ENABLED")
    assert geo_routing_enabled() is True  # saved file
    bare = Scenario(name="n", proposition="p")
    assert "geo_routing" not in bare.model_fields_set
    forced = Scenario(name="n", proposition="p", geo_routing=False)
    assert geo_routing_enabled(forced) is False
    set_cli_geo_override(True)
    assert geo_routing_enabled(forced) is True
    set_cli_geo_override(None)


def test_bad_env_value_is_rejected(monkeypatch):
    monkeypatch.setenv("INFINITUM_GEO_ENABLED", "maybe")
    with pytest.raises(Exception, match="INFINITUM_GEO_ENABLED"):
        geo_routing_enabled()


def test_settings_command_persists(capsys):
    assert main(["settings"]) == 0
    assert "cyber geo-routing: off" in capsys.readouterr().out
    assert main(["settings", "--geo-routing"]) == 0
    assert "cyber geo-routing: on" in capsys.readouterr().out
    assert geo_routing_enabled() is True
    assert main(["settings", "--no-geo-routing"]) == 0
    assert geo_routing_enabled() is False


def test_off_forecast_is_unchanged_and_on_only_annotates(tmp_path, monkeypatch):
    monkeypatch.setenv("ZERKALO_GEO_DIR", str(FIXTURES))
    sc = load_scenario(EXAMPLE)
    sc.n_agents, sc.n_llm_agents, sc.rounds = 24, 0, 2
    sc.proposition = "Residents mention www.example.com and 10.1.2.9"
    set_cli_geo_override(False)
    off = run(Simulation(sc, MockLLM()).run())
    off_dir = off.save(tmp_path / "off")
    set_cli_geo_override(True)
    on = run(Simulation(sc, MockLLM()).run())
    on_dir = on.save(tmp_path / "on")

    assert off.timeline == on.timeline
    assert off.summary() == on.summary()
    assert [a.stance for a in off.agents] == [a.stance for a in on.agents]
    assert not (off_dir / "geo.json").exists()
    saved = json.loads((off_dir / "scenario.json").read_text(encoding="utf-8"))
    assert list(saved) == SCENARIO_KEYS
    assert "geo_routing" not in (off_dir / "summary.json").read_text(encoding="utf-8")
    annotated = json.loads((on_dir / "geo.json").read_text(encoding="utf-8"))
    assert annotated["policy"] == "cyber-geo-routing"
    by_value = {item["value"]: item for item in annotated["matches"]}
    assert by_value["10.1.2.9"]["country"] == "CN"
    assert by_value["www.example.com"]["categories"] == ["news"]
    # the saved forecast files, aside from the annotation, match
    for name in ("summary.json", "timeline.json", "posts.jsonl", "agents.json"):
        assert (off_dir / name).read_bytes() == (on_dir / name).read_bytes()


def test_missing_data_does_not_change_the_run(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("ZERKALO_GEO_DIR", str(tmp_path / "empty"))
    sc = load_scenario(EXAMPLE)
    sc.n_agents, sc.n_llm_agents, sc.rounds = 16, 0, 1
    set_cli_geo_override(False)
    off = run(Simulation(sc, MockLLM()).run())
    set_cli_geo_override(True)
    with caplog.at_level("WARNING"):
        on = run(Simulation(sc, MockLLM()).run())
        on_dir = on.save(tmp_path / "on")
    assert off.timeline == on.timeline
    payload = json.loads((on_dir / "geo.json").read_text(encoding="utf-8"))
    assert "error" in payload
    assert "cyber geo-routing" in caplog.text


def test_cli_flag_and_saved_switch(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ZERKALO_GEO_DIR", str(FIXTURES))
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        "name: geo\nproposition: 'See www.example.com from 10.1.2.9'\n"
        "n_agents: 12\nn_llm_agents: 0\nrounds: 1\nrandom_seed: 1\n",
        encoding="utf-8",
    )
    out = tmp_path / "runs"
    assert main(["run", str(scenario), "--mock", "--out", str(out)]) == 0
    first = capsys.readouterr().out
    assert "cyber geo-routing" not in first
    assert list(out.rglob("geo.json")) == []

    assert main(["run", str(scenario), "--mock", "--geo-routing", "--out", str(out)]) == 0
    second = capsys.readouterr().out
    assert "cyber geo-routing: annotated" in second
    written = list(out.rglob("geo.json"))
    assert len(written) == 1
    body = json.loads(written[0].read_text(encoding="utf-8"))
    assert any(item["country"] == "CN" for item in body["matches"])

    save_settings(Settings(geo_routing=True))
    assert main(["run", str(scenario), "--mock", "--no-geo-routing", "--out", str(out)]) == 0
    third = capsys.readouterr().out
    assert "cyber geo-routing" not in third


def test_cli_rejects_bad_env(monkeypatch, tmp_path):
    monkeypatch.setenv("INFINITUM_GEO_ENABLED", "maybe")
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("name: geo\nproposition: p\n", encoding="utf-8")
    assert main(["run", str(scenario), "--mock", "--rounds", "1", "--out", str(tmp_path / "out")]) == 2


def test_init_template_mentions_the_switch(tmp_path):
    assert main(["init", str(tmp_path / "fresh")]) == 0
    text = (tmp_path / "fresh" / "scenario.yaml").read_text(encoding="utf-8")
    assert "geo_routing: false" in text
    assert load_scenario(tmp_path / "fresh" / "scenario.yaml").geo_routing is False
    assert "geo_routing" not in load_scenario(tmp_path / "fresh" / "scenario.yaml").model_fields_set
