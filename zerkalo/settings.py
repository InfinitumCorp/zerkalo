"""Saved options. Cyber geo-routing is off unless a person turns it on."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("zerkalo.settings")

_ON = {"1", "true", "yes", "on"}
_OFF = {"0", "false", "no", "off"}
_cli_override: bool | None = None


class GeoSettingsError(ValueError):
    """INFINITUM_GEO_ENABLED is set to a value this program does not understand."""


@dataclass
class Settings:
    geo_routing: bool = False


def settings_path() -> Path:
    override = os.environ.get("ZERKALO_SETTINGS_PATH")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "zerkalo" / "settings.json"


def set_cli_geo_override(value: bool | None) -> None:
    """One-run override from `--geo-routing` / `--no-geo-routing`. None clears it."""
    global _cli_override
    _cli_override = value


def load_settings(*, strict: bool = False) -> Settings:
    path = settings_path()
    if not path.is_file():
        return Settings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("settings file must be a JSON object")
        if "geo_routing" not in data:
            return Settings()
        value = data["geo_routing"]
        if not isinstance(value, bool):
            raise ValueError("geo_routing must be true or false")
        return Settings(geo_routing=value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        if strict:
            raise GeoSettingsError(f"cannot read settings {path}: {exc}") from exc
        log.warning("cyber geo-routing settings unreadable (%s); staying off", exc)
        return Settings()


def save_settings(settings: Settings) -> Path:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"geo_routing": settings.geo_routing}, indent=2) + "\n", encoding="utf-8")
    return path


def env_geo_override() -> bool | None:
    raw = os.environ.get("INFINITUM_GEO_ENABLED")
    if raw is None or raw.strip() == "":
        return None
    value = raw.strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    raise GeoSettingsError(
        "INFINITUM_GEO_ENABLED must be on or off (1, true, yes, on / 0, false, no, off)"
    )


def geo_routing_enabled(scenario=None) -> bool:
    """Resolve the cyber geo-routing switch. Default is off.

    Precedence: CLI flag, then INFINITUM_GEO_ENABLED, then an explicit
    `geo_routing` value on the scenario, then the saved settings file.
    """
    if _cli_override is not None:
        return _cli_override
    env = env_geo_override()
    if env is not None:
        return env
    if scenario is not None and "geo_routing" in getattr(scenario, "model_fields_set", ()):
        return bool(scenario.geo_routing)
    return load_settings().geo_routing
