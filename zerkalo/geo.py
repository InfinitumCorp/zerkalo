"""v2fly GeoIPList / GeoSiteList loader and cyber geo-routing annotations.

The `.dat` files are protobuf messages, parsed here with the standard library.
Nothing is loaded, logged, or written unless the geo-routing switch is on.
Lookups do not change how a simulation spreads a message.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import sys
from pathlib import Path

from .settings import geo_routing_enabled

log = logging.getLogger("zerkalo.geo")

POLICY = "cyber-geo-routing"
_TOKEN = re.compile(r"[A-Za-z0-9:.-]{2,}")


def geo_data_dir() -> Path:
    override = os.environ.get("ZERKALO_GEO_DIR")
    if override:
        return Path(override)
    repo = Path(__file__).resolve().parents[1] / "data" / "geo"
    cwd = Path.cwd() / "data" / "geo"
    if (cwd / "geoip.dat").is_file() and (cwd / "geosite.dat").is_file():
        return cwd
    return repo


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = 0
    out = 0
    n = len(buf)
    while True:
        if i >= n or shift > 63:
            raise ValueError("truncated protobuf varint")
        byte = buf[i]
        i += 1
        out |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return out, i
        shift += 7


def _iter_fields(buf: bytes):
    i = 0
    n = len(buf)
    while i < n:
        tag, i = _varint(buf, i)
        field, wire = tag >> 3, tag & 7
        if field <= 0:
            raise ValueError("invalid protobuf field number")
        if wire == 0:
            value, i = _varint(buf, i)
            yield field, value
        elif wire == 2:
            length, i = _varint(buf, i)
            end = i + length
            if end > n:
                raise ValueError("truncated protobuf length")
            yield field, buf[i:end]
            i = end
        elif wire == 5:
            if i + 4 > n:
                raise ValueError("truncated protobuf fixed32")
            i += 4
        elif wire == 1:
            if i + 8 > n:
                raise ValueError("truncated protobuf fixed64")
            i += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


class IPIndex:
    """Longest-prefix match from GeoIP CIDR entries. `reverse_match` lists are omitted."""

    def __init__(self) -> None:
        self._buckets: dict[int, dict[int, dict[int, str]]] = {4: {}, 6: {}}

    def add(self, raw_ip: bytes, prefix: int, country: str) -> None:
        version = 4 if len(raw_ip) == 4 else 6 if len(raw_ip) == 16 else 0
        if version == 0 or not country:
            return
        bits = 32 if version == 4 else 128
        if prefix < 0 or prefix > bits:
            return
        ip_i = int.from_bytes(raw_ip, "big")
        net = 0 if prefix == 0 else (ip_i >> (bits - prefix)) << (bits - prefix)
        self._buckets[version].setdefault(prefix, {}).setdefault(net, country)

    def country(self, ip: str) -> str | None:
        addr = ipaddress.ip_address(ip)
        bits = addr.max_prefixlen
        ip_i = int(addr)
        buckets = self._buckets[addr.version]
        for prefix in range(bits, -1, -1):
            bucket = buckets.get(prefix)
            if not bucket:
                continue
            net = 0 if prefix == 0 else (ip_i >> (bits - prefix)) << (bits - prefix)
            found = bucket.get(net)
            if found is not None:
                return found
        return None


class SiteIndex:
    """Domain → category codes from a GeoSiteList.

    Full is an exact host. Domain matches that host and its subdomains.
    Plain is a substring. Regex uses the pattern as published.
    """

    def __init__(self) -> None:
        self.full: dict[str, set[str]] = {}
        self.suffix: dict[str, set[str]] = {}
        self.plain: list[tuple[str, str]] = []
        self.regex: list[tuple[re.Pattern[str], str]] = []

    def add(self, category: str, typ: int, value: str) -> None:
        value = value.strip().lower().rstrip(".")
        if not category or not value:
            return
        if typ == 3:
            self.full.setdefault(value, set()).add(category)
        elif typ == 2:
            self.suffix.setdefault(value, set()).add(category)
        elif typ == 1:
            try:
                self.regex.append((re.compile(value), category))
            except re.error:
                log.warning("skipping invalid geosite regex in %s", category)
        else:
            self.plain.append((value, category))

    def categories(self, domain: str) -> list[str]:
        domain = domain.strip().lower().rstrip(".")
        if not domain:
            return []
        found: set[str] = set(self.full.get(domain, ()))
        parts = domain.split(".")
        for i in range(len(parts)):
            found.update(self.suffix.get(".".join(parts[i:]), ()))
        for needle, category in self.plain:
            if needle in domain:
                found.add(category)
        for pattern, category in self.regex:
            if pattern.search(domain):
                found.add(category)
        return sorted(found)


def parse_geoip(data: bytes) -> IPIndex:
    index = IPIndex()
    for field, value in _iter_fields(data):
        if field != 1 or not isinstance(value, (bytes, bytearray)):
            continue
        country = ""
        cidrs: list[tuple[bytes, int]] = []
        reverse = False
        for sub, inner in _iter_fields(value):
            if sub == 1 and isinstance(inner, (bytes, bytearray)):
                country = inner.decode("utf-8", "replace").strip()
            elif sub == 2 and isinstance(inner, (bytes, bytearray)):
                raw = b""
                prefix = None
                for leaf, item in _iter_fields(inner):
                    if leaf == 1 and isinstance(item, (bytes, bytearray)):
                        raw = bytes(item)
                    elif leaf == 2 and isinstance(item, int):
                        prefix = item
                if raw and prefix is not None:
                    cidrs.append((raw, prefix))
            elif sub == 3 and isinstance(inner, int):
                reverse = bool(inner)
        if reverse or not country:
            continue
        for raw, prefix in cidrs:
            index.add(raw, prefix, country)
    return index


def parse_geosite(data: bytes) -> SiteIndex:
    index = SiteIndex()
    for field, value in _iter_fields(data):
        if field != 1 or not isinstance(value, (bytes, bytearray)):
            continue
        code = ""
        domains: list[tuple[int, str]] = []
        for sub, inner in _iter_fields(value):
            if sub == 1 and isinstance(inner, (bytes, bytearray)):
                code = inner.decode("utf-8", "replace").strip()
            elif sub == 2 and isinstance(inner, (bytes, bytearray)):
                typ = 0
                host = ""
                for leaf, item in _iter_fields(inner):
                    if leaf == 1 and isinstance(item, int):
                        typ = item
                    elif leaf == 2 and isinstance(item, (bytes, bytearray)):
                        host = item.decode("utf-8", "replace")
                domains.append((typ, host))
        if not code:
            continue
        for typ, host in domains:
            index.add(code, typ, host)
    return index


class GeoDB:
    def __init__(self, ips: IPIndex, sites: SiteIndex):
        self.ips = ips
        self.sites = sites

    @classmethod
    def load(cls, directory: Path | None = None) -> "GeoDB":
        directory = Path(directory) if directory is not None else geo_data_dir()
        ip_path = directory / "geoip.dat"
        site_path = directory / "geosite.dat"
        if not ip_path.is_file() or not site_path.is_file():
            raise FileNotFoundError(
                f"geo data not found in {directory}; run python scripts/fetch_geo_dat.py"
            )
        return cls(parse_geoip(ip_path.read_bytes()), parse_geosite(site_path.read_bytes()))

    def country(self, ip: str) -> str | None:
        return self.ips.country(ip)

    def categories(self, domain: str) -> list[str]:
        return self.sites.categories(domain)

    def classify(self, value: str) -> dict | None:
        """Cyber routing label for one IP or domain. None when the token is neither."""
        token = value.strip().strip(".")
        if not token:
            return None
        try:
            ipaddress.ip_address(token)
        except ValueError:
            if not _looks_like_domain(token):
                return None
            categories = self.categories(token)
            if not categories:
                return None
            return {"kind": "domain", "value": token.lower().rstrip("."), "country": None, "categories": categories}
        return {"kind": "ip", "value": token, "country": self.country(token), "categories": []}


def _looks_like_domain(token: str) -> bool:
    if ":" in token or token.count(".") < 1 or len(token) > 253:
        return False
    labels = token.split(".")
    if not labels[-1].isalpha():
        return False
    for label in labels:
        if not label or len(label) > 63 or label.startswith("-") or label.endswith("-"):
            return False
        if not all(ch.isalnum() or ch == "-" for ch in label):
            return False
    return True


def texts_for_run(scenario, posts) -> list[str]:
    parts = [scenario.name, scenario.proposition, scenario.context or ""]
    try:
        parts.append(scenario.seed_text())
    except (OSError, ValueError):
        pass
    for premise in scenario.all_premises():
        parts.append(premise.source)
        parts.append(premise.text)
    for post in posts:
        parts.append(getattr(post, "text", "") or "")
    return [p for p in parts if p]


def _matches(db: GeoDB, texts: list[str]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    found: list[dict] = []
    for text in texts:
        for token in _TOKEN.findall(text):
            item = db.classify(token)
            if item is None:
                continue
            key = (item["kind"], item["value"])
            if key in seen:
                continue
            seen.add(key)
            found.append(item)
    return found


def _write(out_dir: Path, texts: list[str]) -> None:
    out_dir = Path(out_dir)
    path = out_dir / "geo.json"
    try:
        db = GeoDB.load()
        matches = _matches(db, texts)
    except (OSError, ValueError) as exc:
        payload = {"enabled": True, "policy": POLICY, "error": str(exc)}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.warning("cyber geo-routing unavailable: %s", exc)
        return
    payload = {"enabled": True, "policy": POLICY, "matches": matches}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("cyber geo-routing annotated %s match(es) in %s", len(matches), out_dir)


def maybe_annotate(out_dir: Path, scenario, posts) -> None:
    """Write `geo.json` only when cyber geo-routing is on. Off leaves the run untouched."""
    if not geo_routing_enabled(scenario):
        return
    _write(Path(out_dir), texts_for_run(scenario, posts))


def annotate_saved_run(out_dir: Path, scenario=None) -> None:
    if not geo_routing_enabled(scenario):
        return
    out_dir = Path(out_dir)
    texts: list[str] = []
    posts_path = out_dir / "posts.jsonl"
    if posts_path.is_file():
        for line in posts_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                texts.append(str(item.get("text") or ""))
    for name in ("scenario.json", "summary.json", "world.json"):
        path = out_dir / name
        if path.is_file():
            texts.append(path.read_text(encoding="utf-8"))
    _write(out_dir, texts)


def announce(out: Path, scenario=None) -> None:
    """Print the annotation summary for a run directory. Silent when the switch is off."""
    if not geo_routing_enabled(scenario):
        return
    out = Path(out)
    direct = out / "geo.json"
    if direct.is_file():
        paths = [direct]
    elif out.is_dir():
        paths = [child / "geo.json" for child in sorted(out.iterdir()) if (child / "geo.json").is_file()]
    else:
        paths = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        label = path.parent.name if path.parent != out else ""
        prefix = f"cyber geo-routing ({label}): " if label else "cyber geo-routing: "
        if data.get("error"):
            print(prefix + str(data["error"]), file=sys.stderr)
        else:
            print(prefix + f"annotated {len(data.get('matches') or [])} match(es)")
