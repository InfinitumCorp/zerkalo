"""Download the pinned v2fly geoip.dat and geosite.dat files.

Each asset is checked against data/geo.lock.json and the official `.sha256sum`
published with that release. A mismatch stops the install and leaves no data file.
The binaries are written under data/geo/, which git ignores.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

MAX_BYTES = 64 * 1024 * 1024
_SHA = re.compile(r"\b([0-9a-fA-F]{64})\b")
_ALLOWED = {"geoip.dat", "geosite.dat"}
_LOCK_KEYS = ("name", "url", "sha256sum_url", "sha256", "tag")


class ChecksumMismatch(RuntimeError):
    def __init__(self, name: str, *, lock: str, official: str | None, downloaded: str | None):
        self.name = name
        self.lock = lock
        self.official = official
        self.downloaded = downloaded
        lines = [
            f"checksum mismatch for {name}",
            f"  lockfile:  {lock}",
            f"  official:  {official if official is not None else '(unavailable)'}",
        ]
        if downloaded is not None:
            lines.append(f"  downloaded: {downloaded}")
        lines.append("stopped; the data file was not installed")
        super().__init__("\n".join(lines))


def default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "geo.lock.json", root / "data" / "geo"


def parse_sha256sum(text: str) -> str:
    match = _SHA.search(text)
    if not match:
        raise ValueError("official checksum file has no SHA-256 hex digest")
    return match.group(1).lower()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def default_opener(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "zerkalo-geo-fetch"})
    return urllib.request.urlopen(req, timeout=120)


def _read_limited(resp) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_BYTES:
                raise ValueError(f"download exceeded {MAX_BYTES} bytes")
            chunks.append(chunk)
    finally:
        close = getattr(resp, "close", None)
        if close:
            close()
    return b"".join(chunks)


def load_lock(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    datasets = data.get("datasets") if isinstance(data, dict) else None
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"{path} has no datasets")
    cleaned = []
    for item in datasets:
        if not isinstance(item, dict) or any(not item.get(key) for key in _LOCK_KEYS):
            raise ValueError(f"{path} dataset is missing url, tag, or sha256")
        name = str(item["name"])
        if name not in _ALLOWED or Path(name).name != name:
            raise ValueError(f"refusing unexpected dataset name: {name}")
        digest = str(item["sha256"]).lower()
        if not _SHA.fullmatch(digest):
            raise ValueError(f"{name} lockfile sha256 is not 64 hex characters")
        copied = dict(item)
        copied["sha256"] = digest
        cleaned.append(copied)
    return cleaned


def install_datasets(lock_path: Path, dest_dir: Path, opener=default_opener) -> list[tuple[str, str]]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    installed: list[tuple[str, str]] = []
    for item in load_lock(lock_path):
        name = item["name"]
        lock_hash = item["sha256"]
        official_text = _read_limited(opener(item["sha256sum_url"])).decode("utf-8", "replace")
        official = parse_sha256sum(official_text)
        if official != lock_hash:
            raise ChecksumMismatch(name, lock=lock_hash, official=official, downloaded=None)
        payload = _read_limited(opener(item["url"]))
        downloaded = sha256_bytes(payload)
        if downloaded != official or downloaded != lock_hash:
            raise ChecksumMismatch(name, lock=lock_hash, official=official, downloaded=downloaded)
        partial = dest_dir / f".{name}.partial"
        final = dest_dir / name
        try:
            partial.write_bytes(payload)
            partial.replace(final)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        else:
            partial.unlink(missing_ok=True)
        installed.append((name, downloaded))
    return installed


def main(argv: list[str] | None = None) -> int:
    del argv  # the lockfile pins the URLs; there are no extra arguments
    lock_path, dest_dir = default_paths()
    try:
        installed = install_datasets(lock_path, dest_dir)
    except ChecksumMismatch as exc:
        print(exc, file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"geo fetch failed: {exc}", file=sys.stderr)
        return 1
    for name, digest in installed:
        print(f"{name} sha256 {digest} ok")
    print(f"data directory: {dest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
