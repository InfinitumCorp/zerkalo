#!/usr/bin/env python3
"""Download pinned v2fly geoip.dat and geosite.dat into the gitignored data directory.

Verifies each file against data/geo.lock.json and the official .sha256sum asset.
Stops on checksum mismatch and does not install a bad file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerkalo.geo_fetch import main

if __name__ == "__main__":
    raise SystemExit(main())
