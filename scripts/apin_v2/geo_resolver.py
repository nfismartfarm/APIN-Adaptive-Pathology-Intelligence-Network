# scripts/apin_v2/geo_resolver.py
# 9.N.T31 · GPS -> administrative region resolver for the Analytics geo map.
#
# Wraps reverse_geocoder (offline: a ~30 MB bundled KD-tree of ~150k cities).
# This module does RESOLUTION only — turning a (lat, lon) point into region
# NAMES (country code / state / district). The boundary POLYGONS that the
# globe + flat map draw are loaded client-side (GADM / Natural Earth GeoJSON),
# keyed by the cc + admin names produced here. That split keeps the server
# from ever holding gigabytes of polygons or running point-in-polygon at
# request time.
#
# Design:
#   - Lazy, thread-safe one-time init (the KD-tree load is ~1s; we warm it on
#     first use and reuse it for the process lifetime).
#   - Degrades gracefully: if reverse_geocoder is not installed or a lookup
#     throws, every field comes back None and the caller stores NULLs. The
#     feature is additive — a missing resolver never breaks scan ingestion.
#   - mode=1 (single-threaded query). The server is already multi-threaded;
#     mode=2 spawns a process pool which is the wrong trade-off inside a
#     request handler and is flaky on Windows.

import threading

_rg = None              # the reverse_geocoder module once loaded
_unavailable = False    # True once we've determined it cannot be used
_lock = threading.Lock()

_NULL = {"cc": None, "state": None, "district": None}


def _ensure():
    """Lazily import + warm reverse_geocoder. Returns the module or None."""
    global _rg, _unavailable
    if _rg is not None or _unavailable:
        return _rg
    with _lock:
        if _rg is None and not _unavailable:
            try:
                import reverse_geocoder as rg
                # Warm the KD-tree once so the first real lookup isn't slow.
                rg.search([(0.0, 0.0)], mode=1)
                _rg = rg
            except Exception:
                _unavailable = True
    return _rg


def available() -> bool:
    """True if region resolution is usable in this process."""
    return _ensure() is not None


def _project(r):
    if not r:
        return dict(_NULL)
    return {
        "cc":       (r.get("cc") or None),
        "state":    (r.get("admin1") or None),
        "district": (r.get("admin2") or None),
    }


def resolve(lat, lon) -> dict:
    """Resolve one point. Returns {'cc','state','district'} (any may be None)."""
    try:
        rg = _ensure()
        if rg is None or lat is None or lon is None:
            return dict(_NULL)
        res = rg.search([(float(lat), float(lon))], mode=1)
        return _project(res[0] if res else None)
    except Exception:
        return dict(_NULL)


def resolve_batch(coords) -> list:
    """Resolve many points in a single KD-tree query.

    coords: iterable of (lat, lon). Returns a list of dicts aligned to input.
    Coordinates that are None/invalid map to all-None entries while keeping
    alignment, so the caller can zip results back onto rows.
    """
    coords = list(coords)
    if not coords:
        return []
    rg = _ensure()
    if rg is None:
        return [dict(_NULL) for _ in coords]

    # Partition valid coords (reverse_geocoder rejects NaN/None) while
    # remembering their positions so we can scatter results back.
    pts, idx = [], []
    out = [dict(_NULL) for _ in coords]
    for i, c in enumerate(coords):
        try:
            la, lo = float(c[0]), float(c[1])
            if la != la or lo != lo:   # NaN check
                continue
            pts.append((la, lo))
            idx.append(i)
        except Exception:
            continue
    if not pts:
        return out
    try:
        res = rg.search(pts, mode=1)
        for j, r in enumerate(res):
            out[idx[j]] = _project(r)
    except Exception:
        pass
    return out
