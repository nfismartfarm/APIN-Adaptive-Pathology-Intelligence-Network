# scripts/apin_v2/ip_geo_resolver.py
# 9.N.T31 · IP -> administrative region resolver for the Analytics "Requests"
# (request-origin) map.
#
# Wraps the DB-IP City Lite mmdb (CC-BY 4.0) via geoip2 — gives, per public IP,
# country (cc) / state (subdivision) / city (~district) / lat / lon. The DB is
# ~124 MB so it is NOT committed; it is downloaded at build/setup
# (tools/fetch_ip_db.py) to scripts/apin_v2/ip_city.mmdb. Override the path with
# APIN_IP_CITY_MMDB.
#
# Fail-open: if the DB is missing or geoip2 is unavailable, every field is None
# and the caller treats the IP as "unmapped" — the feature degrades, never
# breaks. Private / loopback IPs (localhost, RFC1918, CGNAT) resolve to a
# {"local": True} marker so the UI can bucket "local / internal" separately.

import os
import threading
import ipaddress

_reader = None
_unavailable = False
_lock = threading.Lock()

_NULL = {"cc": None, "state": None, "district": None, "lat": None,
         "lon": None, "local": False}

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "ip_city.mmdb")


def _db_path():
    return os.environ.get("APIN_IP_CITY_MMDB", _DEFAULT_PATH)


def _ensure():
    global _reader, _unavailable
    if _reader is not None or _unavailable:
        return _reader
    with _lock:
        if _reader is None and not _unavailable:
            try:
                import geoip2.database
                path = _db_path()
                if not os.path.exists(path):
                    _unavailable = True
                    return None
                _reader = geoip2.database.Reader(path)
            except Exception:
                _unavailable = True
    return _reader


def available() -> bool:
    return _ensure() is not None


def _is_private(ip) -> bool:
    try:
        a = ipaddress.ip_address(ip)
        return (a.is_private or a.is_loopback or a.is_link_local
                or a.is_reserved or a.is_unspecified
                # CGNAT 100.64.0.0/10
                or (a.version == 4 and ipaddress.ip_address("100.64.0.0")
                    <= a <= ipaddress.ip_address("100.127.255.255")))
    except Exception:
        return False


# ── device (local-egress) location ──────────────────────────────────────────
# A loopback / private IP (127.0.0.1 in local dev) can't be geolocated from the
# connection address. We attribute it to THIS device's location so local traffic
# still lands on the "Requests" map. Sources, in priority order:
#   1. Explicit env override: APIN_DEVICE_LAT + APIN_DEVICE_LON (+ optional
#      APIN_DEVICE_CC / _STATE / _DISTRICT). No network call.
#   2. Auto-detect (default ON): fetch the machine's public egress IP from an
#      echo service, then run it through the same mmdb. Set
#      APIN_DEVICE_GEO_AUTODETECT=0 to disable. Resolved once, then cached.
# In production every client IP is a real public address, so this branch is
# never hit there — it's purely a local-dev convenience.
_device_geo = None
_device_done = False
_device_lock = threading.Lock()


def _env_device_geo():
    lat = os.environ.get("APIN_DEVICE_LAT")
    lon = os.environ.get("APIN_DEVICE_LON")
    if not (lat and lon):
        return None
    try:
        return {
            "cc": (os.environ.get("APIN_DEVICE_CC") or None),
            "state": (os.environ.get("APIN_DEVICE_STATE") or None),
            "district": (os.environ.get("APIN_DEVICE_DISTRICT") or None),
            "lat": float(lat), "lon": float(lon),
            "local": False, "device": True,
        }
    except Exception:
        return None


def _public_egress_ip():
    try:
        import requests
    except Exception:
        return None
    for url in ("https://api.ipify.org", "https://ifconfig.me/ip",
                "https://icanhazip.com"):
        try:
            r = requests.get(url, timeout=2.5)
            if r.status_code == 200:
                ip = (r.text or "").strip()
                ipaddress.ip_address(ip)  # validate (raises if junk)
                if not _is_private(ip):
                    return ip
        except Exception:
            continue
    return None


def device_location():
    """Best-effort location of this device, used for loopback/private IPs.
    Returns a geo dict (cc/state/district/lat/lon, device=True) or None.
    Resolved at most once per process."""
    global _device_geo, _device_done
    if _device_done:
        return _device_geo
    with _device_lock:
        if _device_done:
            return _device_geo
        g = _env_device_geo()
        if g is None and os.environ.get("APIN_DEVICE_GEO_AUTODETECT", "1") != "0":
            ip = _public_egress_ip()
            if ip:
                gg = resolve(ip)  # public IP → mmdb (not private → no recursion)
                if gg and gg.get("cc"):
                    gg = dict(gg); gg["device"] = True
                    g = gg
        _device_geo = g
        _device_done = True
        return _device_geo


def resolve(ip) -> dict:
    """IP string -> {cc,state,district,lat,lon,local}. district is the city
    name (the finest admin level DB-IP provides). For private/loopback IPs we
    attribute the request to THIS device's location (see device_location);
    if that can't be determined, local=True (not geolocatable)."""
    if not ip:
        return dict(_NULL)
    ip = str(ip).strip()
    # X-Forwarded-For may carry a list — take the first hop.
    if "," in ip:
        ip = ip.split(",")[0].strip()
    if _is_private(ip):
        dg = device_location()
        if dg and dg.get("cc"):
            return dict(dg)
        out = dict(_NULL); out["local"] = True; return out
    rd = _ensure()
    if rd is None:
        return dict(_NULL)
    try:
        c = rd.city(ip)
        sub = c.subdivisions.most_specific.name if c.subdivisions else None
        return {
            "cc": (c.country.iso_code or None),
            "state": (sub or None),
            "district": (c.city.name or None),
            "lat": (float(c.location.latitude) if c.location.latitude is not None else None),
            "lon": (float(c.location.longitude) if c.location.longitude is not None else None),
            "local": False,
        }
    except Exception:
        # geoip2.errors.AddressNotFoundError + any reader error -> unmapped
        return dict(_NULL)


_cache = {}
_cache_lock = threading.Lock()


def resolve_cached(ip) -> dict:
    """resolve() with a process-local memo — the analytics aggregation calls
    this for each distinct IP in a window, so a handful of clients hitting the
    API thousands of times only cost one lookup each."""
    key = str(ip or "")
    with _cache_lock:
        v = _cache.get(key)
    if v is not None:
        return v
    v = resolve(ip)
    with _cache_lock:
        if len(_cache) < 50000:
            _cache[key] = v
    return v
