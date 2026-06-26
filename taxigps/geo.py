"""Small geographic helpers with no third-party dependency."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt


def haversine_km(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    """Return great-circle distance in kilometers."""
    radius_km = 6371.0088
    lng1_r, lat1_r, lng2_r, lat2_r = map(radians, [lng1, lat1, lng2, lat2])
    dlng = lng2_r - lng1_r
    dlat = lat2_r - lat1_r
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlng / 2) ** 2
    return 2 * radius_km * asin(sqrt(a))


def speed_to_color(speed_kmh: float) -> str:
    """Road speed color used by congestion maps."""
    if speed_kmh < 10:
        return "#d73027"
    if speed_kmh < 20:
        return "#fc8d59"
    if speed_kmh < 35:
        return "#fee08b"
    return "#1a9850"


def in_shenzhen_bounds(lng: float, lat: float) -> bool:
    """Broad Shenzhen bbox, used only for removing obvious GPS errors."""
    return 113.70 <= lng <= 114.70 and 22.35 <= lat <= 22.90
