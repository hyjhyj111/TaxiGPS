"""HTML map/report generation without folium."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd

from .geo import speed_to_color


def _records(df: pd.DataFrame, limit: int) -> list[dict[str, object]]:
    return json.loads(df.head(limit).to_json(orient="records", date_format="iso", force_ascii=False))


def write_dashboard(
    clean: pd.DataFrame,
    od: pd.DataFrame,
    output_html: Path,
    boundary_geojson: Path | None = None,
    title: str = "Taxi GPS Dashboard",
    point_limit: int = 1200,
) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    boundary = None
    if boundary_geojson and boundary_geojson.exists():
        boundary = json.loads(boundary_geojson.read_text(encoding="utf-8"))

    sample_vehicle = None
    track = []
    if not clean.empty:
        sample_vehicle = int(clean["id"].mode().iloc[0])
        track_df = clean.loc[clean["id"].eq(sample_vehicle)].sort_values("time").head(point_limit)
        track = _records(track_df, point_limit)

    pickups = _records(od.rename(columns={"start_lat": "lat", "start_lng": "lng"}), point_limit)
    positions = _records(clean.sample(min(point_limit, len(clean)), random_state=7) if len(clean) else clean, point_limit)
    hourly = []
    if not od.empty:
        hourly = _records(od.assign(hour=od["start_time"].dt.hour).groupby("hour", as_index=False).size(), 48)

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; color:#1f2937; }}
    header {{ padding:16px 22px; border-bottom:1px solid #dde3ec; background:#f8fafc; }}
    h1 {{ margin:0; font-size:22px; }}
    .meta {{ display:flex; gap:18px; flex-wrap:wrap; margin-top:8px; color:#475569; font-size:14px; }}
    #map {{ height: 68vh; min-height: 520px; }}
    .panel {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; padding:14px 18px; }}
    .card {{ border:1px solid #dde3ec; border-radius:8px; padding:12px; background:white; }}
    .bar {{ height:8px; background:#e5e7eb; margin:6px 0; }}
    .fill {{ height:8px; background:#2563eb; }}
    code {{ background:#eef2f7; padding:2px 4px; border-radius:4px; }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <div class="meta">
      <span>GPS points: {len(clean):,}</span>
      <span>OD orders: {len(od):,}</span>
      <span>Sample vehicle: {sample_vehicle if sample_vehicle is not None else "N/A"}</span>
    </div>
  </header>
  <div id="map"></div>
  <section class="panel" id="stats"></section>
  <script>
    const boundary = {json.dumps(boundary, ensure_ascii=False)};
    const track = {json.dumps(track, ensure_ascii=False)};
    const pickups = {json.dumps(pickups, ensure_ascii=False)};
    const positions = {json.dumps(positions, ensure_ascii=False)};
    const hourly = {json.dumps(hourly, ensure_ascii=False)};

    const map = L.map('map').setView([22.52847, 114.05454], 11);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);
    if (boundary) {{
      L.geoJSON(boundary, {{ style: {{ color:'#334155', weight:1, fillOpacity:0.03 }} }}).addTo(map);
    }}
    if (track.length) {{
      const line = track.map(p => [p.lat, p.lng]);
      L.polyline(line, {{ color:'#2563eb', weight:3 }}).addTo(map).bindPopup('车辆轨迹');
      L.circleMarker(line[0], {{ radius:6, color:'#16a34a' }}).addTo(map).bindPopup('轨迹起点');
      L.circleMarker(line[line.length - 1], {{ radius:6, color:'#dc2626' }}).addTo(map).bindPopup('轨迹终点');
    }}
    positions.forEach(p => {{
      const color = p.status === 1 ? '#ef4444' : '#0ea5e9';
      L.circleMarker([p.lat, p.lng], {{ radius:2, color, fillOpacity:0.45, weight:1 }}).addTo(map);
    }});
    pickups.forEach(p => {{
      L.circleMarker([p.lat, p.lng], {{ radius:4, color:'#f97316', fillOpacity:0.7, weight:1 }}).addTo(map)
        .bindPopup(`上车点<br>ID: ${{p.id}}<br>${{p.start_time || ''}}`);
    }});

    const stats = document.getElementById('stats');
    const maxHour = Math.max(1, ...hourly.map(x => x.size || 0));
    hourly.forEach(x => {{
      const div = document.createElement('div');
      div.className = 'card';
      div.innerHTML = `<strong>${{x.hour}}:00</strong><div class="bar"><div class="fill" style="width:${{(x.size || 0) / maxHour * 100}}%"></div></div><span>${{x.size || 0}} orders</span>`;
      stats.appendChild(div);
    }});

    window.speedToColor = function(speed) {{
      if (speed < 10) return '{speed_to_color(5)}';
      if (speed < 20) return '{speed_to_color(15)}';
      if (speed < 35) return '{speed_to_color(25)}';
      return '{speed_to_color(40)}';
    }};
  </script>
</body>
</html>"""
    output_html.write_text(html, encoding="utf-8")
