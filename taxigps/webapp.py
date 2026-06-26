"""Generate and serve the Taxi GPS frontend application."""

from __future__ import annotations

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from typing import Any

import pandas as pd


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if df.empty:
        return []
    data = df.head(limit) if limit else df
    return json.loads(data.to_json(orient="records", date_format="iso", force_ascii=False))


def _vehicle_tracks(output_dir: Path, max_points: int = 1200) -> dict[str, list[dict[str, Any]]]:
    vehicle_dir = output_dir / "cache" / "vehicles"
    tracks: dict[str, list[dict[str, Any]]] = {}
    if not vehicle_dir.exists():
        return tracks
    for path in sorted(vehicle_dir.glob("*.csv")):
        df = pd.read_csv(path, parse_dates=["time"])
        if len(df) > max_points:
            step = max(1, len(df) // max_points)
            df = df.iloc[::step].head(max_points)
        tracks[path.stem] = _records(df)
    return tracks


def _minute_sample(minute: pd.DataFrame, per_time: int = 160) -> list[dict[str, Any]]:
    if minute.empty:
        return []
    frames = []
    for _, group in minute.groupby("minute", sort=True):
        frames.append(group.head(per_time))
    return _records(pd.concat(frames, ignore_index=True), 6000) if frames else []


def build_frontend_data(output_dir: Path, project_root: Path) -> dict[str, Any]:
    cleaning = _read_json(output_dir / "cleaning_summary.json", {})
    summary = _read_json(output_dir / "stats" / "summary.json", {})
    eta = _read_json(output_dir / "stats" / "eta_sample.json", {})
    boundary = _read_json(project_root / "深圳市.json", None)

    hourly = _read_csv(output_dir / "stats" / "hourly_orders.csv")
    categories = _read_csv(output_dir / "stats" / "trip_distance_categories.csv")
    fleet = _read_csv(output_dir / "stats" / "vehicle_operation_stats.csv")
    hotspots = _read_csv(output_dir / "stats" / "pickup_hotspots.csv", parse_dates=["time_slice"])
    minute = _read_csv(output_dir / "cache" / "minute_positions.csv", parse_dates=["minute", "time"])
    od = _read_csv(output_dir / "od_orders.csv", parse_dates=["start_time", "end_time"])

    if not fleet.empty:
        fleet = fleet.sort_values("total_km", ascending=False)
    if not hotspots.empty:
        hotspots = hotspots.sort_values(["time_slice", "count"], ascending=[True, False])

    return {
        "meta": {
            "title": "Taxi GPS 数据分析与可视化系统",
            "generated_from": str(output_dir),
            "vehicle_count_cached": len(list((output_dir / "cache" / "vehicles").glob("*.csv")))
            if (output_dir / "cache" / "vehicles").exists()
            else 0,
        },
        "cleaning": cleaning,
        "summary": summary,
        "eta": eta,
        "boundary": boundary,
        "hourly": _records(hourly),
        "categories": _records(categories),
        "fleet": _records(fleet, 60),
        "hotspots": _records(hotspots, 900),
        "minute_times": [str(x) for x in minute["minute"].drop_duplicates().head(120).tolist()] if not minute.empty else [],
        "minute_positions": _minute_sample(minute),
        "od": _records(od, 1200),
        "tracks": _vehicle_tracks(output_dir),
    }


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Taxi GPS 数据分析与可视化系统</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <link rel="stylesheet" href="assets/app.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.9/dist/chart.umd.min.js"></script>
  <script src="https://unpkg.com/lucide@0.468.0/dist/umd/lucide.min.js"></script>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">TG</div>
        <div>
          <strong>Taxi GPS</strong>
          <span>实训 V 系统</span>
        </div>
      </div>
      <nav class="nav">
        <button class="nav-item active" data-page="overview"><i data-lucide="layout-dashboard"></i><span>总览</span></button>
        <button class="nav-item" data-page="cleaning"><i data-lucide="filter"></i><span>数据清洗</span></button>
        <button class="nav-item" data-page="od"><i data-lucide="route"></i><span>OD 与缓存</span></button>
        <button class="nav-item" data-page="trajectory"><i data-lucide="navigation"></i><span>轨迹查询</span></button>
        <button class="nav-item" data-page="minute"><i data-lucide="clock-3"></i><span>分钟位置</span></button>
        <button class="nav-item" data-page="heatmap"><i data-lucide="flame"></i><span>热点热力</span></button>
        <button class="nav-item" data-page="stats"><i data-lucide="bar-chart-3"></i><span>统计分析</span></button>
        <button class="nav-item" data-page="network"><i data-lucide="map"></i><span>路网 ETA</span></button>
        <button class="nav-item" data-page="report"><i data-lucide="file-text"></i><span>验收材料</span></button>
      </nav>
    </aside>
    <main class="main">
      <header class="topbar">
        <div>
          <h1 id="pageTitle">总览</h1>
          <p id="pageSub">清洗、订单、热点、轨迹与 ETA 一体化工作台</p>
        </div>
        <div class="toolbar">
          <label>车辆<select id="vehicleSelect"></select></label>
          <label>时间片<select id="minuteSelect"></select></label>
          <button id="playBtn" class="icon-btn" title="播放轨迹"><i data-lucide="play"></i></button>
        </div>
      </header>
      <section id="overview" class="page active">
        <div class="metrics" id="metrics"></div>
        <div class="split">
          <section class="map-panel"><div id="map"></div></section>
          <section class="table-panel">
            <h2>最近 OD 订单</h2>
            <div id="odTable"></div>
          </section>
        </div>
      </section>
      <section id="cleaning" class="page">
        <div class="metrics" id="cleaningMetrics"></div>
        <section class="wide-panel">
          <h2>清洗口径</h2>
          <div class="timeline">
            <span>字段统一</span><span>时间补全</span><span>重复处理</span><span>异常状态剔除</span><span>深圳范围过滤</span>
          </div>
        </section>
      </section>
      <section id="od" class="page">
        <div class="split">
          <section class="chart-panel"><canvas id="distanceChart"></canvas></section>
          <section class="table-panel"><h2>OD 缓存样例</h2><div id="odCacheTable"></div></section>
        </div>
      </section>
      <section id="trajectory" class="page">
        <div class="split">
          <section class="map-panel"><div id="trackMap"></div></section>
          <section class="table-panel"><h2>轨迹点</h2><div id="trackTable"></div></section>
        </div>
      </section>
      <section id="minute" class="page">
        <div class="split">
          <section class="map-panel"><div id="minuteMap"></div></section>
          <section class="table-panel"><h2>分钟位置</h2><div id="minuteTable"></div></section>
        </div>
      </section>
      <section id="heatmap" class="page">
        <div class="split">
          <section class="map-panel"><div id="hotspotMap"></div></section>
          <section class="table-panel"><h2>DBSCAN 热点</h2><div id="hotspotTable"></div></section>
        </div>
      </section>
      <section id="stats" class="page">
        <div class="charts-grid">
          <section class="chart-panel"><canvas id="hourlyChart"></canvas></section>
          <section class="chart-panel"><canvas id="fleetChart"></canvas></section>
        </div>
        <section class="wide-panel"><h2>车辆运营排行</h2><div id="fleetTable"></div></section>
      </section>
      <section id="network" class="page">
        <div class="split">
          <section class="map-panel"><div id="etaMap"></div></section>
          <section class="table-panel"><h2>ETA 样例</h2><div id="etaBox"></div></section>
        </div>
      </section>
      <section id="report" class="page">
        <div class="report-grid">
          <section class="wide-panel"><h2>交付物</h2><div id="deliverables"></div></section>
          <section class="wide-panel"><h2>验收命令</h2><pre>.venv/bin/python run_project.py pipeline
.venv/bin/python run_project.py webapp --serve</pre></section>
        </div>
      </section>
    </main>
  </div>
  <script src="assets/app.js"></script>
</body>
</html>
"""


APP_CSS = """
:root {
  --bg: #f4f7fb;
  --panel: #ffffff;
  --line: #d9e1ec;
  --text: #172033;
  --muted: #667085;
  --blue: #2563eb;
  --green: #0f9f6e;
  --orange: #f97316;
  --red: #dc2626;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
}
.shell { display: grid; grid-template-columns: 248px minmax(0, 1fr); min-height: 100vh; }
.sidebar { background: #101827; color: #d8e1ee; padding: 18px 12px; border-right: 1px solid #0b1220; }
.brand { display: flex; align-items: center; gap: 12px; padding: 4px 8px 18px; }
.brand-mark { width: 38px; height: 38px; display: grid; place-items: center; border-radius: 8px; background: #2563eb; color: white; font-weight: 800; }
.brand strong { display: block; color: #fff; font-size: 15px; }
.brand span { display: block; color: #93a4bb; font-size: 12px; margin-top: 3px; }
.nav { display: grid; gap: 4px; }
.nav-item {
  height: 42px;
  border: 0;
  border-radius: 8px;
  padding: 0 10px;
  display: flex;
  align-items: center;
  gap: 10px;
  color: #b9c6d8;
  background: transparent;
  cursor: pointer;
  font: inherit;
  text-align: left;
}
.nav-item svg { width: 18px; height: 18px; }
.nav-item.active, .nav-item:hover { background: #1d2a3f; color: #fff; }
.main { min-width: 0; padding: 18px; }
.topbar {
  min-height: 72px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
}
h1 { margin: 0; font-size: 24px; letter-spacing: 0; }
h2 { margin: 0 0 12px; font-size: 16px; letter-spacing: 0; }
p { margin: 6px 0 0; color: var(--muted); }
.toolbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
label { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 13px; }
select, button {
  height: 36px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
  color: var(--text);
}
select { min-width: 148px; padding: 0 10px; }
.icon-btn { width: 38px; display: grid; place-items: center; cursor: pointer; }
.icon-btn svg { width: 17px; height: 17px; }
.page { display: none; }
.page.active { display: block; }
.metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(160px, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}
.metric, .map-panel, .table-panel, .chart-panel, .wide-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.metric span { color: var(--muted); font-size: 12px; }
.metric strong { display: block; margin-top: 8px; font-size: 24px; }
.split { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(330px, .65fr); gap: 12px; }
.charts-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-bottom: 12px; }
.report-grid { display: grid; grid-template-columns: 1.2fr .8fr; gap: 12px; }
#map, #trackMap, #minuteMap, #hotspotMap, #etaMap { height: calc(100vh - 146px); min-height: 560px; border-radius: 6px; }
.table-wrap { overflow: auto; max-height: calc(100vh - 190px); }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { padding: 8px 7px; border-bottom: 1px solid #edf1f6; text-align: left; white-space: nowrap; }
th { color: #475569; background: #f8fafc; position: sticky; top: 0; }
.timeline { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }
.timeline span { padding: 12px; border: 1px solid var(--line); border-radius: 8px; text-align: center; background: #f8fafc; }
.pill { display: inline-flex; align-items: center; height: 26px; border-radius: 999px; padding: 0 9px; background: #eaf1ff; color: #1d4ed8; font-size: 12px; }
pre { margin: 0; overflow: auto; background: #101827; color: #e5edf7; padding: 14px; border-radius: 8px; }
@media (max-width: 980px) {
  .shell { grid-template-columns: 1fr; }
  .sidebar { position: static; }
  .nav { grid-template-columns: repeat(3, 1fr); }
  .split, .charts-grid, .report-grid, .metrics { grid-template-columns: 1fr; }
  #map, #trackMap, #minuteMap, #hotspotMap, #etaMap { height: 62vh; }
}
"""


APP_JS = """
let DATA;
const pageTitles = {
  overview: ['总览', '清洗、订单、热点、轨迹与 ETA 一体化工作台'],
  cleaning: ['数据清洗', '字段统一、时间补全、重复处理与异常剔除'],
  od: ['OD 与缓存', '订单表、车辆缓存、分钟缓存和 OD 缓存'],
  trajectory: ['轨迹查询', '按车辆查看载客/空载轨迹与动画'],
  minute: ['分钟位置', '按分钟查看车辆位置快照'],
  heatmap: ['热点热力', 'DBSCAN 上车热点与时间片分布'],
  stats: ['统计分析', '小时订单、车辆运营、短中长途统计'],
  network: ['路网 ETA', '最近道路节点、最短路径和 ETA 样例'],
  report: ['验收材料', '交付物、运行命令与项目文件']
};
const charts = {};
const maps = {};
const layers = {};

fetch('data.json').then(r => r.json()).then(data => {
  DATA = data;
  initNavigation();
  initControls();
  renderAll();
  lucide.createIcons();
});

function fmt(n, digits = 0) {
  if (n === undefined || n === null || Number.isNaN(Number(n))) return '-';
  return Number(n).toLocaleString('zh-CN', { maximumFractionDigits: digits });
}

function table(rows, columns) {
  const body = rows.map(row => `<tr>${columns.map(c => `<td>${c.f ? c.f(row[c.k], row) : (row[c.k] ?? '')}</td>`).join('')}</tr>`).join('');
  return `<div class="table-wrap"><table><thead><tr>${columns.map(c => `<th>${c.t}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function initNavigation() {
  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => showPage(btn.dataset.page));
  });
}

function showPage(id) {
  document.querySelectorAll('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.page === id));
  document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === id));
  document.getElementById('pageTitle').textContent = pageTitles[id][0];
  document.getElementById('pageSub').textContent = pageTitles[id][1];
  setTimeout(() => Object.values(maps).forEach(m => m.invalidateSize()), 40);
}

function initControls() {
  const vehicleSelect = document.getElementById('vehicleSelect');
  Object.keys(DATA.tracks).forEach(id => vehicleSelect.add(new Option(id, id)));
  vehicleSelect.addEventListener('change', renderTrackPage);
  const minuteSelect = document.getElementById('minuteSelect');
  DATA.minute_times.forEach(t => minuteSelect.add(new Option(t.slice(0, 16).replace('T', ' '), t)));
  minuteSelect.addEventListener('change', renderMinutePage);
  document.getElementById('playBtn').addEventListener('click', playTrack);
}

function metric(label, value, detail = '') {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong>${detail ? `<p>${detail}</p>` : ''}</div>`;
}

function renderAll() {
  renderMetrics();
  renderOverviewMap();
  renderTables();
  renderCharts();
  renderTrackPage();
  renderMinutePage();
  renderHotspotPage();
  renderEtaPage();
  renderReportPage();
}

function renderMetrics() {
  const c = DATA.cleaning;
  const s = DATA.summary;
  document.getElementById('metrics').innerHTML = [
    metric('清洗后 GPS', fmt(c.clean_rows), `原始 ${fmt(c.raw_rows)} 行`),
    metric('OD 订单', fmt(s.orders), `平均 ${fmt(s.avg_order_distance_km, 2)} km`),
    metric('车辆数量', fmt(c.vehicle_count), `缓存 ${fmt(DATA.meta.vehicle_count_cached)} 辆`),
    metric('DBSCAN 热点', fmt(s.hotspot_rows), s.hotspot_method || '')
  ].join('');
  document.getElementById('cleaningMetrics').innerHTML = [
    metric('原始行数', fmt(c.raw_rows)),
    metric('清洗后行数', fmt(c.clean_rows)),
    metric('删除行数', fmt(c.dropped_rows)),
    metric('异常状态', fmt(c.abnormal_status_rows_dropped))
  ].join('');
}

function makeMap(id) {
  if (maps[id]) return maps[id];
  const map = L.map(id, { preferCanvas: true }).setView([22.52847, 114.05454], 11);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap' }).addTo(map);
  if (DATA.boundary) L.geoJSON(DATA.boundary, { style: { color:'#334155', weight:1, fillOpacity:0.03 } }).addTo(map);
  maps[id] = map;
  return map;
}

function clearLayer(key) {
  if (layers[key]) {
    layers[key].remove();
    layers[key] = null;
  }
}

function renderOverviewMap() {
  const map = makeMap('map');
  const group = L.layerGroup();
  DATA.hotspots.slice(0, 160).forEach(p => L.circleMarker([p.lat, p.lng], {
    radius: Math.min(18, 4 + p.count),
    color: '#f97316',
    fillOpacity: 0.45,
    weight: 1
  }).bindPopup(`热点 ${p.count} 单<br>${String(p.time_slice).slice(0,16)}`).addTo(group));
  const firstVehicle = Object.keys(DATA.tracks)[0];
  const track = DATA.tracks[firstVehicle] || [];
  if (track.length) L.polyline(track.map(p => [p.lat, p.lng]), { color:'#2563eb', weight:3 }).addTo(group);
  group.addTo(map);
}

function renderTables() {
  const odCols = [
    {k:'id', t:'车辆'}, {k:'start_time', t:'开始', f:v => String(v).slice(11,19)},
    {k:'end_time', t:'结束', f:v => String(v).slice(11,19)}, {k:'distance_km', t:'km', f:v => fmt(v,2)},
    {k:'avg_speed_kmh', t:'km/h', f:v => fmt(v,1)}
  ];
  document.getElementById('odTable').innerHTML = table(DATA.od.slice(0, 16), odCols);
  document.getElementById('odCacheTable').innerHTML = table(DATA.od.slice(0, 60), odCols);
}

function renderCharts() {
  const hourlyCtx = document.getElementById('hourlyChart');
  charts.hourly = new Chart(hourlyCtx, {
    type: 'bar',
    data: { labels: DATA.hourly.map(x => `${x.hour}:00`), datasets: [{ label: '订单数', data: DATA.hourly.map(x => x.order_count), backgroundColor: '#2563eb' }] },
    options: { responsive: true, plugins: { legend: { display: false } } }
  });
  charts.distance = new Chart(document.getElementById('distanceChart'), {
    type: 'doughnut',
    data: { labels: DATA.categories.map(x => x.category), datasets: [{ data: DATA.categories.map(x => x.order_count), backgroundColor: ['#10b981', '#f97316', '#ef4444'] }] },
    options: { responsive: true }
  });
  charts.fleet = new Chart(document.getElementById('fleetChart'), {
    type: 'bar',
    data: {
      labels: DATA.fleet.slice(0, 12).map(x => x.id),
      datasets: [{ label: '总里程 km', data: DATA.fleet.slice(0, 12).map(x => x.total_km), backgroundColor: '#0f9f6e' }]
    },
    options: { responsive: true, plugins: { legend: { display: false } } }
  });
  document.getElementById('fleetTable').innerHTML = table(DATA.fleet.slice(0, 30), [
    {k:'id', t:'车辆'}, {k:'gps_points', t:'点数', f:v => fmt(v)}, {k:'total_km', t:'总里程', f:v => fmt(v,1)},
    {k:'occupied_km', t:'载客里程', f:v => fmt(v,1)}, {k:'empty_km', t:'空驶里程', f:v => fmt(v,1)}, {k:'occupancy_rate', t:'载客率', f:v => `${fmt(v * 100,1)}%`}
  ]);
}

function renderTrackPage() {
  const vehicle = document.getElementById('vehicleSelect').value || Object.keys(DATA.tracks)[0];
  const track = DATA.tracks[vehicle] || [];
  const map = makeMap('trackMap');
  clearLayer('track');
  const group = L.layerGroup();
  if (track.length) {
    const occupied = track.filter(p => Number(p.status) === 1).map(p => [p.lat, p.lng]);
    const empty = track.filter(p => Number(p.status) === 0).map(p => [p.lat, p.lng]);
    if (occupied.length) L.polyline(occupied, { color:'#dc2626', weight:3 }).addTo(group);
    if (empty.length) L.polyline(empty, { color:'#0ea5e9', weight:3 }).addTo(group);
    L.circleMarker([track[0].lat, track[0].lng], { radius:6, color:'#16a34a' }).bindPopup('起点').addTo(group);
    L.circleMarker([track[track.length - 1].lat, track[track.length - 1].lng], { radius:6, color:'#dc2626' }).bindPopup('终点').addTo(group);
    map.fitBounds(track.map(p => [p.lat, p.lng]), { padding:[24,24] });
  }
  layers.track = group.addTo(map);
  document.getElementById('trackTable').innerHTML = table(track.slice(0, 80), [
    {k:'time', t:'时间', f:v => String(v).slice(11,19)}, {k:'lng', t:'经度', f:v => fmt(v,5)}, {k:'lat', t:'纬度', f:v => fmt(v,5)},
    {k:'status', t:'状态', f:v => Number(v) === 1 ? '<span class="pill">载客</span>' : '空载'}, {k:'speed', t:'速度', f:v => fmt(v,1)}
  ]);
}

function playTrack() {
  const vehicle = document.getElementById('vehicleSelect').value || Object.keys(DATA.tracks)[0];
  const track = DATA.tracks[vehicle] || [];
  if (!track.length) return;
  const map = makeMap('trackMap');
  clearLayer('play');
  let i = 0;
  const marker = L.marker([track[0].lat, track[0].lng]).addTo(map);
  layers.play = marker;
  const timer = setInterval(() => {
    i += 1;
    if (i >= track.length) return clearInterval(timer);
    marker.setLatLng([track[i].lat, track[i].lng]);
  }, 35);
}

function renderMinutePage() {
  const selected = document.getElementById('minuteSelect').value || DATA.minute_times[0];
  const rows = DATA.minute_positions.filter(p => p.minute === selected).slice(0, 180);
  const map = makeMap('minuteMap');
  clearLayer('minute');
  const group = L.layerGroup();
  rows.forEach(p => L.circleMarker([p.lat, p.lng], {
    radius: 4,
    color: Number(p.status) === 1 ? '#ef4444' : '#0ea5e9',
    fillOpacity: 0.65,
    weight: 1
  }).bindPopup(`车辆 ${p.id}<br>${Number(p.status) === 1 ? '载客' : '空载'}<br>${p.speed} km/h`).addTo(group));
  layers.minute = group.addTo(map);
  document.getElementById('minuteTable').innerHTML = table(rows, [
    {k:'id', t:'车辆'}, {k:'time', t:'采样时间', f:v => String(v).slice(11,19)}, {k:'lng', t:'经度', f:v => fmt(v,5)},
    {k:'lat', t:'纬度', f:v => fmt(v,5)}, {k:'status', t:'状态', f:v => Number(v) === 1 ? '载客' : '空载'}, {k:'speed', t:'速度', f:v => fmt(v,1)}
  ]);
}

function renderHotspotPage() {
  const map = makeMap('hotspotMap');
  clearLayer('hotspots');
  const group = L.layerGroup();
  DATA.hotspots.forEach(p => L.circleMarker([p.lat, p.lng], {
    radius: Math.min(22, 4 + Number(p.count)),
    color: '#f97316',
    fillColor: '#f97316',
    fillOpacity: 0.42,
    weight: 1
  }).bindPopup(`时间 ${String(p.time_slice).slice(0,16)}<br>上车 ${p.count} 单`).addTo(group));
  layers.hotspots = group.addTo(map);
  document.getElementById('hotspotTable').innerHTML = table(DATA.hotspots.slice(0, 120), [
    {k:'time_slice', t:'时间片', f:v => String(v).slice(0,16).replace('T',' ')}, {k:'lat', t:'纬度', f:v => fmt(v,5)},
    {k:'lng', t:'经度', f:v => fmt(v,5)}, {k:'count', t:'订单'}, {k:'cluster', t:'簇'}
  ]);
}

function renderEtaPage() {
  const map = makeMap('etaMap');
  const first = DATA.od[0];
  if (first) {
    L.polyline([[first.start_lat, first.start_lng], [first.end_lat, first.end_lng]], { color:'#2563eb', weight:4 }).addTo(map);
    L.marker([first.start_lat, first.start_lng]).bindPopup('起点').addTo(map);
    L.marker([first.end_lat, first.end_lng]).bindPopup('终点').addTo(map);
    map.fitBounds([[first.start_lat, first.start_lng], [first.end_lat, first.end_lng]], { padding:[40,40] });
  }
  const eta = DATA.eta || {};
  document.getElementById('etaBox').innerHTML = [
    metric('方法', eta.method || '-'),
    metric('路网距离', `${fmt(eta.distance_km,2)} km`),
    metric('估计速度', `${fmt(eta.speed_kmh,1)} km/h`),
    metric('ETA', `${fmt(eta.eta_min,1)} min`)
  ].join('');
}

function renderReportPage() {
  document.getElementById('deliverables').innerHTML = table([
    {name:'前端应用', path:'outputs/web/index.html'},
    {name:'清洗数据', path:'outputs/cleaned/taxi_clean.csv'},
    {name:'OD 订单', path:'outputs/od_orders.csv'},
    {name:'车辆与分钟缓存', path:'outputs/cache/'},
    {name:'统计结果', path:'outputs/stats/'},
    {name:'报告模板', path:'reports/project_report.md'}
  ], [{k:'name', t:'名称'}, {k:'path', t:'位置'}]);
}
"""


def generate_frontend(output_dir: Path, web_dir: Path, project_root: Path) -> Path:
    assets = web_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    data = build_frontend_data(output_dir, project_root)
    (web_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (assets / "app.css").write_text(APP_CSS, encoding="utf-8")
    (assets / "app.js").write_text(APP_JS, encoding="utf-8")
    with (web_dir / "data.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return web_dir / "index.html"


def serve_directory(directory: Path, host: str, port: int) -> None:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

    class ReusableServer(ThreadingHTTPServer):
        allow_reuse_address = True

    with ReusableServer((host, port), Handler) as server:
        print(f"Serving frontend at http://{host}:{port}/")
        server.serve_forever()
