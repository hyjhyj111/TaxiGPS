
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
