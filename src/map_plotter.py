#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import pickle
import hashlib
import sys
from functools import lru_cache
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import osmnx as ox
except ImportError:
    ox = None

try:
    import folium
    from folium.plugins import MarkerCluster
except ImportError:
    print("请先安装 folium: pip install folium")
    sys.exit(1)


CONFIG = {
    "MAP_CENTER": [22.52847, 114.05454],
    "MAP_ZOOM": 13,
    "VEHICLE_CACHE_DIR": "cache/vehicles/",
    "MINUTE_CACHE_DIR": "cache/minutes/",
    "OD_TABLE_PATH": "data/processed/od_table.csv",
    "ROAD_NETWORK_PATHS": [
        "shenzhen_drive.pkl",
        "data/shenzhen_drive.pkl",
        "cache/shenzhen_drive.pkl",
        "shenzhen_drive.graphml",
        "data/shenzhen_drive.graphml",
        "cache/shenzhen_drive.graphml",
        "d:/shenzhen_drive.pkl",
        "d:/shenzhen_drive.graphml",
    ],
    "BOUNDARY_PATHS": [
        "data/深圳市.json",
        "data/shenzhen.json",
        "docs/深圳市.json",
    ],
    "OUTPUT_MAP_DIR": "pages/maps/",
    "LOG_PATH": "logs/map_query.log",
    "MAX_TRAJECTORY_POINTS": 1200,
    "MAX_VEHICLE_DISPLAY": 1200,
    "MAX_OD_POINTS": 1200,
    "ROAD_CORRECTED_CACHE_DIR": "cache/road_corrected",
    "MAX_ROAD_CORRECTION_VEHICLES": 3,
    "MAX_ROAD_CORRECTION_INPUT_POINTS": 220,
    "MAX_ROAD_CORRECTION_CACHE_POINTS": 1200,
    "MAX_CORRECTED_PATH_POINTS": 1600,
    "MAX_CONGESTION_VEHICLES": 8,
    "MAX_CONGESTION_INPUT_POINTS": 90,
    "MAX_CONGESTION_SEGMENTS": 500,
    "DEFAULT_CONGESTION_BUCKET_MINUTES": 15,
    "DEFAULT_ETA_SPEED_KMH": 28.0,
    "BASELINE_SPEED_CACHE_PATH": "cache/edge_baseline_speed.csv",
    "DEFAULT_HIGHWAY_SPEED_KPH": {
        "motorway": 80.0,
        "motorway_link": 40.0,
        "trunk": 60.0,
        "trunk_link": 35.0,
        "primary": 50.0,
        "primary_link": 30.0,
        "secondary": 40.0,
        "secondary_link": 25.0,
        "tertiary": 35.0,
        "tertiary_link": 25.0,
        "unclassified": 30.0,
        "residential": 25.0,
        "living_street": 15.0,
    },
    "ANIMATION_MIN_DELAY_MS": 80,
    "ANIMATION_MAX_DELAY_MS": 2500,
    "ANIMATION_GAP_THRESHOLD_S": 1800,
    "ANIMATION_SPEED_CAP_KMH": 120,
    "ANIMATION_DRIFT_CAP_KM": 8.0,
    "ANIMATION_LOW_SPEED_DISPLAY_THRESHOLD_KMH": 5.0,
    "ANIMATION_MIN_DRIVING_DISPLAY_SPEED_KMH": 15.0,
}


def setup_logging():
    log_dir = os.path.dirname(CONFIG["LOG_PATH"])
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(CONFIG["LOG_PATH"], encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


log = setup_logging()


def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_html(content, save_path):
    ensure_parent_dir(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)
    return save_path


def safe_datetime(value):
    if value is None or value == "":
        return None
    return pd.to_datetime(value)


def format_time_tag(value):
    dt = safe_datetime(value)
    if dt is None:
        return "all"
    return dt.strftime("%Y%m%d_%H%M%S")


def haversine_distance(lat1, lng1, lat2, lng2):
    if pd.isna(lat1) or pd.isna(lng1) or pd.isna(lat2) or pd.isna(lng2):
        return np.nan

    r = 6371.0
    lat1_rad = radians(float(lat1))
    lng1_rad = radians(float(lng1))
    lat2_rad = radians(float(lat2))
    lng2_rad = radians(float(lng2))
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlng / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c


def resample_trajectory(df, max_points=1000):
    if df is None or len(df) <= max_points:
        return df.copy() if df is not None else df
    idx = np.linspace(0, len(df) - 1, num=max_points, dtype=int)
    return df.iloc[idx].drop_duplicates().reset_index(drop=True)


def normalize_vehicle_df(df):
    if df is None or len(df) == 0:
        return df

    rename_map = {}
    if "id" in df.columns and "vehicle_id" not in df.columns:
        rename_map["id"] = "vehicle_id"
    if "lon" in df.columns and "long" not in df.columns:
        rename_map["lon"] = "long"
    if "lng" in df.columns and "long" not in df.columns:
        rename_map["lng"] = "long"
    if "lat" in df.columns and "lati" not in df.columns:
        rename_map["lat"] = "lati"
    if "timestamp" in df.columns and "time" not in df.columns:
        rename_map["timestamp"] = "time"
    df = df.rename(columns=rename_map).copy()

    required = ["time", "long", "lati"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"车辆缓存缺少字段: {missing}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df["lati"] = pd.to_numeric(df["lati"], errors="coerce")
    if "status" in df.columns:
        df["status"] = pd.to_numeric(df["status"], errors="coerce").fillna(0).astype(int)
    else:
        df["status"] = 0
    if "speed" in df.columns:
        df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0)
    else:
        df["speed"] = 0.0
    if "vehicle_id" in df.columns:
        df["vehicle_id"] = df["vehicle_id"].astype(str).str.replace(r"\.0$", "", regex=True)

    df = df.dropna(subset=["time", "long", "lati"]).sort_values("time").reset_index(drop=True)
    return df


def normalize_minute_df(df, target_dt):
    if df is None or len(df) == 0:
        return df

    rename_map = {}
    if "id" in df.columns and "vehicle_id" not in df.columns:
        rename_map["id"] = "vehicle_id"
    if "lon" in df.columns and "long" not in df.columns:
        rename_map["lon"] = "long"
    if "lng" in df.columns and "long" not in df.columns:
        rename_map["lng"] = "long"
    if "lat" in df.columns and "lati" not in df.columns:
        rename_map["lat"] = "lati"
    df = df.rename(columns=rename_map).copy()

    required = ["vehicle_id", "long", "lati"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"分钟缓存缺少字段: {missing}")

    df["vehicle_id"] = df["vehicle_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df["lati"] = pd.to_numeric(df["lati"], errors="coerce")
    if "status" in df.columns:
        df["status"] = pd.to_numeric(df["status"], errors="coerce").fillna(0).astype(int)
    else:
        df["status"] = 0
    if "speed" in df.columns:
        df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0)
    else:
        df["speed"] = 0.0
    df["time"] = target_dt
    df = df.dropna(subset=["long", "lati"]).reset_index(drop=True)
    return df


def normalize_od_df(df):
    if df is None or len(df) == 0:
        return df

    rename_map = {}
    if "vehicle_id" in df.columns and "O_TAXI_ID" not in df.columns:
        rename_map["vehicle_id"] = "O_TAXI_ID"
    if "pickup_lat" in df.columns and "O_lat" not in df.columns:
        rename_map["pickup_lat"] = "O_lat"
    if "pickup_lng" in df.columns and "O_lng" not in df.columns:
        rename_map["pickup_lng"] = "O_lng"
    if "dropoff_lat" in df.columns and "D_lat" not in df.columns:
        rename_map["dropoff_lat"] = "D_lat"
    if "dropoff_lng" in df.columns and "D_lng" not in df.columns:
        rename_map["dropoff_lng"] = "D_lng"
    df = df.rename(columns=rename_map).copy()

    required = ["O_time", "D_time", "O_lat", "O_lng", "D_lat", "D_lng", "O_TAXI_ID"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"OD表缺少字段: {missing}")

    df["O_TAXI_ID"] = df["O_TAXI_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["O_time"] = pd.to_datetime(df["O_time"], errors="coerce")
    df["D_time"] = pd.to_datetime(df["D_time"], errors="coerce")
    for col in ["O_lat", "O_lng", "D_lat", "D_lng"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "OD_Time_s" not in df.columns:
        df["OD_Time_s"] = (df["D_time"] - df["O_time"]).dt.total_seconds()
    else:
        df["OD_Time_s"] = pd.to_numeric(df["OD_Time_s"], errors="coerce")

    if "OD_Dist_km" not in df.columns:
        df["OD_Dist_km"] = df.apply(
            lambda row: haversine_distance(row["O_lat"], row["O_lng"], row["D_lat"], row["D_lng"]),
            axis=1,
        )
    else:
        df["OD_Dist_km"] = pd.to_numeric(df["OD_Dist_km"], errors="coerce")

    df = df.dropna(subset=["O_time", "D_time", "O_lat", "O_lng", "D_lat", "D_lng"]).reset_index(drop=True)
    return df


@lru_cache(maxsize=256)
def _read_vehicle_cache(vehicle_file):
    if not os.path.exists(vehicle_file):
        return None
    return normalize_vehicle_df(pd.read_csv(vehicle_file))


@lru_cache(maxsize=256)
def _read_minute_cache(minute_file):
    if not os.path.exists(minute_file):
        return None
    return normalize_minute_df(pd.read_csv(minute_file), pd.Timestamp("1970-01-01"))


@lru_cache(maxsize=8)
def _read_od_cache(od_path, version_key=None):
    if not os.path.exists(od_path):
        return None
    return normalize_od_df(pd.read_csv(od_path))


def _file_version_key(path):
    if not os.path.exists(path):
        return "missing"
    stat = os.stat(path)
    return f"{int(stat.st_mtime)}-{stat.st_size}"


def add_shenzhen_boundary(m):
    for path in CONFIG["BOUNDARY_PATHS"]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    geojson = json.load(f)
                folium.GeoJson(
                    geojson,
                    name="深圳边界",
                    style_function=lambda feature: {
                        "fillColor": "#ffffff",
                        "color": "#5f6368",
                        "weight": 2,
                        "fillOpacity": 0.03,
                    },
                ).add_to(m)
                log.info("已加载深圳边界文件: %s", path)
                return
            except Exception as exc:
                log.warning("边界文件加载失败: %s, %s", path, exc)
                return


def add_map_layers(m, include_boundary=True, include_picker=True):
    if include_boundary:
        add_shenzhen_boundary(m)
    if include_picker:
        add_click_coordinate_picker(m)
    add_leaflet_layout_stabilizer(m)
    folium.LayerControl(collapsed=False).add_to(m)


def add_leaflet_layout_stabilizer(m):
    map_name = m.get_name()
    target_bounds = getattr(m, "_taxigps_fit_bounds", None)
    target_bounds_json = json.dumps(target_bounds, ensure_ascii=False) if target_bounds else "null"
    stabilizer_js = f"""
    (function __taxigpsInvalidateMapSize() {{
        var map = window[{json.dumps(map_name)}];
        if (!map || !map.invalidateSize) {{
            window.setTimeout(__taxigpsInvalidateMapSize, 50);
            return;
        }}
        var targetBounds = {target_bounds_json};
        var fitCount = 0;
        var pending = null;

        function refreshLeafletLayout(reason) {{
            if (!map || !map.invalidateSize) return;
            map.invalidateSize({{pan: false}});
            if (targetBounds && map.fitBounds && fitCount < 8) {{
                map.fitBounds(targetBounds, {{padding: [20, 20], animate: false}});
                fitCount += 1;
            }}
        }}

        function scheduleRefresh(reason, delay) {{
            if (pending) window.clearTimeout(pending);
            pending = window.setTimeout(function() {{
                pending = null;
                refreshLeafletLayout(reason);
            }}, delay || 0);
        }}

        [0, 80, 180, 360, 720, 1200].forEach(function(delay) {{
            window.setTimeout(function() {{ refreshLeafletLayout('startup-' + delay); }}, delay);
        }});
        window.addEventListener('load', function() {{ scheduleRefresh('load', 0); }});
        window.addEventListener('resize', function() {{ scheduleRefresh('window-resize', 80); }});

        var container = map.getContainer ? map.getContainer() : null;
        if (container && typeof ResizeObserver !== 'undefined') {{
            var observer = new ResizeObserver(function() {{
                scheduleRefresh('container-resize', 80);
            }});
            observer.observe(container);
        }}
    }})();
    """
    m.get_root().script.add_child(folium.Element(stabilizer_js))


def add_click_coordinate_picker(m):
    map_name = m.get_name()
    picker_id = f"coord-picker-{map_name}"
    picker_html = f"""
    <div id="{picker_id}" style="position:fixed;left:16px;bottom:18px;z-index:9999;background:rgba(255,255,255,0.96);border:1px solid rgba(148,163,184,0.45);border-radius:12px;padding:10px 12px;box-shadow:0 12px 30px rgba(15,23,42,0.14);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#0f172a;min-width:210px;">
      <div style="font-size:12px;font-weight:700;margin-bottom:4px;">地图选点</div>
      <div data-role="coord" style="font-size:12px;color:#475569;line-height:1.45;">点击地图显示经纬度</div>
    </div>
    """
    picker_js = f"""
    (function bindCoordinatePicker() {{
        var map = window[{json.dumps(map_name)}];
        var panel = document.getElementById({json.dumps(picker_id)});
        if (!map || !panel) {{
            window.setTimeout(bindCoordinatePicker, 50);
            return;
        }}
        var coordNode = panel.querySelector('[data-role="coord"]');
        var marker = null;
        map.on('click', function(e) {{
            var lat = e.latlng.lat;
            var lng = e.latlng.lng;
            coordNode.innerHTML = '纬度: <strong>' + lat.toFixed(6) + '</strong><br>经度: <strong>' + lng.toFixed(6) + '</strong>';
            if (!marker) {{
                marker = L.marker(e.latlng, {{title: '地图选点'}}).addTo(map);
            }} else {{
                marker.setLatLng(e.latlng);
            }}
            marker.bindPopup('纬度: ' + lat.toFixed(6) + '<br>经度: ' + lng.toFixed(6)).openPopup();
        }});
    }})();
    """
    m.get_root().html.add_child(folium.Element(picker_html))
    m.get_root().script.add_child(folium.Element(picker_js))


def build_map(df=None, zoom=None):
    m = folium.Map(location=CONFIG["MAP_CENTER"], zoom_start=zoom or CONFIG["MAP_ZOOM"], control_scale=True)
    if df is not None and len(df) > 0:
        try:
            if {"lati", "long"}.issubset(df.columns):
                bounds = [[df["lati"].min(), df["long"].min()], [df["lati"].max(), df["long"].max()]]
            elif {"O_lat", "O_lng", "D_lat", "D_lng"}.issubset(df.columns):
                lat_min = min(df["O_lat"].min(), df["D_lat"].min())
                lat_max = max(df["O_lat"].max(), df["D_lat"].max())
                lng_min = min(df["O_lng"].min(), df["D_lng"].min())
                lng_max = max(df["O_lng"].max(), df["D_lng"].max())
                bounds = [[lat_min, lng_min], [lat_max, lng_max]]
            else:
                bounds = None
            if bounds is not None:
                m.fit_bounds(bounds, padding=(20, 20))
                m._taxigps_fit_bounds = bounds
        except Exception:
            pass
    return m


def load_vehicle_trajectory(vehicle_id, start_time=None, end_time=None, log_result=True):
    vehicle_file = os.path.join(CONFIG["VEHICLE_CACHE_DIR"], f"{vehicle_id}.csv")
    cached = _read_vehicle_cache(vehicle_file)
    if cached is None:
        log.error(f"车辆 {vehicle_id} 的缓存文件不存在: {vehicle_file}")
        return None

    df = cached.copy()
    start_dt = safe_datetime(start_time)
    end_dt = safe_datetime(end_time)
    if start_dt is not None:
        df = df[df["time"] >= start_dt]
    if end_dt is not None:
        df = df[df["time"] <= end_dt]

    df = df.sort_values("time").reset_index(drop=True)
    if log_result:
        log.info(
            "车辆轨迹载入完成: vehicle_id=%s, start=%s, end=%s, points=%s",
            vehicle_id,
            start_dt,
            end_dt,
            len(df),
        )
    return df


def _road_network_candidates():
    env_path = os.environ.get("TAXIGPS_ROAD_NETWORK_PATH")
    paths = [env_path] if env_path else []
    paths.extend(CONFIG["ROAD_NETWORK_PATHS"])
    cleaned = []
    for path in paths:
        if not path:
            continue
        normalized = os.path.expanduser(str(path))
        if normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def find_road_network_path():
    for path in _road_network_candidates():
        if os.path.exists(path):
            return path
    return None


def road_network_status():
    path = find_road_network_path()
    return {
        "available": bool(path),
        "path": path,
        "candidates": _road_network_candidates(),
        "osmnx_available": ox is not None,
        "networkx_available": nx is not None,
    }


@lru_cache(maxsize=2)
def _load_road_network_cached(path, version_key):
    del version_key
    start = perf_counter()
    if path.lower().endswith(".pkl"):
        with open(path, "rb") as f:
            graph = pickle.load(f)
        method = "pkl"
    elif path.lower().endswith(".graphml"):
        if ox is None:
            raise RuntimeError("加载 graphml 路网需要安装 osmnx。")
        graph = ox.load_graphml(path)
        method = "graphml"
    else:
        raise ValueError(f"不支持的路网文件格式: {path}")

    load_seconds = perf_counter() - start
    node_count = graph.number_of_nodes() if hasattr(graph, "number_of_nodes") else 0
    edge_count = graph.number_of_edges() if hasattr(graph, "number_of_edges") else 0
    log.info(
        "路网加载完成: path=%s, method=%s, nodes=%s, edges=%s, seconds=%.2f",
        path,
        method,
        node_count,
        edge_count,
        load_seconds,
    )
    return graph, {
        "path": path,
        "method": method,
        "load_seconds": load_seconds,
        "node_count": node_count,
        "edge_count": edge_count,
    }


def load_road_network(path=None):
    network_path = path or find_road_network_path()
    if not network_path:
        return None, {
            "available": False,
            "error": "未找到 shenzhen_drive.pkl 或 shenzhen_drive.graphml。",
            "candidates": _road_network_candidates(),
        }
    return _load_road_network_cached(network_path, _file_version_key(network_path))


def _nearest_node_bruteforce(graph, lng, lat):
    best_node = None
    best_dist = float("inf")
    for node, attrs in graph.nodes(data=True):
        node_lng = attrs.get("x", attrs.get("lon", attrs.get("lng")))
        node_lat = attrs.get("y", attrs.get("lat"))
        if node_lng is None or node_lat is None:
            continue
        dist = (float(node_lng) - float(lng)) ** 2 + (float(node_lat) - float(lat)) ** 2
        if dist < best_dist:
            best_dist = dist
            best_node = node
    if best_node is None:
        raise ValueError("路网节点缺少 x/y 或 lon/lat 坐标，无法执行最近邻匹配。")
    return best_node


def nearest_road_node(graph, lng, lat):
    if ox is not None:
        try:
            return ox.distance.nearest_nodes(graph, float(lng), float(lat))
        except Exception:
            log.exception("OSMnx 最近邻匹配失败，改用节点遍历匹配。")
    return _nearest_node_bruteforce(graph, lng, lat)


def _node_lat_lng(graph, node):
    attrs = graph.nodes[node]
    lng = attrs.get("x", attrs.get("lon", attrs.get("lng")))
    lat = attrs.get("y", attrs.get("lat"))
    if lng is None or lat is None:
        return None
    return [float(lat), float(lng)]


def _shortest_path_nodes(graph, source, target, undirected_graph=None, weight="length"):
    if source == target:
        return [source], "same_node"
    if nx is None:
        raise RuntimeError("路网最短路径需要安装 networkx。")
    try:
        return nx.shortest_path(graph, source, target, weight=weight), "directed"
    except Exception as exc:
        if undirected_graph is None:
            raise exc
        return nx.shortest_path(undirected_graph, source, target, weight=weight), "undirected"


def normalize_highway(value):
    if isinstance(value, (list, tuple, set)):
        value = next(iter(value), None)
    return str(value or "road")


def _coerce_edge_part(value):
    if pd.isna(value):
        return value
    try:
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
    except (TypeError, ValueError):
        pass
    return str(value)


def _edge_identity(u, v, key=0):
    return (_coerce_edge_part(u), _coerce_edge_part(v), _coerce_edge_part(key))


def _iter_graph_edges(graph):
    if graph is None:
        return []
    try:
        return graph.edges(keys=True, data=True)
    except TypeError:
        return ((u, v, 0, data) for u, v, data in graph.edges(data=True))


def _graph_edge_rows(graph):
    rows = []
    for u, v, key, data in _iter_graph_edges(graph):
        data = data or {}
        rows.append(
            {
                "edge_u": _coerce_edge_part(u),
                "edge_v": _coerce_edge_part(v),
                "edge_key": _coerce_edge_part(key),
                "length": float(data.get("length", 0.0) or 0.0),
                "highway_type": normalize_highway(data.get("highway")),
            }
        )
    return pd.DataFrame(rows)


def build_edge_baseline_speed_cache(matched_track, graph, min_samples=3, save_path=None):
    required = {"id", "speed", "edge_u", "edge_v", "edge_key"}
    if matched_track is None:
        matched_track = pd.DataFrame()
    missing = required - set(matched_track.columns)
    if missing:
        raise ValueError(f"校正轨迹缺少字段: {sorted(missing)}")

    track = matched_track.copy()
    for col in ["edge_u", "edge_v", "edge_key"]:
        track[col] = track[col].map(_coerce_edge_part)
    track["speed"] = pd.to_numeric(track["speed"], errors="coerce")
    valid = track[track["speed"].between(1, CONFIG["ANIMATION_SPEED_CAP_KMH"])].dropna(subset=["edge_u", "edge_v", "edge_key"])

    if len(valid) == 0:
        speed_stats = pd.DataFrame(columns=["edge_u", "edge_v", "edge_key", "avg_speed", "sample_count", "vehicle_count"])
    else:
        speed_stats = (
            valid.groupby(["edge_u", "edge_v", "edge_key"], as_index=False)
            .agg(
                avg_speed=("speed", "mean"),
                sample_count=("speed", "size"),
                vehicle_count=("id", pd.Series.nunique),
            )
            .reset_index(drop=True)
        )

    edge_info = _graph_edge_rows(graph)
    if len(edge_info) > 0:
        speed_stats = edge_info.merge(speed_stats, on=["edge_u", "edge_v", "edge_key"], how="left")
    else:
        speed_stats["length"] = np.nan
        speed_stats["highway_type"] = "road"

    speed_stats["avg_speed"] = pd.to_numeric(speed_stats.get("avg_speed"), errors="coerce")
    speed_stats["sample_count"] = pd.to_numeric(speed_stats.get("sample_count"), errors="coerce").fillna(0).astype(int)
    speed_stats["vehicle_count"] = pd.to_numeric(speed_stats.get("vehicle_count"), errors="coerce").fillna(0).astype(int)
    speed_stats["highway_type"] = speed_stats["highway_type"].map(normalize_highway)
    speed_stats["reliable"] = (speed_stats["sample_count"] >= int(min_samples)) & speed_stats["avg_speed"].between(1, CONFIG["ANIMATION_SPEED_CAP_KMH"])
    speed_stats["route_cost"] = np.where(
        speed_stats["reliable"],
        pd.to_numeric(speed_stats["length"], errors="coerce") / speed_stats["avg_speed"] * 3.6,
        np.nan,
    )

    reliable = speed_stats[speed_stats["reliable"]].copy()
    highway_median_speed = reliable.groupby("highway_type")["avg_speed"].median().dropna().to_dict()
    meta = {
        "success": True,
        "edge_rows": int(len(speed_stats)),
        "observed_edges": int((speed_stats["sample_count"] > 0).sum()),
        "reliable_edges": int(speed_stats["reliable"].sum()),
        "min_samples": int(min_samples),
        "highway_median_speed": {str(k): float(v) for k, v in highway_median_speed.items()},
        "method": "全日校正轨迹道路边平均速度；样本不足时按 highway 中位数、道路类型默认速度、全局速度回退。",
    }
    if save_path:
        ensure_parent_dir(save_path)
        speed_stats.to_csv(save_path, index=False)
        meta["cache_path"] = save_path
    return speed_stats, meta


def apply_baseline_route_cost(graph, speed_stats, highway_median_speed=None, default_speed_kph=None, fallback_speed=30.0, min_samples=3):
    if graph is None:
        raise ValueError("路网未加载，无法写入 route_cost。")
    highway_median_speed = highway_median_speed or {}
    default_speed_kph = default_speed_kph or CONFIG["DEFAULT_HIGHWAY_SPEED_KPH"]
    observed_speed = {}
    if speed_stats is not None and len(speed_stats) > 0:
        stats = speed_stats.copy()
        for col in ["edge_u", "edge_v", "edge_key"]:
            stats[col] = stats[col].map(_coerce_edge_part)
        stats["avg_speed"] = pd.to_numeric(stats.get("avg_speed"), errors="coerce")
        stats["sample_count"] = pd.to_numeric(stats.get("sample_count"), errors="coerce").fillna(0)
        for row in stats.itertuples(index=False):
            if getattr(row, "sample_count", 0) >= min_samples and pd.notna(getattr(row, "avg_speed", np.nan)):
                speed = float(getattr(row, "avg_speed"))
                if 1 <= speed <= CONFIG["ANIMATION_SPEED_CAP_KMH"]:
                    observed_speed[_edge_identity(row.edge_u, row.edge_v, row.edge_key)] = speed

    updated = 0
    fallback_edges = 0
    observed_edges = 0
    for u, v, key, data in _iter_graph_edges(graph):
        data = data or {}
        edge_id = _edge_identity(u, v, key)
        road_type = normalize_highway(data.get("highway"))
        speed = observed_speed.get(edge_id)
        if speed is None:
            speed = highway_median_speed.get(road_type)
        if speed is None:
            speed = default_speed_kph.get(road_type)
        if speed is None:
            speed = float(fallback_speed)
        speed = max(1.0, min(CONFIG["ANIMATION_SPEED_CAP_KMH"], float(speed)))
        length = float(data.get("length", 0.0) or 0.0)
        if length <= 0:
            point_a = _node_lat_lng(graph, u)
            point_b = _node_lat_lng(graph, v)
            if point_a and point_b:
                length = max(1.0, haversine_distance(point_a[0], point_a[1], point_b[0], point_b[1]) * 1000.0)
            else:
                length = 1.0
        data["baseline_speed_kph"] = speed
        data["route_cost"] = length / speed * 3.6
        updated += 1
        if edge_id in observed_speed:
            observed_edges += 1
        else:
            fallback_edges += 1
    return {"updated_edges": updated, "observed_edges": observed_edges, "fallback_edges": fallback_edges}


def _best_edge_data(graph, source, target, weight="length"):
    edge_data = graph.get_edge_data(source, target) if hasattr(graph, "get_edge_data") else None
    if not edge_data:
        return None, None
    if isinstance(edge_data, dict) and all(isinstance(value, dict) for value in edge_data.values()):
        candidates = [(key, attrs) for key, attrs in edge_data.items() if attrs is not None]
    else:
        candidates = [(0, edge_data)]
    if not candidates:
        return None, None
    def candidate_weight(item):
        _, attrs = item
        value = attrs.get(weight, attrs.get("length", 1.0))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")
    return min(candidates, key=candidate_weight)


def _geometry_points_for_edge(graph, source, target, weight="length"):
    _, data = _best_edge_data(graph, source, target, weight=weight)
    if data:
        geom = data.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            return [[float(lat), float(lng)] for lng, lat in geom.coords]
    start = _node_lat_lng(graph, source)
    end = _node_lat_lng(graph, target)
    return [point for point in [start, end] if point]


def _route_summary(graph, path_nodes, weight="length"):
    points = []
    edges = []
    distance_m = 0.0
    route_cost_s = 0.0
    for idx, node in enumerate(path_nodes or []):
        if idx == 0:
            point = _node_lat_lng(graph, node)
            if point:
                points.append(point)
            continue
        source = path_nodes[idx - 1]
        target = node
        key, data = _best_edge_data(graph, source, target, weight=weight)
        data = data or {}
        length = float(data.get("length", 0.0) or 0.0)
        cost = float(data.get("route_cost", length) or 0.0)
        distance_m += length
        route_cost_s += cost
        edge_points = _geometry_points_for_edge(graph, source, target, weight=weight)
        for point in edge_points:
            if points and points[-1] == point:
                continue
            points.append(point)
        edges.append(
            {
                "u": source,
                "v": target,
                "key": key,
                "length_m": length,
                "route_cost_s": cost,
                "baseline_speed_kph": float(data.get("baseline_speed_kph", 0.0) or 0.0),
                "highway": normalize_highway(data.get("highway")),
            }
        )
    return {
        "nodes": list(path_nodes or []),
        "edges": edges,
        "edge_count": len(edges),
        "distance_m": float(distance_m),
        "route_cost_s": float(route_cost_s),
        "points": points,
    }


def plan_baseline_routes(graph, origin_node, dest_node):
    if graph is None:
        return {"success": False, "error": "路网未加载。"}
    if nx is None:
        return {"success": False, "error": "路线规划需要安装 networkx。"}
    try:
        shortest_nodes = nx.shortest_path(graph, origin_node, dest_node, weight="length")
        fastest_nodes = nx.shortest_path(graph, origin_node, dest_node, weight="route_cost")
    except nx.NodeNotFound as exc:
        return {"success": False, "error": f"起终点节点不存在: {exc}"}
    except nx.NetworkXNoPath as exc:
        return {"success": False, "error": f"起终点之间没有可用路网路径: {exc}"}
    return {
        "success": True,
        "origin_node": origin_node,
        "dest_node": dest_node,
        "shortest": _route_summary(graph, shortest_nodes, weight="length"),
        "fastest": _route_summary(graph, fastest_nodes, weight="route_cost"),
        "method": "同一路网分别使用 length 和静态 route_cost 执行 Dijkstra 路径搜索。",
    }


def build_baseline_speed_cache_from_vehicles(vehicle_ids, query_date=None, graph=None, network_meta=None, min_samples=3, save_path=None):
    cleaned_vehicle_ids = []
    for vehicle_id in vehicle_ids or []:
        vehicle_id = str(vehicle_id).strip()
        if vehicle_id and vehicle_id not in cleaned_vehicle_ids:
            cleaned_vehicle_ids.append(vehicle_id)

    if graph is None:
        graph, network_meta = load_road_network()
    network_meta = network_meta or {}
    if graph is None:
        return pd.DataFrame(), {"success": False, "error": network_meta.get("error", "路网未加载。"), "network": network_meta}

    start_time = None
    end_time = None
    if query_date is not None:
        start_time = pd.Timestamp(query_date).normalize()
        end_time = start_time + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    matched_frames = []
    matching_rows = []
    for vehicle_id in cleaned_vehicle_ids:
        df = load_vehicle_trajectory(vehicle_id, start_time, end_time)
        if df is None or len(df) < 2:
            matching_rows.append({"vehicle_id": vehicle_id, "success": False, "message": "全天轨迹点不足。"})
            continue
        df = df.copy()
        df["vehicle_id"] = vehicle_id
        result = approximate_hmm_match_trajectory(df, graph=graph, network_meta=network_meta)
        meta = result["meta"]
        matching_rows.append(
            {
                "vehicle_id": vehicle_id,
                "success": bool(meta.get("success")),
                "raw_points": meta.get("raw_points", len(df)),
                "sampled_points": meta.get("sampled_points", 0),
                "matched_nodes": meta.get("matched_nodes", 0),
                "segments": meta.get("segments", 0),
                "message": meta.get("error", "基准速度样本已生成。"),
            }
        )
        segments = result.get("segments")
        if segments is not None and len(segments) > 0:
            track = segments.rename(
                columns={
                    "vehicle_id": "id",
                    "speed_kmh": "speed",
                }
            )
            matched_frames.append(track)

    if matched_frames:
        matched_track = pd.concat(matched_frames, ignore_index=True)
    else:
        matched_track = pd.DataFrame(columns=["id", "speed", "edge_u", "edge_v", "edge_key"])

    speed_stats, meta = build_edge_baseline_speed_cache(
        matched_track,
        graph,
        min_samples=min_samples,
        save_path=save_path or CONFIG["BASELINE_SPEED_CACHE_PATH"],
    )
    meta.update(
        {
            "success": True,
            "vehicles": len(cleaned_vehicle_ids),
            "query_date": str(pd.Timestamp(query_date).date()) if query_date is not None else "all",
            "matching": matching_rows,
            "network": network_meta,
            "track_source": "车辆缓存 + 既有近似匹配结果；未读取原始 GPS 大表。",
        }
    )
    return speed_stats, meta


def plan_baseline_routes_between_points(origin_lat, origin_lng, dest_lat, dest_lng, vehicle_ids=None, query_date=None, graph=None, network_meta=None):
    if graph is None:
        graph, network_meta = load_road_network()
    network_meta = network_meta or {}
    if graph is None:
        return {"success": False, "error": network_meta.get("error", "路网未加载。"), "network": network_meta}

    try:
        origin_node = nearest_road_node(graph, origin_lng, origin_lat)
        dest_node = nearest_road_node(graph, dest_lng, dest_lat)
    except Exception as exc:
        log.exception("基准路线起终点最近节点匹配失败")
        return {"success": False, "error": f"起终点无法匹配到路网节点: {exc}", "network": network_meta}

    speed_stats, speed_meta = build_baseline_speed_cache_from_vehicles(
        vehicle_ids or [],
        query_date=query_date,
        graph=graph,
        network_meta=network_meta,
        save_path=CONFIG["BASELINE_SPEED_CACHE_PATH"],
    )
    cost_meta = apply_baseline_route_cost(graph, speed_stats, speed_meta.get("highway_median_speed", {}))
    result = plan_baseline_routes(graph, origin_node, dest_node)
    result.update(
        {
            "origin": {"lat": float(origin_lat), "lng": float(origin_lng), "node": origin_node},
            "destination": {"lat": float(dest_lat), "lng": float(dest_lng), "node": dest_node},
            "speed_meta": speed_meta,
            "cost_meta": cost_meta,
            "network": network_meta,
        }
    )
    return result


def _baseline_route_legend_html(result):
    network_label = result.get("network", {}).get("path") or "未加载"
    speed_meta = result.get("speed_meta", {})
    return f"""
    <div style="position:fixed;right:16px;top:16px;z-index:9999;background:rgba(255,255,255,0.96);border:1px solid rgba(148,163,184,0.35);border-radius:12px;padding:12px 14px;box-shadow:0 12px 32px rgba(15,23,42,0.12);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-width:240px;max-width:360px;">
      <div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:6px;">最短 / 最快路线</div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;"><span style="width:28px;border-top:5px solid #2563eb;"></span><span style="font-size:12px;color:#334155;">最短距离路线 length</span></div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;"><span style="width:28px;border-top:5px solid #16a34a;"></span><span style="font-size:12px;color:#334155;">基准最快路线 route_cost</span></div>
      <div style="font-size:11px;color:#64748b;line-height:1.45;border-top:1px solid rgba(148,163,184,0.25);padding-top:8px;">路网: {network_label}<br>可靠速度边: {speed_meta.get('reliable_edges', 0)} / {speed_meta.get('edge_rows', 0)}</div>
    </div>
    """


def add_dual_point_picker(m):
    map_name = m.get_name()
    picker_id = f"dual-route-picker-{map_name}"
    picker_html = f"""
    <div id="{picker_id}" style="position:fixed;left:16px;bottom:18px;z-index:9999;background:rgba(255,255,255,0.96);border:1px solid rgba(148,163,184,0.45);border-radius:12px;padding:10px 12px;box-shadow:0 12px 30px rgba(15,23,42,0.14);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#0f172a;min-width:260px;">
      <div style="font-size:12px;font-weight:700;margin-bottom:4px;">连续选点</div>
      <div data-role="coord" style="font-size:12px;color:#475569;line-height:1.45;">依次点击起点、终点；第三次点击会重新开始。</div>
    </div>
    """
    picker_js = f"""
    (function bindDualRoutePicker() {{
        var map = window[{json.dumps(map_name)}];
        var panel = document.getElementById({json.dumps(picker_id)});
        if (!map || !panel || typeof L === 'undefined') {{
            window.setTimeout(bindDualRoutePicker, 50);
            return;
        }}
        var coordNode = panel.querySelector('[data-role="coord"]');
        var points = [];
        var markers = [];
        var preview = null;
        function clearSelection() {{
            markers.forEach(function(marker) {{ map.removeLayer(marker); }});
            markers = [];
            points = [];
            if (preview) map.removeLayer(preview);
            preview = null;
        }}
        map.on('click', function(e) {{
            if (points.length === 2) clearSelection();
            var label = points.length === 0 ? '起点' : '终点';
            var point = {{lat: e.latlng.lat, lng: e.latlng.lng}};
            points.push(point);
            var marker = L.marker(e.latlng, {{title: label}}).addTo(map).bindTooltip(label).openTooltip();
            markers.push(marker);
            if (points.length === 1) {{
                coordNode.innerHTML = '起点 纬度: <strong>' + point.lat.toFixed(6) + '</strong><br>起点 经度: <strong>' + point.lng.toFixed(6) + '</strong><br>继续点击终点';
                return;
            }}
            preview = L.polyline([[points[0].lat, points[0].lng], [points[1].lat, points[1].lng]], {{
                color: '#64748b',
                weight: 3,
                dashArray: '6,6',
                opacity: 0.8
            }}).addTo(map);
            coordNode.innerHTML = '起点: ' + points[0].lat.toFixed(6) + ', ' + points[0].lng.toFixed(6) + '<br>终点: ' + points[1].lat.toFixed(6) + ', ' + points[1].lng.toFixed(6) + '<br>将坐标填入页面可重新计算路线';
        }});
    }})();
    """
    m.get_root().html.add_child(folium.Element(picker_html))
    m.get_root().script.add_child(folium.Element(picker_js))


def plot_baseline_route_comparison(route_result, save_path=None):
    if not route_result or not route_result.get("success"):
        return None
    shortest_points = route_result.get("shortest", {}).get("points", [])
    fastest_points = route_result.get("fastest", {}).get("points", [])
    all_points = shortest_points + fastest_points
    map_df = pd.DataFrame({"lati": [point[0] for point in all_points], "long": [point[1] for point in all_points]})
    m = build_map(map_df)

    if shortest_points:
        folium.PolyLine(
            shortest_points,
            color="#2563eb",
            weight=6,
            opacity=0.86,
            tooltip=f"最短距离路线 {route_result['shortest']['distance_m'] / 1000:.2f} km",
        ).add_to(m)
    if fastest_points:
        folium.PolyLine(
            fastest_points,
            color="#16a34a",
            weight=5,
            opacity=0.9,
            dash_array="8,6" if fastest_points == shortest_points else None,
            tooltip=f"基准最快路线 {route_result['fastest']['route_cost_s'] / 60:.1f} min",
        ).add_to(m)

    origin = route_result.get("origin", {})
    destination = route_result.get("destination", {})
    if origin:
        folium.Marker([origin["lat"], origin["lng"]], tooltip=f"起点 node={origin.get('node')}", icon=folium.Icon(color="green", icon="play")).add_to(m)
    if destination:
        folium.Marker([destination["lat"], destination["lng"]], tooltip=f"终点 node={destination.get('node')}", icon=folium.Icon(color="red", icon="flag")).add_to(m)

    m.get_root().html.add_child(folium.Element(_baseline_route_legend_html(route_result)))
    add_dual_point_picker(m)
    add_map_layers(m, include_boundary=True, include_picker=False)
    output_path = save_path or os.path.join(CONFIG["OUTPUT_MAP_DIR"], "baseline_route_comparison.html")
    ensure_parent_dir(output_path)
    m.save(output_path)
    return output_path


def _edge_length_km(graph, source, target):
    if source == target:
        return 0.0
    edge_data = graph.get_edge_data(source, target) if hasattr(graph, "get_edge_data") else None
    if not edge_data and hasattr(graph, "get_edge_data"):
        edge_data = graph.get_edge_data(target, source)
    if not edge_data:
        point_a = _node_lat_lng(graph, source)
        point_b = _node_lat_lng(graph, target)
        if point_a and point_b:
            return haversine_distance(point_a[0], point_a[1], point_b[0], point_b[1])
        return np.nan
    if isinstance(edge_data, dict) and all(isinstance(value, dict) for value in edge_data.values()):
        lengths = [attrs.get("length") for attrs in edge_data.values() if attrs and attrs.get("length") is not None]
    else:
        lengths = [edge_data.get("length")] if edge_data.get("length") is not None else []
    if not lengths:
        point_a = _node_lat_lng(graph, source)
        point_b = _node_lat_lng(graph, target)
        if point_a and point_b:
            return haversine_distance(point_a[0], point_a[1], point_b[0], point_b[1])
        return np.nan
    return float(min(lengths)) / 1000.0


def _path_length_km(graph, path_nodes):
    total = 0.0
    for idx in range(1, len(path_nodes or [])):
        length_km = _edge_length_km(graph, path_nodes[idx - 1], path_nodes[idx])
        if pd.notna(length_km):
            total += float(length_km)
    return total


def _segment_key(source, target):
    return f"{source}->{target}"


def _safe_cache_token(value):
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _road_network_cache_token(network_meta=None):
    network_meta = network_meta or {}
    path = network_meta.get("path") or "missing-road-network"
    version = _file_version_key(path) if path and os.path.exists(path) else "missing"
    payload = f"road-corrected-v5-edge-geometry-time-speed|{path}|{version}|{CONFIG['MAX_ROAD_CORRECTION_CACHE_POINTS']}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def road_corrected_vehicle_cache_path(vehicle_id, network_meta=None):
    cache_dir = CONFIG["ROAD_CORRECTED_CACHE_DIR"]
    token = _road_network_cache_token(network_meta)
    filename = f"{_safe_cache_token(vehicle_id)}_{token}.csv"
    return os.path.join(cache_dir, filename)


def road_corrected_vehicle_coverage_path(vehicle_id, network_meta=None):
    return road_corrected_vehicle_cache_path(vehicle_id, network_meta) + ".coverage.json"


def _serialize_ts(value):
    dt = safe_datetime(value)
    return pd.Timestamp(dt).isoformat() if dt is not None else None


def _read_road_cache_coverage(path):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        intervals = payload.get("coverage", []) if isinstance(payload, dict) else []
    except Exception:
        log.exception("车辆路网校正缓存覆盖信息读取失败: %s", path)
        return []
    cleaned = []
    for item in intervals:
        start = safe_datetime(item.get("start")) if isinstance(item, dict) else None
        end = safe_datetime(item.get("end")) if isinstance(item, dict) else None
        if start is not None and end is not None and start <= end:
            cleaned.append((pd.Timestamp(start), pd.Timestamp(end)))
    return cleaned


def _merge_coverage_intervals(intervals):
    cleaned = sorted(
        [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in intervals if start is not None and end is not None and start <= end],
        key=lambda item: item[0],
    )
    if not cleaned:
        return []
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _write_road_cache_coverage(path, intervals):
    ensure_parent_dir(path)
    payload = {
        "coverage": [
            {
                "start": _serialize_ts(start),
                "end": _serialize_ts(end),
            }
            for start, end in _merge_coverage_intervals(intervals)
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _missing_coverage_intervals(start_time, end_time, coverage):
    start = safe_datetime(start_time)
    end = safe_datetime(end_time)
    if start is None or end is None:
        return []
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start > end:
        return []
    missing = []
    cursor = start
    for covered_start, covered_end in _merge_coverage_intervals(coverage):
        if covered_end < cursor:
            continue
        if covered_start > end:
            break
        if covered_start > cursor:
            missing.append((cursor, min(covered_start, end)))
        if covered_end >= cursor:
            cursor = max(cursor, covered_end)
        if cursor >= end:
            break
    if cursor < end:
        missing.append((cursor, end))
    return [(item_start, item_end) for item_start, item_end in missing if item_start < item_end]


def _corrected_rows_to_points(rows):
    if rows is None or len(rows) == 0:
        return []
    if not {"matched_lat", "matched_lon"}.issubset(rows.columns):
        return []
    points = []
    for _, row in rows.dropna(subset=["matched_lat", "matched_lon"]).iterrows():
        point = [float(row["matched_lat"]), float(row["matched_lon"])]
        if points and points[-1] == point:
            continue
        points.append(point)
    return points


def _distance_ratios_for_points(points):
    if not points:
        return []
    if len(points) == 1:
        return [0.0]

    cumulative = [0.0]
    total_distance = 0.0
    for idx in range(1, len(points)):
        prev = points[idx - 1]
        current = points[idx]
        distance = haversine_distance(prev[0], prev[1], current[0], current[1])
        if pd.isna(distance):
            distance = 0.0
        total_distance += float(distance)
        cumulative.append(total_distance)

    if total_distance <= 0:
        return [idx / max(1, len(points) - 1) for idx in range(len(points))]
    return [distance / total_distance for distance in cumulative]


def _interpolate_times_for_points(points, start_time, end_time):
    if not points:
        return []
    start_ts = pd.Timestamp(start_time)
    end_ts = pd.Timestamp(end_time)
    if len(points) == 1 or end_ts <= start_ts:
        return [start_ts for _ in points]

    duration_ns = end_ts.value - start_ts.value
    return [pd.Timestamp(start_ts.value + int(round(duration_ns * ratio))) for ratio in _distance_ratios_for_points(points)]


def _interpolate_values_for_points(points, start_value, end_value):
    if not points:
        return []
    try:
        start_number = float(start_value)
    except (TypeError, ValueError):
        start_number = 0.0
    try:
        end_number = float(end_value)
    except (TypeError, ValueError):
        end_number = start_number
    return [start_number + (end_number - start_number) * ratio for ratio in _distance_ratios_for_points(points)]


def _edge_geometry_points(graph, source, target, weight="length"):
    points = _geometry_points_for_edge(graph, source, target, weight=weight)
    if points:
        return points
    start = _node_lat_lng(graph, source)
    end = _node_lat_lng(graph, target)
    return [point for point in [start, end] if point]


def _normalize_corrected_cache_rows(rows, vehicle_id=None):
    if rows is None:
        rows = pd.DataFrame()
    rows = rows.copy()
    if len(rows) == 0:
        return pd.DataFrame(
            columns=[
                "vehicle_id",
                "time",
                "status",
                "speed",
                "raw_lon",
                "raw_lat",
                "matched_lon",
                "matched_lat",
                "matched_node",
                "edge_u",
                "edge_v",
                "edge_key",
                "path_mode",
                "sequence",
            ]
        )
    if vehicle_id is not None:
        rows["vehicle_id"] = str(vehicle_id)
    if "time" in rows.columns:
        rows["time"] = pd.to_datetime(rows["time"], errors="coerce")
    for col in ["status", "matched_node", "edge_u", "edge_v", "edge_key", "sequence"]:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    for col in ["speed", "raw_lon", "raw_lat", "matched_lon", "matched_lat"]:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    if "sequence" not in rows.columns:
        rows["sequence"] = range(len(rows))
    rows = rows.dropna(subset=["time", "matched_lon", "matched_lat"]).sort_values(["time", "sequence"]).reset_index(drop=True)
    return rows


def speed_to_color(speed_kmh):
    if speed_kmh is None or pd.isna(speed_kmh):
        return "#9ca3af"
    speed = float(speed_kmh)
    if speed < 15:
        return "#dc2626"
    if speed < 25:
        return "#f97316"
    if speed < 40:
        return "#eab308"
    return "#16a34a"


def speed_to_level(speed_kmh):
    if speed_kmh is None or pd.isna(speed_kmh):
        return "未知"
    speed = float(speed_kmh)
    if speed < 15:
        return "严重拥堵"
    if speed < 25:
        return "拥堵"
    if speed < 40:
        return "缓行"
    return "畅通"


def approximate_hmm_match_trajectory(df, graph=None, network_meta=None):
    if df is None or len(df) < 2:
        return {
            "segments": pd.DataFrame(),
            "matched_nodes": [],
            "meta": {"success": False, "error": "轨迹点不足，无法执行地图匹配。"},
        }

    if graph is None:
        graph, network_meta = load_road_network()
    network_meta = network_meta or {}
    if graph is None:
        return {
            "segments": pd.DataFrame(),
            "matched_nodes": [],
            "meta": {"success": False, "error": network_meta.get("error", "路网未加载。"), "network": network_meta},
        }

    sample_df = resample_trajectory(df, CONFIG["MAX_CONGESTION_INPUT_POINTS"])
    matched_nodes = []
    nearest_failures = 0
    for _, row in sample_df.iterrows():
        try:
            matched_nodes.append(nearest_road_node(graph, row["long"], row["lati"]))
        except Exception:
            nearest_failures += 1
            matched_nodes.append(None)
            log.exception("HMM 近似匹配最近节点失败: lng=%s, lat=%s", row.get("long"), row.get("lati"))

    undirected_graph = None
    rows = []
    path_failures = 0
    undirected_segments = 0
    same_node_segments = 0

    for idx in range(1, len(sample_df)):
        source = matched_nodes[idx - 1]
        target = matched_nodes[idx]
        if source is None or target is None:
            continue

        prev_row = sample_df.iloc[idx - 1]
        row = sample_df.iloc[idx]
        time_diff_s = (row["time"] - prev_row["time"]).total_seconds() if pd.notna(row["time"]) and pd.notna(prev_row["time"]) else 0
        if time_diff_s <= 0:
            continue

        gps_distance_km = haversine_distance(prev_row["lati"], prev_row["long"], row["lati"], row["long"])
        if pd.isna(gps_distance_km) or gps_distance_km > CONFIG["ANIMATION_DRIFT_CAP_KM"]:
            continue

        observed_speed = float(row.get("speed", 0.0) or 0.0)
        if observed_speed <= 0:
            observed_speed = float(gps_distance_km / (time_diff_s / 3600.0)) if time_diff_s > 0 else 0.0
        if observed_speed > CONFIG["ANIMATION_SPEED_CAP_KMH"]:
            continue

        if undirected_graph is None and source != target and hasattr(graph, "is_directed") and graph.is_directed():
            try:
                undirected_graph = graph.to_undirected(as_view=True)
            except TypeError:
                undirected_graph = graph.to_undirected()
        try:
            path_nodes, mode = _shortest_path_nodes(graph, source, target, undirected_graph=undirected_graph)
            if mode == "undirected":
                undirected_segments += 1
            elif mode == "same_node":
                same_node_segments += 1
        except Exception:
            path_failures += 1
            log.exception("HMM 近似匹配转移路径失败: source=%s, target=%s", source, target)
            continue

        path_distance_km = _path_length_km(graph, path_nodes)
        if path_distance_km <= 0 or pd.isna(path_distance_km):
            path_distance_km = max(float(gps_distance_km), 0.001)
        segment_speed = min(CONFIG["ANIMATION_SPEED_CAP_KMH"], max(1.0, path_distance_km / (time_diff_s / 3600.0)))
        if observed_speed > 0:
            segment_speed = min(CONFIG["ANIMATION_SPEED_CAP_KMH"], max(1.0, 0.65 * segment_speed + 0.35 * observed_speed))

        for node_idx in range(1, len(path_nodes)):
            edge_source = path_nodes[node_idx - 1]
            edge_target = path_nodes[node_idx]
            edge_key, _ = _best_edge_data(graph, edge_source, edge_target, weight="length")
            point_a = _node_lat_lng(graph, edge_source)
            point_b = _node_lat_lng(graph, edge_target)
            if not point_a or not point_b:
                continue
            edge_length_km = _edge_length_km(graph, edge_source, edge_target)
            if pd.isna(edge_length_km):
                edge_length_km = haversine_distance(point_a[0], point_a[1], point_b[0], point_b[1])
            rows.append(
                {
                    "time": row["time"],
                    "time_bucket": None,
                    "vehicle_id": str(row.get("vehicle_id", "")),
                    "segment_key": _segment_key(edge_source, edge_target),
                    "edge_u": edge_source,
                    "edge_v": edge_target,
                    "edge_key": 0 if edge_key is None else edge_key,
                    "source": edge_source,
                    "target": edge_target,
                    "lat1": point_a[0],
                    "lng1": point_a[1],
                    "lat2": point_b[0],
                    "lng2": point_b[1],
                    "speed_kmh": segment_speed,
                    "edge_length_km": float(edge_length_km) if pd.notna(edge_length_km) else 0.0,
                    "path_mode": mode,
                }
            )

    segments = pd.DataFrame(rows)
    meta = {
        "success": len(segments) > 0,
        "method": "近似 HMM: 最近道路节点作为发射近似，最短路连续性作为转移约束。",
        "raw_points": len(df),
        "sampled_points": len(sample_df),
        "matched_nodes": sum(1 for node in matched_nodes if node is not None),
        "segments": len(segments),
        "nearest_failures": nearest_failures,
        "path_failures": path_failures,
        "undirected_segments": undirected_segments,
        "same_node_segments": same_node_segments,
        "network": network_meta,
    }
    if not meta["success"]:
        meta["error"] = "没有生成可聚合的匹配路段。"
    return {"segments": segments, "matched_nodes": matched_nodes, "meta": meta}


def aggregate_road_speed_segments(segments_df, bucket_minutes=15):
    if segments_df is None or len(segments_df) == 0:
        return pd.DataFrame()

    bucket_minutes = int(bucket_minutes or CONFIG["DEFAULT_CONGESTION_BUCKET_MINUTES"])
    bucket_minutes = max(1, bucket_minutes)
    df = segments_df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "segment_key", "speed_kmh", "lat1", "lng1", "lat2", "lng2"])
    if len(df) == 0:
        return pd.DataFrame()

    df["time_bucket"] = df["time"].dt.floor(f"{bucket_minutes}min")
    grouped = (
        df.groupby(["time_bucket", "segment_key"], as_index=False)
        .agg(
            source=("source", "first"),
            target=("target", "first"),
            lat1=("lat1", "first"),
            lng1=("lng1", "first"),
            lat2=("lat2", "first"),
            lng2=("lng2", "first"),
            avg_speed_kmh=("speed_kmh", "mean"),
            min_speed_kmh=("speed_kmh", "min"),
            max_speed_kmh=("speed_kmh", "max"),
            observation_count=("speed_kmh", "size"),
            vehicle_count=("vehicle_id", pd.Series.nunique),
            edge_length_km=("edge_length_km", "mean"),
        )
        .sort_values(["time_bucket", "avg_speed_kmh", "observation_count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    grouped["color"] = grouped["avg_speed_kmh"].apply(speed_to_color)
    grouped["level"] = grouped["avg_speed_kmh"].apply(speed_to_level)
    return grouped


def _build_congestion_frames(aggregated_df, max_segments=None):
    if aggregated_df is None or len(aggregated_df) == 0:
        return []

    max_segments = int(max_segments or CONFIG["MAX_CONGESTION_SEGMENTS"])
    frames = []
    for bucket, group in aggregated_df.groupby("time_bucket", sort=True):
        display_group = group.sort_values(["observation_count", "avg_speed_kmh"], ascending=[False, True]).head(max_segments)
        segments = []
        for _, row in display_group.iterrows():
            segments.append(
                {
                    "key": str(row["segment_key"]),
                    "points": [[float(row["lat1"]), float(row["lng1"])], [float(row["lat2"]), float(row["lng2"])]],
                    "speed": round(float(row["avg_speed_kmh"]), 1),
                    "count": int(row["observation_count"]),
                    "vehicles": int(row["vehicle_count"]),
                    "level": str(row["level"]),
                    "color": str(row["color"]),
                }
            )
        frames.append(
            {
                "label": pd.Timestamp(bucket).strftime("%Y-%m-%d %H:%M"),
                "segments": segments,
            }
        )
    return frames


def _add_congestion_player(m, frames, bucket_minutes):
    if not frames:
        return

    map_name = m.get_name()
    control_id = f"congestion-player-{map_name}"
    frames_json = json.dumps(frames, ensure_ascii=False)
    html = f"""
    <div id="{control_id}" style="position:fixed;left:50%;bottom:18px;transform:translateX(-50%);z-index:9999;background:rgba(255,255,255,0.96);border:1px solid rgba(148,163,184,0.35);border-radius:12px;padding:10px 12px;box-shadow:0 12px 32px rgba(15,23,42,0.16);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-width:360px;max-width:680px;">
      <div style="display:flex;align-items:center;gap:10px;">
        <button type="button" data-role="toggle" style="border:1px solid #cbd5e1;background:#0f172a;color:#fff;border-radius:8px;padding:6px 10px;cursor:pointer;">播放</button>
        <input type="range" data-role="slider" min="0" max="{len(frames) - 1}" value="0" step="1" style="flex:1;">
        <span data-role="label" style="font-size:12px;color:#334155;min-width:128px;text-align:right;">{frames[0]['label']}</span>
      </div>
      <div style="font-size:11px;color:#64748b;margin-top:6px;">按 {bucket_minutes} 分钟聚合路段平均速度；红/橙/黄/绿分别表示严重拥堵、拥堵、缓行、畅通。</div>
    </div>
    """
    script = f"""
    (function() {{
      const frames = {frames_json};
      const map = {map_name};
      const container = document.getElementById({json.dumps(control_id)});
      if (!map || !container || !window.L || !frames.length) return;
      const slider = container.querySelector('[data-role="slider"]');
      const label = container.querySelector('[data-role="label"]');
      const button = container.querySelector('[data-role="toggle"]');
      const layer = L.layerGroup().addTo(map);
      let timer = null;
      let index = 0;

      function renderFrame(nextIndex) {{
        index = Math.max(0, Math.min(frames.length - 1, Number(nextIndex) || 0));
        const frame = frames[index];
        layer.clearLayers();
        frame.segments.forEach(function(seg) {{
          L.polyline(seg.points, {{
            color: seg.color,
            weight: Math.min(9, 4 + Math.log2(Math.max(1, seg.count))),
            opacity: 0.86,
            lineCap: 'round',
            lineJoin: 'round'
          }}).bindTooltip(
            '路段 ' + seg.key + '<br>平均速度: ' + seg.speed + ' km/h<br>状态: ' + seg.level + '<br>样本: ' + seg.count + ' / 车辆: ' + seg.vehicles,
            {{sticky: true}}
          ).addTo(layer);
        }});
        slider.value = String(index);
        label.textContent = frame.label + ' (' + frame.segments.length + '段)';
      }}

      function stop() {{
        if (timer) window.clearInterval(timer);
        timer = null;
        button.textContent = '播放';
      }}

      function play() {{
        stop();
        button.textContent = '暂停';
        timer = window.setInterval(function() {{
          const next = index >= frames.length - 1 ? 0 : index + 1;
          renderFrame(next);
        }}, 900);
      }}

      slider.addEventListener('input', function() {{
        stop();
        renderFrame(slider.value);
      }});
      button.addEventListener('click', function() {{
        if (timer) stop();
        else play();
      }});
      renderFrame(0);
    }})();
    """
    m.get_root().html.add_child(folium.Element(html))
    m.get_root().script.add_child(folium.Element(script))


def estimate_eta_between_points(start_lat, start_lng, end_lat, end_lng, departure_time=None, historical_segments=None, graph=None, network_meta=None):
    if graph is None:
        graph, network_meta = load_road_network()
    network_meta = network_meta or {}
    if graph is None:
        return {"success": False, "error": network_meta.get("error", "路网未加载。"), "network": network_meta}

    try:
        source = nearest_road_node(graph, start_lng, start_lat)
        target = nearest_road_node(graph, end_lng, end_lat)
    except Exception as exc:
        log.exception("ETA 起终点最近节点匹配失败")
        return {"success": False, "error": f"起终点无法匹配到路网节点: {exc}", "network": network_meta}

    undirected_graph = None
    if source != target and hasattr(graph, "is_directed") and graph.is_directed():
        try:
            undirected_graph = graph.to_undirected(as_view=True)
        except TypeError:
            undirected_graph = graph.to_undirected()
    try:
        path_nodes, mode = _shortest_path_nodes(graph, source, target, undirected_graph=undirected_graph)
    except Exception as exc:
        log.exception("ETA 最短路径失败: source=%s, target=%s", source, target)
        return {"success": False, "error": f"起终点之间没有可用道路路径: {exc}", "network": network_meta}

    route_points = [_node_lat_lng(graph, node) for node in path_nodes]
    route_points = [point for point in route_points if point]
    distance_km = _path_length_km(graph, path_nodes)
    if distance_km <= 0 or pd.isna(distance_km):
        distance_km = haversine_distance(start_lat, start_lng, end_lat, end_lng)

    speed_candidates = []
    if historical_segments is not None and len(historical_segments) > 0:
        hist = historical_segments.copy()
        route_keys = set()
        for idx in range(1, len(path_nodes)):
            route_keys.add(_segment_key(path_nodes[idx - 1], path_nodes[idx]))
            route_keys.add(_segment_key(path_nodes[idx], path_nodes[idx - 1]))
        if route_keys and "segment_key" in hist.columns:
            matched = hist[hist["segment_key"].isin(route_keys)]
            speed_candidates.extend(matched["avg_speed_kmh"].dropna().astype(float).tolist())
        if not speed_candidates and "avg_speed_kmh" in hist.columns:
            speed_candidates.extend(hist["avg_speed_kmh"].dropna().astype(float).tolist())

    avg_speed_kmh = float(np.mean(speed_candidates)) if speed_candidates else CONFIG["DEFAULT_ETA_SPEED_KMH"]
    avg_speed_kmh = max(5.0, min(CONFIG["ANIMATION_SPEED_CAP_KMH"], avg_speed_kmh))
    duration_minutes = float(distance_km / avg_speed_kmh * 60.0) if avg_speed_kmh > 0 else np.nan
    departure_dt = safe_datetime(departure_time)
    arrival_time = None
    if departure_dt is not None and pd.notna(duration_minutes):
        arrival_time = departure_dt + pd.to_timedelta(duration_minutes, unit="m")

    return {
        "success": True,
        "method": "路网最短距离 + 当前窗口历史路段平均速度；若路线无历史样本，则使用全局历史均速或默认速度。",
        "source_node": source,
        "target_node": target,
        "path_mode": mode,
        "distance_km": float(distance_km),
        "avg_speed_kmh": float(avg_speed_kmh),
        "duration_minutes": duration_minutes,
        "arrival_time": arrival_time,
        "route_points": route_points,
        "historical_speed_samples": len(speed_candidates),
        "network": network_meta,
    }


def plot_congestion_roads_and_eta(
    vehicle_ids,
    start_time=None,
    end_time=None,
    bucket_minutes=15,
    eta_start=None,
    eta_end=None,
    save_path=None,
):
    cleaned_vehicle_ids = []
    for vehicle_id in vehicle_ids or []:
        vehicle_id = str(vehicle_id).strip()
        if vehicle_id and vehicle_id not in cleaned_vehicle_ids:
            cleaned_vehicle_ids.append(vehicle_id)
    cleaned_vehicle_ids = cleaned_vehicle_ids[: CONFIG["MAX_CONGESTION_VEHICLES"]]
    if not cleaned_vehicle_ids:
        return None, {"success": False, "error": f"请选择 1-{CONFIG['MAX_CONGESTION_VEHICLES']} 辆车用于 HMM 拥堵道路示例。"}

    bucket_minutes = int(bucket_minutes or CONFIG["DEFAULT_CONGESTION_BUCKET_MINUTES"])
    graph, network_meta = load_road_network()
    if graph is None:
        return None, {"success": False, "error": network_meta.get("error", "路网未加载。"), "network": network_meta}

    loaded_frames = []
    all_segments = []
    match_summaries = []
    per_vehicle_limit = max(60, min(CONFIG["MAX_CONGESTION_INPUT_POINTS"], CONFIG["MAX_TRAJECTORY_POINTS"] // max(1, len(cleaned_vehicle_ids))))
    for index, vehicle_id in enumerate(cleaned_vehicle_ids):
        df = load_vehicle_trajectory(vehicle_id, start_time, end_time)
        if df is None or len(df) < 2:
            match_summaries.append({"vehicle_id": vehicle_id, "success": False, "message": "当前时间范围轨迹点不足。"})
            continue
        df = resample_trajectory(df, per_vehicle_limit).assign(vehicle_id=vehicle_id)
        loaded_frames.append(df)
        match_result = approximate_hmm_match_trajectory(df, graph=graph, network_meta=network_meta)
        meta = match_result["meta"]
        match_summaries.append(
            {
                "vehicle_id": vehicle_id,
                "success": bool(meta.get("success")),
                "method": meta.get("method", ""),
                "raw_points": meta.get("raw_points", len(df)),
                "sampled_points": meta.get("sampled_points", 0),
                "matched_nodes": meta.get("matched_nodes", 0),
                "segments": meta.get("segments", 0),
                "nearest_failures": meta.get("nearest_failures", 0),
                "path_failures": meta.get("path_failures", 0),
                "undirected_segments": meta.get("undirected_segments", 0),
                "message": meta.get("error", "匹配完成。"),
            }
        )
        if len(match_result["segments"]) > 0:
            all_segments.append(match_result["segments"].assign(vehicle_id=vehicle_id))

    if not loaded_frames:
        return None, {"success": False, "error": "所选车辆在当前时间范围内没有轨迹数据。", "network": network_meta, "matches": match_summaries}

    combined_df = pd.concat(loaded_frames, ignore_index=True)
    m = build_map(combined_df)
    if not all_segments:
        add_map_layers(m)
        return None, {
            "success": False,
            "error": "HMM 近似匹配没有生成可聚合路段。",
            "network": network_meta,
            "matches": match_summaries,
        }

    segments_df = pd.concat(all_segments, ignore_index=True)
    aggregated = aggregate_road_speed_segments(segments_df, bucket_minutes=bucket_minutes)
    frames = _build_congestion_frames(aggregated, max_segments=CONFIG["MAX_CONGESTION_SEGMENTS"])

    matched_layer = folium.FeatureGroup(name="匹配后的道路轨迹", show=False).add_to(m)
    display_segments = segments_df.head(CONFIG["MAX_CONGESTION_SEGMENTS"])
    for _, row in display_segments.iterrows():
        folium.PolyLine(
            [[row["lat1"], row["lng1"]], [row["lat2"], row["lng2"]]],
            color=speed_to_color(row["speed_kmh"]),
            weight=3,
            opacity=0.42,
            tooltip=f"车辆 {row['vehicle_id']} 匹配路段 {row['segment_key']}",
        ).add_to(matched_layer)

    eta_result = None
    if eta_start and eta_end:
        eta_result = estimate_eta_between_points(
            eta_start[0],
            eta_start[1],
            eta_end[0],
            eta_end[1],
            departure_time=start_time,
            historical_segments=aggregated,
            graph=graph,
            network_meta=network_meta,
        )
        if eta_result.get("success") and len(eta_result.get("route_points", [])) >= 2:
            route_points = eta_result["route_points"]
            if len(route_points) > 800:
                idx = np.linspace(0, len(route_points) - 1, num=800, dtype=int)
                route_points = [route_points[i] for i in idx]
            folium.PolyLine(
                route_points,
                color="#2563eb",
                weight=6,
                opacity=0.9,
                tooltip=f"ETA 路径: {eta_result['distance_km']:.2f} km / {eta_result['duration_minutes']:.1f} 分钟",
            ).add_to(m)
            folium.Marker(
                [eta_start[0], eta_start[1]],
                icon=folium.Icon(color="green", icon="play"),
                tooltip="ETA 起点",
            ).add_to(m)
            folium.Marker(
                [eta_end[0], eta_end[1]],
                icon=folium.Icon(color="red", icon="flag"),
                tooltip="ETA 终点",
            ).add_to(m)

    legend_html = f"""
    <div style="position:fixed;right:16px;top:16px;z-index:9999;background:rgba(255,255,255,0.96);border:1px solid rgba(148,163,184,0.35);border-radius:12px;padding:12px 14px;box-shadow:0 12px 32px rgba(15,23,42,0.12);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-width:220px;max-width:340px;">
      <div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:6px;">拥堵道路图例</div>
      <div style="display:grid;grid-template-columns:18px 1fr;gap:6px 8px;font-size:12px;color:#334155;align-items:center;">
        <span style="height:4px;background:#dc2626;"></span><span>&lt;15 km/h 严重拥堵</span>
        <span style="height:4px;background:#f97316;"></span><span>15-25 km/h 拥堵</span>
        <span style="height:4px;background:#eab308;"></span><span>25-40 km/h 缓行</span>
        <span style="height:4px;background:#16a34a;"></span><span>&gt;=40 km/h 畅通</span>
        <span style="height:4px;background:#2563eb;"></span><span>ETA 最短路</span>
      </div>
      <div style="font-size:11px;color:#64748b;line-height:1.45;border-top:1px solid rgba(148,163,184,0.25);padding-top:8px;margin-top:8px;">近似 HMM: 最近道路节点 + 最短路转移约束。点击地图可显示经纬度，底部滑块播放时间片。</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    _add_congestion_player(m, frames, bucket_minutes)
    add_map_layers(m)

    start_str = format_time_tag(start_time)
    end_str = format_time_tag(end_time)
    vehicle_tag = "-".join(cleaned_vehicle_ids[:3])
    output_path = save_path or os.path.join(
        CONFIG["OUTPUT_MAP_DIR"],
        f"congestion_hmm_eta_{vehicle_tag}_{start_str}_{end_str}_{bucket_minutes}min.html",
    )
    ensure_parent_dir(output_path)
    m.save(output_path)

    info = {
        "success": bool(frames),
        "output_path": output_path,
        "vehicles": len(loaded_frames),
        "bucket_minutes": bucket_minutes,
        "matched_segment_rows": len(segments_df),
        "aggregated_segments": len(aggregated),
        "frame_count": len(frames),
        "matches": match_summaries,
        "eta": eta_result,
        "network": network_meta,
    }
    log.info(
        "HMM拥堵道路与ETA地图已保存: output=%s, vehicles=%s, frames=%s, segments=%s",
        output_path,
        len(loaded_frames),
        len(frames),
        len(aggregated),
    )
    return output_path, info


def correct_trajectory_with_road_network(df, graph=None, network_meta=None, max_points=None):
    if df is None or len(df) < 2:
        return {
            "points": [],
            "matched_nodes": [],
            "meta": {
                "success": False,
                "error": "轨迹点不足，无法校正。",
            },
        }

    if graph is None:
        graph, network_meta = load_road_network()
    network_meta = network_meta or {}
    if graph is None:
        return {
            "points": [],
            "matched_nodes": [],
            "meta": {
                "success": False,
                "error": network_meta.get("error", "路网未加载。"),
                "network": network_meta,
            },
        }

    sample_limit = int(max_points or CONFIG["MAX_ROAD_CORRECTION_INPUT_POINTS"])
    sample_df = resample_trajectory(df, sample_limit)
    matched_nodes = []
    nearest_failures = 0
    for _, row in sample_df.iterrows():
        try:
            matched_nodes.append(nearest_road_node(graph, row["long"], row["lati"]))
        except Exception:
            nearest_failures += 1
            matched_nodes.append(None)
            log.exception("GPS 点最近邻匹配失败: lng=%s, lat=%s", row.get("long"), row.get("lati"))

    if sum(1 for node in matched_nodes if node is not None) < 2:
        return {
            "points": [],
            "matched_nodes": matched_nodes,
            "meta": {
                "success": False,
                "error": "有效匹配节点不足，无法拼接校正轨迹。",
                "nearest_failures": nearest_failures,
                "network": network_meta,
            },
        }

    undirected_graph = None
    corrected_points = []
    path_failures = 0
    undirected_segments = 0
    same_node_segments = 0
    path_rows = []
    sequence = 0

    for idx in range(1, len(matched_nodes)):
        source = matched_nodes[idx - 1]
        target = matched_nodes[idx]
        if source is None or target is None:
            continue
        prev_row = sample_df.iloc[idx - 1]
        row = sample_df.iloc[idx]
        if undirected_graph is None and source != target and hasattr(graph, "is_directed") and graph.is_directed():
            try:
                undirected_graph = graph.to_undirected(as_view=True)
            except TypeError:
                undirected_graph = graph.to_undirected()
        try:
            path_nodes, mode = _shortest_path_nodes(graph, source, target, undirected_graph=undirected_graph)
            if mode == "undirected":
                undirected_segments += 1
            elif mode == "same_node":
                same_node_segments += 1
        except Exception:
            path_failures += 1
            log.exception("路网最短路径失败: source=%s, target=%s", source, target)
            fallback = _node_lat_lng(graph, source)
            if fallback:
                if not corrected_points or corrected_points[-1] != fallback:
                    corrected_points.append(fallback)
            continue

        if len(path_nodes) == 1:
            point = _node_lat_lng(graph, path_nodes[0])
            edge_points = [point] if point else []
            edge_source = path_nodes[0]
            edge_target = path_nodes[0]
            edge_key = 0
            edge_node = path_nodes[0]
            edge_items = [(edge_source, edge_target, edge_key, edge_node, edge_points)]
        else:
            edge_items = []
            for node_idx in range(1, len(path_nodes)):
                edge_source = path_nodes[node_idx - 1]
                edge_target = path_nodes[node_idx]
                edge_key, _ = _best_edge_data(graph, edge_source, edge_target, weight="length")
                edge_key = 0 if edge_key is None else edge_key
                edge_items.append((edge_source, edge_target, edge_key, edge_target, _edge_geometry_points(graph, edge_source, edge_target, weight="length")))

        segment_items = []
        for edge_source, edge_target, edge_key, edge_node, edge_points in edge_items:
            for point in edge_points or []:
                if not point:
                    continue
                if segment_items and segment_items[-1]["point"] == point:
                    continue
                segment_items.append(
                    {
                        "point": point,
                        "edge_source": edge_source,
                        "edge_target": edge_target,
                        "edge_key": edge_key,
                        "edge_node": edge_node,
                    }
                )

        segment_times = _interpolate_times_for_points(
            [item["point"] for item in segment_items],
            prev_row["time"],
            row["time"],
        )
        segment_speeds = _interpolate_values_for_points(
            [item["point"] for item in segment_items],
            prev_row.get("speed", 0.0),
            row.get("speed", prev_row.get("speed", 0.0)),
        )

        for item, point_time, point_speed in zip(segment_items, segment_times, segment_speeds):
            point = item["point"]
            if corrected_points and corrected_points[-1] == point:
                if path_rows:
                    path_rows[-1]["time"] = min(pd.Timestamp(path_rows[-1]["time"]), pd.Timestamp(point_time))
                    path_rows[-1]["speed"] = float(point_speed)
                continue
            vehicle_value = row.get("vehicle_id", row.get("id", prev_row.get("vehicle_id", prev_row.get("id", ""))))
            corrected_points.append(point)
            path_rows.append(
                {
                    "vehicle_id": str(vehicle_value),
                    "time": point_time,
                    "status": int(row.get("status", prev_row.get("status", 0)) or 0),
                    "speed": float(point_speed),
                    "raw_lon": float(row.get("long", np.nan)),
                    "raw_lat": float(row.get("lati", np.nan)),
                    "matched_lon": point[1],
                    "matched_lat": point[0],
                    "matched_node": item["edge_node"],
                    "edge_u": item["edge_source"],
                    "edge_v": item["edge_target"],
                    "edge_key": item["edge_key"],
                    "path_mode": mode,
                    "sequence": sequence,
                }
            )
            sequence += 1

    if len(corrected_points) > CONFIG["MAX_CORRECTED_PATH_POINTS"]:
        idx = np.linspace(0, len(corrected_points) - 1, num=CONFIG["MAX_CORRECTED_PATH_POINTS"], dtype=int)
        corrected_points = [corrected_points[i] for i in idx]

    success = len(corrected_points) >= 2
    meta = {
        "success": success,
        "raw_points": len(df),
        "sampled_points": len(sample_df),
        "matched_nodes": sum(1 for node in matched_nodes if node is not None),
        "corrected_points": len(corrected_points),
        "nearest_failures": nearest_failures,
        "path_failures": path_failures,
        "undirected_segments": undirected_segments,
        "same_node_segments": same_node_segments,
        "network": network_meta,
    }
    if not success:
        meta["error"] = "最短路径拼接后没有足够点位可显示。"
    log.info(
        "路网校正完成: raw=%s, sampled=%s, matched=%s, corrected=%s, nearest_failures=%s, path_failures=%s, undirected_segments=%s",
        meta["raw_points"],
        meta["sampled_points"],
        meta["matched_nodes"],
        meta["corrected_points"],
        nearest_failures,
        path_failures,
        undirected_segments,
    )
    return {"points": corrected_points, "matched_nodes": matched_nodes, "path_rows": pd.DataFrame(path_rows), "meta": meta}


def load_or_build_road_corrected_vehicle_cache(vehicle_id, start_time=None, end_time=None, graph=None, network_meta=None, force_rebuild=False):
    vehicle_id = str(vehicle_id).strip()
    if not vehicle_id:
        return {"rows": pd.DataFrame(), "points": [], "meta": {"success": False, "error": "未提供车辆ID。"}}
    if graph is None:
        graph, network_meta = load_road_network()
    network_meta = network_meta or {}
    if graph is None:
        return {
            "rows": pd.DataFrame(),
            "points": [],
            "meta": {"success": False, "error": network_meta.get("error", "路网未加载。"), "network": network_meta},
        }

    cache_path = road_corrected_vehicle_cache_path(vehicle_id, network_meta)
    coverage_path = road_corrected_vehicle_coverage_path(vehicle_id, network_meta)
    cached_rows = pd.DataFrame()
    coverage = [] if force_rebuild else _read_road_cache_coverage(coverage_path)
    if not force_rebuild and os.path.exists(cache_path):
        try:
            cached_rows = _normalize_corrected_cache_rows(pd.read_csv(cache_path), vehicle_id=vehicle_id)
        except Exception:
            log.exception("车辆路网校正缓存读取失败，将重新生成: %s", cache_path)
            cached_rows = pd.DataFrame()
            coverage = []

    query_start = safe_datetime(start_time)
    query_end = safe_datetime(end_time)
    missing_intervals = _missing_coverage_intervals(query_start, query_end, coverage)
    if query_start is None or query_end is None:
        if os.path.exists(cache_path) and len(cached_rows) > 0 and not force_rebuild:
            missing_intervals = []
        else:
            df_all = load_vehicle_trajectory(vehicle_id, None, None)
            if df_all is not None and len(df_all) > 0:
                query_start = df_all["time"].min()
                query_end = df_all["time"].max()
                missing_intervals = [(pd.Timestamp(query_start), pd.Timestamp(query_end))]

    new_frames = []
    processed_intervals = []
    skipped_intervals = []
    for interval_start, interval_end in missing_intervals:
        df = load_vehicle_trajectory(vehicle_id, interval_start, interval_end)
        if df is None or len(df) < 2:
            skipped_intervals.append((interval_start, interval_end))
            processed_intervals.append((interval_start, interval_end))
            continue
        cache_df = resample_trajectory(df, CONFIG["MAX_ROAD_CORRECTION_CACHE_POINTS"])
        cache_df = cache_df.copy()
        cache_df["vehicle_id"] = vehicle_id
        result = correct_trajectory_with_road_network(
            cache_df,
            graph=graph,
            network_meta=network_meta,
            max_points=CONFIG["MAX_ROAD_CORRECTION_CACHE_POINTS"],
        )
        rows = _normalize_corrected_cache_rows(result.get("path_rows"), vehicle_id=vehicle_id)
        if len(rows) > 0:
            new_frames.append(rows)
        processed_intervals.append((interval_start, interval_end))

    if new_frames:
        combined_rows = pd.concat([cached_rows] + new_frames, ignore_index=True) if len(cached_rows) > 0 else pd.concat(new_frames, ignore_index=True)
        combined_rows = _normalize_corrected_cache_rows(combined_rows, vehicle_id=vehicle_id)
        dedupe_cols = [
            col
            for col in ["vehicle_id", "time", "matched_lon", "matched_lat", "matched_node", "edge_u", "edge_v", "edge_key", "path_mode"]
            if col in combined_rows.columns
        ]
        if dedupe_cols:
            combined_rows = combined_rows.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)
        combined_rows["sequence"] = range(len(combined_rows))
        ensure_parent_dir(cache_path)
        combined_rows.to_csv(cache_path, index=False)
        cached_rows = combined_rows
    elif len(cached_rows) == 0 and processed_intervals:
        ensure_parent_dir(cache_path)
        _normalize_corrected_cache_rows(pd.DataFrame(), vehicle_id=vehicle_id).to_csv(cache_path, index=False)

    if processed_intervals:
        coverage = _merge_coverage_intervals(list(coverage) + processed_intervals)
        _write_road_cache_coverage(coverage_path, coverage)

    rows = slice_road_corrected_rows(cached_rows, start_time=query_start, end_time=query_end)
    points = _corrected_rows_to_points(rows)
    cache_hit = len(missing_intervals) == 0 and len(cached_rows) > 0
    if len(points) < 2 and len(cached_rows) == 0 and not processed_intervals:
        return {
            "rows": pd.DataFrame(),
            "points": [],
            "meta": {"success": False, "cache_hit": False, "cache_path": cache_path, "coverage_path": coverage_path, "error": "车辆缓存轨迹点不足，无法生成路网校正缓存。"},
        }
    meta = {
        "success": len(points) >= 2,
        "cache_hit": cache_hit,
        "cache_path": cache_path,
        "coverage_path": coverage_path,
        "cache_rows": len(cached_rows),
        "corrected_points": len(points),
        "processed_intervals": len(processed_intervals),
        "skipped_intervals": len(skipped_intervals),
        "coverage": [{"start": _serialize_ts(start), "end": _serialize_ts(end)} for start, end in _merge_coverage_intervals(coverage)],
        "network": network_meta,
        "message": "已读取查询缓存。" if cache_hit else "已补齐查询缺失区间并更新缓存。",
    }
    if len(points) < 2:
        meta.setdefault("error", "当前查询区间的车辆路网校正缓存没有足够点位可显示。")
    log.info(
        "车辆路网校正查询缓存完成: vehicle_id=%s, cache_hit=%s, processed=%s, rows=%s, cache=%s",
        vehicle_id,
        cache_hit,
        len(processed_intervals),
        len(cached_rows),
        cache_path,
    )
    return {"rows": rows, "points": points, "meta": meta}


def slice_road_corrected_rows(rows, start_time=None, end_time=None):
    rows = _normalize_corrected_cache_rows(rows)
    if len(rows) == 0:
        return rows
    start_dt = safe_datetime(start_time)
    end_dt = safe_datetime(end_time)
    if start_dt is not None:
        rows = rows[rows["time"] >= start_dt]
    if end_dt is not None:
        rows = rows[rows["time"] <= end_dt]
    return rows.sort_values(["time", "sequence"]).reset_index(drop=True)


def load_road_corrected_vehicle_slice(vehicle_id, start_time=None, end_time=None, graph=None, network_meta=None, force_rebuild=False):
    cached = load_or_build_road_corrected_vehicle_cache(
        vehicle_id,
        start_time=start_time,
        end_time=end_time,
        graph=graph,
        network_meta=network_meta,
        force_rebuild=force_rebuild,
    )
    rows = slice_road_corrected_rows(cached.get("rows"), start_time=start_time, end_time=end_time)
    points = _corrected_rows_to_points(rows)
    if len(points) > CONFIG["MAX_CORRECTED_PATH_POINTS"]:
        idx = np.linspace(0, len(points) - 1, num=CONFIG["MAX_CORRECTED_PATH_POINTS"], dtype=int)
        points = [points[i] for i in idx]
    meta = dict(cached.get("meta", {}))
    meta.update(
        {
            "window_rows": len(rows),
            "window_corrected_points": len(points),
            "success": len(points) >= 2,
        }
    )
    if len(points) < 2:
        meta.setdefault("error", "当前时间窗口内的车辆路网校正缓存点不足。")
    return {"rows": rows, "points": points, "meta": meta}


def _trajectory_segment_rows(df):
    if df is None or len(df) < 2:
        return []

    segments = []
    current_status = int(df.iloc[0]["status"])
    current_points = [[df.iloc[0]["lati"], df.iloc[0]["long"]]]

    for idx in range(1, len(df)):
        prev_row = df.iloc[idx - 1]
        row = df.iloc[idx]
        time_diff = (row["time"] - prev_row["time"]).total_seconds() if pd.notna(row["time"]) and pd.notna(prev_row["time"]) else 0
        distance_km = haversine_distance(prev_row["lati"], prev_row["long"], row["lati"], row["long"])

        if pd.isna(distance_km) or pd.isna(time_diff):
            continue

        gap_break = time_diff >= CONFIG["ANIMATION_GAP_THRESHOLD_S"]
        drift_break = time_diff > 0 and distance_km > CONFIG["ANIMATION_DRIFT_CAP_KM"] and (distance_km / (time_diff / 3600.0)) > CONFIG["ANIMATION_SPEED_CAP_KMH"]
        status_change = int(row["status"]) != current_status

        if gap_break or drift_break or status_change:
            if len(current_points) >= 2:
                segments.append({"status": current_status, "points": current_points})
            current_status = int(row["status"])
            current_points = [[prev_row["lati"], prev_row["long"]], [row["lati"], row["long"]]]
        else:
            current_points.append([row["lati"], row["long"]])

    if len(current_points) >= 2:
        segments.append({"status": current_status, "points": current_points})
    return segments


def _trajectory_segment_dfs(df):
    if df is None or len(df) < 2:
        return []

    segments = []
    current_status = int(df.iloc[0]["status"])
    current_rows = [df.iloc[0]]

    for idx in range(1, len(df)):
        prev_row = df.iloc[idx - 1]
        row = df.iloc[idx]
        time_diff = (row["time"] - prev_row["time"]).total_seconds() if pd.notna(row["time"]) and pd.notna(prev_row["time"]) else 0
        distance_km = haversine_distance(prev_row["lati"], prev_row["long"], row["lati"], row["long"])
        if pd.isna(distance_km) or pd.isna(time_diff):
            continue

        gap_break = time_diff >= CONFIG["ANIMATION_GAP_THRESHOLD_S"]
        drift_break = time_diff > 0 and distance_km > CONFIG["ANIMATION_DRIFT_CAP_KM"] and (distance_km / (time_diff / 3600.0)) > CONFIG["ANIMATION_SPEED_CAP_KMH"]
        status_change = int(row["status"]) != current_status

        if gap_break or drift_break or status_change:
            if len(current_rows) >= 2:
                segments.append({"status": current_status, "df": pd.DataFrame(current_rows).reset_index(drop=True)})
            current_status = int(row["status"])
            current_rows = [prev_row, row]
        else:
            current_rows.append(row)

    if len(current_rows) >= 2:
        segments.append({"status": current_status, "df": pd.DataFrame(current_rows).reset_index(drop=True)})
    return segments


def _calibrated_segment_points(df, max_points=None, smooth_window=3, max_step_km=0.25):
    if df is None or len(df) < 2:
        return []

    segment_df = df.copy().sort_values("time").reset_index(drop=True)
    if len(segment_df) < 2:
        return []

    filtered_rows = [segment_df.iloc[0]]
    for idx in range(1, len(segment_df)):
        prev_row = filtered_rows[-1]
        row = segment_df.iloc[idx]
        distance_km = haversine_distance(prev_row["lati"], prev_row["long"], row["lati"], row["long"])
        time_diff_s = (row["time"] - prev_row["time"]).total_seconds() if pd.notna(row["time"]) and pd.notna(prev_row["time"]) else 0
        speed_kmh = (distance_km / (time_diff_s / 3600.0)) if time_diff_s and time_diff_s > 0 and pd.notna(distance_km) else 0.0
        if pd.isna(distance_km):
            continue
        if distance_km > CONFIG["ANIMATION_DRIFT_CAP_KM"]:
            continue
        if speed_kmh > CONFIG["ANIMATION_SPEED_CAP_KMH"] * 2:
            continue
        filtered_rows.append(row)

    if len(filtered_rows) < 2:
        filtered_rows = [segment_df.iloc[i] for i in range(len(segment_df))]

    working_df = pd.DataFrame(filtered_rows).reset_index(drop=True)
    if len(working_df) >= smooth_window:
        working_df["lati"] = working_df["lati"].rolling(window=smooth_window, center=True, min_periods=1).median()
        working_df["long"] = working_df["long"].rolling(window=smooth_window, center=True, min_periods=1).median()

    points = [[float(row["lati"]), float(row["long"])] for _, row in working_df.iterrows()]
    if len(points) < 2:
        return points

    densified = [points[0]]
    for idx in range(1, len(points)):
        prev_lat, prev_lng = densified[-1]
        lat, lng = points[idx]
        distance_km = haversine_distance(prev_lat, prev_lng, lat, lng)
        if pd.isna(distance_km):
            continue
        steps = max(1, int(np.ceil(distance_km / max_step_km)))
        for step_idx in range(1, steps + 1):
            t = step_idx / steps
            densified.append([
                float(prev_lat + (lat - prev_lat) * t),
                float(prev_lng + (lng - prev_lng) * t),
            ])

    if max_points and len(densified) > max_points:
        idx = np.linspace(0, len(densified) - 1, num=max_points, dtype=int)
        densified = [densified[i] for i in idx]

    return densified


def _trajectory_summary(df):
    if df is None or len(df) == 0:
        return {}

    occupied_points = int((df["status"] == 1).sum())
    vacant_points = int((df["status"] == 0).sum())
    total_distance = 0.0
    active_speed_samples = []

    prev = None
    for _, row in df.iterrows():
        if prev is not None and pd.notna(prev["time"]) and pd.notna(row["time"]):
            dt = (row["time"] - prev["time"]).total_seconds()
            dist = haversine_distance(prev["lati"], prev["long"], row["lati"], row["long"])
            if pd.notna(dt) and dt > 0 and pd.notna(dist):
                total_distance += dist
                speed = (dist / (dt / 3600.0)) if dt > 0 else 0
                if 0 <= speed <= CONFIG["ANIMATION_SPEED_CAP_KMH"] * 2:
                    active_speed_samples.append(speed)
        prev = row

    return {
        "points": len(df),
        "occupied_points": occupied_points,
        "vacant_points": vacant_points,
        "total_distance_km": total_distance,
        "avg_speed_kmh": float(np.mean(active_speed_samples)) if active_speed_samples else 0.0,
        "start_time": df.iloc[0]["time"],
        "end_time": df.iloc[-1]["time"],
    }


def plot_vehicle_trajectory(vehicle_id, start_time=None, end_time=None, save_path=None):
    log.info(
        "查询车辆轨迹: vehicle_id=%s, start_time=%s, end_time=%s",
        vehicle_id,
        start_time,
        end_time,
    )

    df = load_vehicle_trajectory(vehicle_id, start_time, end_time)
    if df is None or len(df) == 0:
        log.warning("未找到符合条件的轨迹数据")
        return None

    df = resample_trajectory(df, CONFIG["MAX_TRAJECTORY_POINTS"])
    m = build_map(df)

    segments = _trajectory_segment_dfs(df)
    for segment in segments:
        segment_df = segment["df"]
        color = "#d73027" if segment["status"] == 1 else "#2c7bb6"
        label = "载客轨迹" if segment["status"] == 1 else "空载轨迹"
        calibrated_points = _calibrated_segment_points(segment_df, max_points=240)
        if len(calibrated_points) < 2:
            calibrated_points = [[row["lati"], row["long"]] for _, row in segment_df.iterrows()]
        folium.PolyLine(
            calibrated_points,
            color=color,
            weight=4,
            opacity=0.8,
            tooltip=label,
        ).add_to(m)

    start_row = df.iloc[0]
    end_row = df.iloc[-1]
    folium.Marker(
        location=[start_row["lati"], start_row["long"]],
        icon=folium.Icon(color="green", icon="play"),
        popup=f"<b>起点</b><br>时间: {start_row['time']}<br>速度: {start_row['speed']} km/h<br>状态: {'载客' if start_row['status'] == 1 else '空载'}",
        tooltip="起点",
    ).add_to(m)
    folium.Marker(
        location=[end_row["lati"], end_row["long"]],
        icon=folium.Icon(color="red", icon="flag"),
        popup=f"<b>终点</b><br>时间: {end_row['time']}<br>速度: {end_row['speed']} km/h<br>状态: {'载客' if end_row['status'] == 1 else '空载'}",
        tooltip="终点",
    ).add_to(m)

    add_map_layers(m)

    start_str = format_time_tag(start_time)
    end_str = format_time_tag(end_time)
    output_path = save_path or os.path.join(
        CONFIG["OUTPUT_MAP_DIR"],
        f"vehicle_{vehicle_id}_trajectory_{start_str}_{end_str}.html",
    )
    ensure_parent_dir(output_path)
    m.save(output_path)
    log.info("轨迹地图已保存至: %s", output_path)
    return output_path


TRAJECTORY_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#be185d",
    "#4f46e5",
    "#0f766e",
    "#d97706",
]


def _trajectory_color(index):
    return TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)]


def _add_vehicle_markers(group, df, vehicle_id, color):
    start_row = df.iloc[0]
    end_row = df.iloc[-1]

    folium.CircleMarker(
        location=[start_row["lati"], start_row["long"]],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.95,
        popup=f"<b>车辆 {vehicle_id} 起点</b><br>时间: {start_row['time']}<br>状态: {'载客' if start_row['status'] == 1 else '空载'}",
        tooltip=f"车辆 {vehicle_id} 起点",
    ).add_to(group)
    folium.CircleMarker(
        location=[end_row["lati"], end_row["long"]],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.95,
        popup=f"<b>车辆 {vehicle_id} 终点</b><br>时间: {end_row['time']}<br>状态: {'载客' if end_row['status'] == 1 else '空载'}",
        tooltip=f"车辆 {vehicle_id} 终点",
    ).add_to(group)


def _add_vehicle_mid_label(group, df, vehicle_id, color):
    if df is None or len(df) == 0:
        return

    mid_row = df.iloc[len(df) // 2]
    folium.Marker(
        location=[mid_row["lati"], mid_row["long"]],
        icon=folium.DivIcon(
            html=f"""
            <div style="
                transform: translate(-50%, -50%);
                background: rgba(255,255,255,0.96);
                border: 1px solid {color};
                border-radius: 999px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 700;
                color: {color};
                box-shadow: 0 4px 10px rgba(15,23,42,0.18);
                white-space: nowrap;
            ">{vehicle_id}</div>
            """
        ),
        tooltip=f"车辆 {vehicle_id}",
    ).add_to(group)


def _trajectory_summary_stats(df):
    if df is None or len(df) == 0:
        return {
            "points": 0,
            "avg_speed": 0.0,
            "occupied_ratio": 0.0,
            "current_status": "未知",
            "start_time": None,
            "end_time": None,
        }

    occupied_points = int((df["status"] == 1).sum()) if "status" in df.columns else 0
    total_points = len(df)
    avg_speed = float(pd.to_numeric(df["speed"], errors="coerce").fillna(0.0).mean()) if "speed" in df.columns else 0.0
    current_status = "载客" if int(df.iloc[-1]["status"]) == 1 else "空载"
    return {
        "points": total_points,
        "avg_speed": avg_speed,
        "occupied_ratio": occupied_points / total_points if total_points else 0.0,
        "current_status": current_status,
        "start_time": df.iloc[0]["time"].strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": df.iloc[-1]["time"].strftime("%Y-%m-%d %H:%M:%S"),
    }


def _clamp_animation_time_scale(time_scale):
    try:
        value = float(time_scale)
    except (TypeError, ValueError):
        value = 1.0
    return max(1.0, min(5.0, value))


def _animation_display_speed(row):
    observed_speed = float(row["speed"]) if "speed" in row and pd.notna(row["speed"]) else 0.0
    geometry_speed = float(row.get("speed_kmh", 0.0)) if pd.notna(row.get("speed_kmh", 0.0)) else 0.0
    distance_km = float(row.get("distance_km", 0.0)) if pd.notna(row.get("distance_km", 0.0)) else 0.0
    low_threshold = float(CONFIG["ANIMATION_LOW_SPEED_DISPLAY_THRESHOLD_KMH"])
    min_driving_speed = float(CONFIG["ANIMATION_MIN_DRIVING_DISPLAY_SPEED_KMH"])
    cap = float(CONFIG["ANIMATION_SPEED_CAP_KMH"])

    is_moving = geometry_speed >= low_threshold or distance_km >= 0.005
    if is_moving and 0 <= observed_speed < low_threshold:
        reference_speed = geometry_speed if geometry_speed > 0 else min_driving_speed
        return min(cap, max(min_driving_speed, reference_speed))
    return min(cap, max(0.0, observed_speed))


def _vehicle_animation_features(df):
    features = []
    for idx, row in df.iterrows():
        speed_value = _animation_display_speed(row)
        time_ms = int(pd.Timestamp(row["time"]).timestamp() * 1000)
        features.append(
            {
                "lat": float(row["lati"]),
                "lng": float(row["long"]),
                "time": row["time"].strftime("%Y-%m-%d %H:%M:%S"),
                "timeMs": time_ms,
                "status": int(row["status"]),
                "speed": speed_value,
                "index": int(idx),
            }
        )
    return features


def _corrected_rows_to_animation_df(rows, fallback_start_time=None, fallback_end_time=None):
    rows = _normalize_corrected_cache_rows(rows)
    if len(rows) < 2 or not {"matched_lat", "matched_lon", "time"}.issubset(rows.columns):
        return pd.DataFrame()
    df = rows.dropna(subset=["matched_lat", "matched_lon", "time"]).copy()
    if len(df) < 2:
        return pd.DataFrame()
    df["lati"] = df["matched_lat"].astype(float)
    df["long"] = df["matched_lon"].astype(float)
    df["time"] = pd.to_datetime(df["time"])
    df["status"] = pd.to_numeric(df.get("status", 0), errors="coerce").fillna(0).astype(int)
    df["speed"] = pd.to_numeric(df.get("speed", 0.0), errors="coerce").fillna(0.0)
    df = df.sort_values(["time", "sequence"]).drop_duplicates(subset=["time", "lati", "long", "status"]).reset_index(drop=True)
    if df["time"].nunique() < len(df):
        start_ts = safe_datetime(fallback_start_time) or df["time"].min()
        end_ts = safe_datetime(fallback_end_time) or df["time"].max()
        if pd.Timestamp(end_ts) > pd.Timestamp(start_ts):
            df["time"] = _interpolate_times_for_points(df[["lati", "long"]].values.tolist(), start_ts, end_ts)
    return df[["time", "long", "lati", "status", "speed"]]


def _prepare_animation_dataframe(df):
    if df is None or len(df) < 2:
        return pd.DataFrame()
    df = resample_trajectory(df, CONFIG["MAX_TRAJECTORY_POINTS"])
    df = df.copy()
    df["prev_time"] = df["time"].shift(1)
    df["time_diff"] = (df["time"] - df["prev_time"]).dt.total_seconds().fillna(0)
    df["prev_lat"] = df["lati"].shift(1)
    df["prev_lng"] = df["long"].shift(1)
    df["distance_km"] = df.apply(
        lambda row: haversine_distance(row["prev_lat"], row["prev_lng"], row["lati"], row["long"]) if pd.notna(row["prev_time"]) else 0,
        axis=1,
    ).fillna(0.0)
    df["speed_kmh"] = df.apply(
        lambda row: (row["distance_km"] / (row["time_diff"] / 3600.0)) if row["time_diff"] and row["time_diff"] > 0 else 0.0,
        axis=1,
    ).replace([np.inf, -np.inf], 0.0)
    df.loc[df["speed_kmh"] > CONFIG["ANIMATION_SPEED_CAP_KMH"], "speed_kmh"] = CONFIG["ANIMATION_SPEED_CAP_KMH"]
    return df


def _load_animation_trajectory(vehicle_id, start_time=None, end_time=None, graph=None, network_meta=None):
    if graph is None:
        graph, network_meta = load_road_network()
    if graph is not None:
        corrected = load_road_corrected_vehicle_slice(
            vehicle_id,
            start_time=start_time,
            end_time=end_time,
            graph=graph,
            network_meta=network_meta,
        )
        corrected_df = _corrected_rows_to_animation_df(corrected.get("rows"), fallback_start_time=start_time, fallback_end_time=end_time)
        if len(corrected_df) >= 2:
            log.info("动画轨迹使用路网校正缓存: vehicle_id=%s, points=%s", vehicle_id, len(corrected_df))
            return _prepare_animation_dataframe(corrected_df)
        log.warning(
            "车辆 %s 路网校正动画点不足，回退原始轨迹: %s",
            vehicle_id,
            corrected.get("meta", {}).get("error", "校正结果不可用"),
        )

    raw_df = load_vehicle_trajectory(vehicle_id, start_time, end_time)
    return _prepare_animation_dataframe(raw_df)


def _lighten_hex_color(color):
    color = str(color).lstrip("#")
    if len(color) != 6:
        return "#9ca3af"
    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except ValueError:
        return "#9ca3af"
    r = int(r + (255 - r) * 0.68)
    g = int(g + (255 - g) * 0.68)
    b = int(b + (255 - b) * 0.68)
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_multi_animation_html(vehicle_entries, time_scale, bounds, global_start_ms, global_end_ms):
    playback_scale = _clamp_animation_time_scale(time_scale)
    payload = json.dumps(
        {
            "vehicles": [
                {
                    "vehicleId": vehicle_id,
                    "color": color,
                    "features": _vehicle_animation_features(df),
                    "summary": _trajectory_summary_stats(df),
                }
                for vehicle_id, df, color in vehicle_entries
            ]
        },
        ensure_ascii=False,
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>多车辆动画轨迹</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        :root {{
            --panel-bg: #ffffff;
            --panel-border: rgba(15, 23, 42, 0.12);
            --panel-shadow: 0 18px 50px rgba(15, 23, 42, 0.12);
            --text: #0f172a;
            --muted: #475569;
            --accent: #ef4444;
        }}
        html, body {{
            margin: 0;
            padding: 0;
            min-height: 100%;
            background: #f5f5f5;
            color: var(--text);
            overflow-x: hidden;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        body {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        #controls, #info {{
            position: relative;
            z-index: 2;
            margin: 12px 12px 0;
            padding: 12px 14px 10px;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            box-shadow: var(--panel-shadow);
        }}
        #controls .row, #info .row {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }}
        #controls .row + .row {{
            margin-top: 8px;
        }}
        #controls select {{
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            padding: 7px 10px;
            background: #fff;
            color: var(--text);
            min-width: 150px;
        }}
        #map {{
            width: calc(100% - 24px);
            height: 66vh;
            min-height: 460px;
            margin: 0 12px;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: var(--panel-shadow);
            border: 1px solid var(--panel-border);
        }}
        #info {{
            margin: 0 12px 12px;
        }}
        #statusGrid {{
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        .vehicle-status-card {{
            background: #fafafa;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 12px;
            padding: 10px 12px;
        }}
        .vehicle-status-card .title {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 700;
            margin-bottom: 6px;
        }}
        .vehicle-status-card .dot {{
            width: 11px;
            height: 11px;
            border-radius: 50%;
            display: inline-block;
            border: 1px solid rgba(0,0,0,0.18);
            flex: 0 0 auto;
        }}
        .vehicle-status-card .line {{
            font-size: 12px;
            color: var(--muted);
            line-height: 1.45;
        }}
        .vehicle-status-card .line strong {{
            color: var(--text);
        }}
        .label {{
            color: var(--muted);
            margin-right: 6px;
        }}
        .value {{
            font-variant-numeric: tabular-nums;
            font-weight: 700;
        }}
        @media (max-width: 1200px) {{
            #statusGrid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}
        @media (max-width: 768px) {{
            #statusGrid {{
                grid-template-columns: 1fr;
            }}
        }}
        .btn {{
            border: 0;
            border-radius: 999px;
            padding: 8px 14px;
            margin-right: 8px;
            background: #0f172a;
            color: white;
            cursor: pointer;
        }}
        .btn.secondary {{
            background: #475569;
        }}
        input[type="range"] {{
            width: 190px;
            vertical-align: middle;
            accent-color: var(--accent);
        }}
    </style>
</head>
<body>
    <div id="controls">
        <div class="row">
            <button class="btn" onclick="playAnimation()">播放</button>
            <button class="btn secondary" onclick="pauseAnimation()">暂停</button>
            <button class="btn secondary" onclick="resetAnimation()">重置</button>
        </div>
        <div class="row">
            <span class="label">速度缩放</span>
            <input type="range" id="speedSlider" min="1" max="5" step="0.5" value="{playback_scale:.1f}" onchange="updateSpeed()">
            <span class="value" id="speedValue">{playback_scale:.1f}x</span>
        </div>
    </div>
    <div id="map"></div>
    <div id="info">
        <div class="row">
            <span><span class="label">当前时间</span><span class="value" id="currentTime">--</span></span>
            <span><span class="label">全局进度</span><span class="value" id="globalProgress">0%</span></span>
            <span><span class="label">车辆数量</span><span class="value">{len(vehicle_entries)}</span></span>
        </div>
        <div id="statusGrid"></div>
    </div>

    <script>
        var fleetData = {payload}.vehicles;
        var speedBaseMultiplier = 1.0;
        var speedMultiplier = {playback_scale:.1f} * speedBaseMultiplier;
        var maxPlaybackSpeedKmh = 60.0;
        var minMovementSpeedKmh = 1.0;
        var targetFrameRate = 60;
        var statusUpdateIntervalMs = 100;
        var minDelay = {CONFIG["ANIMATION_MIN_DELAY_MS"]};
        var animationFrameId = null;
        var isPlaying = false;
        var pausedVirtualElapsed = 0;
        var playStartedAt = 0;
        var totalDurationMs = Math.max(1, {int(global_end_ms)} - {int(global_start_ms)});
        var globalStartMs = {int(global_start_ms)};
        var globalEndMs = {int(global_end_ms)};
        var lastFrameTimestamp = 0;
        var lastFleetStatusUpdateAt = 0;
        var frameStats = {{
            targetFps: targetFrameRate,
            fps: 0,
            belowTarget: false,
            sampleStart: 0,
            sampleFrames: 0
        }};
        window.__taxigpsAnimationStats = frameStats;

        var map = L.map('map', {{
            preferCanvas: true,
            zoomSnap: 0.5,
            zoomDelta: 0.5,
            fadeAnimation: false,
            markerZoomAnimation: false,
            zoomAnimation: false
        }}).setView([{CONFIG["MAP_CENTER"][0]}, {CONFIG["MAP_CENTER"][1]}], {CONFIG["MAP_ZOOM"]});
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors',
            maxZoom: 18,
            detectRetina: true,
            tileSize: 512,
            zoomOffset: -1
        }}).addTo(map);

        var bounds = L.latLngBounds([[{bounds["minLat"]}, {bounds["minLng"]}], [{bounds["maxLat"]}, {bounds["maxLng"]}]]);
        map.fitBounds(bounds.pad(0.12));

        function clamp(value, minValue, maxValue) {{
            return Math.max(minValue, Math.min(maxValue, value));
        }}

        function lerp(a, b, t) {{
            return a + (b - a) * t;
        }}

        function recordFrameSample(frameTimestamp) {{
            if (!isFinite(frameTimestamp) || frameTimestamp <= 0) return;
            if (!frameStats.sampleStart) {{
                frameStats.sampleStart = frameTimestamp;
                frameStats.sampleFrames = 0;
            }}
            frameStats.sampleFrames += 1;
            var elapsed = frameTimestamp - frameStats.sampleStart;
            if (elapsed >= 1000) {{
                frameStats.fps = frameStats.sampleFrames * 1000 / elapsed;
                frameStats.belowTarget = frameStats.fps < targetFrameRate;
                frameStats.sampleStart = frameTimestamp;
                frameStats.sampleFrames = 0;
            }}
        }}

        function segmentDurationMs(rawDuration, segmentDistance, startSpeed, endSpeed) {{
            if (segmentDistance <= 0) {{
                return rawDuration;
            }}
            var startObservedSpeed = Math.max(0, Number(startSpeed || 0));
            var endObservedSpeed = Math.max(0, Number(endSpeed || 0));
            var averageObservedSpeed = Math.max(minMovementSpeedKmh, (startObservedSpeed + endObservedSpeed) / 2);
            var observedDuration = (segmentDistance / averageObservedSpeed) * 3600000;
            var minimumDuration = (segmentDistance / maxPlaybackSpeedKmh) * 3600000;
            return Math.max(rawDuration, observedDuration, minimumDuration);
        }}

        function speedAwareProgress(t, startSpeed, endSpeed) {{
            var x = clamp(t, 0, 1);
            var startObservedSpeed = Math.max(0, Number(startSpeed || 0));
            var endObservedSpeed = Math.max(0, Number(endSpeed || 0));
            var averageObservedSpeed = (startObservedSpeed + endObservedSpeed) / 2;
            if (averageObservedSpeed <= 0.01) {{
                return x;
            }}
            var distanceRatio = (startObservedSpeed * x + 0.5 * (endObservedSpeed - startObservedSpeed) * x * x) / averageObservedSpeed;
            return clamp(distanceRatio, 0, 1);
        }}

        function easeInOutCubic(t) {{
            t = clamp(t, 0, 1);
            return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        }}

        function getEasingDerivative(t) {{
            var x = clamp(t, 0, 1);
            return x < 0.5 ? 12 * x * x : 12 * (1 - x) * (1 - x);
        }}

        function haversine(lat1, lng1, lat2, lng2) {{
            var R = 6371.0;
            function toRad(v) {{ return v * Math.PI / 180; }}
            var dLat = toRad(lat2 - lat1);
            var dLng = toRad(lng2 - lng1);
            var a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
                Math.sin(dLng / 2) * Math.sin(dLng / 2);
            var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
            return R * c;
        }}

        function formatSpeedValue(speed) {{
            if (!isFinite(speed) || speed < 0) return '0.0';
            return speed < 10 ? speed.toFixed(2) : speed.toFixed(1);
        }}

        function formatTimeFromMs(timeMs) {{
            var dt = new Date(timeMs);
            var year = dt.getFullYear();
            var month = String(dt.getMonth() + 1).padStart(2, '0');
            var day = String(dt.getDate()).padStart(2, '0');
            var hour = String(dt.getHours()).padStart(2, '0');
            var minute = String(dt.getMinutes()).padStart(2, '0');
            var second = String(dt.getSeconds()).padStart(2, '0');
            return year + '-' + month + '-' + day + ' ' + hour + ':' + minute + ':' + second;
        }}

        function buildVehicleState(vehicle) {{
            vehicle.features = vehicle.features || [];
            vehicle.pathLatLngs = [];
            vehicle.segments = [];
            vehicle.previewLine = null;
            vehicle.travelLine = null;
            vehicle.travelLineState = {{fixedIndex: 0, hasMovingPoint: false}};
            vehicle.marker = null;
            vehicle.currentSegmentIndex = 0;
            vehicle.currentTimeMs = vehicle.features.length > 0 ? vehicle.features[0].timeMs : 0;
            vehicle.lastStatusUpdateAt = 0;

            if (vehicle.features.length === 0) {{
                return;
            }}

            var timelineCursor = Math.max(0, Number(vehicle.features[0].timeMs || globalStartMs) - globalStartMs);
            vehicle.timelineStartMs = timelineCursor;
            vehicle.timelineEndMs = timelineCursor;

            for (var i = 0; i < vehicle.features.length; i++) {{
                var feature = vehicle.features[i];
                vehicle.pathLatLngs.push([feature.lat, feature.lng]);
                if (i === 0) continue;
                var prev = vehicle.features[i - 1];
                var rawDuration = Math.max(1, feature.timeMs - prev.timeMs);
                var segmentDistance = haversine(prev.lat, prev.lng, feature.lat, feature.lng);
                var startDisplaySpeed = Number(prev.speed || 0);
                var endDisplaySpeed = Number(feature.speed || 0);
                var duration = segmentDurationMs(rawDuration, segmentDistance, startDisplaySpeed, endDisplaySpeed);
                var segmentSpeed = duration > 0 && segmentDistance > 0
                    ? Math.min({CONFIG["ANIMATION_SPEED_CAP_KMH"]}, segmentDistance / (duration / 3600000))
                    : feature.speed;
                vehicle.segments.push({{
                    startIndex: i - 1,
                    endIndex: i,
                    startLat: prev.lat,
                    startLng: prev.lng,
                    endLat: feature.lat,
                    endLng: feature.lng,
                    startTimeMs: timelineCursor,
                    endTimeMs: timelineCursor + duration,
                    durationMs: duration,
                    speedKmh: segmentSpeed,
                    startDisplaySpeed: startDisplaySpeed,
                    endDisplaySpeed: endDisplaySpeed,
                    status: prev.status
                }});
                timelineCursor += duration;
            }}
            vehicle.timelineEndMs = Math.max(vehicle.timelineStartMs + 1, timelineCursor);
            totalDurationMs = Math.max(totalDurationMs, vehicle.timelineEndMs);

            var first = vehicle.features[0];
            var markerIcon = L.divIcon({{
                className: 'car-marker',
                html: '<div style="width: 16px; height: 16px; background: ' + vehicle.color + '; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 8px rgba(0,0,0,0.35);"></div>',
                iconSize: [16, 16],
                iconAnchor: [8, 8]
            }});
            vehicle.previewLine = L.polyline(vehicle.pathLatLngs, {{
                color: '{_lighten_hex_color("#64748b")}',
                weight: 2,
                opacity: 0.35,
                dashArray: '6,6'
            }}).addTo(map);
            vehicle.travelLine = L.polyline([vehicle.pathLatLngs[0]], {{
                color: vehicle.color,
                weight: 4,
                opacity: 0.92
            }}).addTo(map);
            vehicle.marker = L.marker(vehicle.pathLatLngs[0], {{icon: markerIcon}}).addTo(map);
            vehicle.marker.bindTooltip('车辆 ' + vehicle.vehicleId, {{sticky: true}});
            vehicle.marker.bindPopup('<b>车辆 ' + vehicle.vehicleId + '</b><br>开始时间: ' + first.time + '<br>状态: ' + (first.status === 1 ? '载客' : '空载'));
        }}

        function resetTravelLineState(line, state, firstLatLng) {{
            state.fixedIndex = 0;
            state.hasMovingPoint = false;
            line.setLatLngs([firstLatLng]);
        }}

        function updateTravelLineEndpoint(line, pathLatLngs, targetIndex, currentLatLng, travelLineState) {{
            if (!pathLatLngs || pathLatLngs.length === 0) return;
            var latLngs = line.getLatLngs();
            if (latLngs.length === 0) {{
                line.addLatLng(pathLatLngs[0]);
                latLngs = line.getLatLngs();
            }}
            while (travelLineState.fixedIndex < targetIndex && travelLineState.fixedIndex + 1 < pathLatLngs.length) {{
                var nextFixed = pathLatLngs[travelLineState.fixedIndex + 1];
                if (travelLineState.hasMovingPoint && latLngs.length > 0) {{
                    latLngs[latLngs.length - 1] = L.latLng(nextFixed[0], nextFixed[1]);
                    travelLineState.hasMovingPoint = false;
                    line.redraw();
                }} else {{
                    line.addLatLng(nextFixed);
                    latLngs = line.getLatLngs();
                }}
                travelLineState.fixedIndex += 1;
            }}
            if (travelLineState.hasMovingPoint && latLngs.length > 0) {{
                latLngs[latLngs.length - 1] = L.latLng(currentLatLng[0], currentLatLng[1]);
                line.redraw();
            }} else {{
                line.addLatLng(currentLatLng);
                travelLineState.hasMovingPoint = true;
            }}
        }}

        function buildStatusGrid() {{
            var container = document.getElementById('statusGrid');
            if (!container) return;
            var html = '';
            for (var i = 0; i < fleetData.length; i++) {{
                var vehicle = fleetData[i];
                html += '' +
                    '<div class="vehicle-status-card" id="card-' + vehicle.vehicleId + '">' +
                    '<div class="title"><span class="dot" style="background:' + vehicle.color + '"></span>车辆 ' + vehicle.vehicleId + '</div>' +
                    '<div class="line">速度 <strong id="speed-' + vehicle.vehicleId + '">-- km/h</strong></div>' +
                    '<div class="line">状态 <strong id="status-' + vehicle.vehicleId + '">--</strong></div>' +
                    '<div class="line">进度 <strong id="progress-' + vehicle.vehicleId + '">0%</strong></div>' +
                    '</div>';
            }}
            container.innerHTML = html;
        }}

        function updateVehicleCard(vehicle, currentTimeMs, currentSpeed, currentStatus, progress) {{
            var speedNode = document.getElementById('speed-' + vehicle.vehicleId);
            var statusNode = document.getElementById('status-' + vehicle.vehicleId);
            var progressNode = document.getElementById('progress-' + vehicle.vehicleId);
            if (speedNode) speedNode.textContent = formatSpeedValue(currentSpeed) + ' km/h';
            if (statusNode) statusNode.textContent = currentStatus;
            if (progressNode) progressNode.textContent = Math.round(progress * 100) + '%';
        }}

        function maybeUpdateVehicleCard(vehicle, currentTimeMs, currentSpeed, currentStatus, progress, frameTimestamp, forceUpdate) {{
            var now = frameTimestamp || performance.now();
            if (!forceUpdate && now - vehicle.lastStatusUpdateAt < statusUpdateIntervalMs) {{
                return;
            }}
            vehicle.lastStatusUpdateAt = now;
            updateVehicleCard(vehicle, currentTimeMs, currentSpeed, currentStatus, progress);
        }}

        function renderVehicle(vehicle, currentElapsed, frameTimestamp, forceStatusUpdate) {{
            if (!vehicle.features || vehicle.features.length === 0) return;

            var first = vehicle.features[0];
            var last = vehicle.features[vehicle.features.length - 1];
            var renderLat = first.lat;
            var renderLng = first.lng;
            var travelTargetIndex = 0;
            var currentSpeed = first.speed || 0;
            var currentStatus = first.status === 1 ? '载客' : '空载';
            var progress = 0;

            if (currentElapsed <= vehicle.timelineStartMs) {{
                vehicle.currentSegmentIndex = 0;
            }} else if (currentElapsed >= vehicle.timelineEndMs) {{
                renderLat = last.lat;
                renderLng = last.lng;
                travelTargetIndex = vehicle.pathLatLngs.length - 1;
                currentSpeed = last.speed || 0;
                currentStatus = last.status === 1 ? '载客' : '空载';
                progress = 1;
                vehicle.currentSegmentIndex = vehicle.segments.length - 1;
            }} else {{
                while (vehicle.currentSegmentIndex < vehicle.segments.length - 1 && currentElapsed > vehicle.segments[vehicle.currentSegmentIndex].endTimeMs) {{
                    vehicle.currentSegmentIndex += 1;
                }}
                while (vehicle.currentSegmentIndex > 0 && currentElapsed < vehicle.segments[vehicle.currentSegmentIndex].startTimeMs) {{
                    vehicle.currentSegmentIndex -= 1;
                }}

                var segment = vehicle.segments[vehicle.currentSegmentIndex];
                var localElapsed = currentElapsed - segment.startTimeMs;
                var rawT = segment.durationMs > 0 ? clamp(localElapsed / segment.durationMs, 0, 1) : 1;
                var easedT = speedAwareProgress(rawT, segment.startDisplaySpeed, segment.endDisplaySpeed);
                renderLat = lerp(segment.startLat, segment.endLat, easedT);
                renderLng = lerp(segment.startLng, segment.endLng, easedT);
                currentSpeed = Math.max(0, lerp(segment.startDisplaySpeed, segment.endDisplaySpeed, rawT));
                currentStatus = segment.status === 1 ? '载客' : '空载';
                progress = clamp((currentElapsed - vehicle.timelineStartMs) / Math.max(1, vehicle.timelineEndMs - vehicle.timelineStartMs), 0, 1);
                travelTargetIndex = segment.startIndex;
            }}

            updateTravelLineEndpoint(vehicle.travelLine, vehicle.pathLatLngs, travelTargetIndex, [renderLat, renderLng], vehicle.travelLineState);
            vehicle.marker.setLatLng([renderLat, renderLng]);
            maybeUpdateVehicleCard(vehicle, currentElapsed, currentSpeed, currentStatus, progress, frameTimestamp, forceStatusUpdate);
        }}

        function maybeUpdateFleetStatus(currentElapsed, frameTimestamp, forceStatusUpdate) {{
            var now = frameTimestamp || performance.now();
            if (!forceStatusUpdate && now - lastFleetStatusUpdateAt < statusUpdateIntervalMs) {{
                return;
            }}
            lastFleetStatusUpdateAt = now;
            var currentTimeMs = globalStartMs + clamp(currentElapsed, 0, totalDurationMs);
            document.getElementById('currentTime').textContent = formatTimeFromMs(currentTimeMs);
            document.getElementById('globalProgress').textContent = Math.round(clamp(currentElapsed / totalDurationMs, 0, 1) * 100) + '%';
        }}

        function renderFleet(currentElapsed, frameTimestamp, forceStatusUpdate) {{
            maybeUpdateFleetStatus(currentElapsed, frameTimestamp, forceStatusUpdate);
            for (var i = 0; i < fleetData.length; i++) {{
                renderVehicle(fleetData[i], currentElapsed, frameTimestamp, forceStatusUpdate);
            }}
        }}

        function getVirtualElapsed(now) {{
            if (!isPlaying) {{
                return pausedVirtualElapsed;
            }}
            return (now - playStartedAt) * speedMultiplier;
        }}

        function tick(currentTime) {{
            if (!isPlaying) return;
            lastFrameTimestamp = currentTime;
            var virtualElapsed = getVirtualElapsed(currentTime);
            if (!isFinite(virtualElapsed)) {{
                virtualElapsed = pausedVirtualElapsed;
            }}
            if (virtualElapsed >= totalDurationMs) {{
                pausedVirtualElapsed = totalDurationMs;
                recordFrameSample(currentTime);
                renderFleet(totalDurationMs, currentTime, true);
                isPlaying = false;
                if (animationFrameId) {{
                    cancelAnimationFrame(animationFrameId);
                    animationFrameId = null;
                }}
                return;
            }}
            pausedVirtualElapsed = virtualElapsed;
            recordFrameSample(currentTime);
            renderFleet(virtualElapsed, currentTime, false);
            animationFrameId = requestAnimationFrame(tick);
        }}

        function playAnimation() {{
            if (isPlaying) return;
            isPlaying = true;
            if (pausedVirtualElapsed >= totalDurationMs) {{
                pausedVirtualElapsed = 0;
            }}
            lastFrameTimestamp = performance.now();
            playStartedAt = lastFrameTimestamp - pausedVirtualElapsed / Math.max(speedMultiplier, 0.001);
            animationFrameId = requestAnimationFrame(tick);
        }}

        function pauseAnimation() {{
            isPlaying = false;
            if (animationFrameId) {{
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }}
        }}

        function resetAnimation() {{
            pauseAnimation();
            pausedVirtualElapsed = 0;
            lastFrameTimestamp = 0;
            playStartedAt = 0;
            for (var i = 0; i < fleetData.length; i++) {{
                var vehicle = fleetData[i];
                if (vehicle.pathLatLngs && vehicle.pathLatLngs.length > 0) {{
                    resetTravelLineState(vehicle.travelLine, vehicle.travelLineState, vehicle.pathLatLngs[0]);
                    vehicle.currentSegmentIndex = 0;
                    vehicle.lastStatusUpdateAt = 0;
                }}
            }}
            lastFleetStatusUpdateAt = 0;
            renderFleet(0, performance.now(), true);
        }}

        function updateSpeed() {{
            if (isPlaying) {{
                var now = performance.now();
                pausedVirtualElapsed = clamp(getVirtualElapsed(now), 0, totalDurationMs);
                lastFrameTimestamp = now;
            }}
            var selectedSpeed = parseFloat(document.getElementById('speedSlider').value);
            speedMultiplier = selectedSpeed * speedBaseMultiplier;
            if (isPlaying) {{
                playStartedAt = lastFrameTimestamp - pausedVirtualElapsed / Math.max(speedMultiplier, 0.001);
            }}
            document.getElementById('speedValue').textContent = selectedSpeed.toFixed(1) + 'x';
            if (!isPlaying) {{
                renderFleet(pausedVirtualElapsed, performance.now(), true);
            }}
        }}

        for (var i = 0; i < fleetData.length; i++) {{
            buildVehicleState(fleetData[i]);
        }}
        buildStatusGrid();
        resetAnimation();
    </script>
</body>
</html>
"""
    return html


def plot_vehicle_trajectories(vehicle_ids, start_time=None, end_time=None, save_path=None):
    cleaned_vehicle_ids = []
    for vehicle_id in vehicle_ids or []:
        vehicle_id = str(vehicle_id).strip()
        if vehicle_id and vehicle_id not in cleaned_vehicle_ids:
            cleaned_vehicle_ids.append(vehicle_id)
    cleaned_vehicle_ids = cleaned_vehicle_ids[:10]

    log.info(
        "查询多车辆轨迹: vehicle_ids=%s, start_time=%s, end_time=%s",
        cleaned_vehicle_ids,
        start_time,
        end_time,
    )

    if not cleaned_vehicle_ids:
        log.warning("未提供车辆ID，无法生成多车轨迹")
        return None

    loaded_frames = []
    vehicle_entries = []
    per_vehicle_limit = max(80, min(400, CONFIG["MAX_TRAJECTORY_POINTS"] // max(1, len(cleaned_vehicle_ids))))

    for index, vehicle_id in enumerate(cleaned_vehicle_ids):
        df = load_vehicle_trajectory(vehicle_id, start_time, end_time)
        if df is None or len(df) == 0:
            log.warning("车辆 %s 在当前时间范围内没有轨迹数据", vehicle_id)
            continue

        df = resample_trajectory(df, per_vehicle_limit)
        df = df.copy()
        color = _trajectory_color(index)
        df["vehicle_color"] = color
        loaded_frames.append(df)
        vehicle_entries.append((vehicle_id, df, color))

    if not vehicle_entries:
        log.warning("所有车辆都没有可用轨迹数据")
        return None

    combined_df = pd.concat(loaded_frames, ignore_index=True)
    m = build_map(combined_df)

    for index, (vehicle_id, df, color) in enumerate(vehicle_entries):
        group = folium.FeatureGroup(name=f"车辆 {vehicle_id}", show=True)
        group.add_to(m)

        segments = _trajectory_segment_dfs(df)
        if segments:
            for segment in segments:
                segment_df = segment["df"]
                occupied = segment["status"] == 1
                calibrated_points = _calibrated_segment_points(segment_df, max_points=180)
                if len(calibrated_points) < 2:
                    calibrated_points = [[row["lati"], row["long"]] for _, row in segment_df.iterrows()]
                folium.PolyLine(
                    calibrated_points,
                    color=color,
                    weight=4 if occupied else 3,
                    opacity=0.88 if occupied else 0.6,
                    dash_array=None if occupied else "6,6",
                    tooltip=f"车辆 {vehicle_id} - {'载客' if occupied else '空载'}",
                ).add_to(group)
        else:
            folium.PolyLine(
                [[row["lati"], row["long"]] for _, row in df.iterrows()],
                color=color,
                weight=5,
                opacity=0.95,
                tooltip=f"车辆 {vehicle_id}",
            ).add_to(group)

        _add_vehicle_markers(group, df, vehicle_id, color)
        _add_vehicle_mid_label(group, df, vehicle_id, color)

    legend_rows = "".join(
        f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
          <span style="width:12px;height:12px;border-radius:50%;background:{entry[2]};display:inline-block;border:1px solid rgba(0,0,0,0.18);"></span>
          <span style="font-size:12px;color:#334155;">车辆 {entry[0]}</span>
        </div>
        """
        for entry in vehicle_entries
    )
    legend_html = f"""
    <div style="position:fixed;right:16px;top:16px;z-index:9999;background:rgba(255,255,255,0.96);border:1px solid rgba(148,163,184,0.35);border-radius:12px;padding:12px 14px;box-shadow:0 12px 32px rgba(15,23,42,0.12);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-width:150px;">
      <div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:6px;">轨迹图例</div>
      <div style="font-size:12px;color:#64748b;line-height:1.5;margin-bottom:8px;">颜色区分车辆，实线表示载客，虚线表示空载。</div>
      {legend_rows}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    add_map_layers(m)

    start_str = format_time_tag(start_time)
    end_str = format_time_tag(end_time)
    vehicle_tag = "-".join(cleaned_vehicle_ids[:3])
    if len(cleaned_vehicle_ids) > 3:
        vehicle_tag += "-more"
    output_path = save_path or os.path.join(
        CONFIG["OUTPUT_MAP_DIR"],
        f"vehicles_{vehicle_tag}_trajectory_{start_str}_{end_str}.html",
    )
    ensure_parent_dir(output_path)
    m.save(output_path)
    log.info("多车辆轨迹地图已保存至: %s", output_path)
    return output_path


def plot_road_corrected_trajectories(vehicle_ids, start_time=None, end_time=None, enable_correction=True, save_path=None):
    cleaned_vehicle_ids = []
    for vehicle_id in vehicle_ids or []:
        vehicle_id = str(vehicle_id).strip()
        if vehicle_id and vehicle_id not in cleaned_vehicle_ids:
            cleaned_vehicle_ids.append(vehicle_id)
    cleaned_vehicle_ids = cleaned_vehicle_ids[: CONFIG["MAX_ROAD_CORRECTION_VEHICLES"]]

    log.info(
        "生成路网校正轨迹: vehicle_ids=%s, start_time=%s, end_time=%s, enable_correction=%s",
        cleaned_vehicle_ids,
        start_time,
        end_time,
        enable_correction,
    )

    if not cleaned_vehicle_ids:
        return None, {"success": False, "error": "未提供车辆ID。"}

    vehicle_entries = []
    loaded_frames = []
    per_vehicle_limit = max(80, min(320, CONFIG["MAX_TRAJECTORY_POINTS"] // max(1, len(cleaned_vehicle_ids))))
    for index, vehicle_id in enumerate(cleaned_vehicle_ids):
        df = load_vehicle_trajectory(vehicle_id, start_time, end_time)
        if df is None or len(df) == 0:
            continue
        df = resample_trajectory(df, per_vehicle_limit)
        color = _trajectory_color(index)
        vehicle_entries.append({"vehicle_id": vehicle_id, "df": df, "color": color})
        loaded_frames.append(df.assign(vehicle_id=vehicle_id))

    if not vehicle_entries:
        return None, {"success": False, "error": "所选车辆在当前时间范围内没有轨迹数据。"}

    combined_df = pd.concat(loaded_frames, ignore_index=True)
    m = build_map(combined_df)
    graph = None
    network_meta = {"available": False}
    correction_available = False
    if enable_correction:
        graph, network_meta = load_road_network()
        correction_available = graph is not None

    corrections = []
    for entry in vehicle_entries:
        vehicle_id = entry["vehicle_id"]
        df = entry["df"]
        color = entry["color"]

        raw_group = folium.FeatureGroup(name=f"车辆 {vehicle_id} 原始轨迹", show=True)
        raw_group.add_to(m)
        for segment in _trajectory_segment_dfs(df):
            segment_df = segment["df"]
            occupied = segment["status"] == 1
            folium.PolyLine(
                [[row["lati"], row["long"]] for _, row in segment_df.iterrows()],
                color=color,
                weight=3 if occupied else 2,
                opacity=0.36,
                dash_array=None if occupied else "6,6",
                tooltip=f"车辆 {vehicle_id} 原始{'载客' if occupied else '空载'}轨迹",
            ).add_to(raw_group)
        _add_vehicle_markers(raw_group, df, vehicle_id, color)
        _add_vehicle_mid_label(raw_group, df, vehicle_id, color)

        if not enable_correction:
            corrections.append(
                {
                    "vehicle_id": vehicle_id,
                    "success": False,
                    "raw_points": len(df),
                    "corrected_points": 0,
                    "path_failures": 0,
                    "nearest_failures": 0,
                    "message": "未启用路网校正。",
                }
            )
            continue

        if not correction_available:
            corrections.append(
                {
                    "vehicle_id": vehicle_id,
                    "success": False,
                    "raw_points": len(df),
                    "corrected_points": 0,
                    "path_failures": 0,
                    "nearest_failures": 0,
                    "message": network_meta.get("error", "路网未加载。"),
                }
            )
            continue

        result = load_road_corrected_vehicle_slice(
            vehicle_id,
            start_time=start_time,
            end_time=end_time,
            graph=graph,
            network_meta=network_meta,
        )
        meta = result["meta"]
        corrected_points = result["points"]
        corrections.append(
            {
                "vehicle_id": vehicle_id,
                "success": bool(meta.get("success")),
                "raw_points": meta.get("raw_points", len(df)),
                "sampled_points": meta.get("sampled_points", 0),
                "cache_hit": bool(meta.get("cache_hit")),
                "cache_rows": meta.get("cache_rows", 0),
                "window_rows": meta.get("window_rows", len(corrected_points)),
                "processed_intervals": meta.get("processed_intervals", 0),
                "corrected_points": meta.get("window_corrected_points", len(corrected_points)),
                "matched_nodes": meta.get("matched_nodes", 0),
                "path_failures": meta.get("path_failures", 0),
                "nearest_failures": meta.get("nearest_failures", 0),
                "undirected_segments": meta.get("undirected_segments", 0),
                "cache_path": meta.get("cache_path", ""),
                "coverage_path": meta.get("coverage_path", ""),
                "message": meta.get("error") or meta.get("message", "校正完成。"),
            }
        )
        if len(corrected_points) >= 2:
            corrected_group = folium.FeatureGroup(name=f"车辆 {vehicle_id} 路网校正", show=True)
            corrected_group.add_to(m)
            folium.PolyLine(
                corrected_points,
                color=color,
                weight=6,
                opacity=0.9,
                tooltip=f"车辆 {vehicle_id} 路网校正轨迹",
            ).add_to(corrected_group)
            folium.PolyLine(
                corrected_points,
                color="#ffffff",
                weight=2,
                opacity=0.72,
                tooltip=f"车辆 {vehicle_id} 路网校正轨迹",
            ).add_to(corrected_group)

    legend_rows = "".join(
        f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
          <span style="width:26px;height:0;border-top:3px solid {entry['color']};opacity:.36;"></span>
          <span style="font-size:12px;color:#334155;">车辆 {entry['vehicle_id']} 原始轨迹</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
          <span style="width:26px;height:0;border-top:6px solid {entry['color']};"></span>
          <span style="font-size:12px;color:#334155;">车辆 {entry['vehicle_id']} 校正轨迹</span>
        </div>
        """
        for entry in vehicle_entries
    )
    network_label = network_meta.get("path") or "未加载"
    legend_html = f"""
    <div style="position:fixed;right:16px;top:16px;z-index:9999;background:rgba(255,255,255,0.96);border:1px solid rgba(148,163,184,0.35);border-radius:12px;padding:12px 14px;box-shadow:0 12px 32px rgba(15,23,42,0.12);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-width:190px;max-width:320px;">
      <div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:6px;">路网校正图例</div>
      <div style="font-size:12px;color:#64748b;line-height:1.5;margin-bottom:8px;">淡线为原始 GPS 轨迹，粗线为道路节点最短路拼接结果。点击地图可查看经纬度。</div>
      {legend_rows}
      <div style="font-size:11px;color:#64748b;line-height:1.45;border-top:1px solid rgba(148,163,184,0.25);padding-top:8px;">路网: {network_label}</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    add_map_layers(m)

    start_str = format_time_tag(start_time)
    end_str = format_time_tag(end_time)
    vehicle_tag = "-".join(cleaned_vehicle_ids[:3])
    output_path = save_path or os.path.join(
        CONFIG["OUTPUT_MAP_DIR"],
        f"road_corrected_{vehicle_tag}_{start_str}_{end_str}.html",
    )
    ensure_parent_dir(output_path)
    m.save(output_path)

    successful = sum(1 for item in corrections if item["success"])
    info = {
        "success": successful > 0 if enable_correction else True,
        "enabled": enable_correction,
        "vehicles": len(vehicle_entries),
        "successful_corrections": successful,
        "corrections": corrections,
        "network": network_meta,
        "output_path": output_path,
    }
    log.info(
        "路网校正地图已保存: output=%s, vehicles=%s, successful=%s, network=%s",
        output_path,
        len(vehicle_entries),
        successful,
        network_label,
    )
    return output_path, info


def build_congestion_segments(vehicle_ids, start_time=None, end_time=None, bucket_minutes=15):
    cleaned_vehicle_ids = []
    for vehicle_id in vehicle_ids or []:
        vehicle_id = str(vehicle_id).strip()
        if vehicle_id and vehicle_id not in cleaned_vehicle_ids:
            cleaned_vehicle_ids.append(vehicle_id)
    cleaned_vehicle_ids = cleaned_vehicle_ids[: CONFIG["MAX_CONGESTION_VEHICLES"]]

    graph, network_meta = load_road_network()
    if graph is None:
        return pd.DataFrame(), {"success": False, "error": network_meta.get("error", "路网未加载。"), "network": network_meta}

    matched_frames = []
    matching_rows = []
    for vehicle_id in cleaned_vehicle_ids:
        df = load_vehicle_trajectory(vehicle_id, start_time, end_time)
        if df is None or len(df) < 2:
            matching_rows.append({"vehicle_id": vehicle_id, "success": False, "message": "当前时间范围内轨迹点不足。"})
            continue
        df = df.copy()
        df["vehicle_id"] = vehicle_id
        result = approximate_hmm_match_trajectory(df, graph=graph, network_meta=network_meta)
        meta = result["meta"]
        matching_rows.append(
            {
                "vehicle_id": vehicle_id,
                "success": bool(meta.get("success")),
                "raw_points": meta.get("raw_points", len(df)),
                "sampled_points": meta.get("sampled_points", 0),
                "matched_nodes": meta.get("matched_nodes", 0),
                "segments": meta.get("segments", 0),
                "path_failures": meta.get("path_failures", 0),
                "nearest_failures": meta.get("nearest_failures", 0),
                "message": meta.get("error", "匹配完成。"),
            }
        )
        segments = result["segments"]
        if segments is not None and len(segments) > 0:
            matched_frames.append(segments)

    if not matched_frames:
        return pd.DataFrame(), {
            "success": False,
            "error": "没有生成可聚合的路段速度。",
            "network": network_meta,
            "matching": matching_rows,
        }

    matched_df = pd.concat(matched_frames, ignore_index=True)
    bucket_minutes = int(bucket_minutes or 15)
    matched_df["time_bucket"] = pd.to_datetime(matched_df["time"]).dt.floor(f"{bucket_minutes}min")
    grouped = (
        matched_df.groupby(["time_bucket", "segment_key"], as_index=False)
        .agg(
            speed_kmh=("speed_kmh", "mean"),
            vehicle_count=("vehicle_id", "nunique"),
            sample_count=("speed_kmh", "size"),
            edge_length_km=("edge_length_km", "mean"),
            lat1=("lat1", "first"),
            lng1=("lng1", "first"),
            lat2=("lat2", "first"),
            lng2=("lng2", "first"),
        )
        .sort_values(["time_bucket", "sample_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    grouped["color"] = grouped["speed_kmh"].apply(speed_to_color)
    grouped["level"] = grouped["speed_kmh"].apply(speed_to_level)
    if len(grouped) > CONFIG["MAX_CONGESTION_SEGMENTS"]:
        grouped = (
            grouped.sort_values(["time_bucket", "sample_count", "edge_length_km"], ascending=[True, False, False])
            .groupby("time_bucket", group_keys=False)
            .head(max(20, CONFIG["MAX_CONGESTION_SEGMENTS"] // max(1, grouped["time_bucket"].nunique())))
            .head(CONFIG["MAX_CONGESTION_SEGMENTS"])
            .reset_index(drop=True)
        )

    meta = {
        "success": True,
        "method": "近似 HMM: 最近道路节点作为发射近似，最短路连续性作为转移约束。",
        "vehicles": len(cleaned_vehicle_ids),
        "bucket_minutes": bucket_minutes,
        "segment_rows": len(grouped),
        "time_slices": int(grouped["time_bucket"].nunique()),
        "network": network_meta,
        "matching": matching_rows,
    }
    return grouped, meta


def _congestion_legend_html(title, subtitle, network_label=None):
    network_line = f'<div style="font-size:11px;color:#64748b;line-height:1.45;border-top:1px solid rgba(148,163,184,0.25);padding-top:8px;">路网: {network_label}</div>' if network_label else ""
    return f"""
    <div style="position:fixed;right:16px;top:16px;z-index:9999;background:rgba(255,255,255,0.96);border:1px solid rgba(148,163,184,0.35);border-radius:12px;padding:12px 14px;box-shadow:0 12px 32px rgba(15,23,42,0.12);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-width:210px;max-width:340px;">
      <div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:6px;">{title}</div>
      <div style="font-size:12px;color:#64748b;line-height:1.5;margin-bottom:8px;">{subtitle}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;"><span style="width:24px;border-top:5px solid #dc2626;"></span><span style="font-size:12px;color:#334155;">&lt; 15 km/h 严重拥堵</span></div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;"><span style="width:24px;border-top:5px solid #f97316;"></span><span style="font-size:12px;color:#334155;">15-25 km/h 拥堵</span></div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;"><span style="width:24px;border-top:5px solid #eab308;"></span><span style="font-size:12px;color:#334155;">25-40 km/h 缓行</span></div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;"><span style="width:24px;border-top:5px solid #16a34a;"></span><span style="font-size:12px;color:#334155;">≥ 40 km/h 畅通</span></div>
      {network_line}
    </div>
    """


def _inject_congestion_player(m, grouped):
    map_name = m.get_name()
    panel_id = f"congestion-player-{map_name}"
    rows = []
    for _, row in grouped.iterrows():
        rows.append(
            {
                "bucket": pd.Timestamp(row["time_bucket"]).strftime("%H:%M"),
                "segmentKey": row["segment_key"],
                "points": [[float(row["lat1"]), float(row["lng1"])], [float(row["lat2"]), float(row["lng2"])]],
                "speed": round(float(row["speed_kmh"]), 1),
                "color": row["color"],
                "level": row["level"],
                "samples": int(row["sample_count"]),
                "vehicles": int(row["vehicle_count"]),
            }
        )
    player_html = f"""
    <div id="{panel_id}" style="position:fixed;left:16px;top:16px;z-index:9999;background:rgba(255,255,255,0.96);border:1px solid rgba(148,163,184,0.45);border-radius:12px;padding:10px 12px;box-shadow:0 12px 30px rgba(15,23,42,0.14);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#0f172a;min-width:260px;">
      <div style="font-size:12px;font-weight:800;margin-bottom:6px;">路段速度播放</div>
      <div data-role="bucket" style="font-size:12px;color:#475569;margin-bottom:8px;">准备中</div>
      <div style="display:flex;align-items:center;gap:8px;">
        <button data-role="play" style="border:1px solid #cbd5e1;background:#fff;border-radius:8px;padding:4px 10px;cursor:pointer;">播放</button>
        <input data-role="slider" type="range" min="0" max="0" value="0" style="width:150px;accent-color:#2563eb;">
      </div>
    </div>
    """
    player_js = f"""
    (function bindCongestionPlayer() {{
        var map = window[{json.dumps(map_name)}];
        var panel = document.getElementById({json.dumps(panel_id)});
        if (!map || !panel || typeof L === 'undefined') {{
            window.setTimeout(bindCongestionPlayer, 50);
            return;
        }}
        var rows = {json.dumps(rows, ensure_ascii=False)};
        var buckets = Array.from(new Set(rows.map(function(row) {{ return row.bucket; }})));
        var layerGroup = L.layerGroup().addTo(map);
        var slider = panel.querySelector('[data-role="slider"]');
        var playButton = panel.querySelector('[data-role="play"]');
        var label = panel.querySelector('[data-role="bucket"]');
        var timer = null;
        slider.max = Math.max(0, buckets.length - 1);
        function render(index) {{
            layerGroup.clearLayers();
            var bucket = buckets[index] || '';
            var bucketRows = rows.filter(function(row) {{ return row.bucket === bucket; }});
            bucketRows.forEach(function(row) {{
                L.polyline(row.points, {{
                    color: row.color,
                    weight: 6,
                    opacity: 0.86,
                    lineCap: 'round'
                }}).bindTooltip(
                    row.segmentKey + '<br>' + row.level + ' ' + row.speed + ' km/h<br>样本 ' + row.samples + ' / 车辆 ' + row.vehicles
                ).addTo(layerGroup);
            }});
            label.textContent = bucket ? ('时间片 ' + bucket + '，路段 ' + bucketRows.length + ' 条') : '无可播放路段';
            slider.value = index;
        }}
        slider.addEventListener('input', function() {{
            render(Number(slider.value || 0));
        }});
        playButton.addEventListener('click', function() {{
            if (timer) {{
                window.clearInterval(timer);
                timer = null;
                playButton.textContent = '播放';
                return;
            }}
            playButton.textContent = '暂停';
            timer = window.setInterval(function() {{
                var next = (Number(slider.value || 0) + 1) % Math.max(1, buckets.length);
                render(next);
            }}, 900);
        }});
        render(0);
    }})();
    """
    m.get_root().html.add_child(folium.Element(player_html))
    m.get_root().script.add_child(folium.Element(player_js))


def plot_congestion_roads(vehicle_ids, start_time=None, end_time=None, bucket_minutes=15, save_path=None):
    grouped, meta = build_congestion_segments(vehicle_ids, start_time, end_time, bucket_minutes=bucket_minutes)
    if grouped is None or len(grouped) == 0:
        return None, meta

    map_df = pd.DataFrame(
        {
            "lati": pd.concat([grouped["lat1"], grouped["lat2"]], ignore_index=True),
            "long": pd.concat([grouped["lng1"], grouped["lng2"]], ignore_index=True),
        }
    )
    m = build_map(map_df)
    _inject_congestion_player(m, grouped)
    network_label = meta.get("network", {}).get("path") or "未加载"
    m.get_root().html.add_child(
        folium.Element(
            _congestion_legend_html(
                "拥堵道路图例",
                "颜色来自每个时间片内匹配路段的平均速度。拖动左侧滑块可查看不同时间片。",
                network_label=network_label,
            )
        )
    )
    add_map_layers(m)

    start_str = format_time_tag(start_time)
    end_str = format_time_tag(end_time)
    vehicle_tag = "-".join([str(item).strip() for item in (vehicle_ids or [])][:3]) or "all"
    output_path = save_path or os.path.join(
        CONFIG["OUTPUT_MAP_DIR"],
        f"congestion_roads_{vehicle_tag}_{start_str}_{end_str}_{int(bucket_minutes)}m.html",
    )
    ensure_parent_dir(output_path)
    m.save(output_path)
    meta["output_path"] = output_path
    return output_path, meta


def _historical_speed_lookup(grouped):
    if grouped is None or len(grouped) == 0:
        return {}, CONFIG["DEFAULT_ETA_SPEED_KMH"]
    speeds = pd.to_numeric(grouped["speed_kmh"], errors="coerce").dropna()
    global_speed = float(speeds[(speeds > 0) & (speeds <= CONFIG["ANIMATION_SPEED_CAP_KMH"])].mean()) if len(speeds) else np.nan
    if pd.isna(global_speed) or global_speed <= 0:
        global_speed = CONFIG["DEFAULT_ETA_SPEED_KMH"]
    lookup = {}
    for _, row in grouped.iterrows():
        speed = float(row["speed_kmh"]) if pd.notna(row["speed_kmh"]) else global_speed
        lookup[row["segment_key"]] = max(5.0, min(CONFIG["ANIMATION_SPEED_CAP_KMH"], speed))
    return lookup, global_speed


def estimate_eta(origin_lat, origin_lng, dest_lat, dest_lng, vehicle_ids=None, start_time=None, end_time=None, bucket_minutes=15):
    graph, network_meta = load_road_network()
    if graph is None:
        return {"success": False, "error": network_meta.get("error", "路网未加载。"), "network": network_meta}

    try:
        origin_node = nearest_road_node(graph, origin_lng, origin_lat)
        dest_node = nearest_road_node(graph, dest_lng, dest_lat)
    except Exception as exc:
        log.exception("ETA 起终点最近节点匹配失败")
        return {"success": False, "error": f"起终点最近道路节点匹配失败: {exc}", "network": network_meta}

    undirected_graph = None
    if hasattr(graph, "is_directed") and graph.is_directed():
        try:
            undirected_graph = graph.to_undirected(as_view=True)
        except TypeError:
            undirected_graph = graph.to_undirected()
    try:
        path_nodes, path_mode = _shortest_path_nodes(graph, origin_node, dest_node, undirected_graph=undirected_graph)
    except Exception as exc:
        log.exception("ETA 最短路径失败")
        return {"success": False, "error": f"起终点之间没有可用路网路径: {exc}", "network": network_meta}

    grouped, speed_meta = build_congestion_segments(vehicle_ids or [], start_time, end_time, bucket_minutes=bucket_minutes)
    speed_lookup, global_speed = _historical_speed_lookup(grouped)
    route_points = []
    route_segments = []
    eta_hours = 0.0
    total_distance_km = 0.0
    for idx, node in enumerate(path_nodes):
        point = _node_lat_lng(graph, node)
        if point:
            route_points.append(point)
        if idx == 0:
            continue
        source = path_nodes[idx - 1]
        target = node
        length_km = _edge_length_km(graph, source, target)
        if pd.isna(length_km) or length_km <= 0:
            prev_point = _node_lat_lng(graph, source)
            cur_point = _node_lat_lng(graph, target)
            length_km = haversine_distance(prev_point[0], prev_point[1], cur_point[0], cur_point[1]) if prev_point and cur_point else 0.0
        speed = speed_lookup.get(_segment_key(source, target), global_speed)
        total_distance_km += float(length_km)
        eta_hours += float(length_km) / max(5.0, float(speed))
        route_segments.append({"segment_key": _segment_key(source, target), "length_km": float(length_km), "speed_kmh": float(speed)})

    return {
        "success": len(route_points) >= 2,
        "origin_node": origin_node,
        "dest_node": dest_node,
        "path_mode": path_mode,
        "distance_km": total_distance_km,
        "eta_minutes": eta_hours * 60.0,
        "avg_speed_kmh": total_distance_km / eta_hours if eta_hours > 0 else global_speed,
        "route_points": route_points,
        "route_segments": route_segments,
        "historical_speed_kmh": global_speed,
        "method": "路网最短路径距离 + 当前查询时段匹配路段历史平均速度；缺失路段使用查询样本全局均速。",
        "network": network_meta,
        "speed_meta": speed_meta,
    }


def plot_eta_route(eta_result, save_path=None):
    if not eta_result or not eta_result.get("success"):
        return None
    route_points = eta_result.get("route_points", [])
    map_df = pd.DataFrame({"lati": [point[0] for point in route_points], "long": [point[1] for point in route_points]})
    m = build_map(map_df)
    folium.PolyLine(
        route_points,
        color="#2563eb",
        weight=6,
        opacity=0.88,
        tooltip=f"ETA {eta_result.get('eta_minutes', 0):.1f} 分钟 / {eta_result.get('distance_km', 0):.2f} km",
    ).add_to(m)
    if route_points:
        folium.Marker(route_points[0], tooltip="ETA 起点", icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker(route_points[-1], tooltip="ETA 终点", icon=folium.Icon(color="red", icon="flag")).add_to(m)
    network_label = eta_result.get("network", {}).get("path") or "未加载"
    m.get_root().html.add_child(
        folium.Element(
            _congestion_legend_html(
                "ETA 路线",
                f"预计 {eta_result.get('eta_minutes', 0):.1f} 分钟，距离 {eta_result.get('distance_km', 0):.2f} km，均速 {eta_result.get('avg_speed_kmh', 0):.1f} km/h。",
                network_label=network_label,
            )
        )
    )
    add_map_layers(m)
    output_path = save_path or os.path.join(CONFIG["OUTPUT_MAP_DIR"], "eta_route.html")
    ensure_parent_dir(output_path)
    m.save(output_path)
    return output_path


def load_minute_data(target_time):
    target_dt = safe_datetime(target_time)
    if target_dt is None:
        log.warning("分钟查询时间无效: %s", target_time)
        return None

    date_str = target_dt.strftime("%Y-%m-%d")
    hour_str = target_dt.strftime("%H")
    minute_str = target_dt.strftime("%M")
    minute_file = os.path.join(CONFIG["MINUTE_CACHE_DIR"], date_str, hour_str, f"{minute_str}.csv")

    cached = _read_minute_cache(minute_file)
    if cached is None:
        log.warning("分钟缓存文件不存在: %s", minute_file)
        return None

    df = cached.copy()
    df["time"] = target_dt
    log.info("分钟缓存载入完成: time=%s, rows=%s", target_dt, len(df))
    return df


@lru_cache(maxsize=4096)
def _cached_future_trajectory_points(vehicle_id, target_time_str):
    target_dt = safe_datetime(target_time_str)
    if target_dt is None:
        return tuple()

    preview_cache_dir = os.path.join(CONFIG["MINUTE_CACHE_DIR"], "_future_preview")
    preview_cache_path = os.path.join(preview_cache_dir, target_time_str.replace(":", "").replace("-", "").replace(" ", "_"), f"{vehicle_id}.json")
    if os.path.exists(preview_cache_path):
        try:
            with open(preview_cache_path, "r", encoding="utf-8") as f:
                cached_points = json.load(f)
            if isinstance(cached_points, list) and cached_points:
                return tuple((float(lat), float(lng)) for lat, lng in cached_points)
        except Exception:
            pass

    preview_df = load_vehicle_trajectory(vehicle_id, target_dt, None, log_result=False)
    if preview_df is None or len(preview_df) < 2:
        return tuple()

    preview_df = resample_trajectory(preview_df, 80)
    preview_points = _calibrated_segment_points(preview_df, max_points=80)
    if len(preview_points) < 2:
        preview_points = [[float(r["lati"]), float(r["long"])] for _, r in preview_df.iterrows()]
    preview_points = tuple((float(lat), float(lng)) for lat, lng in preview_points)

    try:
        ensure_parent_dir(preview_cache_path)
        with open(preview_cache_path, "w", encoding="utf-8") as f:
            json.dump([[lat, lng] for lat, lng in preview_points], f, ensure_ascii=False)
    except Exception:
        pass

    return preview_points


def _prefetch_future_trajectory_points(vehicle_ids, target_time_key, point_limit=80, max_workers=8):
    unique_vehicle_ids = []
    for vehicle_id in vehicle_ids or []:
        vehicle_id = str(vehicle_id).strip()
        if vehicle_id and vehicle_id not in unique_vehicle_ids:
            unique_vehicle_ids.append(vehicle_id)

    if not unique_vehicle_ids or not target_time_key:
        return {}

    results = {}
    worker_count = max(1, min(int(max_workers), len(unique_vehicle_ids)))

    def load_preview(vehicle_id):
        cached_points = _cached_future_trajectory_points(vehicle_id, target_time_key)
        if not cached_points:
            return vehicle_id, []
        if point_limit and len(cached_points) > point_limit:
            idx = np.linspace(0, len(cached_points) - 1, num=point_limit, dtype=int)
            sampled = [cached_points[i] for i in idx]
        else:
            sampled = list(cached_points)
        return vehicle_id, sampled

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(load_preview, vehicle_id) for vehicle_id in unique_vehicle_ids]
        for future in as_completed(futures):
            vehicle_id, points = future.result()
            if points:
                results[vehicle_id] = points
    return results


def plot_minute_vehicles(target_time, vehicle_id=None, vehicle_ids=None, save_path=None, display_limit=None):
    log.info("查询分钟车辆位置: target_time=%s, vehicle_id=%s, vehicle_ids=%s", target_time, vehicle_id, vehicle_ids)

    df = load_minute_data(target_time)
    if df is None or len(df) == 0:
        log.warning("未找到该分钟的车辆数据")
        return None

    target_dt = safe_datetime(target_time)
    selected_vehicle_ids = []
    if vehicle_ids:
        selected_vehicle_ids = [str(item).strip() for item in vehicle_ids if str(item).strip()]
    elif vehicle_id:
        selected_vehicle_ids = [str(vehicle_id)]

    if selected_vehicle_ids:
        if len(selected_vehicle_ids) == 1:
            df = df[df["vehicle_id"] == selected_vehicle_ids[0]].copy()
            if len(df) == 0:
                log.warning("车辆 %s 在该时间点没有数据", selected_vehicle_ids[0])
                return None
        else:
            df = df[df["vehicle_id"].isin(selected_vehicle_ids)].copy()
            if len(df) == 0:
                log.warning("所选车辆在该时间点没有数据: %s", selected_vehicle_ids)
                return None
    else:
        limit = display_limit or CONFIG["MAX_VEHICLE_DISPLAY"]
        if len(df) > limit:
            df = df.sample(n=limit, random_state=42).copy()
            log.info("车辆数量过多，抽样展示 %s / %s 辆", limit, len(df))

    m = build_map(df)

    occupied_df = df[df["status"] == 1]
    vacant_df = df[df["status"] == 0]

    occupied_cluster = MarkerCluster(name="载客车辆").add_to(m)
    vacant_cluster = MarkerCluster(name="空载车辆").add_to(m)

    target_time_key = pd.to_datetime(target_dt).strftime("%Y-%m-%d %H:%M:%S") if target_dt is not None else ""
    preview_point_limit = 16 if not selected_vehicle_ids else 80
    preview_vehicle_ids = [str(v) for v in df["vehicle_id"].astype(str).unique().tolist()]
    preview_points_map = _prefetch_future_trajectory_points(
        preview_vehicle_ids,
        target_time_key,
        point_limit=preview_point_limit,
        max_workers=8,
    )

    def build_future_preview(vehicle_key):
        if not target_time_key:
            return None
        cached_points = preview_points_map.get(str(vehicle_key))
        if not cached_points:
            return None
        return [[lat, lng] for lat, lng in cached_points]

    trajectory_show_handlers = []

    for _, row in occupied_df.iterrows():
        vehicle_key = str(row["vehicle_id"])
        preview_points = build_future_preview(vehicle_key)
        future_hint = ""
        if preview_points is not None:
            future_hint = f"<br>后续轨迹点: {len(preview_points)}"
            preview_line = folium.PolyLine(
                preview_points,
                color="#9ca3af",
                weight=2,
                opacity=0.0,
                dash_array="5,6",
                tooltip=f"车辆 {row['vehicle_id']} 后续轨迹",
            ).add_to(m)
        marker = folium.CircleMarker(
            location=[row["lati"], row["long"]],
            radius=4,
            color="#d73027",
            fill=True,
            fill_color="#d73027",
            fill_opacity=0.8,
            popup=f"<b>车辆 ID: {row['vehicle_id']}</b><br>时间: {pd.to_datetime(target_time)}<br>速度: {row['speed']} km/h<br>状态: 载客{future_hint}",
            tooltip=f"车辆 {row['vehicle_id']} - 载客",
        ).add_to(occupied_cluster)
        if preview_points is not None:
            trajectory_show_handlers.append({"marker": marker.get_name(), "line": preview_line.get_name()})

    for _, row in vacant_df.iterrows():
        vehicle_key = str(row["vehicle_id"])
        preview_points = build_future_preview(vehicle_key)
        future_hint = ""
        if preview_points is not None:
            future_hint = f"<br>后续轨迹点: {len(preview_points)}"
            preview_line = folium.PolyLine(
                preview_points,
                color="#9ca3af",
                weight=2,
                opacity=0.0,
                dash_array="5,6",
                tooltip=f"车辆 {row['vehicle_id']} 后续轨迹",
            ).add_to(m)
        marker = folium.CircleMarker(
            location=[row["lati"], row["long"]],
            radius=4,
            color="#2c7bb6",
            fill=True,
            fill_color="#2c7bb6",
            fill_opacity=0.8,
            popup=f"<b>车辆 ID: {row['vehicle_id']}</b><br>时间: {pd.to_datetime(target_time)}<br>速度: {row['speed']} km/h<br>状态: 空载{future_hint}",
            tooltip=f"车辆 {row['vehicle_id']} - 空载",
        ).add_to(vacant_cluster)
        if preview_points is not None:
            trajectory_show_handlers.append({"marker": marker.get_name(), "line": preview_line.get_name()})

    if trajectory_show_handlers:
        bindings_json = json.dumps(trajectory_show_handlers, ensure_ascii=False)
        handler_js = f"""
        window.__minuteTrajectoryBindings = {bindings_json};
        window.__activeMinuteTrajectoryLine = null;
        window.__hideActiveMinuteTrajectory = function() {{
            if (window.__activeMinuteTrajectoryLine && window.__activeMinuteTrajectoryLine.setStyle) {{
                window.__activeMinuteTrajectoryLine.setStyle({{opacity: 0.0, weight: 2}});
            }}
            window.__activeMinuteTrajectoryLine = null;
        }};
        function bindMinuteTrajectoryHandlers() {{
            (window.__minuteTrajectoryBindings || []).forEach(function(binding) {{
                var marker = window[binding.marker];
                var line = window[binding.line];
                if (!marker || !line) {{
                    try {{
                        marker = eval(binding.marker);
                        line = eval(binding.line);
                    }} catch (err) {{
                        marker = marker || null;
                        line = line || null;
                    }}
                }}
                if (!marker || !line || !marker.on) return;
                if (marker.__minuteTrajectoryBound) return;
                marker.__minuteTrajectoryBound = true;
                marker.on('click', function() {{
                    window.__hideActiveMinuteTrajectory();
                    if (line.setStyle) {{
                        line.setStyle({{opacity: 0.9, weight: 4}});
                    }}
                    if (line.bringToFront) line.bringToFront();
                    window.__activeMinuteTrajectoryLine = line;
                }});
            }});
        }}
        bindMinuteTrajectoryHandlers();
        window.addEventListener('load', bindMinuteTrajectoryHandlers);
        setTimeout(bindMinuteTrajectoryHandlers, 300);
        """
        m.get_root().script.add_child(folium.Element(handler_js))

    add_map_layers(m)

    output_time = pd.to_datetime(target_time).strftime("%Y%m%d_%H%M")
    output_path = save_path or os.path.join(
        CONFIG["OUTPUT_MAP_DIR"],
        f"minute_positions_{output_time}.html"
        if not selected_vehicle_ids
        else f"minute_positions_{output_time}_vehicle_{'-'.join(selected_vehicle_ids[:3])}.html",
    )
    ensure_parent_dir(output_path)
    m.save(output_path)
    log.info("分钟车辆位置地图已保存至: %s", output_path)
    return output_path


def load_od_data(start_time, end_time, vehicle_id=None, vehicle_ids=None):
    cached = _read_od_cache(CONFIG["OD_TABLE_PATH"], _file_version_key(CONFIG["OD_TABLE_PATH"]))
    if cached is None:
        log.error("OD表文件不存在: %s", CONFIG["OD_TABLE_PATH"])
        return None

    df = cached.copy()
    start_dt = safe_datetime(start_time)
    end_dt = safe_datetime(end_time)

    if start_dt is not None:
        df = df[df["O_time"] >= start_dt]
    if end_dt is not None:
        df = df[df["D_time"] <= end_dt]
    selected_vehicle_ids = []
    if vehicle_ids:
        selected_vehicle_ids = [str(item).strip() for item in vehicle_ids if str(item).strip()]
    elif vehicle_id:
        selected_vehicle_ids = [str(vehicle_id)]

    if selected_vehicle_ids:
        if len(selected_vehicle_ids) == 1:
            df = df[df["O_TAXI_ID"] == selected_vehicle_ids[0]]
        else:
            df = df[df["O_TAXI_ID"].isin(selected_vehicle_ids)]

    df = df.sort_values("O_time").reset_index(drop=True)
    log.info(
        "OD数据载入完成: start=%s, end=%s, vehicle_id=%s, vehicle_ids=%s, rows=%s",
        start_dt,
        end_dt,
        vehicle_id,
        selected_vehicle_ids,
        len(df),
    )
    return df


def plot_od_points(start_time, end_time, vehicle_id=None, vehicle_ids=None, save_path=None, use_cluster=True):
    log.info(
        "查询OD点: start_time=%s, end_time=%s, vehicle_id=%s, vehicle_ids=%s",
        start_time,
        end_time,
        vehicle_id,
        vehicle_ids,
    )

    df = load_od_data(start_time, end_time, vehicle_id, vehicle_ids=vehicle_ids)
    if df is None or len(df) == 0:
        log.warning("未找到符合条件的OD数据")
        return None

    if len(df) > CONFIG["MAX_OD_POINTS"]:
        df = df.sample(n=CONFIG["MAX_OD_POINTS"], random_state=42).copy()
        log.info("OD订单过多，抽样显示 %s 条", len(df))

    m = build_map(df)

    if use_cluster and len(df) > 100:
        pickup_layer = MarkerCluster(name="上车点聚类", show=True).add_to(m)
        dropoff_layer = MarkerCluster(name="下车点聚类", show=True).add_to(m)
    else:
        pickup_layer = folium.FeatureGroup(name="上车点").add_to(m)
        dropoff_layer = folium.FeatureGroup(name="下车点").add_to(m)

    sample_lines = df.sample(n=min(80, len(df)), random_state=42) if len(df) > 80 else df

    for _, row in df.iterrows():
        folium.Marker(
            location=[row["O_lat"], row["O_lng"]],
            icon=folium.Icon(color="green", icon="plus"),
            popup=(
                f"<b>上车点</b><br>车辆ID: {row['O_TAXI_ID']}<br>时间: {row['O_time']}<br>"
                f"订单时长: {row['OD_Time_s']:.0f}秒<br>订单距离: {row['OD_Dist_km']:.2f}公里"
            ),
            tooltip="上车点",
        ).add_to(pickup_layer)

        folium.Marker(
            location=[row["D_lat"], row["D_lng"]],
            icon=folium.Icon(color="red", icon="minus"),
            popup=(
                f"<b>下车点</b><br>车辆ID: {row['O_TAXI_ID']}<br>时间: {row['D_time']}<br>"
                f"订单时长: {row['OD_Time_s']:.0f}秒<br>订单距离: {row['OD_Dist_km']:.2f}公里"
            ),
            tooltip="下车点",
        ).add_to(dropoff_layer)

    for _, row in sample_lines.iterrows():
        folium.PolyLine(
            [[row["O_lat"], row["O_lng"]], [row["D_lat"], row["D_lng"]]],
            color="orange",
            weight=2,
            opacity=0.6,
            dash_array="5,5",
        ).add_to(m)

    add_map_layers(m)

    start_str = format_time_tag(start_time)
    end_str = format_time_tag(end_time)
    output_path = save_path or os.path.join(
        CONFIG["OUTPUT_MAP_DIR"],
        f"od_points_{start_str}_{end_str}.html"
        if not vehicle_id and not vehicle_ids
        else f"od_points_{start_str}_{end_str}_vehicle_{vehicle_id or '-'.join([str(v) for v in (vehicle_ids or [])][:3])}.html",
    )
    ensure_parent_dir(output_path)
    m.save(output_path)
    log.info("OD点地图已保存至: %s", output_path)
    return output_path


def _build_animation_html(vehicle_id, df, time_scale):
    playback_scale = _clamp_animation_time_scale(time_scale)
    features = []
    for idx, row in df.iterrows():
        speed_value = _animation_display_speed(row)
        time_ms = int(pd.Timestamp(row["time"]).timestamp() * 1000)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(row["long"]), float(row["lati"])]},
                "properties": {
                    "time": row["time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "timeMs": time_ms,
                    "status": int(row["status"]),
                    "speed": speed_value,
                    "index": int(idx),
                },
            }
        )

    payload = json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False)
    first = df.iloc[0]
    bounds = {
        "minLat": float(df["lati"].min()),
        "maxLat": float(df["lati"].max()),
        "minLng": float(df["long"].min()),
        "maxLng": float(df["long"].max()),
    }

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>车辆 {vehicle_id} 动画轨迹</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        :root {{
            --panel-bg: #ffffff;
            --panel-border: rgba(15, 23, 42, 0.12);
            --panel-shadow: 0 18px 50px rgba(15, 23, 42, 0.12);
            --text: #0f172a;
            --muted: #475569;
            --accent: #ef4444;
            --accent-soft: rgba(239, 68, 68, 0.12);
        }}
        html, body {{ margin: 0; padding: 0; min-height: 100%; background: #f5f5f5; color: var(--text); overflow-x: hidden; }}
        body {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        #controls {{
            position: relative;
            z-index: 2;
            margin: 12px 12px 0;
            padding: 12px 14px 10px;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            box-shadow: var(--panel-shadow);
            backdrop-filter: blur(10px);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 14px;
            min-width: 320px;
        }}
        #controls .row {{ margin-top: 8px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
        #controls select {{
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            padding: 7px 10px;
            background: #fff;
            color: var(--text);
            min-width: 150px;
        }}
        #map {{ width: calc(100% - 24px); height: 74vh; min-height: 540px; margin: 0 12px; border-radius: 16px; overflow: hidden; box-shadow: var(--panel-shadow); border: 1px solid var(--panel-border); }}
        #info {{
            position: relative;
            z-index: 2;
            margin: 0 12px 12px;
            padding: 10px 14px;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            box-shadow: var(--panel-shadow);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            align-items: center;
        }}
        #speedPanel {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid rgba(148, 163, 184, 0.2);
        }}
        .btn {{
            border: 0;
            border-radius: 999px;
            padding: 8px 14px;
            margin-right: 8px;
            background: #0f172a;
            color: white;
            cursor: pointer;
        }}
        .btn.secondary {{ background: #475569; }}
        input[type="range"] {{ width: 190px; vertical-align: middle; accent-color: var(--accent); }}
        .label {{ color: var(--muted); margin-right: 6px; }}
        .value {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
    </style>
</head>
<body>
    <div id="controls">
        <div style="margin-bottom: 8px;">
            <button class="btn" onclick="playAnimation()">播放</button>
            <button class="btn secondary" onclick="pauseAnimation()">暂停</button>
            <button class="btn secondary" onclick="resetAnimation()">重置</button>
        </div>
        <div>
            <span class="label">速度缩放</span>
            <input type="range" id="speedSlider" min="1" max="5" step="0.5" value="{playback_scale:.1f}" onchange="updateSpeed()">
            <span class="value" id="speedValue">{playback_scale:.1f}x</span>
        </div>
    </div>
    <div id="map"></div>
    <div id="info">
        <span><span class="label">时间</span><span class="value" id="currentTime">--</span></span>
        <span><span class="label">速度</span><span class="value" id="currentSpeed">-- km/h</span></span>
        <span><span class="label">状态</span><span class="value" id="currentStatus">--</span></span>
        <span><span class="label">进度</span><span class="value" id="progress">0%</span></span>
    </div>

    <script>
        var trajectoryData = {payload};
        var speedBaseMultiplier = 1.0;
        var speedMultiplier = {playback_scale:.1f} * speedBaseMultiplier;
        var maxPlaybackSpeedKmh = 60.0;
        var minMovementSpeedKmh = 1.0;
        var targetFrameRate = 60;
        var statusUpdateIntervalMs = 100;
        var minDelay = {CONFIG["ANIMATION_MIN_DELAY_MS"]};
        var animationFrameId = null;
        var isPlaying = false;
        var pausedVirtualElapsed = 0;
        var playStartedAt = 0;
        var segmentIndex = 0;
        var totalDurationMs = 0;
        var segments = [];
        var pathLatLngs = [];
        var basePath = [];
        var travelLineState = {{fixedIndex: 0, hasMovingPoint: false}};
        var lastFrameTimestamp = 0;
        var lastStatusUpdateAt = 0;
        var frameStats = {{
            targetFps: targetFrameRate,
            fps: 0,
            belowTarget: false,
            sampleStart: 0,
            sampleFrames: 0
        }};
        window.__taxigpsAnimationStats = frameStats;

        var map = L.map('map', {{
            preferCanvas: true,
            zoomSnap: 0.5,
            zoomDelta: 0.5,
            fadeAnimation: false,
            markerZoomAnimation: false,
            zoomAnimation: false
        }}).setView([{CONFIG["MAP_CENTER"][0]}, {CONFIG["MAP_CENTER"][1]}], {CONFIG["MAP_ZOOM"]});
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors',
            maxZoom: 18,
            detectRetina: true,
            tileSize: 512,
            zoomOffset: -1
        }}).addTo(map);

        var previewLine = L.polyline([], {{color: '#9ca3af', weight: 2, opacity: 0.45, dashArray: '6,6'}}).addTo(map);
        var traveledLine = L.polyline([], {{color: '#16a34a', weight: 5, opacity: 0.9}}).addTo(map);

        var markerIcon = L.divIcon({{
            className: 'car-marker',
            html: '<div style="width: 18px; height: 18px; background: #ef4444; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 8px rgba(0,0,0,0.35);"></div>',
            iconSize: [18, 18],
            iconAnchor: [9, 9]
        }});
        var carMarker = L.marker([{first['lati']}, {first['long']}], {{icon: markerIcon}}).addTo(map);

        var bounds = L.latLngBounds([[{bounds["minLat"]}, {bounds["minLng"]}], [{bounds["maxLat"]}, {bounds["maxLng"]}]]);
        map.fitBounds(bounds.pad(0.15));

        function clamp(value, minValue, maxValue) {{
            return Math.max(minValue, Math.min(maxValue, value));
        }}

        function lerp(a, b, t) {{
            return a + (b - a) * t;
        }}

        function recordFrameSample(frameTimestamp) {{
            if (!isFinite(frameTimestamp) || frameTimestamp <= 0) return;
            if (!frameStats.sampleStart) {{
                frameStats.sampleStart = frameTimestamp;
                frameStats.sampleFrames = 0;
            }}
            frameStats.sampleFrames += 1;
            var elapsed = frameTimestamp - frameStats.sampleStart;
            if (elapsed >= 1000) {{
                frameStats.fps = frameStats.sampleFrames * 1000 / elapsed;
                frameStats.belowTarget = frameStats.fps < targetFrameRate;
                frameStats.sampleStart = frameTimestamp;
                frameStats.sampleFrames = 0;
            }}
        }}

        function segmentDurationMs(rawDuration, segmentDistance, startSpeed, endSpeed) {{
            if (segmentDistance <= 0) {{
                return rawDuration;
            }}
            var startObservedSpeed = Math.max(0, Number(startSpeed || 0));
            var endObservedSpeed = Math.max(0, Number(endSpeed || 0));
            var averageObservedSpeed = Math.max(minMovementSpeedKmh, (startObservedSpeed + endObservedSpeed) / 2);
            var observedDuration = (segmentDistance / averageObservedSpeed) * 3600000;
            var minimumDuration = (segmentDistance / maxPlaybackSpeedKmh) * 3600000;
            return Math.max(rawDuration, observedDuration, minimumDuration);
        }}

        function speedAwareProgress(t, startSpeed, endSpeed) {{
            var x = clamp(t, 0, 1);
            var startObservedSpeed = Math.max(0, Number(startSpeed || 0));
            var endObservedSpeed = Math.max(0, Number(endSpeed || 0));
            var averageObservedSpeed = (startObservedSpeed + endObservedSpeed) / 2;
            if (averageObservedSpeed <= 0.01) {{
                return x;
            }}
            var distanceRatio = (startObservedSpeed * x + 0.5 * (endObservedSpeed - startObservedSpeed) * x * x) / averageObservedSpeed;
            return clamp(distanceRatio, 0, 1);
        }}

        function easeLinear(t) {{
            return t;
        }}

        function easeInQuad(t) {{
            return t * t;
        }}

        function easeOutQuad(t) {{
            return 1 - (1 - t) * (1 - t);
        }}

        function easeInOutCubic(t) {{
            return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        }}

        function getEasingDerivative(t) {{
            var x = clamp(t, 0, 1);
            return x < 0.5 ? 12 * x * x : 12 * (1 - x) * (1 - x);
        }}

        function getEasingValue(t) {{
            return easeInOutCubic(clamp(t, 0, 1));
        }}

        function buildSegments() {{
            var features = trajectoryData.features || [];
            if (features.length < 2) return;

            pathLatLngs = [];
            totalDurationMs = 0;
            segments = [];

            for (var i = 0; i < features.length; i++) {{
                var feature = features[i];
                var lat = feature.geometry.coordinates[1];
                var lng = feature.geometry.coordinates[0];
                var speed = Number(feature.properties.speed || 0);
                var timeMs = Number(feature.properties.timeMs || 0);
                pathLatLngs.push([lat, lng]);

                if (i === 0) continue;
                var prev = features[i - 1];
                var prevLat = prev.geometry.coordinates[1];
                var prevLng = prev.geometry.coordinates[0];
                var prevTimeMs = Number(prev.properties.timeMs || 0);
                var rawDuration = Math.max(1, timeMs - prevTimeMs);
                var segmentDistance = haversine(prevLat, prevLng, lat, lng);
                var startDisplaySpeed = Number(prev.properties.speed || 0);
                var endDisplaySpeed = speed;
                var duration = segmentDurationMs(rawDuration, segmentDistance, startDisplaySpeed, endDisplaySpeed);
                var segmentSpeed = 0;
                if (duration > 0) {{
                    segmentSpeed = segmentDistance > 0 ? Math.min({CONFIG["ANIMATION_SPEED_CAP_KMH"]}, segmentDistance / (duration / 3600000)) : speed;
                }}
                segments.push({{
                    startIndex: i - 1,
                    endIndex: i,
                    startLat: prevLat,
                    startLng: prevLng,
                    endLat: lat,
                    endLng: lng,
                    startSpeed: startDisplaySpeed,
                    endSpeed: endDisplaySpeed,
                    durationMs: duration,
                    startTimeMs: totalDurationMs,
                    endTimeMs: totalDurationMs + duration,
                    speedKmh: segmentSpeed,
                    startDisplaySpeed: startDisplaySpeed,
                    endDisplaySpeed: endDisplaySpeed,
                }});
                totalDurationMs += duration;
            }}
            previewLine.setLatLngs(pathLatLngs);
            basePath = pathLatLngs.slice(0, 1);
        }}

        function resetTravelLineState(firstLatLng) {{
            travelLineState.fixedIndex = 0;
            travelLineState.hasMovingPoint = false;
            traveledLine.setLatLngs([firstLatLng]);
        }}

        function updateTravelLineEndpoint(targetIndex, currentLatLng) {{
            if (pathLatLngs.length === 0) return;
            var latLngs = traveledLine.getLatLngs();
            if (latLngs.length === 0) {{
                traveledLine.addLatLng(pathLatLngs[0]);
                latLngs = traveledLine.getLatLngs();
            }}
            while (travelLineState.fixedIndex < targetIndex && travelLineState.fixedIndex + 1 < pathLatLngs.length) {{
                var nextFixed = pathLatLngs[travelLineState.fixedIndex + 1];
                if (travelLineState.hasMovingPoint && latLngs.length > 0) {{
                    latLngs[latLngs.length - 1] = L.latLng(nextFixed[0], nextFixed[1]);
                    travelLineState.hasMovingPoint = false;
                    traveledLine.redraw();
                }} else {{
                    traveledLine.addLatLng(nextFixed);
                    latLngs = traveledLine.getLatLngs();
                }}
                travelLineState.fixedIndex += 1;
            }}
            if (travelLineState.hasMovingPoint && latLngs.length > 0) {{
                latLngs[latLngs.length - 1] = L.latLng(currentLatLng[0], currentLatLng[1]);
                traveledLine.redraw();
            }} else {{
                traveledLine.addLatLng(currentLatLng);
                travelLineState.hasMovingPoint = true;
            }}
        }}

        function haversine(lat1, lng1, lat2, lng2) {{
            var R = 6371.0;
            function toRad(v) {{ return v * Math.PI / 180; }}
            var dLat = toRad(lat2 - lat1);
            var dLng = toRad(lng2 - lng1);
            var a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
                Math.sin(dLng / 2) * Math.sin(dLng / 2);
            var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
            return R * c;
        }}

        function updatePlaybackInfo(currentElapsed, currentSpeed) {{
            var progress = totalDurationMs > 0 ? clamp(currentElapsed / totalDurationMs, 0, 1) : 0;
            document.getElementById('currentSpeed').textContent = formatSpeedValue(currentSpeed) + ' km/h';
            document.getElementById('progress').textContent = Math.round(progress * 100) + '%';
        }}

        function formatSpeedValue(speed) {{
            if (!isFinite(speed) || speed < 0) return '0.00';
            return speed < 10 ? speed.toFixed(2) : speed.toFixed(1);
        }}

        function formatTimeFromMs(timeMs) {{
            var dt = new Date(timeMs);
            var year = dt.getFullYear();
            var month = String(dt.getMonth() + 1).padStart(2, '0');
            var day = String(dt.getDate()).padStart(2, '0');
            var hour = String(dt.getHours()).padStart(2, '0');
            var minute = String(dt.getMinutes()).padStart(2, '0');
            var second = String(dt.getSeconds()).padStart(2, '0');
            return year + '-' + month + '-' + day + ' ' + hour + ':' + minute + ':' + second;
        }}

        function updateInfoByPosition(lat, lng, currentElapsed, currentSpeed, status) {{
            document.getElementById('currentTime').textContent = formatTimeFromMs(trajectoryData.features[0].properties.timeMs + currentElapsed);
            document.getElementById('currentStatus').textContent = status === 1 ? '载客' : '空载';
            updatePlaybackInfo(currentElapsed, currentSpeed);
        }}

        function maybeUpdateInfoByPosition(lat, lng, currentElapsed, currentSpeed, status, frameTimestamp, forceUpdate) {{
            var now = frameTimestamp || performance.now();
            if (!forceUpdate && now - lastStatusUpdateAt < statusUpdateIntervalMs) {{
                return;
            }}
            lastStatusUpdateAt = now;
            updateInfoByPosition(lat, lng, currentElapsed, currentSpeed, status);
        }}

        function renderFrame(currentElapsed, frameTimestamp, forceStatusUpdate) {{
            if (segments.length === 0) return;
            var total = totalDurationMs;
            var elapsed = clamp(currentElapsed, 0, total);

            while (segmentIndex < segments.length - 1 && elapsed > segments[segmentIndex].endTimeMs) {{
                segmentIndex += 1;
            }}
            while (segmentIndex > 0 && elapsed < segments[segmentIndex].startTimeMs) {{
                segmentIndex -= 1;
            }}

            var segment = segments[segmentIndex];
            var localElapsed = elapsed - segment.startTimeMs;
            var rawT = segment.durationMs > 0 ? clamp(localElapsed / segment.durationMs, 0, 1) : 1;
            var easedT = speedAwareProgress(rawT, segment.startDisplaySpeed, segment.endDisplaySpeed);
            var lat = lerp(segment.startLat, segment.endLat, easedT);
            var lng = lerp(segment.startLng, segment.endLng, easedT);
            var currentSpeed = Math.max(0, lerp(segment.startDisplaySpeed, segment.endDisplaySpeed, rawT));
            var status = trajectoryData.features[segment.startIndex].properties.status;

            updateTravelLineEndpoint(segment.startIndex, [lat, lng]);
            carMarker.setLatLng([lat, lng]);
            maybeUpdateInfoByPosition(lat, lng, elapsed, currentSpeed, status, frameTimestamp, forceStatusUpdate);
        }}

        function getVirtualElapsed(now) {{
            if (!isPlaying) {{
                return pausedVirtualElapsed;
            }}
            return (now - playStartedAt) * speedMultiplier;
        }}

        function tick(currentTime) {{
            if (!isPlaying) {{
                return;
            }}

            lastFrameTimestamp = currentTime;
            var virtualElapsed = getVirtualElapsed(currentTime);
            if (!isFinite(virtualElapsed)) {{
                virtualElapsed = pausedVirtualElapsed;
            }}

            if (virtualElapsed >= totalDurationMs) {{
                pausedVirtualElapsed = totalDurationMs;
                recordFrameSample(currentTime);
                renderFrame(totalDurationMs, currentTime, true);
                isPlaying = false;
                if (animationFrameId) {{
                    cancelAnimationFrame(animationFrameId);
                    animationFrameId = null;
                }}
                return;
            }}

            pausedVirtualElapsed = virtualElapsed;
            recordFrameSample(currentTime);
            renderFrame(virtualElapsed, currentTime, false);
            animationFrameId = requestAnimationFrame(tick);
        }}

        function handleVisibilityChange() {{
            if (document.hidden) {{
                lastFrameTimestamp = 0;
            }} else if (isPlaying) {{
                lastFrameTimestamp = performance.now();
            }}
        }}

        document.addEventListener('visibilitychange', handleVisibilityChange);

        function playAnimation() {{
            if (isPlaying) return;

            isPlaying = true;
            if (pausedVirtualElapsed >= totalDurationMs) {{
                pausedVirtualElapsed = 0;
                segmentIndex = 0;
            }}

            lastFrameTimestamp = performance.now();
            playStartedAt = lastFrameTimestamp - pausedVirtualElapsed / Math.max(speedMultiplier, 0.001);
            animationFrameId = requestAnimationFrame(tick);
        }}

        function pauseAnimation() {{
            isPlaying = false;
            if (animationFrameId) {{
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }}
        }}

        function resetAnimation() {{
            pauseAnimation();
            pausedVirtualElapsed = 0;
            segmentIndex = 0;
            lastFrameTimestamp = 0;
            playStartedAt = 0;
            if (trajectoryData.features.length > 0 && pathLatLngs.length > 0) {{
                var first = trajectoryData.features[0];
                var firstLatLng = pathLatLngs[0];
                carMarker.setLatLng(firstLatLng);
                resetTravelLineState(firstLatLng);
                updateInfoByPosition(firstLatLng[0], firstLatLng[1], 0, Number(first.properties.speed || 0), Number(first.properties.status || 0));
                document.getElementById('progress').textContent = '0%';
            }}
        }}

        function updateSpeed() {{
            if (isPlaying) {{
                var now = performance.now();
                pausedVirtualElapsed = clamp(getVirtualElapsed(now), 0, totalDurationMs);
                lastFrameTimestamp = now;
            }}
            var selectedSpeed = parseFloat(document.getElementById('speedSlider').value);
            speedMultiplier = selectedSpeed * speedBaseMultiplier;
            if (isPlaying) {{
                playStartedAt = lastFrameTimestamp - pausedVirtualElapsed / Math.max(speedMultiplier, 0.001);
            }}
            document.getElementById('speedValue').textContent = selectedSpeed.toFixed(1) + 'x';
            if (!isPlaying) {{
                renderFrame(pausedVirtualElapsed, performance.now(), true);
            }}
        }}

        buildSegments();
        resetAnimation();
    </script>
</body>
</html>
"""
    return html


def plot_animated_trajectory(vehicle_id, start_time=None, end_time=None, time_scale=1.0, save_path=None):
    log.info("生成动画轨迹: vehicle_id=%s, time_scale=%s", vehicle_id, time_scale)

    df = _load_animation_trajectory(vehicle_id, start_time, end_time)
    if df is None or len(df) < 2:
        log.warning("轨迹数据不足，无法生成动画")
        return None

    m = build_map(df)
    add_map_layers(m)

    # 先绘制载客/空载静态轨迹，动画层再逐步增长。
    occupied_segments = []
    vacant_segments = []
    for segment in _trajectory_segment_dfs(df):
        calibrated_points = _calibrated_segment_points(segment["df"], max_points=260)
        if len(calibrated_points) < 2:
            calibrated_points = [[row["lati"], row["long"]] for _, row in segment["df"].iterrows()]
        if segment["status"] == 1:
            occupied_segments.append(calibrated_points)
        else:
            vacant_segments.append(calibrated_points)

    for segment in occupied_segments:
        folium.PolyLine(segment, color="#d73027", weight=4, opacity=0.28, tooltip="载客轨迹").add_to(m)
    for segment in vacant_segments:
        folium.PolyLine(segment, color="#2c7bb6", weight=4, opacity=0.28, tooltip="空载轨迹").add_to(m)

    start_row = df.iloc[0]
    end_row = df.iloc[-1]
    folium.Marker(
        location=[start_row["lati"], start_row["long"]],
        icon=folium.Icon(color="green", icon="play"),
        popup=f"<b>起点</b><br>时间: {start_row['time']}<br>状态: {'载客' if start_row['status'] == 1 else '空载'}",
        tooltip="起点",
    ).add_to(m)
    folium.Marker(
        location=[end_row["lati"], end_row["long"]],
        icon=folium.Icon(color="red", icon="flag"),
        popup=f"<b>终点</b><br>时间: {end_row['time']}<br>状态: {'载客' if end_row['status'] == 1 else '空载'}",
        tooltip="终点",
    ).add_to(m)

    output_path = save_path or os.path.join(
        CONFIG["OUTPUT_MAP_DIR"],
        f"vehicle_{vehicle_id}_animation_{format_time_tag(start_time)}_{format_time_tag(end_time)}.html",
    )
    ensure_parent_dir(output_path)

    html_content = _build_animation_html(vehicle_id, df, float(time_scale))
    save_html(html_content, output_path)
    log.info("动画轨迹地图已保存至: %s", output_path)
    return output_path


def plot_multi_vehicle_animated_trajectory(vehicle_ids, start_time=None, end_time=None, time_scale=1.0, save_path=None):
    cleaned_vehicle_ids = []
    for vehicle_id in vehicle_ids or []:
        vehicle_id = str(vehicle_id).strip()
        if vehicle_id and vehicle_id not in cleaned_vehicle_ids:
            cleaned_vehicle_ids.append(vehicle_id)
    cleaned_vehicle_ids = cleaned_vehicle_ids[:10]

    log.info(
        "生成多车辆动画轨迹: vehicle_ids=%s, time_scale=%s",
        cleaned_vehicle_ids,
        time_scale,
    )

    if not cleaned_vehicle_ids:
        log.warning("轨迹车辆为空，无法生成多车辆动画")
        return None

    loaded_entries = []
    loaded_frames = []
    graph, network_meta = load_road_network()
    for index, vehicle_id in enumerate(cleaned_vehicle_ids):
        df = _load_animation_trajectory(vehicle_id, start_time, end_time, graph=graph, network_meta=network_meta)
        if df is None or len(df) < 2:
            log.warning("车辆 %s 轨迹点不足，跳过动画", vehicle_id)
            continue

        color = _trajectory_color(index)
        loaded_entries.append((vehicle_id, df, color))
        loaded_frames.append(df)

    if not loaded_entries:
        log.warning("多车辆动画没有可用数据")
        return None

    combined_df = pd.concat(loaded_frames, ignore_index=True)
    bounds = {
        "minLat": float(combined_df["lati"].min()),
        "maxLat": float(combined_df["lati"].max()),
        "minLng": float(combined_df["long"].min()),
        "maxLng": float(combined_df["long"].max()),
    }
    global_start_ms = int(combined_df["time"].min().timestamp() * 1000)
    global_end_ms = int(combined_df["time"].max().timestamp() * 1000)

    html_content = _build_multi_animation_html(loaded_entries, float(time_scale), bounds, global_start_ms, global_end_ms)

    start_str = format_time_tag(start_time)
    end_str = format_time_tag(end_time)
    vehicle_tag = "-".join(cleaned_vehicle_ids[:3])
    if len(cleaned_vehicle_ids) > 3:
        vehicle_tag += "-more"
    output_path = save_path or os.path.join(
        CONFIG["OUTPUT_MAP_DIR"],
        f"vehicles_{vehicle_tag}_animation_{start_str}_{end_str}.html",
    )
    save_html(html_content, output_path)
    log.info("多车辆动画轨迹地图已保存至: %s", output_path)
    return output_path


def main():
    log.info("=" * 70)
    log.info("第四阶段：地图基础与轨迹查询")
    log.info("=" * 70)

    test_vehicle_id = "22223"
    test_time = "2023-10-12 09:30:00"
    test_start_time = "2023-10-12 08:00:00"
    test_end_time = "2023-10-12 10:00:00"
    test_short_time = "2023-10-12 08:00:00"

    log.info("【测试轨迹查询】")
    plot_vehicle_trajectory(test_vehicle_id, test_start_time, test_end_time)

    log.info("【测试分钟位置查询】")
    plot_minute_vehicles(test_time)

    log.info("【测试OD点标注 - 短时间范围】")
    plot_od_points(test_short_time, test_time)

    log.info("【测试动画轨迹】")
    plot_animated_trajectory(test_vehicle_id, test_start_time, test_end_time, time_scale=2.0)

    log.info("=" * 70)
    log.info("测试完成 - 所有地图已保存至 pages/maps/")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
