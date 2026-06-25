#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import sys
from functools import lru_cache
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

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
    "ANIMATION_MIN_DELAY_MS": 80,
    "ANIMATION_MAX_DELAY_MS": 2500,
    "ANIMATION_GAP_THRESHOLD_S": 1800,
    "ANIMATION_SPEED_CAP_KMH": 120,
    "ANIMATION_DRIFT_CAP_KM": 8.0,
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


def add_map_layers(m, include_boundary=True):
    if include_boundary:
        add_shenzhen_boundary(m)
    folium.LayerControl(collapsed=False).add_to(m)


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


def _vehicle_animation_features(df):
    features = []
    for idx, row in df.iterrows():
        speed_value = float(row["speed_kmh"]) if "speed_kmh" in row and pd.notna(row["speed_kmh"]) else float(row.get("speed", 0.0))
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
            <input type="range" id="speedSlider" min="0.5" max="5" step="0.1" value="{time_scale}" onchange="updateSpeed()">
            <span class="value" id="speedValue">{time_scale}x</span>
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
        var speedBaseMultiplier = 3.0;
        var speedMultiplier = {float(time_scale)} * speedBaseMultiplier;
        var minDelay = {CONFIG["ANIMATION_MIN_DELAY_MS"]};
        var animationFrameId = null;
        var isPlaying = false;
        var pausedVirtualElapsed = 0;
        var totalDurationMs = Math.max(1, {int(global_end_ms)} - {int(global_start_ms)});
        var globalStartMs = {int(global_start_ms)};
        var globalEndMs = {int(global_end_ms)};
        var frameGapCapMs = 50;
        var deltaSmoothingAlpha = 0.18;
        var lastFrameTimestamp = 0;
        var smoothedDeltaMs = 16.67;

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
            vehicle.marker = null;
            vehicle.currentSegmentIndex = 0;
            vehicle.currentTimeMs = vehicle.features.length > 0 ? vehicle.features[0].timeMs : 0;

            if (vehicle.features.length === 0) {{
                return;
            }}

            for (var i = 0; i < vehicle.features.length; i++) {{
                var feature = vehicle.features[i];
                vehicle.pathLatLngs.push([feature.lat, feature.lng]);
                if (i === 0) continue;
                var prev = vehicle.features[i - 1];
                var rawDuration = Math.max(1, feature.timeMs - prev.timeMs);
                var segmentDistance = haversine(prev.lat, prev.lng, feature.lat, feature.lng);
                var segmentSpeed = rawDuration > 0 && segmentDistance > 0
                    ? Math.min({CONFIG["ANIMATION_SPEED_CAP_KMH"]}, segmentDistance / (rawDuration / 3600000))
                    : feature.speed;
                vehicle.segments.push({{
                    startIndex: i - 1,
                    endIndex: i,
                    startLat: prev.lat,
                    startLng: prev.lng,
                    endLat: feature.lat,
                    endLng: feature.lng,
                    startTimeMs: prev.timeMs,
                    endTimeMs: feature.timeMs,
                    durationMs: rawDuration,
                    speedKmh: segmentSpeed,
                    status: prev.status
                }});
            }}

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

        function renderVehicle(vehicle, currentTimeMs) {{
            if (!vehicle.features || vehicle.features.length === 0) return;

            var first = vehicle.features[0];
            var last = vehicle.features[vehicle.features.length - 1];
            var renderLat = first.lat;
            var renderLng = first.lng;
            var traveledLatLngs = [vehicle.pathLatLngs[0]];
            var currentSpeed = first.speed || 0;
            var currentStatus = first.status === 1 ? '载客' : '空载';
            var progress = 0;

            if (currentTimeMs <= first.timeMs) {{
                vehicle.currentSegmentIndex = 0;
            }} else if (currentTimeMs >= last.timeMs) {{
                renderLat = last.lat;
                renderLng = last.lng;
                traveledLatLngs = vehicle.pathLatLngs.slice();
                currentSpeed = last.speed || 0;
                currentStatus = last.status === 1 ? '载客' : '空载';
                progress = 1;
                vehicle.currentSegmentIndex = vehicle.segments.length - 1;
            }} else {{
                while (vehicle.currentSegmentIndex < vehicle.segments.length - 1 && currentTimeMs > vehicle.segments[vehicle.currentSegmentIndex].endTimeMs) {{
                    vehicle.currentSegmentIndex += 1;
                }}
                while (vehicle.currentSegmentIndex > 0 && currentTimeMs < vehicle.segments[vehicle.currentSegmentIndex].startTimeMs) {{
                    vehicle.currentSegmentIndex -= 1;
                }}

                var segment = vehicle.segments[vehicle.currentSegmentIndex];
                var localElapsed = currentTimeMs - segment.startTimeMs;
                var rawT = segment.durationMs > 0 ? clamp(localElapsed / segment.durationMs, 0, 1) : 1;
                var easedT = easeInOutCubic(rawT);
                renderLat = lerp(segment.startLat, segment.endLat, easedT);
                renderLng = lerp(segment.startLng, segment.endLng, easedT);
                currentSpeed = Math.min({CONFIG["ANIMATION_SPEED_CAP_KMH"]}, Math.max(0, segment.speedKmh * getEasingDerivative(rawT)));
                currentStatus = segment.status === 1 ? '载客' : '空载';
                progress = clamp((currentTimeMs - first.timeMs) / Math.max(1, last.timeMs - first.timeMs), 0, 1);
                traveledLatLngs = vehicle.pathLatLngs.slice(0, segment.startIndex + 1).concat([[renderLat, renderLng]]);
            }}

            vehicle.travelLine.setLatLngs(traveledLatLngs);
            vehicle.marker.setLatLng([renderLat, renderLng]);
            updateVehicleCard(vehicle, currentTimeMs, currentSpeed, currentStatus, progress);
        }}

        function renderFleet(currentElapsed) {{
            var currentTimeMs = globalStartMs + clamp(currentElapsed, 0, totalDurationMs);
            document.getElementById('currentTime').textContent = formatTimeFromMs(currentTimeMs);
            document.getElementById('globalProgress').textContent = Math.round(clamp(currentElapsed / totalDurationMs, 0, 1) * 100) + '%';
            for (var i = 0; i < fleetData.length; i++) {{
                renderVehicle(fleetData[i], currentTimeMs);
            }}
        }}

        function tick(currentTime) {{
            if (!isPlaying) return;
            if (!lastFrameTimestamp) {{
                lastFrameTimestamp = currentTime;
            }}
            var deltaReal = currentTime - lastFrameTimestamp;
            lastFrameTimestamp = currentTime;
            if (!isFinite(deltaReal) || deltaReal < 0) {{
                deltaReal = 0;
            }}
            var cappedDelta = clamp(deltaReal, 0, frameGapCapMs);
            smoothedDeltaMs = smoothedDeltaMs > 0
                ? (smoothedDeltaMs * (1 - deltaSmoothingAlpha) + cappedDelta * deltaSmoothingAlpha)
                : cappedDelta;
            var virtualElapsed = pausedVirtualElapsed + smoothedDeltaMs * speedMultiplier;
            if (!isFinite(virtualElapsed)) {{
                virtualElapsed = pausedVirtualElapsed;
            }}
            if (virtualElapsed >= totalDurationMs) {{
                pausedVirtualElapsed = totalDurationMs;
                renderFleet(totalDurationMs);
                isPlaying = false;
                if (animationFrameId) {{
                    cancelAnimationFrame(animationFrameId);
                    animationFrameId = null;
                }}
                return;
            }}
            pausedVirtualElapsed = virtualElapsed;
            renderFleet(virtualElapsed);
            animationFrameId = requestAnimationFrame(tick);
        }}

        function playAnimation() {{
            if (isPlaying) return;
            isPlaying = true;
            if (pausedVirtualElapsed >= totalDurationMs) {{
                pausedVirtualElapsed = 0;
            }}
            lastFrameTimestamp = performance.now();
            smoothedDeltaMs = 16.67;
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
            smoothedDeltaMs = 16.67;
            renderFleet(0);
        }}

        function updateSpeed() {{
            if (isPlaying) {{
                lastFrameTimestamp = performance.now();
            }}
            var selectedSpeed = parseFloat(document.getElementById('speedSlider').value);
            speedMultiplier = selectedSpeed * speedBaseMultiplier;
            document.getElementById('speedValue').textContent = selectedSpeed.toFixed(1) + 'x';
            if (!isPlaying) {{
                renderFleet(pausedVirtualElapsed);
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
    features = []
    for idx, row in df.iterrows():
        speed_value = float(row["speed_kmh"]) if "speed_kmh" in row and pd.notna(row["speed_kmh"]) else float(row.get("speed", 0.0))
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
            <input type="range" id="speedSlider" min="0.5" max="5" step="0.1" value="{time_scale}" onchange="updateSpeed()">
            <span class="value" id="speedValue">{time_scale}x</span>
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
        var speedBaseMultiplier = 3.0;
        var speedMultiplier = {float(time_scale)} * speedBaseMultiplier;
        var minDelay = {CONFIG["ANIMATION_MIN_DELAY_MS"]};
        var animationFrameId = null;
        var isPlaying = false;
        var pausedVirtualElapsed = 0;
        var segmentIndex = 0;
        var totalDurationMs = 0;
        var segments = [];
        var pathLatLngs = [];
        var basePath = [];
        var frameGapCapMs = 50;
        var deltaSmoothingAlpha = 0.18;
        var lastFrameTimestamp = 0;
        var smoothedDeltaMs = 16.67;

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
                var duration = rawDuration;
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
                    startSpeed: Number(prev.properties.speed || 0),
                    endSpeed: speed,
                    durationMs: duration,
                    startTimeMs: totalDurationMs,
                    endTimeMs: totalDurationMs + duration,
                    speedKmh: segmentSpeed,
                }});
                totalDurationMs += duration;
            }}
            previewLine.setLatLngs(pathLatLngs);
            basePath = pathLatLngs.slice(0, 1);
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

        function renderFrame(currentElapsed) {{
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
            var easedT = getEasingValue(rawT);
            var lat = lerp(segment.startLat, segment.endLat, easedT);
            var lng = lerp(segment.startLng, segment.endLng, easedT);
            var motionFactor = getEasingDerivative(rawT);
            var currentSpeed = Math.min({CONFIG["ANIMATION_SPEED_CAP_KMH"]}, Math.max(0, segment.speedKmh * motionFactor));
            var status = trajectoryData.features[segment.startIndex].properties.status;

            traveledLine.setLatLngs(pathLatLngs.slice(0, segment.startIndex + 1).concat([[lat, lng]]));
            carMarker.setLatLng([lat, lng]);
            updateInfoByPosition(lat, lng, elapsed, currentSpeed, status);
        }}

        function tick(currentTime) {{
            if (!isPlaying) {{
                return;
            }}

            if (!lastFrameTimestamp) {{
                lastFrameTimestamp = currentTime;
            }}

            var deltaReal = currentTime - lastFrameTimestamp;
            lastFrameTimestamp = currentTime;

            if (!isFinite(deltaReal) || deltaReal < 0) {{
                deltaReal = 0;
            }}

            var cappedDelta = clamp(deltaReal, 0, frameGapCapMs);
            smoothedDeltaMs = smoothedDeltaMs > 0
                ? (smoothedDeltaMs * (1 - deltaSmoothingAlpha) + cappedDelta * deltaSmoothingAlpha)
                : cappedDelta;

            var virtualElapsed = pausedVirtualElapsed + smoothedDeltaMs * speedMultiplier;
            if (!isFinite(virtualElapsed)) {{
                virtualElapsed = pausedVirtualElapsed;
            }}

            if (virtualElapsed >= totalDurationMs) {{
                pausedVirtualElapsed = totalDurationMs;
                renderFrame(totalDurationMs);
                isPlaying = false;
                if (animationFrameId) {{
                    cancelAnimationFrame(animationFrameId);
                    animationFrameId = null;
                }}
                return;
            }}

            pausedVirtualElapsed = virtualElapsed;
            renderFrame(virtualElapsed);
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
            smoothedDeltaMs = 16.67;
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
            smoothedDeltaMs = 16.67;
            if (trajectoryData.features.length > 0 && pathLatLngs.length > 0) {{
                var first = trajectoryData.features[0];
                var firstLatLng = pathLatLngs[0];
                carMarker.setLatLng(firstLatLng);
                traveledLine.setLatLngs([firstLatLng]);
                updateInfoByPosition(firstLatLng[0], firstLatLng[1], first, 0, Number(first.properties.speed || 0));
                document.getElementById('progress').textContent = '0%';
            }}
        }}

        function updateSpeed() {{
            if (isPlaying) {{
                lastFrameTimestamp = performance.now();
            }}
            var selectedSpeed = parseFloat(document.getElementById('speedSlider').value);
            speedMultiplier = selectedSpeed * speedBaseMultiplier;
            document.getElementById('speedValue').textContent = selectedSpeed.toFixed(1) + 'x';
            if (!isPlaying) {{
                renderFrame(pausedVirtualElapsed);
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

    df = load_vehicle_trajectory(vehicle_id, start_time, end_time)
    if df is None or len(df) < 2:
        log.warning("轨迹数据不足，无法生成动画")
        return None

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
    for index, vehicle_id in enumerate(cleaned_vehicle_ids):
        df = load_vehicle_trajectory(vehicle_id, start_time, end_time)
        if df is None or len(df) < 2:
            log.warning("车辆 %s 轨迹点不足，跳过动画", vehicle_id)
            continue

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
