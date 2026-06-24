#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import sys
from functools import lru_cache
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt

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
def _read_od_cache(od_path):
    if not os.path.exists(od_path):
        return None
    return normalize_od_df(pd.read_csv(od_path))


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


def load_vehicle_trajectory(vehicle_id, start_time=None, end_time=None):
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

    segments = _trajectory_segment_rows(df)
    for segment in segments:
        color = "#d73027" if segment["status"] == 1 else "#2c7bb6"
        label = "载客轨迹" if segment["status"] == 1 else "空载轨迹"
        folium.PolyLine(
            segment["points"],
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


def plot_minute_vehicles(target_time, vehicle_id=None, save_path=None, display_limit=None):
    log.info("查询分钟车辆位置: target_time=%s, vehicle_id=%s", target_time, vehicle_id)

    df = load_minute_data(target_time)
    if df is None or len(df) == 0:
        log.warning("未找到该分钟的车辆数据")
        return None

    if vehicle_id:
        vehicle_id = str(vehicle_id)
        df = df[df["vehicle_id"] == vehicle_id].copy()
        if len(df) == 0:
            log.warning("车辆 %s 在该时间点没有数据", vehicle_id)
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

    for _, row in occupied_df.iterrows():
        folium.CircleMarker(
            location=[row["lati"], row["long"]],
            radius=4,
            color="#d73027",
            fill=True,
            fill_color="#d73027",
            fill_opacity=0.8,
            popup=f"<b>车辆 ID: {row['vehicle_id']}</b><br>时间: {pd.to_datetime(target_time)}<br>速度: {row['speed']} km/h<br>状态: 载客",
            tooltip=f"车辆 {row['vehicle_id']} - 载客",
        ).add_to(occupied_cluster)

    for _, row in vacant_df.iterrows():
        folium.CircleMarker(
            location=[row["lati"], row["long"]],
            radius=4,
            color="#2c7bb6",
            fill=True,
            fill_color="#2c7bb6",
            fill_opacity=0.8,
            popup=f"<b>车辆 ID: {row['vehicle_id']}</b><br>时间: {pd.to_datetime(target_time)}<br>速度: {row['speed']} km/h<br>状态: 空载",
            tooltip=f"车辆 {row['vehicle_id']} - 空载",
        ).add_to(vacant_cluster)

    vehicle_preview = None
    if vehicle_id:
        vehicle_preview = load_vehicle_trajectory(vehicle_id)
    elif "vehicle_id" in df.columns:
        preview_vehicle = str(df.iloc[0]["vehicle_id"])
        vehicle_preview = load_vehicle_trajectory(preview_vehicle)

    if vehicle_preview is not None and len(vehicle_preview) > 0:
        vehicle_preview = resample_trajectory(vehicle_preview, 400)
        preview_points = [[row["lati"], row["long"]] for _, row in vehicle_preview.iterrows()]
        folium.PolyLine(
            preview_points,
            color="#666666",
            weight=2,
            opacity=0.45,
            dash_array="5,5",
            tooltip="当日轨迹预览",
        ).add_to(m)

    add_map_layers(m)

    output_time = pd.to_datetime(target_time).strftime("%Y%m%d_%H%M")
    output_path = save_path or os.path.join(
        CONFIG["OUTPUT_MAP_DIR"],
        f"minute_positions_{output_time}.html" if not vehicle_id else f"minute_positions_{output_time}_vehicle_{vehicle_id}.html",
    )
    ensure_parent_dir(output_path)
    m.save(output_path)
    log.info("分钟车辆位置地图已保存至: %s", output_path)
    return output_path


def load_od_data(start_time, end_time, vehicle_id=None):
    cached = _read_od_cache(CONFIG["OD_TABLE_PATH"])
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
    if vehicle_id:
        df = df[df["O_TAXI_ID"] == str(vehicle_id)]

    df = df.sort_values("O_time").reset_index(drop=True)
    log.info(
        "OD数据载入完成: start=%s, end=%s, vehicle_id=%s, rows=%s",
        start_dt,
        end_dt,
        vehicle_id,
        len(df),
    )
    return df


def plot_od_points(start_time, end_time, vehicle_id=None, save_path=None, use_cluster=True):
    log.info("查询OD点: start_time=%s, end_time=%s, vehicle_id=%s", start_time, end_time, vehicle_id)

    df = load_od_data(start_time, end_time, vehicle_id)
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
        f"od_points_{start_str}_{end_str}.html" if not vehicle_id else f"od_points_{start_str}_{end_str}_vehicle_{vehicle_id}.html",
    )
    ensure_parent_dir(output_path)
    m.save(output_path)
    log.info("OD点地图已保存至: %s", output_path)
    return output_path


def _build_animation_html(vehicle_id, df, time_scale):
    features = []
    for idx, row in df.iterrows():
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(row["long"]), float(row["lati"])]},
                "properties": {
                    "time": row["time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "status": int(row["status"]),
                    "speed": float(row["speed"]),
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
        }}
        html, body {{ margin: 0; padding: 0; height: 100%; background: #eef2f7; color: var(--text); }}
        #controls {{
            position: absolute;
            top: 12px;
            left: 12px;
            z-index: 1000;
            padding: 12px 14px;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            box-shadow: var(--panel-shadow);
            backdrop-filter: blur(10px);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 14px;
            min-width: 300px;
        }}
        #map {{ width: 100%; height: calc(100vh - 78px); margin-top: 78px; }}
        #info {{
            position: absolute;
            left: 12px;
            right: 12px;
            bottom: 12px;
            z-index: 1000;
            padding: 10px 14px;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            box-shadow: var(--panel-shadow);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
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
        input[type="range"] {{ width: 190px; vertical-align: middle; }}
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
        <span><span class="label">当前位置</span><span class="value" id="currentPos">({first['lati']:.6f}, {first['long']:.6f})</span></span>
    </div>

    <script>
        var trajectoryData = {payload};
        var speedMultiplier = {float(time_scale)};
        var minDelay = {CONFIG["ANIMATION_MIN_DELAY_MS"]};
        var maxDelay = {CONFIG["ANIMATION_MAX_DELAY_MS"]};
        var animationTimer = null;
        var isPlaying = false;
        var currentIndex = 0;
        var traveledPoints = [];

        var map = L.map('map').setView([{CONFIG["MAP_CENTER"][0]}, {CONFIG["MAP_CENTER"][1]}], {CONFIG["MAP_ZOOM"]});
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);

        var previewLine = L.polyline([], {{color: '#9ca3af', weight: 2, opacity: 0.45, dashArray: '6,6'}}).addTo(map);
        var traveledLine = L.polyline([], {{color: '#16a34a', weight: 5, opacity: 0.88}}).addTo(map);

        var markerIcon = L.divIcon({{
            className: 'car-marker',
            html: '<div style="width: 18px; height: 18px; background: #ef4444; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 8px rgba(0,0,0,0.35);"></div>',
            iconSize: [18, 18],
            iconAnchor: [9, 9]
        }});
        var carMarker = L.marker([{first['lati']}, {first['long']}], {{icon: markerIcon}}).addTo(map);

        var bounds = L.latLngBounds([[{bounds["minLat"]}, {bounds["minLng"]}], [{bounds["maxLat"]}, {bounds["maxLng"]}]]);
        map.fitBounds(bounds.pad(0.15));

        var staticSegments = [];
        function addStaticSegments() {{
            var points = [];
            for (var i = 0; i < trajectoryData.features.length; i++) {{
                var f = trajectoryData.features[i];
                points.push([f.geometry.coordinates[1], f.geometry.coordinates[0]]);
            }}
            previewLine.setLatLngs(points);
        }}

        function updateInfo(feature, index) {{
            var lat = feature.geometry.coordinates[1];
            var lng = feature.geometry.coordinates[0];
            document.getElementById('currentTime').textContent = feature.properties.time;
            document.getElementById('currentSpeed').textContent = feature.properties.speed.toFixed(1) + ' km/h';
            document.getElementById('currentStatus').textContent = feature.properties.status === 1 ? '载客' : '空载';
            document.getElementById('progress').textContent = Math.round((index / Math.max(1, trajectoryData.features.length - 1)) * 100) + '%';
            document.getElementById('currentPos').textContent = '(' + lat.toFixed(6) + ', ' + lng.toFixed(6) + ')';
        }}

        function step() {{
            if (!isPlaying) return;
            if (currentIndex >= trajectoryData.features.length) {{
                pauseAnimation();
                return;
            }}

            var feature = trajectoryData.features[currentIndex];
            var latlng = [feature.geometry.coordinates[1], feature.geometry.coordinates[0]];
            traveledPoints.push(latlng);
            traveledLine.setLatLngs(traveledPoints);
            carMarker.setLatLng(latlng);
            updateInfo(feature, currentIndex);
            currentIndex += 1;

            if (currentIndex >= trajectoryData.features.length) {{
                pauseAnimation();
                return;
            }}

            var prev = trajectoryData.features[currentIndex - 1];
            var next = trajectoryData.features[currentIndex];
            var prevTime = new Date(prev.properties.time);
            var nextTime = new Date(next.properties.time);
            var timeDiff = Math.max(0, nextTime - prevTime);
            var delay = timeDiff > 0 ? timeDiff / speedMultiplier : minDelay;
            delay = Math.max(minDelay, Math.min(maxDelay, delay));
            animationTimer = setTimeout(step, delay);
        }}

        function playAnimation() {{
            if (isPlaying) return;
            isPlaying = true;
            if (currentIndex >= trajectoryData.features.length) {{
                currentIndex = Math.max(0, trajectoryData.features.length - 1);
            }}
            step();
        }}

        function pauseAnimation() {{
            isPlaying = false;
            if (animationTimer) {{
                clearTimeout(animationTimer);
                animationTimer = null;
            }}
        }}

        function resetAnimation() {{
            pauseAnimation();
            currentIndex = 0;
            traveledPoints = [];
            traveledLine.setLatLngs([]);
            if (trajectoryData.features.length > 0) {{
                var first = trajectoryData.features[0];
                var firstLatLng = [first.geometry.coordinates[1], first.geometry.coordinates[0]];
                carMarker.setLatLng(firstLatLng);
                updateInfo(first, 0);
                document.getElementById('progress').textContent = '0%';
            }}
        }}

        function updateSpeed() {{
            speedMultiplier = parseFloat(document.getElementById('speedSlider').value);
            document.getElementById('speedValue').textContent = speedMultiplier.toFixed(1) + 'x';
        }}

        addStaticSegments();
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
    for segment in _trajectory_segment_rows(df):
        if segment["status"] == 1:
            occupied_segments.append(segment["points"])
        else:
            vacant_segments.append(segment["points"])

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
