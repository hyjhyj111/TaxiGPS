#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from datetime import datetime
from functools import lru_cache
import hashlib
import json

import numpy as np
import pandas as pd

try:
    import folium
    from folium.plugins import HeatMap
except ImportError as exc:
    raise RuntimeError("请先安装 folium: pip install folium") from exc

try:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:
    raise RuntimeError("请先安装 scikit-learn: pip install scikit-learn") from exc

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


from map_plotter import (
    CONFIG,
    _read_minute_cache,
    _read_od_cache,
    _file_version_key,
    add_map_layers,
    build_map,
    ensure_parent_dir,
    format_time_tag,
    haversine_distance,
    load_minute_data,
    load_od_data,
    log,
    safe_datetime,
)


ANALYSIS_CONFIG = {
    "EXPORT_DIR": "exports/heatmap_stats/",
    "ANALYSIS_CACHE_DIR": "cache/analysis/",
    "HEATMAP_RADIUS": 18,
    "HEATMAP_BLUR": 22,
    "HEATMAP_PRECISION": 4,
    "MAX_DYNAMIC_SLICES": 120,
    "MAX_SLICE_POINTS": 700,
    "MAX_DYNAMIC_INPUT_POINTS_MINUTE": 260,
    "MAX_DYNAMIC_INPUT_POINTS_PICKUP": 420,
    "MAX_DYNAMIC_RENDER_POINTS": 420,
    "MIN_DYNAMIC_VISIBLE_WEIGHT": 0.05,
    "MIN_INTERPOLATED_POINTS": 180,
    "DYNAMIC_WEIGHT_GAMMA": 0.65,
    "DYNAMIC_WEIGHT_FLOOR": 0.16,
    "DYNAMIC_RADIUS_SCALE": 1.35,
    "DYNAMIC_BLUR_SCALE": 1.15,
    "DYNAMIC_MIN_OPACITY": 0.30,
    "DYNAMIC_HEAT_MAX": 0.85,
    "MAX_CLUSTER_POINTS": 3000,
    "CITY_LAT_RANGE": (21.8, 23.2),
    "CITY_LNG_RANGE": (113.3, 114.9),
    "DISPLAY_LAT_RANGE": (22.42, 22.89),
    "DISPLAY_LNG_RANGE": (113.75, 114.63),
    "DRIFT_CAP_KM": 8.0,
    "DRIFT_SPEED_CAP_KMH": 160.0,
    "ANIMATION_TARGET_FPS": 60,
    "ANIMATION_TOTAL_MS": 12000,
    "ANIMATION_MIN_TRANSITION_MS": 260,
    "ANIMATION_MAX_TRANSITION_MS": 850,
    "ANIMATION_STATUS_THROTTLE_MS": 80,
}


def vectorized_haversine(lat1, lng1, lat2, lng2):
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lng1 = np.radians(np.asarray(lng1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lng2 = np.radians(np.asarray(lng2, dtype=float))
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371.0 * c


def _iter_minutes(start_dt, end_dt):
    current = pd.Timestamp(start_dt).floor("min")
    end_limit = pd.Timestamp(end_dt).floor("min")
    while current <= end_limit:
        yield current
        current += pd.Timedelta(minutes=1)


def _minute_cache_path(target_dt):
    ts = pd.Timestamp(target_dt)
    return os.path.join(
        CONFIG["MINUTE_CACHE_DIR"],
        ts.strftime("%Y-%m-%d"),
        ts.strftime("%H"),
        f"{ts.strftime('%M')}.csv",
    )


def _load_minute_data_silent(target_dt):
    minute_file = _minute_cache_path(target_dt)
    cached = _read_minute_cache(minute_file)
    if cached is None:
        return None
    df = cached.copy()
    df["time"] = pd.Timestamp(target_dt)
    return df


def _clip_weights(points_df, threshold_quantile=0.92):
    if points_df is None or len(points_df) == 0:
        return points_df, 0.0

    weights = pd.to_numeric(points_df["weight"], errors="coerce").fillna(0.0)
    positive = weights[weights > 0]
    if len(positive) == 0:
        points_df = points_df.copy()
        points_df["weight_scaled"] = 0.0
        return points_df, 0.0

    cap_value = float(np.quantile(positive, threshold_quantile))
    if cap_value <= 0:
        cap_value = float(positive.max())
    points_df = points_df.copy()
    points_df["weight_scaled"] = np.clip(weights, 0, cap_value) / max(cap_value, 1e-9)
    return points_df, cap_value


def _enhance_dynamic_visual_weights(weight_series):
    weights = np.asarray(weight_series, dtype=float)
    weights = np.clip(weights, 0.0, 1.0)
    enhanced = np.power(weights, ANALYSIS_CONFIG["DYNAMIC_WEIGHT_GAMMA"])
    positive_mask = enhanced > 0
    if positive_mask.any():
        floor_value = float(ANALYSIS_CONFIG["DYNAMIC_WEIGHT_FLOOR"])
        enhanced[positive_mask] = floor_value + (1.0 - floor_value) * enhanced[positive_mask]
    return np.clip(enhanced, 0.0, 1.0)


def _normalize_bounds(df, lat_col="lat", lng_col="lng"):
    lat_min, lat_max = ANALYSIS_CONFIG["CITY_LAT_RANGE"]
    lng_min, lng_max = ANALYSIS_CONFIG["CITY_LNG_RANGE"]
    return df[
        df[lat_col].between(lat_min, lat_max)
        & df[lng_col].between(lng_min, lng_max)
    ].copy()


def _normalize_display_bounds(df, lat_col="lat", lng_col="lng"):
    lat_min, lat_max = ANALYSIS_CONFIG["DISPLAY_LAT_RANGE"]
    lng_min, lng_max = ANALYSIS_CONFIG["DISPLAY_LNG_RANGE"]
    return df[
        df[lat_col].between(lat_min, lat_max)
        & df[lng_col].between(lng_min, lng_max)
    ].copy()


def _is_minute_source(source_type):
    return str(source_type) == "minute"


def filter_drift_points(df, lat_col="lat", lng_col="lng", time_col="time", vehicle_col="vehicle_id"):
    if df is None or len(df) == 0:
        return df, {"input_points": 0, "bounds_removed": 0, "drift_removed": 0, "output_points": 0}

    filtered = _normalize_bounds(df, lat_col=lat_col, lng_col=lng_col)
    bounds_removed = int(len(df) - len(filtered))

    if vehicle_col not in filtered.columns or time_col not in filtered.columns:
        stats = {
            "input_points": int(len(df)),
            "bounds_removed": bounds_removed,
            "drift_removed": 0,
            "output_points": int(len(filtered)),
        }
        return filtered.reset_index(drop=True), stats

    filtered = filtered.sort_values([vehicle_col, time_col]).reset_index(drop=True)
    keep_indices = []
    drift_removed = 0

    for _, group in filtered.groupby(vehicle_col, sort=False):
        previous = None
        for index, row in group.iterrows():
            if previous is None:
                keep_indices.append(index)
                previous = row
                continue

            dt_seconds = (row[time_col] - previous[time_col]).total_seconds()
            distance_km = haversine_distance(previous[lat_col], previous[lng_col], row[lat_col], row[lng_col])
            speed_kmh = (distance_km / (dt_seconds / 3600.0)) if dt_seconds and dt_seconds > 0 and pd.notna(distance_km) else 0.0
            drift_point = (
                pd.notna(distance_km)
                and dt_seconds > 0
                and distance_km > ANALYSIS_CONFIG["DRIFT_CAP_KM"]
                and speed_kmh > ANALYSIS_CONFIG["DRIFT_SPEED_CAP_KMH"]
            )

            if drift_point:
                drift_removed += 1
                continue

            keep_indices.append(index)
            previous = row

    result = filtered.loc[keep_indices].reset_index(drop=True)
    stats = {
        "input_points": int(len(df)),
        "bounds_removed": bounds_removed,
        "drift_removed": drift_removed,
        "output_points": int(len(result)),
    }
    return result, stats


def load_minute_range(start_time, end_time, vehicle_ids=None):
    start_dt = safe_datetime(start_time)
    end_dt = safe_datetime(end_time)
    if start_dt is None or end_dt is None or start_dt > end_dt:
        return pd.DataFrame()

    selected_vehicle_ids = {str(item).strip() for item in (vehicle_ids or []) if str(item).strip()}
    frames = []
    for minute_dt in _iter_minutes(start_dt, end_dt):
        minute_df = _load_minute_data_silent(minute_dt)
        if minute_df is None or len(minute_df) == 0:
            continue
        current = minute_df.copy()
        current["time"] = pd.Timestamp(minute_dt)
        if selected_vehicle_ids:
            current = current[current["vehicle_id"].astype(str).isin(selected_vehicle_ids)].copy()
        if len(current):
            frames.append(current)

    if not frames:
        log.warning("分钟范围热力图未命中数据: start=%s, end=%s", start_dt, end_dt)
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    log.info(
        "分钟范围数据载入完成: start=%s, end=%s, vehicle_count=%s, rows=%s",
        start_dt,
        end_dt,
        len(selected_vehicle_ids) if selected_vehicle_ids else "all",
        len(result),
    )
    return result


@lru_cache(maxsize=64)
def _cached_aggregate_minute_range_fast(start_time_key, end_time_key, vehicle_ids_key, precision, time_bucket_minutes):
    start_dt = pd.Timestamp(start_time_key)
    end_dt = pd.Timestamp(end_time_key)
    selected_vehicle_ids = set(vehicle_ids_key or ())
    time_bucket = None if time_bucket_minutes is None else int(time_bucket_minutes)
    lat_min, lat_max = ANALYSIS_CONFIG["CITY_LAT_RANGE"]
    lng_min, lng_max = ANALYSIS_CONFIG["CITY_LNG_RANGE"]
    sampling_stride = 1
    if time_bucket is not None:
        if time_bucket >= 60:
            sampling_stride = 5
        elif time_bucket >= 30:
            sampling_stride = 3
        elif time_bucket >= 15:
            sampling_stride = 2
    total_input_points = 0
    total_output_points = 0
    bounds_removed = 0
    chunk_minutes = time_bucket or 60
    chunk_map = {}
    for minute_dt in _iter_minutes(start_dt, end_dt):
        chunk_key = pd.Timestamp(minute_dt).floor(f"{int(chunk_minutes)}min")
        chunk_map.setdefault(chunk_key, []).append(pd.Timestamp(minute_dt))

    grouped_frames = []
    for chunk_key, minute_list in chunk_map.items():
        frames = []
        sampled_minutes = minute_list[::sampling_stride] if sampling_stride > 1 else minute_list
        for minute_dt in sampled_minutes:
            minute_df = _load_minute_data_silent(minute_dt)
            if minute_df is None or len(minute_df) == 0:
                continue
            current = minute_df[["vehicle_id", "lati", "long"]].copy()
            total_input_points += int(len(current))
            current["vehicle_id"] = current["vehicle_id"].astype(str)
            if selected_vehicle_ids:
                current = current[current["vehicle_id"].isin(selected_vehicle_ids)].copy()
            if len(current) == 0:
                continue
            current["time"] = pd.Timestamp(minute_dt)
            frames.append(current)

        if not frames:
            continue

        chunk_df = pd.concat(frames, ignore_index=True)
        chunk_df["lati"] = pd.to_numeric(chunk_df["lati"], errors="coerce")
        chunk_df["long"] = pd.to_numeric(chunk_df["long"], errors="coerce")
        valid_mask = chunk_df["lati"].between(lat_min, lat_max) & chunk_df["long"].between(lng_min, lng_max)
        bounds_removed += int((~valid_mask).sum())
        chunk_df = chunk_df[valid_mask].copy()
        if len(chunk_df) == 0:
            continue

        total_output_points += int(len(chunk_df))
        chunk_df["lat_key"] = chunk_df["lati"].round(int(precision))
        chunk_df["lng_key"] = chunk_df["long"].round(int(precision))
        if time_bucket:
            chunk_df["time_slice"] = pd.to_datetime(chunk_df["time"]).dt.floor(f"{time_bucket}min")
        else:
            chunk_df["time_slice"] = pd.NaT

        grouped = (
            chunk_df.groupby(["lat_key", "lng_key", "time_slice"], dropna=False)
            .agg(
                lat=("lati", "mean"),
                lng=("long", "mean"),
                weight=("vehicle_id", "size"),
                time_start=("time", "min"),
                time_end=("time", "max"),
            )
            .reset_index()
        )
        if sampling_stride > 1:
            grouped["weight"] = grouped["weight"] * sampling_stride
        grouped["point_count"] = grouped["weight"].astype(int)
        grouped_frames.append(grouped)

    if grouped_frames:
        grouped_df = pd.concat(grouped_frames, ignore_index=True)
        if not time_bucket:
            grouped_df = (
                grouped_df.groupby(["lat_key", "lng_key"], dropna=False)
                .agg(
                    lat=("lat", "mean"),
                    lng=("lng", "mean"),
                    weight=("weight", "sum"),
                    point_count=("point_count", "sum"),
                    time_start=("time_start", "min"),
                    time_end=("time_end", "max"),
                )
                .reset_index()
            )
            grouped_df["time_slice"] = pd.NaT
        else:
            grouped_df = (
                grouped_df.groupby(["lat_key", "lng_key", "time_slice"], dropna=False)
                .agg(
                    lat=("lat", "mean"),
                    lng=("lng", "mean"),
                    weight=("weight", "sum"),
                    point_count=("point_count", "sum"),
                    time_start=("time_start", "min"),
                    time_end=("time_end", "max"),
                )
                .reset_index()
            )
    else:
        grouped_df = pd.DataFrame(columns=["lat_key", "lng_key", "time_slice", "lat", "lng", "weight", "point_count", "time_start", "time_end"])

    meta = {
        "source_label": "分钟缓存车辆位置",
        "filter_stats": {
            "input_points": total_input_points,
            "bounds_removed": bounds_removed,
            "drift_removed": 0,
            "output_points": total_output_points,
        },
        "lat_range": (
            float(grouped_df["lat"].min()) if len(grouped_df) else None,
            float(grouped_df["lat"].max()) if len(grouped_df) else None,
        ),
        "lng_range": (
            float(grouped_df["lng"].min()) if len(grouped_df) else None,
            float(grouped_df["lng"].max()) if len(grouped_df) else None,
        ),
    }
    return grouped_df, meta


def aggregate_minute_range_fast(start_time, end_time, vehicle_ids=None, precision=4, time_bucket_minutes=None):
    start_time_key = pd.Timestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    end_time_key = pd.Timestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
    vehicle_ids_key = _normalize_vehicle_ids_arg(vehicle_ids)
    grouped_df, meta = _cached_aggregate_minute_range_fast(
        start_time_key,
        end_time_key,
        vehicle_ids_key,
        int(precision),
        None if time_bucket_minutes is None else int(time_bucket_minutes),
    )
    return grouped_df.copy(), dict(meta)


def _source_version_token(path):
    try:
        stat = os.stat(path)
        raw = f"{os.path.abspath(path)}|{int(stat.st_mtime)}|{stat.st_size}"
    except OSError:
        raw = f"{os.path.abspath(path)}|missing"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _daily_operation_cache_path(query_date, vehicle_ids=None):
    date_key = pd.Timestamp(query_date).strftime("%Y%m%d")
    vehicle_ids_key = _normalize_vehicle_ids_arg(vehicle_ids)
    scope_key = "all" if not vehicle_ids_key else hashlib.md5("|".join(vehicle_ids_key).encode("utf-8")).hexdigest()[:12]
    source_token = _source_version_token(CONFIG["OD_TABLE_PATH"])
    return os.path.join(
        ANALYSIS_CONFIG["ANALYSIS_CACHE_DIR"],
        "daily_operation",
        f"{date_key}_{scope_key}_{source_token}_v3.json",
    )


def _normalize_vehicle_ids_arg(vehicle_ids=None):
    if not vehicle_ids:
        return tuple()
    return tuple(sorted({str(item).strip() for item in vehicle_ids if str(item).strip()}))


@lru_cache(maxsize=64)
def _cached_standardize_heatmap_source(source_type, start_time_key, end_time_key, vehicle_ids_key):
    vehicle_ids = list(vehicle_ids_key) if vehicle_ids_key else None
    result_df, meta = standardize_heatmap_source(source_type, start_time_key, end_time_key, vehicle_ids=vehicle_ids)
    return result_df.copy(), dict(meta)


def cached_standardize_heatmap_source(source_type, start_time, end_time, vehicle_ids=None):
    start_time_key = pd.Timestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    end_time_key = pd.Timestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
    vehicle_ids_key = _normalize_vehicle_ids_arg(vehicle_ids)
    df, meta = _cached_standardize_heatmap_source(source_type, start_time_key, end_time_key, vehicle_ids_key)
    return df.copy(), dict(meta)


@lru_cache(maxsize=128)
def _cached_aggregate_spatial_points(source_type, start_time_key, end_time_key, vehicle_ids_key, precision, time_bucket_minutes):
    std_df, _ = _cached_standardize_heatmap_source(source_type, start_time_key, end_time_key, vehicle_ids_key)
    grouped = aggregate_spatial_points(std_df, precision=precision, time_bucket_minutes=time_bucket_minutes)
    return grouped.copy()


def cached_aggregate_spatial_points(source_type, start_time, end_time, vehicle_ids=None, precision=4, time_bucket_minutes=None):
    start_time_key = pd.Timestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    end_time_key = pd.Timestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
    vehicle_ids_key = _normalize_vehicle_ids_arg(vehicle_ids)
    grouped = _cached_aggregate_spatial_points(
        source_type,
        start_time_key,
        end_time_key,
        vehicle_ids_key,
        int(precision),
        None if time_bucket_minutes is None else int(time_bucket_minutes),
    )
    return grouped.copy()


def standardize_heatmap_source(source_type, start_time, end_time, vehicle_ids=None):
    if source_type == "minute":
        raw_df = load_minute_range(start_time, end_time, vehicle_ids=vehicle_ids)
        if raw_df is None or len(raw_df) == 0:
            return pd.DataFrame(), {"source_label": "分钟缓存车辆位置", "filter_stats": {}}

        std_df = raw_df.rename(columns={"lati": "lat", "long": "lng"}).copy()
        std_df["vehicle_id"] = std_df["vehicle_id"].astype(str)
        std_df["weight"] = 1.0
        std_df["source_label"] = "分钟缓存车辆位置"
        std_df, filter_stats = filter_drift_points(std_df, lat_col="lat", lng_col="lng", time_col="time", vehicle_col="vehicle_id")
        return std_df[["time", "vehicle_id", "lat", "lng", "weight", "status", "speed", "source_label"]], {
            "source_label": "分钟缓存车辆位置",
            "filter_stats": filter_stats,
        }

    if source_type == "pickup":
        raw_df = load_od_data(start_time, end_time, None, vehicle_ids=vehicle_ids)
        if raw_df is None or len(raw_df) == 0:
            return pd.DataFrame(), {"source_label": "OD 上车点", "filter_stats": {}}

        std_df = pd.DataFrame(
            {
                "time": pd.to_datetime(raw_df["O_time"], errors="coerce"),
                "vehicle_id": raw_df["O_TAXI_ID"].astype(str),
                "lat": pd.to_numeric(raw_df["O_lat"], errors="coerce"),
                "lng": pd.to_numeric(raw_df["O_lng"], errors="coerce"),
                "weight": 1.0,
                "source_label": "OD 上车点",
            }
        )
        std_df = std_df.dropna(subset=["time", "lat", "lng"]).reset_index(drop=True)
        std_df, filter_stats = filter_drift_points(std_df, lat_col="lat", lng_col="lng", time_col="time", vehicle_col="vehicle_id")
        return std_df[["time", "vehicle_id", "lat", "lng", "weight", "source_label"]], {
            "source_label": "OD 上车点",
            "filter_stats": filter_stats,
        }

    raise ValueError(f"不支持的数据源: {source_type}")


def aggregate_spatial_points(df, precision=4, time_bucket_minutes=None):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    working = df.copy()
    working["lat_key"] = pd.to_numeric(working["lat"], errors="coerce").round(precision)
    working["lng_key"] = pd.to_numeric(working["lng"], errors="coerce").round(precision)
    group_cols = ["lat_key", "lng_key"]
    if time_bucket_minutes:
        working["time_slice"] = pd.to_datetime(working["time"]).dt.floor(f"{int(time_bucket_minutes)}min")
        group_cols.append("time_slice")

    grouped = (
        working.groupby(group_cols, dropna=True)
        .agg(
            lat=("lat", "mean"),
            lng=("lng", "mean"),
            weight=("weight", "sum"),
            point_count=("weight", "size"),
            time_start=("time", "min"),
            time_end=("time", "max"),
        )
        .reset_index()
    )
    return grouped


def recommend_dbscan_params(points_df):
    if points_df is None or len(points_df) < 8:
        return {"eps_km": 0.35, "min_samples": 8}

    sample_df = points_df.sample(n=min(len(points_df), 1200), random_state=42).copy()
    coords_rad = np.radians(sample_df[["lat", "lng"]].to_numpy())
    neighbors = min(5, len(sample_df))
    model = NearestNeighbors(n_neighbors=neighbors, metric="haversine", algorithm="ball_tree")
    model.fit(coords_rad)
    distances, _ = model.kneighbors(coords_rad)
    kth = distances[:, -1] * 6371.0
    eps_km = float(np.clip(np.quantile(kth, 0.75) * 1.25, 0.15, 1.20))
    min_samples = int(np.clip(round(np.sqrt(len(sample_df)) / 2.0), 6, 18))
    return {"eps_km": round(eps_km, 3), "min_samples": min_samples}


def cluster_points(points_df, eps_km, min_samples):
    if points_df is None or len(points_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    if len(points_df) > ANALYSIS_CONFIG["MAX_CLUSTER_POINTS"]:
        working = points_df.nlargest(ANALYSIS_CONFIG["MAX_CLUSTER_POINTS"], "weight").copy()
        log.info("聚类输入点过多，已截断为 %s 个高权重点", len(working))
    else:
        working = points_df.copy()

    coords_rad = np.radians(working[["lat", "lng"]].to_numpy())
    model = DBSCAN(
        eps=max(float(eps_km), 0.01) / 6371.0,
        min_samples=max(int(min_samples), 2),
        metric="haversine",
        algorithm="ball_tree",
    )
    labels = model.fit_predict(coords_rad)
    working["cluster_id"] = labels
    clustered = working[working["cluster_id"] >= 0].copy()
    noise = working[working["cluster_id"] < 0].copy()

    if len(clustered) == 0:
        return pd.DataFrame(), noise

    def _center_lat(group):
        return np.average(group["lat"], weights=group["weight"])

    def _center_lng(group):
        return np.average(group["lng"], weights=group["weight"])

    clusters = (
        clustered.groupby("cluster_id")
        .apply(
            lambda group: pd.Series(
                {
                    "center_lat": _center_lat(group),
                    "center_lng": _center_lng(group),
                    "heat_value": float(group["weight"].sum()),
                    "point_count": int(group["point_count"].sum()) if "point_count" in group.columns else int(len(group)),
                    "time_start": group["time_start"].min() if "time_start" in group.columns else pd.NaT,
                    "time_end": group["time_end"].max() if "time_end" in group.columns else pd.NaT,
                }
            )
        )
        .reset_index()
        .sort_values("heat_value", ascending=False)
        .reset_index(drop=True)
    )
    return clusters, noise


def _build_generic_map(points_df):
    display_df = _normalize_display_bounds(points_df, lat_col="lat", lng_col="lng") if points_df is not None and len(points_df) else points_df
    base_df = display_df.rename(columns={"lat": "lati", "lng": "long"}).copy() if display_df is not None and len(display_df) else None
    m = build_map(base_df)
    m.fit_bounds(
        [
            [ANALYSIS_CONFIG["DISPLAY_LAT_RANGE"][0], ANALYSIS_CONFIG["DISPLAY_LNG_RANGE"][0]],
            [ANALYSIS_CONFIG["DISPLAY_LAT_RANGE"][1], ANALYSIS_CONFIG["DISPLAY_LNG_RANGE"][1]],
        ],
        padding=(12, 12),
    )
    return m


def _resolve_dynamic_transition_ms(slice_count):
    segment_count = max(int(slice_count) - 1, 1)
    target_total_ms = float(ANALYSIS_CONFIG["ANIMATION_TOTAL_MS"])
    transition_ms = target_total_ms / segment_count
    return int(
        np.clip(
            transition_ms,
            ANALYSIS_CONFIG["ANIMATION_MIN_TRANSITION_MS"],
            ANALYSIS_CONFIG["ANIMATION_MAX_TRANSITION_MS"],
        )
    )


def _append_before_closing_tag(html_text, tag, content):
    closing_tag = f"</{tag}>"
    if closing_tag in html_text:
        return html_text.replace(closing_tag, f"{content}\n{closing_tag}", 1)
    return html_text + content


def _build_dynamic_heatmap_player_assets(map_name, frames, source_label, radius, blur, transition_ms):
    control_id = f"heatmap-player-{map_name}"
    settings = {
        "targetFps": int(ANALYSIS_CONFIG["ANIMATION_TARGET_FPS"]),
        "frameBudgetMs": round(1000.0 / max(int(ANALYSIS_CONFIG["ANIMATION_TARGET_FPS"]), 1), 2),
        "transitionMs": int(transition_ms),
        "statusThrottleMs": int(ANALYSIS_CONFIG["ANIMATION_STATUS_THROTTLE_MS"]),
        "minVisibleWeight": float(ANALYSIS_CONFIG["MIN_DYNAMIC_VISIBLE_WEIGHT"]),
        "maxInterpolatedPoints": int(ANALYSIS_CONFIG["MAX_DYNAMIC_RENDER_POINTS"]),
        "minInterpolatedPoints": int(ANALYSIS_CONFIG["MIN_INTERPOLATED_POINTS"]),
        "radius": int(round(radius * ANALYSIS_CONFIG["DYNAMIC_RADIUS_SCALE"])),
        "blur": int(round(blur * ANALYSIS_CONFIG["DYNAMIC_BLUR_SCALE"])),
        "minOpacity": float(ANALYSIS_CONFIG["DYNAMIC_MIN_OPACITY"]),
        "heatMax": float(ANALYSIS_CONFIG["DYNAMIC_HEAT_MAX"]),
    }

    control_html = f"""
    <style>
        #{control_id} {{
            position: absolute;
            left: 18px;
            right: 18px;
            bottom: 18px;
            z-index: 9999;
            padding: 12px 14px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(10px);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.14);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        #{control_id} .heatmap-player-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }}
        #{control_id} .heatmap-player-title {{
            font-size: 13px;
            color: #0f172a;
            font-weight: 700;
            min-width: 116px;
        }}
        #{control_id} .heatmap-player-meta {{
            font-size: 12px;
            color: #475569;
            min-width: 148px;
        }}
        #{control_id} .heatmap-player-button {{
            border: none;
            border-radius: 999px;
            background: #ef4444;
            color: #fff;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
        }}
        #{control_id} .heatmap-player-button:hover {{
            background: #dc2626;
        }}
        #{control_id} input[type="range"] {{
            flex: 1;
            min-width: 180px;
            accent-color: #ef4444;
        }}
        #{control_id} .heatmap-player-hint {{
            margin-top: 6px;
            font-size: 12px;
            color: #64748b;
        }}
    </style>
    <div id="{control_id}">
        <div class="heatmap-player-row">
            <div class="heatmap-player-title">{source_label}动态热力图</div>
            <button type="button" class="heatmap-player-button" data-role="toggle">播放动画</button>
            <input type="range" min="0" max="{max(len(frames) - 1, 0)}" value="0" step="1" data-role="scrubber" />
            <div class="heatmap-player-meta" data-role="time">--:--</div>
            <div class="heatmap-player-meta" data-role="fps">目标 60fps</div>
        </div>
        <div class="heatmap-player-hint">采用逐帧插值和平滑切片过渡；拖动时间轴可立即预览，播放期间会自动按设备性能调节点数预算以降低卡顿和闪烁。</div>
    </div>
    """

    script_html = f"""
    <script src="https://cdn.jsdelivr.net/npm/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    <script>
    (function() {{
        const map = {map_name};
        const frames = {json.dumps(frames, ensure_ascii=False)};
        const settings = {json.dumps(settings, ensure_ascii=False)};
        const container = document.getElementById("{control_id}");
        if (!map || !container || !window.L || !L.heatLayer || !frames.length) {{
            return;
        }}

        const toggleButton = container.querySelector('[data-role="toggle"]');
        const scrubber = container.querySelector('[data-role="scrubber"]');
        const timeLabel = container.querySelector('[data-role="time"]');
        const fpsLabel = container.querySelector('[data-role="fps"]');

        const heatLayer = L.heatLayer(frames[0].points || [], {{
            radius: settings.radius,
            blur: settings.blur,
            maxZoom: 16,
            minOpacity: settings.minOpacity,
            max: settings.heatMax,
            gradient: {{
                0.06: '#38bdf8',
                0.22: '#22d3ee',
                0.42: '#84cc16',
                0.60: '#facc15',
                0.78: '#fb923c',
                0.90: '#f97316',
                1.00: '#ef4444'
            }}
        }}).addTo(map);

        let playing = false;
        let currentFrameIndex = 0;
        let segmentStartTs = 0;
        let rafId = null;
        let lastPaintTs = 0;
        let lastStatusTs = 0;
        let adaptivePointBudget = settings.maxInterpolatedPoints;
        let smoothedFrameCost = settings.frameBudgetMs;
        const transitionCache = new Map();

        function clamp(value, minValue, maxValue) {{
            return Math.min(Math.max(value, minValue), maxValue);
        }}

        function easeInOutSine(t) {{
            return -(Math.cos(Math.PI * t) - 1) / 2;
        }}

        function updateButton() {{
            toggleButton.textContent = playing ? '暂停动画' : '播放动画';
        }}

        function updateStatus(force, ts) {{
            if (!force && ts - lastStatusTs < settings.statusThrottleMs) {{
                return;
            }}
            lastStatusTs = ts;
            const frame = frames[currentFrameIndex] || frames[0];
            timeLabel.textContent = frame ? frame.time : '--:--';
            const fps = clamp(1000 / Math.max(smoothedFrameCost, 1), 1, settings.targetFps).toFixed(0);
            fpsLabel.textContent = '约 ' + fps + 'fps · 点预算 ' + adaptivePointBudget;
        }}

        function buildTransition(index) {{
            if (transitionCache.has(index)) {{
                return transitionCache.get(index);
            }}
            const startPoints = (frames[index] && frames[index].points) || [];
            const endPoints = (frames[index + 1] && frames[index + 1].points) || startPoints;
            const pointMap = new Map();

            startPoints.forEach(function(point) {{
                const key = point[0].toFixed(4) + ',' + point[1].toFixed(4);
                pointMap.set(key, {{ lat: point[0], lng: point[1], start: point[2], end: 0 }});
            }});
            endPoints.forEach(function(point) {{
                const key = point[0].toFixed(4) + ',' + point[1].toFixed(4);
                if (pointMap.has(key)) {{
                    pointMap.get(key).end = point[2];
                }} else {{
                    pointMap.set(key, {{ lat: point[0], lng: point[1], start: 0, end: point[2] }});
                }}
            }});

            const rows = Array.from(pointMap.values())
                .filter(function(item) {{
                    return Math.max(item.start, item.end) >= settings.minVisibleWeight;
                }})
                .sort(function(a, b) {{
                    return Math.max(b.start, b.end) - Math.max(a.start, a.end);
                }});

            transitionCache.set(index, rows);
            return rows;
        }}

        function renderPoints(points, ts, force) {{
            heatLayer.setLatLngs(points);
            scrubber.value = String(currentFrameIndex);
            updateStatus(force, ts);
        }}

        function renderStaticFrame(index, ts, force) {{
            currentFrameIndex = clamp(index, 0, Math.max(frames.length - 1, 0));
            renderPoints((frames[currentFrameIndex] && frames[currentFrameIndex].points) || [], ts, force);
        }}

        function renderInterpolated(segmentIndex, progress, ts) {{
            const transitionRows = buildTransition(segmentIndex);
            const eased = easeInOutSine(clamp(progress, 0, 1));
            const budget = Math.min(adaptivePointBudget, transitionRows.length);
            const output = [];

            for (let i = 0; i < budget; i += 1) {{
                const row = transitionRows[i];
                const weight = row.start + (row.end - row.start) * eased;
                if (weight >= settings.minVisibleWeight) {{
                    output.push([row.lat, row.lng, Math.min(Math.max(weight, 0), 1)]);
                }}
            }}

            renderPoints(output, ts, false);
        }}

        function adaptBudget(renderCost) {{
            smoothedFrameCost = smoothedFrameCost * 0.82 + renderCost * 0.18;
            if (smoothedFrameCost > 18 && adaptivePointBudget > settings.minInterpolatedPoints) {{
                adaptivePointBudget = Math.max(settings.minInterpolatedPoints, adaptivePointBudget - 18);
            }} else if (smoothedFrameCost < 11 && adaptivePointBudget < settings.maxInterpolatedPoints) {{
                adaptivePointBudget = Math.min(settings.maxInterpolatedPoints, adaptivePointBudget + 8);
            }}
        }}

        function pause(forceRender) {{
            playing = false;
            updateButton();
            if (rafId) {{
                cancelAnimationFrame(rafId);
                rafId = null;
            }}
            segmentStartTs = 0;
            if (forceRender) {{
                renderStaticFrame(currentFrameIndex, performance.now(), true);
            }}
        }}

        function play() {{
            if (frames.length <= 1) {{
                renderStaticFrame(0, performance.now(), true);
                return;
            }}
            if (currentFrameIndex >= frames.length - 1) {{
                currentFrameIndex = 0;
            }}
            playing = true;
            updateButton();
            segmentStartTs = 0;
            lastPaintTs = 0;
            rafId = requestAnimationFrame(tick);
        }}

        function tick(ts) {{
            if (!playing) {{
                return;
            }}
            if (!segmentStartTs) {{
                segmentStartTs = ts;
            }}
            if (ts - lastPaintTs < settings.frameBudgetMs - 1) {{
                rafId = requestAnimationFrame(tick);
                return;
            }}
            lastPaintTs = ts;

            const segmentIndex = clamp(currentFrameIndex, 0, Math.max(frames.length - 2, 0));
            const progress = (ts - segmentStartTs) / settings.transitionMs;
            const paintStartedAt = performance.now();

            if (progress >= 1) {{
                currentFrameIndex = Math.min(segmentIndex + 1, frames.length - 1);
                renderStaticFrame(currentFrameIndex, ts, true);
                adaptBudget(performance.now() - paintStartedAt);
                segmentStartTs = ts;
                if (currentFrameIndex >= frames.length - 1) {{
                    pause(false);
                    return;
                }}
            }} else {{
                renderInterpolated(segmentIndex, progress, ts);
                adaptBudget(performance.now() - paintStartedAt);
            }}
            rafId = requestAnimationFrame(tick);
        }}

        toggleButton.addEventListener('click', function() {{
            if (playing) {{
                pause(true);
            }} else {{
                play();
            }}
        }});

        scrubber.addEventListener('input', function(event) {{
            const nextIndex = Number(event.target.value || 0);
            pause(false);
            renderStaticFrame(nextIndex, performance.now(), true);
        }});

        map.on('zoomstart movestart', function() {{
            pause(false);
        }});

        renderStaticFrame(0, performance.now(), true);
    }})();
    </script>
    """
    return control_html, script_html


def _postprocess_dynamic_heatmap_html(output_path, map_name, frames, source_label, radius, blur, transition_ms):
    with open(output_path, "r", encoding="utf-8") as f:
        html_text = f.read()

    control_html, script_html = _build_dynamic_heatmap_player_assets(
        map_name,
        frames,
        source_label,
        radius=radius,
        blur=blur,
        transition_ms=transition_ms,
    )
    html_text = _append_before_closing_tag(html_text, "body", control_html)
    html_text = _append_before_closing_tag(html_text, "html", script_html)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_text)


def build_static_heatmap(
    source_type,
    start_time,
    end_time,
    vehicle_ids=None,
    enable_cluster=False,
    eps_km=0.35,
    min_samples=8,
    threshold_quantile=0.92,
    blur=22,
    save_path=None,
):
    if _is_minute_source(source_type):
        points_df, meta = aggregate_minute_range_fast(
            start_time,
            end_time,
            vehicle_ids=vehicle_ids,
            precision=ANALYSIS_CONFIG["HEATMAP_PRECISION"],
            time_bucket_minutes=None,
        )
        std_df = points_df[["lat", "lng", "weight"]].copy() if len(points_df) else pd.DataFrame()
    else:
        std_df, meta = cached_standardize_heatmap_source(source_type, start_time, end_time, vehicle_ids=vehicle_ids)
        if std_df is None or len(std_df) == 0:
            return None, meta
        points_df = cached_aggregate_spatial_points(
            source_type,
            start_time,
            end_time,
            vehicle_ids=vehicle_ids,
            precision=ANALYSIS_CONFIG["HEATMAP_PRECISION"],
            time_bucket_minutes=None,
        )

    if points_df is None or len(points_df) == 0:
        return None, meta
    cluster_df = pd.DataFrame()
    noise_df = pd.DataFrame()
    heat_df = points_df

    if enable_cluster:
        cluster_df, noise_df = cluster_points(points_df, eps_km=eps_km, min_samples=min_samples)
        if len(cluster_df):
            heat_df = cluster_df.rename(columns={"center_lat": "lat", "center_lng": "lng", "heat_value": "weight"}).copy()
            heat_df["point_count"] = cluster_df["point_count"]

    heat_df = _normalize_display_bounds(heat_df, lat_col="lat", lng_col="lng")
    std_df = _normalize_display_bounds(std_df, lat_col="lat", lng_col="lng") if std_df is not None and len(std_df) else std_df
    if heat_df is None or len(heat_df) == 0:
        return None, meta

    heat_df, cap_value = _clip_weights(heat_df, threshold_quantile=threshold_quantile)
    m = _build_generic_map(heat_df[["lat", "lng"]].copy())
    heat_points = heat_df[["lat", "lng", "weight_scaled"]].to_numpy().tolist()

    radius = ANALYSIS_CONFIG["HEATMAP_RADIUS"]
    HeatMap(
        heat_points,
        name=f"{meta['source_label']}热力图",
        radius=int(radius),
        blur=int(blur),
        min_opacity=0.25,
        max_zoom=16,
    ).add_to(m)

    if enable_cluster and len(cluster_df):
        cluster_layer = folium.FeatureGroup(name="聚类中心", show=True).add_to(m)
        for _, row in cluster_df.iterrows():
            popup = (
                f"<b>聚类簇 {int(row['cluster_id'])}</b><br>"
                f"中心: {row['center_lat']:.6f}, {row['center_lng']:.6f}<br>"
                f"热力值: {row['heat_value']:.0f}<br>"
                f"覆盖点数: {int(row['point_count'])}<br>"
                f"时间范围: {row['time_start']} - {row['time_end']}"
            )
            folium.CircleMarker(
                location=[row["center_lat"], row["center_lng"]],
                radius=min(14, max(5, int(np.sqrt(row["heat_value"])))),
                color="#111827",
                fill=True,
                fill_color="#f97316",
                fill_opacity=0.85,
                popup=popup,
                tooltip=f"簇 {int(row['cluster_id'])} / 热力值 {row['heat_value']:.0f}",
            ).add_to(cluster_layer)

    add_map_layers(m)

    source_tag = "minute" if source_type == "minute" else "pickup"
    output_path = save_path or os.path.join(
        CONFIG["OUTPUT_MAP_DIR"],
        f"static_heatmap_{source_tag}_{format_time_tag(start_time)}_{format_time_tag(end_time)}.html",
    )
    ensure_parent_dir(output_path)
    m.save(output_path)

    meta.update(
        {
            "output_path": output_path,
            "input_points": int(meta.get("filter_stats", {}).get("output_points", len(std_df))),
            "heat_points": int(len(points_df)),
            "cluster_count": int(len(cluster_df)),
            "noise_count": int(len(noise_df)),
            "threshold_cap": cap_value,
            "cluster_heat_value_definition": "簇内空间聚合点权重之和；分钟热力值表示累计车辆出现次数，OD 热力值表示累计上车订单数。",
        }
    )
    log.info(
        "静态热力图已生成: source=%s, cluster=%s, eps_km=%.3f, min_samples=%s, threshold_q=%.2f, output=%s",
        meta["source_label"],
        enable_cluster,
        eps_km,
        min_samples,
        threshold_quantile,
        output_path,
    )
    return output_path, meta


def resolve_dynamic_granularity(start_time, end_time, requested_minutes):
    start_dt = safe_datetime(start_time)
    end_dt = safe_datetime(end_time)
    total_minutes = max(int((end_dt - start_dt).total_seconds() // 60), 1)
    candidates = [1, 15, 30, 60]
    requested = int(requested_minutes)
    actual = requested
    for candidate in [item for item in candidates if item >= requested]:
        slices = int(np.ceil(total_minutes / candidate))
        if slices <= ANALYSIS_CONFIG["MAX_DYNAMIC_SLICES"]:
            actual = candidate
            break
    final_slices = int(np.ceil(total_minutes / actual))
    auto_adjusted = actual != requested
    return {
        "requested_minutes": requested,
        "actual_minutes": actual,
        "estimated_slices": final_slices,
        "auto_adjusted": auto_adjusted,
    }


def _slice_index_range(start_dt, end_dt, minutes):
    current = pd.Timestamp(start_dt).floor(f"{minutes}min")
    end_floor = pd.Timestamp(end_dt).floor(f"{minutes}min")
    labels = []
    while current <= end_floor:
        labels.append(current)
        current += pd.Timedelta(minutes=minutes)
    return labels


def _smooth_slice_frames(slice_frames, method="EMA", alpha=0.55, wma_window=3):
    if not slice_frames:
        return []

    smoothed_frames = []
    previous = {}
    history = []

    for frame in slice_frames:
        current = {(row["lat"], row["lng"]): float(row["weight"]) for _, row in frame.iterrows()}
        if method == "WMA":
            history.append(current)
            history = history[-max(int(wma_window), 2):]
            keys = set()
            for item in history:
                keys.update(item.keys())
            weights = np.arange(1, len(history) + 1, dtype=float)
            weighted = {}
            for key in keys:
                values = np.array([item.get(key, 0.0) for item in history], dtype=float)
                weighted[key] = float(np.dot(values, weights) / weights.sum())
            previous = weighted
        else:
            keys = set(previous.keys()) | set(current.keys())
            previous = {
                key: float(alpha * current.get(key, 0.0) + (1.0 - alpha) * previous.get(key, 0.0))
                for key in keys
            }

        smoothed_frames.append(
            pd.DataFrame(
                [{"lat": key[0], "lng": key[1], "weight": value} for key, value in previous.items() if value > 0.01]
            )
        )

    return smoothed_frames


def build_dynamic_heatmap(
    source_type,
    start_time,
    end_time,
    requested_granularity=15,
    vehicle_ids=None,
    smoothing_method="EMA",
    ema_alpha=0.55,
    wma_window=3,
    threshold_quantile=0.92,
    save_path=None,
):
    radius = ANALYSIS_CONFIG["HEATMAP_RADIUS"]
    blur = ANALYSIS_CONFIG["HEATMAP_BLUR"]
    granularity_meta = resolve_dynamic_granularity(start_time, end_time, requested_granularity)
    actual_minutes = granularity_meta["actual_minutes"]
    if _is_minute_source(source_type):
        grouped, meta = aggregate_minute_range_fast(
            start_time,
            end_time,
            vehicle_ids=vehicle_ids,
            precision=ANALYSIS_CONFIG["HEATMAP_PRECISION"],
            time_bucket_minutes=actual_minutes,
        )
        map_df = grouped[["lat", "lng"]].copy() if len(grouped) else pd.DataFrame()
    else:
        std_df, meta = cached_standardize_heatmap_source(source_type, start_time, end_time, vehicle_ids=vehicle_ids)
        if std_df is None or len(std_df) == 0:
            return None, meta
        grouped = cached_aggregate_spatial_points(
            source_type,
            start_time,
            end_time,
            vehicle_ids=vehicle_ids,
            precision=ANALYSIS_CONFIG["HEATMAP_PRECISION"],
            time_bucket_minutes=actual_minutes,
        )
        map_df = grouped[["lat", "lng"]].copy() if len(grouped) else std_df[["lat", "lng"]].copy()

    if grouped is None or len(grouped) == 0:
        return None, meta

    slice_labels = _slice_index_range(start_time, end_time, actual_minutes)
    raw_slice_frames = []
    slice_payload = []
    input_limit = (
        ANALYSIS_CONFIG["MAX_DYNAMIC_INPUT_POINTS_MINUTE"]
        if _is_minute_source(source_type)
        else ANALYSIS_CONFIG["MAX_DYNAMIC_INPUT_POINTS_PICKUP"]
    )
    for label in slice_labels:
        frame = grouped[grouped["time_slice"] == label][["lat", "lng", "weight"]].copy()
        if len(frame) > input_limit:
            frame = frame.sort_values("weight", ascending=False).head(input_limit).copy()
        raw_slice_frames.append(frame)
    smoothed_frames = _smooth_slice_frames(raw_slice_frames, method=smoothing_method, alpha=ema_alpha, wma_window=wma_window)

    render_frames = []
    for label, frame in zip(slice_labels, smoothed_frames):
        if frame is None or len(frame) == 0:
            render_frames.append({"time": label.strftime("%H:%M"), "points": []})
            slice_payload.append({"time": label.strftime("%Y-%m-%d %H:%M"), "points": []})
            continue
        clipped, _ = _clip_weights(frame, threshold_quantile=threshold_quantile)
        clipped["weight_visual"] = _enhance_dynamic_visual_weights(clipped["weight_scaled"])
        clipped = clipped[clipped["weight_visual"] >= ANALYSIS_CONFIG["MIN_DYNAMIC_VISIBLE_WEIGHT"]].copy()
        clipped = clipped.sort_values("weight_visual", ascending=False).head(ANALYSIS_CONFIG["MAX_SLICE_POINTS"]).copy()
        render_clipped = clipped.head(ANALYSIS_CONFIG["MAX_DYNAMIC_RENDER_POINTS"]).copy()
        triples = render_clipped[["lat", "lng", "weight_visual"]].round(6).to_numpy().tolist()
        raw_points = clipped[["lat", "lng", "weight"]].round(6).to_numpy().tolist()
        render_frames.append({"time": label.strftime("%H:%M"), "points": triples})
        slice_payload.append({"time": label.strftime("%Y-%m-%d %H:%M"), "points": raw_points})

    m = _build_generic_map(map_df)
    transition_ms = _resolve_dynamic_transition_ms(len(render_frames))
    add_map_layers(m)

    source_tag = "minute" if source_type == "minute" else "pickup"
    output_path = save_path or os.path.join(
        CONFIG["OUTPUT_MAP_DIR"],
        f"dynamic_heatmap_{source_tag}_{format_time_tag(start_time)}_{format_time_tag(end_time)}_{actual_minutes}min.html",
    )
    ensure_parent_dir(output_path)
    m.save(output_path)
    _postprocess_dynamic_heatmap_html(
        output_path,
        m.get_name(),
        render_frames,
        meta["source_label"],
        radius=radius,
        blur=blur,
        transition_ms=transition_ms,
    )

    meta.update(
        {
            "output_path": output_path,
            "granularity": granularity_meta,
            "smoothing_method": smoothing_method,
            "time_slices": slice_payload,
            "animation_profile": {
                "target_fps": ANALYSIS_CONFIG["ANIMATION_TARGET_FPS"],
                "transition_ms": transition_ms,
                "render_points_cap": ANALYSIS_CONFIG["MAX_DYNAMIC_RENDER_POINTS"],
                "min_visible_weight": ANALYSIS_CONFIG["MIN_DYNAMIC_VISIBLE_WEIGHT"],
            },
        }
    )
    log.info(
        "动态热力图已生成: source=%s, requested=%s, actual=%s, smoothing=%s, slice_count=%s, output=%s",
        meta["source_label"],
        requested_granularity,
        actual_minutes,
        smoothing_method,
        len(slice_payload),
        output_path,
    )
    return output_path, meta


def compute_order_statistics(start_time, end_time, vehicle_ids=None):
    od_df = load_od_data(start_time, end_time, None, vehicle_ids=vehicle_ids)
    minute_df = load_minute_range(start_time, end_time, vehicle_ids=vehicle_ids)
    if od_df is None:
        od_df = pd.DataFrame()
    if minute_df is None:
        minute_df = pd.DataFrame()

    if len(od_df):
        order_hourly = (
            od_df.assign(hour=pd.to_datetime(od_df["O_time"]).dt.floor("h"))
            .groupby("hour")
            .agg(order_count=("O_TAXI_ID", "size"))
            .reset_index()
        )
        mileage_df = pd.DataFrame(
            {
                "里程区间": ["短途(<4km)", "中途(4-8km)", "长途(>8km)"],
                "订单数": [
                    int((od_df["OD_Dist_km"] < 4).sum()),
                    int(((od_df["OD_Dist_km"] >= 4) & (od_df["OD_Dist_km"] <= 8)).sum()),
                    int((od_df["OD_Dist_km"] > 8).sum()),
                ],
            }
        )
        mileage_df["占比"] = mileage_df["订单数"] / max(int(len(od_df)), 1)
    else:
        order_hourly = pd.DataFrame(columns=["hour", "order_count"])
        mileage_df = pd.DataFrame({"里程区间": ["短途(<4km)", "中途(4-8km)", "长途(>8km)"], "订单数": [0, 0, 0], "占比": [0.0, 0.0, 0.0]})

    if len(minute_df):
        minute_df["hour"] = pd.to_datetime(minute_df["time"]).dt.floor("h")
        total_vehicle_hourly = minute_df.groupby("hour")["vehicle_id"].nunique().rename("total_vehicles")
        occupied_vehicle_hourly = (
            minute_df[minute_df["status"] == 1]
            .groupby("hour")["vehicle_id"]
            .nunique()
            .rename("occupied_vehicles")
        )
        vehicle_hourly = pd.concat([total_vehicle_hourly, occupied_vehicle_hourly], axis=1).fillna(0).reset_index()
        vehicle_hourly["occupied_vehicles"] = vehicle_hourly["occupied_vehicles"].astype(int)
        vehicle_hourly["total_vehicles"] = vehicle_hourly["total_vehicles"].astype(int)
        vehicle_hourly["occupancy_rate"] = vehicle_hourly["occupied_vehicles"] / vehicle_hourly["total_vehicles"].replace(0, np.nan)
        vehicle_hourly["occupancy_rate"] = vehicle_hourly["occupancy_rate"].fillna(0.0)
    else:
        vehicle_hourly = pd.DataFrame(columns=["hour", "total_vehicles", "occupied_vehicles", "occupancy_rate"])

    merged_hourly = pd.merge(order_hourly, vehicle_hourly, on="hour", how="outer").sort_values("hour").fillna(0)
    if len(merged_hourly):
        merged_hourly["order_count"] = merged_hourly["order_count"].astype(int)
        merged_hourly["occupied_vehicles"] = merged_hourly["occupied_vehicles"].astype(int)
        merged_hourly["total_vehicles"] = merged_hourly["total_vehicles"].astype(int)

    summary = {
        "order_count": int(len(od_df)),
        "avg_distance_km": float(pd.to_numeric(od_df.get("OD_Dist_km"), errors="coerce").fillna(0.0).mean()) if len(od_df) else 0.0,
        "occupied_vehicle_peak": int(merged_hourly["occupied_vehicles"].max()) if len(merged_hourly) else 0,
        "occupancy_rate_peak": float(merged_hourly["occupancy_rate"].max()) if len(merged_hourly) else 0.0,
    }
    log.info(
        "订单统计完成: start=%s, end=%s, orders=%s, minute_rows=%s",
        start_time,
        end_time,
        summary["order_count"],
        len(minute_df),
    )
    return {
        "summary": summary,
        "hourly": merged_hourly,
        "distance_buckets": mileage_df,
        "od_rows": int(len(od_df)),
        "minute_rows": int(len(minute_df)),
        "stat_definition": "小时载客率 = 该小时内出现过载客状态的唯一车辆数 / 该小时内出现过的唯一车辆数。",
    }


def compute_daily_operation_statistics(query_date, vehicle_ids=None):
    date_value = pd.Timestamp(query_date).date()
    day_start = pd.Timestamp(datetime.combine(date_value, datetime.min.time()))
    day_end = pd.Timestamp(datetime.combine(date_value, datetime.max.time()))
    selected_vehicle_ids = {str(item).strip() for item in (vehicle_ids or []) if str(item).strip()}
    cache_path = _daily_operation_cache_path(query_date, vehicle_ids=vehicle_ids)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_payload = json.load(f)
            result = {
                "summary_table": pd.DataFrame(cached_payload["summary_table"]),
                "occupancy_ratio": float(cached_payload["occupancy_ratio"]),
                "total_distance_km": float(cached_payload["total_distance_km"]),
                "occupied_distance_km": float(cached_payload["occupied_distance_km"]),
                "empty_distance_km": float(cached_payload["empty_distance_km"]),
                "vehicle_count": int(cached_payload["vehicle_count"]),
                "stat_definition": cached_payload["stat_definition"],
            }
            log.info("车辆运营统计命中磁盘缓存: %s", cache_path)
            return result
        except Exception:
            log.warning("车辆运营统计缓存读取失败，将重新计算: %s", cache_path)

    od_df = _read_od_cache(CONFIG["OD_TABLE_PATH"], _file_version_key(CONFIG["OD_TABLE_PATH"]))
    if od_df is None:
        od_df = pd.DataFrame()
    else:
        od_df = od_df.copy()
        od_df = od_df[(od_df["O_time"] >= day_start) & (od_df["D_time"] <= day_end)].copy()
        if selected_vehicle_ids:
            od_df = od_df[od_df["O_TAXI_ID"].astype(str).isin(selected_vehicle_ids)].copy()

    if len(od_df):
        od_df["OD_Time_s"] = pd.to_numeric(od_df["OD_Time_s"], errors="coerce").fillna((od_df["D_time"] - od_df["O_time"]).dt.total_seconds())
        od_df["OD_Dist_km"] = pd.to_numeric(od_df["OD_Dist_km"], errors="coerce").fillna(0.0)
        grouped = (
            od_df.groupby("O_TAXI_ID")
            .agg(
                first_o=("O_time", "min"),
                last_d=("D_time", "max"),
                occupied_seconds=("OD_Time_s", "sum"),
            )
            .reset_index()
        )
        grouped["active_window_s"] = (grouped["last_d"] - grouped["first_o"]).dt.total_seconds().clip(lower=0)
        total_active_window_s = float(grouped["active_window_s"].sum())
        total_occupied_seconds = float(grouped["occupied_seconds"].sum())
        occupancy_ratio = (total_occupied_seconds / total_active_window_s) if total_active_window_s > 0 else 0.0
        occupied_distance_km = float(od_df["OD_Dist_km"].sum())
        total_distance_km = (occupied_distance_km / occupancy_ratio) if occupancy_ratio > 0 else occupied_distance_km
        empty_distance_km = max(total_distance_km - occupied_distance_km, 0.0)
        vehicle_count = int(grouped["O_TAXI_ID"].astype(str).nunique())
    else:
        occupancy_ratio = 0.0
        occupied_distance_km = 0.0
        total_distance_km = 0.0
        empty_distance_km = 0.0
        vehicle_count = len(selected_vehicle_ids) if selected_vehicle_ids else 0

    summary = pd.DataFrame(
        [
            {"指标": "统计日期", "数值": str(date_value), "说明": "按全天 OD 快速统计"},
            {"指标": "车辆范围", "数值": "指定车辆" if selected_vehicle_ids else "全部车辆", "说明": "无车辆筛选时统计当日有 OD 记录的车辆"},
            {"指标": "运营车辆数", "数值": int(vehicle_count), "说明": "当日有 OD 订单记录的唯一车辆数"},
            {"指标": "全天载客率", "数值": round(occupancy_ratio, 4), "说明": "总载客时长 / 车辆活跃窗口总时长"},
            {"指标": "总里程估计(km)", "数值": round(total_distance_km, 3), "说明": "按载客里程和载客率反推的总运营里程估计值"},
            {"指标": "载客里程(km)", "数值": round(occupied_distance_km, 3), "说明": "OD 表中载客订单里程总和"},
            {"指标": "空载里程估计(km)", "数值": round(empty_distance_km, 3), "说明": "总里程估计值减去载客里程"},
        ]
    )
    log.info(
        "车辆运营统计完成: date=%s, selected_vehicles=%s, vehicles=%s, occupancy_ratio=%.4f, total_distance=%.3f",
        date_value,
        len(selected_vehicle_ids) if selected_vehicle_ids else "all",
        vehicle_count,
        occupancy_ratio,
        total_distance_km,
    )
    result = {
        "summary_table": summary,
        "occupancy_ratio": occupancy_ratio,
        "total_distance_km": total_distance_km,
        "occupied_distance_km": occupied_distance_km,
        "empty_distance_km": empty_distance_km,
        "vehicle_count": vehicle_count,
        "stat_definition": "车辆运营统计已切换为 OD 快速统计口径：载客率 = 总载客时长 / 车辆活跃窗口总时长；载客里程来自 OD 表；总里程按载客里程和载客率反推，用于提升模块响应速度。",
    }
    try:
        ensure_parent_dir(cache_path)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary_table": result["summary_table"].to_dict(orient="records"),
                    "occupancy_ratio": result["occupancy_ratio"],
                    "total_distance_km": result["total_distance_km"],
                    "occupied_distance_km": result["occupied_distance_km"],
                    "empty_distance_km": result["empty_distance_km"],
                    "vehicle_count": result["vehicle_count"],
                    "stat_definition": result["stat_definition"],
                },
                f,
                ensure_ascii=False,
            )
    except Exception:
        log.warning("车辆运营统计缓存写入失败: %s", cache_path)
    return result


def run_pickup_cluster_analysis(start_time, end_time, vehicle_ids=None, eps_km=0.35, min_samples=8, threshold_quantile=0.92):
    std_df, meta = cached_standardize_heatmap_source("pickup", start_time, end_time, vehicle_ids=vehicle_ids)
    if std_df is None or len(std_df) == 0:
        return {"clusters": pd.DataFrame(), "noise": pd.DataFrame(), "meta": meta, "map_path": None}

    points_df = cached_aggregate_spatial_points(
        "pickup",
        start_time,
        end_time,
        vehicle_ids=vehicle_ids,
        precision=ANALYSIS_CONFIG["HEATMAP_PRECISION"],
        time_bucket_minutes=None,
    )
    clusters, noise = cluster_points(points_df, eps_km=eps_km, min_samples=min_samples)
    if len(clusters):
        heat_df = clusters.rename(columns={"center_lat": "lat", "center_lng": "lng", "heat_value": "weight"}).copy()
    else:
        heat_df = pd.DataFrame(columns=["lat", "lng", "weight"])

    clipped, cap_value = _clip_weights(heat_df, threshold_quantile=threshold_quantile)
    m = _build_generic_map(std_df[["lat", "lng"]].copy())
    if len(clipped):
        HeatMap(
            clipped[["lat", "lng", "weight_scaled"]].to_numpy().tolist(),
            name="上车点聚类热力图",
            radius=20,
            blur=24,
            min_opacity=0.25,
            max_zoom=16,
        ).add_to(m)

    cluster_layer = folium.FeatureGroup(name="聚类中心", show=True).add_to(m)
    for _, row in clusters.iterrows():
        folium.CircleMarker(
            location=[row["center_lat"], row["center_lng"]],
            radius=min(16, max(6, int(np.sqrt(row["heat_value"])))),
            color="#111827",
            fill=True,
            fill_color="#f59e0b",
            fill_opacity=0.86,
            popup=(
                f"<b>簇 {int(row['cluster_id'])}</b><br>"
                f"中心: {row['center_lat']:.6f}, {row['center_lng']:.6f}<br>"
                f"热力值: {row['heat_value']:.0f}<br>"
                f"时间: {row['time_start']} - {row['time_end']}"
            ),
            tooltip=f"簇 {int(row['cluster_id'])}",
        ).add_to(cluster_layer)

    add_map_layers(m)
    output_path = os.path.join(
        CONFIG["OUTPUT_MAP_DIR"],
        f"pickup_cluster_heatmap_{format_time_tag(start_time)}_{format_time_tag(end_time)}.html",
    )
    ensure_parent_dir(output_path)
    m.save(output_path)

    meta.update(
        {
            "threshold_cap": cap_value,
            "cluster_heat_value_definition": "簇内空间聚合点权重和，对应时间段内上车订单累计量。",
            "output_path": output_path,
        }
    )
    log.info(
        "上车点聚类分析完成: eps_km=%.3f, min_samples=%s, clusters=%s, noise=%s, output=%s",
        eps_km,
        min_samples,
        len(clusters),
        len(noise),
        output_path,
    )
    return {"clusters": clusters, "noise": noise, "meta": meta, "map_path": output_path}


def export_statistics_bundle(query_date, order_stats, operation_stats, cluster_result=None):
    date_value = pd.Timestamp(query_date).strftime("%Y%m%d")
    export_dir = os.path.join(ANALYSIS_CONFIG["EXPORT_DIR"], date_value)
    os.makedirs(export_dir, exist_ok=True)

    order_hourly_path = os.path.join(export_dir, "order_hourly_stats.csv")
    distance_bucket_path = os.path.join(export_dir, "order_distance_bucket_stats.csv")
    operation_path = os.path.join(export_dir, "vehicle_operation_stats.csv")

    order_stats["hourly"].to_csv(order_hourly_path, index=False, encoding="utf-8-sig")
    order_stats["distance_buckets"].to_csv(distance_bucket_path, index=False, encoding="utf-8-sig")
    operation_stats["summary_table"].to_csv(operation_path, index=False, encoding="utf-8-sig")

    xlsx_path = os.path.join(export_dir, "heatmap_statistics_bundle.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        order_stats["hourly"].to_excel(writer, sheet_name="小时订单统计", index=False)
        order_stats["distance_buckets"].to_excel(writer, sheet_name="里程区间统计", index=False)
        operation_stats["summary_table"].to_excel(writer, sheet_name="车辆运营统计", index=False)
        if cluster_result and len(cluster_result.get("clusters", pd.DataFrame())):
            cluster_result["clusters"].to_excel(writer, sheet_name="上车点聚类", index=False)

    log.info(
        "统计结果已导出: hourly=%s, bucket=%s, operation=%s, workbook=%s",
        order_hourly_path,
        distance_bucket_path,
        operation_path,
        xlsx_path,
    )
    return {
        "export_dir": export_dir,
        "csv_files": [order_hourly_path, distance_bucket_path, operation_path],
        "xlsx_file": xlsx_path,
    }
