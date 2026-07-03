#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线规划模块
支持地图点击选择起终点，计算最短距离和基准最快路线
"""

import altair as alt
import streamlit as st
import streamlit.components.v1 as components
import json
import math
import networkx as nx
from pathlib import Path
from map_plotter import (
    CONFIG,
    apply_baseline_route_cost,
    load_road_network,
    nearest_road_node,
    plan_baseline_routes_between_points,
    road_network_status,
)
import pandas as pd
from order_route_analysis import (
    filter_completed_od_orders,
    ensure_completed_od_cache,
    load_completed_od_cache,
)
from stage08_order_route_comparison import (
    TRACK_CACHE_FALLBACKS,
    Stage08DataError,
    calculate_comparison_metrics,
    compute_actual_route_geometry,
    extract_actual_edges_from_track,
    get_planning_routes,
    normalize_od_dataframe,
    normalize_order,
    resolve_existing_path,
    select_valid_orders as select_stage08_valid_orders,
    vehicle_id_token,
)


ROUTE_CLICK_COMPONENT_DIR = Path(__file__).resolve().parent / "route_click_component"
route_click_map_component = components.declare_component(
    "route_planner_click_map",
    path=str(ROUTE_CLICK_COMPONENT_DIR),
)


@st.cache_data(show_spinner=False, ttl=3600)
def _load_route_boundary_geojson():
    for path in CONFIG.get("BOUNDARY_PATHS", []):
        candidate = Path(path)
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def _route_component_points(points):
    if not points:
        origin = {"lat": CONFIG["MAP_CENTER"][0], "lng": CONFIG["MAP_CENTER"][1]}
        destination = {"lat": 22.6008, "lng": 114.1010}
    elif len(points) == 1:
        origin = points[0]
        destination = {"lat": 22.6008, "lng": 114.1010}
    else:
        origin = points[0]
        destination = points[1]
    return origin, destination


def _component_line_points(points):
    clean_points = []
    for point in points or []:
        try:
            if isinstance(point, dict):
                lat = float(point["lat"])
                lng = float(point["lng"])
            else:
                lat = float(point[0])
                lng = float(point[1])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        clean_points.append([lat, lng])
    return clean_points


def _component_line_segments(lines):
    segments = []
    for line in lines or []:
        points = _component_line_points(line)
        if len(points) >= 2:
            segments.append(points)
    return segments


def _route_lines_for_component(route_result):
    if not route_result or not route_result.get("success"):
        return [], [], "", ""

    shortest = route_result.get("shortest", {})
    fastest = route_result.get("fastest", {})
    shortest_points = _component_line_points(shortest.get("points"))
    fastest_points = _component_line_points(fastest.get("points"))
    shortest_label = (
        f"最短距离路线 {shortest.get('distance_m', 0) / 1000:.2f} km"
        if shortest_points
        else ""
    )
    fastest_label = (
        f"基准最快路线 {fastest.get('route_cost_s', 0) / 60:.1f} min"
        if fastest_points
        else ""
    )
    return shortest_points, fastest_points, shortest_label, fastest_label


def render_route_component_map(points, route_result=None, actual_route=None, key="route_planner_map", height=600, boundary_geojson=None):
    origin, destination = _route_component_points(points)
    target = "destination" if len(points or []) == 1 else "origin"
    shortest_points, fastest_points, shortest_label, fastest_label = _route_lines_for_component(route_result)
    actual_points = _component_line_points((actual_route or {}).get("points"))
    actual_segments = (
        _component_line_segments((actual_route or {}).get("segments"))
        if (actual_route or {}).get("has_break")
        else []
    )
    actual_label = ""
    if actual_points or actual_segments:
        actual_label = f"历史实际路线 {(actual_route or {}).get('distance_m', 0) / 1000:.2f} km"
    center = {
        "lat": (float(origin["lat"]) + float(destination["lat"])) / 2,
        "lng": (float(origin["lng"]) + float(destination["lng"])) / 2,
    }
    return route_click_map_component(
        origin={"lat": float(origin["lat"]), "lng": float(origin["lng"])},
        destination={"lat": float(destination["lat"]), "lng": float(destination["lng"])},
        center=center,
        pickTarget=target,
        selectedCount=len(points or []),
        actualPoints=actual_points,
        actualSegments=actual_segments,
        actualLabel=actual_label,
        shortestPoints=shortest_points,
        fastestPoints=fastest_points,
        shortestLabel=shortest_label,
        fastestLabel=fastest_label,
        boundaryGeoJson=boundary_geojson,
        height=height,
        default=None,
        key=key,
    )


def _points_from_stage08_order(order):
    normalized = normalize_order(order)
    return [
        {
            "lat": float(normalized["pickup_matched_lat"]),
            "lng": float(normalized["pickup_matched_lon"]),
        },
        {
            "lat": float(normalized["dropoff_matched_lat"]),
            "lng": float(normalized["dropoff_matched_lon"]),
        },
    ]


def _flatten_route_lines(lines):
    points = []
    for line in lines or []:
        for lat, lng in line or []:
            current = [float(lat), float(lng)]
            if not points or points[-1] != current:
                points.append(current)
    return points


def _stage08_route_summary(route_gdf):
    if route_gdf is None or len(route_gdf) == 0:
        return {
            "nodes": [],
            "points": [],
            "distance_m": 0.0,
            "route_cost_s": 0.0,
            "edge_count": 0,
        }
    lines = route_gdf["coordinates"].tolist() if "coordinates" in route_gdf.columns else []
    return {
        "nodes": route_gdf.attrs.get("route_nodes", []),
        "points": _flatten_route_lines(lines),
        "distance_m": float(route_gdf["length"].sum()),
        "route_cost_s": float(route_gdf["route_cost"].sum()) if "route_cost" in route_gdf.columns else 0.0,
        "edge_count": int(len(route_gdf)),
    }


def _stage08_track_cache_path():
    return resolve_existing_path(
        "cache/vehicle_corrected",
        TRACK_CACHE_FALLBACKS,
        "车辆校正轨迹",
    )


def _load_stage08_speed_cache():
    speed_path = Path(CONFIG.get("BASELINE_SPEED_CACHE_PATH", "cache/edge_baseline_speed.csv"))
    if not speed_path.exists():
        return pd.DataFrame(), {}
    if speed_path.suffix.lower() == ".parquet":
        speed_stats = pd.read_parquet(speed_path)
    else:
        speed_stats = pd.read_csv(speed_path)
    if "highway_type" not in speed_stats.columns:
        return speed_stats, {}
    reliable = speed_stats.copy()
    reliable["avg_speed"] = pd.to_numeric(reliable.get("avg_speed"), errors="coerce")
    reliable["sample_count"] = pd.to_numeric(reliable.get("sample_count"), errors="coerce").fillna(0)
    reliable = reliable[
        reliable["avg_speed"].between(1, CONFIG["ANIMATION_SPEED_CAP_KMH"], inclusive="both")
        & (reliable["sample_count"] >= 3)
    ]
    highway_median = reliable.groupby("highway_type")["avg_speed"].median().dropna().to_dict()
    return speed_stats, highway_median


def _apply_stage08_route_costs(graph):
    if graph is None:
        return
    if getattr(graph, "graph", {}).get("_stage08_baseline_cost_applied"):
        return
    speed_stats, highway_median = _load_stage08_speed_cache()
    apply_baseline_route_cost(
        graph,
        speed_stats,
        highway_median_speed=highway_median,
        default_speed_kph=CONFIG["DEFAULT_HIGHWAY_SPEED_KPH"],
        min_samples=3,
    )
    graph.graph["_stage08_baseline_cost_applied"] = True


def _raw_vehicle_cache_path(vehicle_id):
    token = vehicle_id_token(vehicle_id)
    for suffix in (".parquet", ".csv"):
        path = Path("cache") / "vehicles" / f"{token}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到车辆 {token} 的单车轨迹缓存: cache/vehicles")


def _read_raw_vehicle_cache_segment(order):
    normalized = normalize_order(order)
    path = _raw_vehicle_cache_path(normalized["vehicle_id"])
    if path.suffix.lower() == ".parquet":
        track_df = pd.read_parquet(path)
    else:
        track_df = pd.read_csv(path)

    work = track_df.copy()
    if "lat" not in work.columns and "lati" in work.columns:
        work["lat"] = work["lati"]
    if "lng" not in work.columns and "long" in work.columns:
        work["lng"] = work["long"]
    if "lng" not in work.columns and "lon" in work.columns:
        work["lng"] = work["lon"]
    if "time" not in work.columns or "lat" not in work.columns or "lng" not in work.columns:
        raise Stage08DataError(f"单车轨迹缓存缺少 time/lat/lng 字段: {path}")
    if "lati" not in work.columns:
        work["lati"] = work["lat"]
    if "long" not in work.columns:
        work["long"] = work["lng"]
    if "vehicle_id" not in work.columns:
        work["vehicle_id"] = normalized["vehicle_id"]

    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    pickup_time = pd.Timestamp(normalized["pickup_time"])
    dropoff_time = pd.Timestamp(normalized["dropoff_time"])
    segment = work[work["time"].between(pickup_time, dropoff_time, inclusive="both")].copy()
    keep_cols = [
        col
        for col in ["time", "lat", "lng", "lati", "long", "status", "speed", "vehicle_id", "id"]
        if col in segment.columns
    ]
    segment = segment[keep_cols].dropna(subset=["time", "lat", "lng", "lati", "long"]).sort_values("time")
    if len(segment) < 2:
        raise Stage08DataError(f"单车轨迹缓存截取点数不足: {len(segment)}")
    return segment, str(path)


def _haversine_m(lat1, lng1, lat2, lng2):
    radius_m = 6371008.8
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lam = math.radians(float(lng2) - float(lng1))
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _polyline_distance_m(points):
    total = 0.0
    for left, right in zip(points[:-1], points[1:]):
        total += _haversine_m(left[0], left[1], right[0], right[1])
    return float(total)


def _clean_actual_track_segment(segment, max_speed_mps=55.6, max_step_m=1200.0):
    """清理单车轨迹缓存中的瞬时跳点，保留完整轨迹字段供路网校正使用。"""
    if segment is None or len(segment) == 0:
        return pd.DataFrame(), 0

    ordered = segment.sort_values("time").reset_index(drop=True)
    kept_indices = []
    dropped = 0
    last_time = None
    for row in ordered.itertuples(index=True):
        current = [float(row.lat), float(row.lng)]
        current_time = pd.Timestamp(row.time)
        if not kept_indices:
            kept_indices.append(row.Index)
            last_time = current_time
            continue

        previous = ordered.loc[kept_indices[-1]]
        distance_m = _haversine_m(previous["lat"], previous["lng"], current[0], current[1])
        elapsed_s = max(float((current_time - last_time).total_seconds()), 1.0)
        speed_mps = distance_m / elapsed_s
        if distance_m <= max_step_m or speed_mps <= max_speed_mps:
            if [float(previous["lat"]), float(previous["lng"])] != current:
                kept_indices.append(row.Index)
            last_time = current_time
        else:
            dropped += 1

    if len(kept_indices) < 2:
        return ordered, 0
    return ordered.loc[kept_indices].reset_index(drop=True), dropped


def _clean_actual_track_points(segment, max_speed_mps=55.6, max_step_m=1200.0):
    """清理单车轨迹缓存中的瞬时跳点，避免历史路线被异常 GPS 点拉歪。"""
    cleaned_segment, dropped = _clean_actual_track_segment(
        segment,
        max_speed_mps=max_speed_mps,
        max_step_m=max_step_m,
    )
    if len(cleaned_segment) == 0:
        return [], 0
    cleaned = [[float(row.lat), float(row.lng)] for row in cleaned_segment.itertuples(index=False)]
    if len(cleaned) < 2:
        cleaned = [[float(row.lat), float(row.lng)] for row in segment.sort_values("time").itertuples(index=False)]
        dropped = 0
    return cleaned, dropped


def _project_local_m(point, ref_lat):
    lat, lng = float(point[0]), float(point[1])
    x = math.radians(lng) * 6371008.8 * math.cos(math.radians(ref_lat))
    y = math.radians(lat) * 6371008.8
    return x, y


def _point_to_segment_distance_m(point_xy, left_xy, right_xy):
    px, py = point_xy
    ax, ay = left_xy
    bx, by = right_xy
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    nearest_x = ax + t * dx
    nearest_y = ay + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def _min_distance_to_polyline_m(point, planning_points, ref_lat):
    if len(planning_points) < 2:
        return math.inf
    point_xy = _project_local_m(point, ref_lat)
    projected = [_project_local_m(item, ref_lat) for item in planning_points]
    return min(
        _point_to_segment_distance_m(point_xy, left, right)
        for left, right in zip(projected[:-1], projected[1:])
    )


def _approximate_polyline_overlap_ratio(actual_points, planning_points, planning_distance_m, tolerance_m=60.0):
    if not actual_points or len(actual_points) < 2 or not planning_points or len(planning_points) < 2:
        return math.nan
    if not planning_distance_m or planning_distance_m <= 0:
        return math.nan
    ref_lat = sum(point[0] for point in actual_points[: min(20, len(actual_points))]) / min(20, len(actual_points))
    overlap_m = 0.0
    for left, right in zip(actual_points[:-1], actual_points[1:]):
        midpoint = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
        segment_m = _haversine_m(left[0], left[1], right[0], right[1])
        if _min_distance_to_polyline_m(midpoint, planning_points, ref_lat) <= tolerance_m:
            overlap_m += segment_m
    return float(min(1.0, overlap_m / planning_distance_m))


def _basic_stage08_valid_orders(od_df, limit=10):
    normalized = normalize_od_dataframe(od_df)
    if len(normalized) == 0:
        return normalized
    valid = normalized[
        normalized["vehicle_id"].notna()
        & normalized["pickup_node"].notna()
        & normalized["dropoff_node"].notna()
        & normalized["pickup_time"].notna()
        & normalized["dropoff_time"].notna()
        & normalized["pickup_matched_lat"].notna()
        & normalized["pickup_matched_lon"].notna()
        & normalized["dropoff_matched_lat"].notna()
        & normalized["dropoff_matched_lon"].notna()
        & (normalized["dropoff_time"] > normalized["pickup_time"])
    ].copy()
    cached = set()
    vehicle_cache_dir = Path("cache") / "vehicles"
    if vehicle_cache_dir.exists():
        for path in vehicle_cache_dir.iterdir():
            if path.suffix.lower() in {".csv", ".parquet"}:
                cached.add(vehicle_id_token(path.stem.split("_", 1)[0]))
    if cached:
        valid = valid[valid["vehicle_id"].map(vehicle_id_token).isin(cached)].copy()
    return valid.head(max(1, min(int(limit), 10))).reset_index(drop=True)


def _stage08_frontend_valid_orders(od_df, limit, track_cache_path):
    strict_orders = select_stage08_valid_orders(od_df, limit=limit, track_cache_path=track_cache_path)
    if len(strict_orders) > 0:
        return strict_orders
    return _basic_stage08_valid_orders(od_df, limit=limit)


def _edge_series_from_corrected_rows(rows):
    if rows is None or len(rows) == 0:
        return pd.DataFrame(columns=["edge_u", "edge_v", "edge_key"])
    required = ["edge_u", "edge_v", "edge_key"]
    if any(col not in rows.columns for col in required):
        return pd.DataFrame(columns=required)
    edge_series = rows[required].dropna().copy()
    if len(edge_series) == 0:
        return edge_series
    edge_series = edge_series[edge_series.ne(edge_series.shift()).any(axis=1)].copy()
    return edge_series.reset_index(drop=True)


def _best_length_edge_key(graph, u, v):
    edge_data = graph.get_edge_data(u, v)
    if not edge_data:
        return 0, 0.0
    best_key = None
    best_length = math.inf
    for key, data in edge_data.items():
        try:
            length = float((data or {}).get("length", 0.0) or 0.0)
        except (TypeError, ValueError):
            length = math.inf
        if length < best_length:
            best_key = key
            best_length = length
    return (0 if best_key is None else best_key), (0.0 if math.isinf(best_length) else best_length)


def _correct_order_track_to_edges(graph, cleaned_segment, max_speed_mps=110.0, max_ratio=8.0, min_allowed_m=1800.0):
    """将订单窗口内的清洗轨迹点转换为连续道路边序列。

    优先使用有向路网最短路；若相邻轨迹点因单行方向、局部断边或拓扑缺口
    无法连通，则使用无向路网做短距离补桥。补桥仍恢复为真实道路边几何，
    不使用两点直线替代。
    """
    rows = []
    meta = {
        "matched_segments": 0,
        "bridged_segments": 0,
        "skipped_segments": 0,
        "path_failures": 0,
        "nearest_failures": 0,
    }
    if graph is None or cleaned_segment is None or len(cleaned_segment) < 2:
        return pd.DataFrame(columns=["edge_u", "edge_v", "edge_key"]), meta

    try:
        undirected_graph = graph.to_undirected(as_view=True) if graph.is_directed() else graph
    except TypeError:
        undirected_graph = graph.to_undirected() if graph.is_directed() else graph

    def build_edges_from_nodes(path_nodes):
        segment_edges = []
        path_distance_m = 0.0
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            if graph.get_edge_data(u, v):
                edge_u, edge_v = u, v
            elif graph.get_edge_data(v, u):
                # 无向补桥时可能需要反向取边几何；仅用于历史路线展示与距离累加。
                edge_u, edge_v = v, u
            else:
                continue
            key, length = _best_length_edge_key(graph, edge_u, edge_v)
            segment_edges.append({"edge_u": edge_u, "edge_v": edge_v, "edge_key": key})
            path_distance_m += length
        return segment_edges, path_distance_m

    def is_plausible(path_distance_m, gps_distance_m, elapsed_s):
        allowed_m = max(float(min_allowed_m), float(gps_distance_m) * float(max_ratio))
        if path_distance_m <= 0:
            return False
        return path_distance_m <= allowed_m and path_distance_m / max(elapsed_s, 1.0) <= max_speed_mps

    for prev_row, row in zip(cleaned_segment.iloc[:-1].itertuples(index=False), cleaned_segment.iloc[1:].itertuples(index=False)):
        try:
            source = nearest_road_node(graph, float(prev_row.long), float(prev_row.lati))
            target = nearest_road_node(graph, float(row.long), float(row.lati))
        except Exception:
            meta["nearest_failures"] += 1
            continue
        if source == target:
            continue

        elapsed_s = max(float((pd.Timestamp(row.time) - pd.Timestamp(prev_row.time)).total_seconds()), 1.0)
        gps_distance_m = _haversine_m(prev_row.lati, prev_row.long, row.lati, row.long)

        accepted_edges = []
        accepted_kind = ""
        try:
            path_nodes = nx.shortest_path(graph, source, target, weight="length")
            candidate_edges, candidate_distance_m = build_edges_from_nodes(path_nodes)
            if candidate_edges and is_plausible(candidate_distance_m, gps_distance_m, elapsed_s):
                accepted_edges = candidate_edges
                accepted_kind = "matched_segments"
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            meta["path_failures"] += 1

        if not accepted_edges:
            try:
                bridge_nodes = nx.shortest_path(undirected_graph, source, target, weight="length")
                bridge_edges, bridge_distance_m = build_edges_from_nodes(bridge_nodes)
                if bridge_edges and is_plausible(bridge_distance_m, gps_distance_m, elapsed_s):
                    accepted_edges = bridge_edges
                    accepted_kind = "bridged_segments"
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                meta["path_failures"] += 1

        if not accepted_edges:
            meta["skipped_segments"] += 1
            continue

        rows.extend(accepted_edges)
        meta[accepted_kind] += 1

    edge_series = pd.DataFrame(rows, columns=["edge_u", "edge_v", "edge_key"])
    if len(edge_series) > 0:
        edge_series = edge_series[edge_series.ne(edge_series.shift()).any(axis=1)].reset_index(drop=True)
    return edge_series, meta

def _stage08_corrected_cache_analysis_for_order(graph, order, strict_error=None):
    normalized = normalize_order(order)
    raw_segment, raw_cache_path = _read_raw_vehicle_cache_segment(normalized)
    cleaned_segment, dropped_points = _clean_actual_track_segment(raw_segment)
    if len(cleaned_segment) < 2:
        raise Stage08DataError("清洗后的订单轨迹点不足，无法执行路网校正。")

    actual_edges, correction_meta = _correct_order_track_to_edges(graph, cleaned_segment)
    if len(actual_edges) == 0:
        raise Stage08DataError("订单轨迹路网校正没有可用道路边序列。")

    _apply_stage08_route_costs(graph)
    actual_distance_m, actual_lines, actual_meta = compute_actual_route_geometry(graph, actual_edges)
    shortest_gdf, fastest_gdf = get_planning_routes(
        graph,
        int(normalized["pickup_node"]),
        int(normalized["dropoff_node"]),
    )
    metrics = calculate_comparison_metrics(
        normalized,
        actual_edges,
        actual_distance_m,
        shortest_gdf,
        fastest_gdf,
    )
    metrics["raw_track_points"] = int(len(raw_segment))
    metrics["cleaned_track_points"] = int(len(cleaned_segment))
    metrics["gps_outlier_points"] = int(dropped_points)
    metrics["actual_used_edges"] = int(actual_meta.get("used_edges", 0))
    metrics["actual_has_break"] = bool(actual_meta.get("has_break", False) or correction_meta.get("skipped_segments", 0) > 0)
    metrics["actual_source"] = "订单轨迹路网校正"
    metrics["actual_cache_path"] = raw_cache_path
    metrics["strict_cache_error"] = str(strict_error) if strict_error else ""
    metrics["matched_road_segments"] = int(correction_meta.get("matched_segments", 0))
    metrics["bridged_road_segments"] = int(correction_meta.get("bridged_segments", 0))
    metrics["skipped_road_segments"] = int(correction_meta.get("skipped_segments", 0))
    return {
        "success": True,
        "order": normalized,
        "metrics": metrics,
        "actual": {
            "points": _flatten_route_lines(actual_lines),
            "segments": actual_lines,
            "distance_m": actual_distance_m,
            "edge_set": set(zip(actual_edges["edge_u"], actual_edges["edge_v"], actual_edges["edge_key"])),
            "raw_point_count": int(len(raw_segment)),
            "has_break": metrics.get("actual_has_break", False),
        },
        "route_result": {
            "success": True,
            "shortest": _stage08_route_summary(shortest_gdf),
            "fastest": _stage08_route_summary(fastest_gdf),
            "method": "使用校正OD端点规划，历史实际路线来自订单轨迹路网校正。",
        },
    }


def _stage08_gps_cache_analysis_for_order(graph, order, strict_error=None):
    try:
        return _stage08_corrected_cache_analysis_for_order(graph, order, strict_error=strict_error)
    except Exception as correction_exc:
        fallback_error = correction_exc

    segment, cache_path = _read_raw_vehicle_cache_segment(order)
    actual_points, dropped_points = _clean_actual_track_points(segment)
    normalized = normalize_order(order)
    _apply_stage08_route_costs(graph)
    shortest_gdf, fastest_gdf = get_planning_routes(
        graph,
        int(normalized["pickup_node"]),
        int(normalized["dropoff_node"]),
    )
    shortest = _stage08_route_summary(shortest_gdf)
    fastest = _stage08_route_summary(fastest_gdf)
    actual_distance_m = _polyline_distance_m(actual_points)
    pickup_time = pd.Timestamp(normalized["pickup_time"])
    dropoff_time = pd.Timestamp(normalized["dropoff_time"])
    duration_s = float((dropoff_time - pickup_time).total_seconds())
    shortest_distance_m = float(shortest["distance_m"]) if shortest.get("edge_count", 0) else math.nan
    fastest_distance_m = float(fastest["distance_m"]) if fastest.get("edge_count", 0) else math.nan
    detour_ratio = (
        float((actual_distance_m - shortest_distance_m) / shortest_distance_m)
        if pd.notna(shortest_distance_m) and shortest_distance_m > 0
        else math.nan
    )
    fastest_overlap_rate = _approximate_polyline_overlap_ratio(
        actual_points,
        fastest.get("points", []),
        fastest_distance_m,
    )
    metrics = {
        "order_id": normalized["id"],
        "vehicle_id": normalized["vehicle_id"],
        "pickup_node": normalized["pickup_node"],
        "dropoff_node": normalized["dropoff_node"],
        "actual_distance_m": float(actual_distance_m),
        "actual_duration_s": duration_s,
        "shortest_distance_m": shortest_distance_m,
        "shortest_route_cost_s": float(shortest.get("route_cost_s", 0.0)) if shortest.get("edge_count", 0) else math.nan,
        "shortest_edge_count": int(shortest.get("edge_count", 0)),
        "fastest_distance_m": fastest_distance_m,
        "fastest_route_cost_s": float(fastest.get("route_cost_s", 0.0)) if fastest.get("edge_count", 0) else math.nan,
        "fastest_edge_count": int(fastest.get("edge_count", 0)),
        "distance_delta_m": float(actual_distance_m - shortest_distance_m) if pd.notna(shortest_distance_m) else math.nan,
        "detour_ratio": detour_ratio,
        "fastest_overlap_rate": fastest_overlap_rate,
        "raw_track_points": int(len(segment)),
        "cleaned_track_points": int(len(actual_points)),
        "gps_outlier_points": int(dropped_points),
        "actual_point_edges": int(max(0, len(actual_points) - 1)),
        "actual_used_edges": int(max(0, len(actual_points) - 1)),
        "actual_has_break": False,
        "actual_source": "单车轨迹缓存，路网校正失败后兜底",
        "actual_cache_path": cache_path,
        "strict_cache_error": str(strict_error or fallback_error),
    }
    return {
        "success": True,
        "order": normalized,
        "metrics": metrics,
        "actual": {
            "points": actual_points,
            "segments": [actual_points] if len(actual_points) >= 2 else [],
            "distance_m": actual_distance_m,
            "edge_set": set(),
            "raw_point_count": int(len(segment)),
            "has_break": False,
        },
        "route_result": {
            "success": True,
            "shortest": shortest,
            "fastest": fastest,
            "method": "使用校正OD端点规划，历史实际路线来自单车轨迹缓存。",
        },
    }


def _stage08_analysis_for_order(graph, order, track_cache_path):
    if Path(str(track_cache_path)).name == "road_corrected":
        return _stage08_corrected_cache_analysis_for_order(graph, order)
    try:
        actual_edges, raw_point_count = extract_actual_edges_from_track(graph, order, track_cache_path)
    except Exception as exc:
        return _stage08_gps_cache_analysis_for_order(graph, order, strict_error=exc)
    actual_distance_m, actual_lines, actual_meta = compute_actual_route_geometry(graph, actual_edges)
    normalized = normalize_order(order)
    _apply_stage08_route_costs(graph)
    shortest_gdf, fastest_gdf = get_planning_routes(
        graph,
        int(normalized["pickup_node"]),
        int(normalized["dropoff_node"]),
    )
    metrics = calculate_comparison_metrics(
        normalized,
        actual_edges,
        actual_distance_m,
        shortest_gdf,
        fastest_gdf,
    )
    metrics["raw_track_points"] = int(raw_point_count)
    metrics["actual_used_edges"] = int(actual_meta.get("used_edges", 0))
    metrics["actual_has_break"] = bool(actual_meta.get("has_break", False))
    return {
        "success": True,
        "order": normalized,
        "metrics": metrics,
        "actual": {
            "points": _flatten_route_lines(actual_lines),
            "segments": actual_lines,
            "distance_m": actual_distance_m,
            "edge_set": set(zip(actual_edges["edge_u"], actual_edges["edge_v"], actual_edges["edge_key"])),
            "raw_point_count": raw_point_count,
            "has_break": actual_meta.get("has_break", False),
        },
        "route_result": {
            "success": True,
            "shortest": _stage08_route_summary(shortest_gdf),
            "fastest": _stage08_route_summary(fastest_gdf),
            "method": "直接使用校正OD缓存中的 pickup_node/dropoff_node 规划，不重新吸附坐标。",
        },
    }


def render_route_planning_view(payload):
    """渲染路线规划视图"""
    st.subheader("路线规划")

    # 检查路网状态
    status = road_network_status()
    if not status["available"]:
        st.warning("未找到路网文件。请将 shenzhen_drive.pkl 或 shenzhen_drive.graphml 放到项目根目录、data/ 或 cache/，或设置 TAXIGPS_ROAD_NETWORK_PATH。")
        return

    st.caption(f"路网文件: {status['path']}")

    # 获取选中的车辆ID（用于生成道路速度缓存）
    if st.session_state.get("route_source_mode") == "历史OD端点":
        selected_vehicle_ids = payload.get("history_od_vehicle_ids") or payload.get("trajectory_vehicle_ids", [])
    else:
        selected_vehicle_ids = payload.get("trajectory_vehicle_ids", [])

    speed_vehicle_ids = selected_vehicle_ids[:CONFIG["MAX_CONGESTION_VEHICLES"]]

    # 初始化状态
    if 'route_points' not in st.session_state:
        st.session_state.route_points = []
    if 'route_result' not in st.session_state:
        st.session_state.route_result = None
    route_source_mode = st.session_state.get("route_source_mode")
    if route_source_mode not in {"地图选点", "历史OD端点"}:
        route_source_mode = "地图选点"

    history_vehicle_ids = payload.get("history_od_vehicle_ids", [])
    if route_source_mode == "历史OD端点":
        render_od_selection_mode(payload, history_vehicle_ids)
        return

    render_map_selection_mode(payload, speed_vehicle_ids)


def render_od_selection_mode(payload, speed_vehicle_ids):
    """历史 OD 端点选择模式"""
    selected_vehicle_ids = payload.get("history_od_vehicle_ids", []) or speed_vehicle_ids

    od_cache_df, od_cache_meta = load_completed_od_cache()
    if not od_cache_meta.get("success"):
        with st.spinner("首次进入历史OD端点，正在生成全量校正OD缓存..."):
            graph, _network_meta = load_road_network()
            od_cache_df, od_cache_meta = ensure_completed_od_cache(graph)
        if not od_cache_meta.get("success"):
            st.warning(od_cache_meta.get("error", "校正OD缓存生成失败。"))
            st.caption("请确认已有 OD 缓存和校正轨迹缓存存在；本页面不会自动读取原始 OD 大表。")
            return

    if not selected_vehicle_ids:
        return

    day_start = pd.Timestamp(payload["start_time"]).normalize()
    day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    od_df = filter_completed_od_orders(
        od_cache_df,
        start_time=day_start,
        end_time=day_end,
        vehicle_ids=selected_vehicle_ids,
    )

    if od_df is None or len(od_df) == 0:
        st.warning("校正OD缓存中当前车辆没有当天已完成订单记录")
        st.caption(f"缓存: {od_cache_meta.get('cache_path')} | 日期: {day_start.strftime('%Y-%m-%d')}")
        return

    st.caption(f"已应用车辆: {', '.join(selected_vehicle_ids[:10])} | OD订单: {len(od_df)} 条 | 缓存: {od_cache_meta.get('cache_path')}")
    render_history_od_stage08_from_sidebar(payload, od_df, selected_vehicle_ids)


def render_history_od_stage08_from_sidebar(payload, od_df, selected_vehicle_ids):
    """左侧控制台驱动的历史OD三路线分析。"""
    try:
        track_cache_path = _stage08_track_cache_path()
    except Exception as exc:
        track_cache_path = "cache/vehicle_corrected"

    vehicle_count = len(selected_vehicle_ids)
    if vehicle_count == 1:
        render_single_vehicle_stage08_result(od_df, selected_vehicle_ids[0], track_cache_path)
    else:
        render_multi_vehicle_stage08_result(od_df, selected_vehicle_ids, track_cache_path)


def render_single_vehicle_stage08_result(od_df, vehicle_id, track_cache_path):
    valid_orders = _stage08_frontend_valid_orders(od_df, limit=10, track_cache_path=track_cache_path)
    if len(valid_orders) == 0:
        st.warning(f"车辆 {vehicle_id} 当天没有可分析的已完成订单。")
        return

    selected_order = valid_orders.iloc[0]
    analysis_key = f"single|{_stage08_order_key(selected_order)}"
    if st.session_state.get("stage08_auto_single_key") != analysis_key:
        with st.spinner("正在计算该车辆首条有效订单的三路线对比..."):
            graph, network_meta = load_road_network()
            try:
                analysis = _stage08_analysis_for_order(graph, selected_order, track_cache_path)
                analysis["route_result"]["network"] = network_meta
                st.session_state.stage08_auto_single_analysis = analysis
                st.session_state.stage08_auto_single_key = analysis_key
            except Exception as exc:
                st.error(f"三路线分析失败: {exc}")
                return

    analysis = st.session_state.get("stage08_auto_single_analysis")
    st.caption(f"展示订单: {_stage08_order_option_label(selected_order, 0)}")
    render_route_component_map(
        _points_from_stage08_order(selected_order),
        analysis.get("route_result") if analysis else None,
        actual_route=analysis.get("actual") if analysis else None,
        key=f"stage08_single_map_{analysis_key}",
        boundary_geojson=_load_route_boundary_geojson(),
    )
    if analysis:
        st.markdown("### 历史订单分析")
        st.caption("红色为历史实际路线，蓝色为距离最短路线，绿色为基准最快路线。")
        display_stage08_metrics(analysis)


def render_multi_vehicle_stage08_result(od_df, selected_vehicle_ids, track_cache_path):
    batch_limit = min(10, max(5, len(selected_vehicle_ids)))
    valid_orders = _stage08_frontend_valid_orders(od_df, limit=batch_limit, track_cache_path=track_cache_path)
    if len(valid_orders) == 0:
        st.warning("所选车辆当天没有可分析的已完成订单。")
        return
    if len(valid_orders) < 5:
        st.warning(f"当前仅找到 {len(valid_orders)} 条可分析订单，少于建议的5条。")

    batch_key = "batch|" + "|".join(_stage08_order_key(row) for _, row in valid_orders.iterrows())
    if st.session_state.get("stage08_auto_batch_key") != batch_key:
        rows = []
        analyses = []
        with st.spinner("正在计算多车辆订单三路线并生成统计..."):
            graph, _network_meta = load_road_network()
            for _, order in valid_orders.iterrows():
                try:
                    analysis = _stage08_analysis_for_order(graph, order, track_cache_path)
                    row = analysis["metrics"].copy()
                    row["success"] = True
                    row["error"] = ""
                    analyses.append(analysis)
                except Exception as exc:
                    normalized = normalize_order(order)
                    row = {
                        "order_id": normalized["id"],
                        "vehicle_id": normalized["vehicle_id"],
                        "success": False,
                        "error": str(exc),
                    }
                rows.append(row)
        results = pd.DataFrame(rows)
        st.session_state.stage08_auto_batch_results = results
        st.session_state.stage08_auto_batch_analyses = analyses
        st.session_state.stage08_auto_batch_key = batch_key

    results = st.session_state.get("stage08_auto_batch_results", pd.DataFrame())
    analyses = st.session_state.get("stage08_auto_batch_analyses", [])

    success_results = results[results["success"] == True].copy() if len(results) and "success" in results.columns else pd.DataFrame()
    top_analysis = None
    if len(success_results) > 0 and analyses:
        top_order_id = success_results.sort_values("detour_ratio", ascending=False).iloc[0]["order_id"]
        top_analysis = next((item for item in analyses if str(item["metrics"].get("order_id")) == str(top_order_id)), analyses[0])
        top_metrics = top_analysis.get("metrics", {})
        top_vehicle = top_metrics.get("vehicle_id", "")
        top_ratio = _format_ratio(top_metrics.get("detour_ratio"))
        st.markdown(f"### 绕行比例最高订单：车辆 {top_vehicle}，订单 {top_order_id}，绕行比例 {top_ratio}")
        render_route_component_map(
            _points_from_stage08_order(top_analysis["order"]),
            top_analysis.get("route_result"),
            actual_route=top_analysis.get("actual"),
            key=f"stage08_top_detour_map_{top_order_id}",
            boundary_geojson=_load_route_boundary_geojson(),
        )

    st.markdown("### 历史订单分析")
    st.caption("红色为历史实际路线，蓝色为距离最短路线，绿色为基准最快路线。")
    display_stage08_batch_results(results)
    if top_analysis is not None:
        display_stage08_metrics(top_analysis)


def _stage08_order_key(order):
    normalized = normalize_order(order)
    return f"{normalized['vehicle_id']}|{pd.Timestamp(normalized['pickup_time']).isoformat()}|{pd.Timestamp(normalized['dropoff_time']).isoformat()}"


def _stage08_order_option_label(order, idx):
    normalized = normalize_order(order)
    return (
        f"订单 {idx + 1} | 车辆 {normalized['vehicle_id']} | "
        f"{pd.Timestamp(normalized['pickup_time']).strftime('%H:%M:%S')} - "
        f"{pd.Timestamp(normalized['dropoff_time']).strftime('%H:%M:%S')}"
    )


def display_stage08_metrics(analysis):
    metrics = analysis["metrics"]
    st.markdown("### 订单指标")
    metric_cols = st.columns(4)
    metric_cols[0].metric("历史实际距离", f"{metrics['actual_distance_m'] / 1000:.2f} km")
    metric_cols[1].metric("历史实际耗时", f"{metrics['actual_duration_s'] / 60:.1f} 分钟")
    metric_cols[2].metric("绕行比例", _format_ratio(metrics.get("detour_ratio")))
    metric_cols[3].metric("最快重合率", _format_ratio(metrics.get("fastest_overlap_rate")))

    rows = [
        {
            "路线": "历史实际",
            "距离(km)": round(metrics["actual_distance_m"] / 1000, 3),
            "耗时/成本(分钟)": round(metrics["actual_duration_s"] / 60, 2),
            "道路边数": metrics.get("actual_used_edges", 0),
            "说明": metrics.get("actual_source", "校正车辆轨迹缓存，连续重复边压缩"),
        },
        {
            "路线": "最短距离",
            "距离(km)": round(metrics["shortest_distance_m"] / 1000, 3) if pd.notna(metrics.get("shortest_distance_m")) else None,
            "耗时/成本(分钟)": round(metrics["shortest_route_cost_s"] / 60, 2) if pd.notna(metrics.get("shortest_route_cost_s")) else None,
            "道路边数": metrics.get("shortest_edge_count", 0),
            "说明": "同一pickup_node/dropoff_node，length权重",
        },
        {
            "路线": "基准最快",
            "距离(km)": round(metrics["fastest_distance_m"] / 1000, 3) if pd.notna(metrics.get("fastest_distance_m")) else None,
            "耗时/成本(分钟)": round(metrics["fastest_route_cost_s"] / 60, 2) if pd.notna(metrics.get("fastest_route_cost_s")) else None,
            "道路边数": metrics.get("fastest_edge_count", 0),
            "说明": "同一pickup_node/dropoff_node，route_cost权重",
        },
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        f"实际轨迹点数: {metrics.get('raw_track_points', 0)} | "
        f"清洗后点数: {metrics.get('cleaned_track_points', metrics.get('raw_track_points', 0))} | "
        f"过滤跳点: {metrics.get('gps_outlier_points', 0)} | "
        f"连续去重后边数: {metrics.get('actual_point_edges', 0)} | "
        f"道路补桥: {metrics.get('bridged_road_segments', 0)} | "
        f"跳过异常转移: {metrics.get('skipped_road_segments', 0)} | "
        f"起终点节点: {metrics.get('pickup_node')} -> {metrics.get('dropoff_node')}"
    )
    if metrics.get("actual_has_break"):
        st.warning("历史实际路线存在路网边缺失或断裂，已在指标中标记。")


def display_stage08_batch_results(results):
    success_df = results[results["success"] == True].copy() if "success" in results.columns else pd.DataFrame()
    metric_cols = st.columns(4)
    metric_cols[0].metric("订单总数", len(results))
    metric_cols[1].metric("成功订单", len(success_df))
    metric_cols[2].metric("失败订单", len(results) - len(success_df))
    metric_cols[3].metric(
        "平均绕行比例",
        _format_ratio(success_df["detour_ratio"].mean()) if len(success_df) and "detour_ratio" in success_df.columns else "无",
    )

    if len(success_df) > 0:
        chart_df = success_df.copy()
        chart_df["order_label"] = chart_df["vehicle_id"].astype(str) + "-" + chart_df["order_id"].astype(str)
        chart_df["actual_distance_km"] = pd.to_numeric(chart_df.get("actual_distance_m"), errors="coerce") / 1000.0
        chart_df["detour_ratio_pct"] = pd.to_numeric(chart_df.get("detour_ratio"), errors="coerce") * 100.0
        chart_df["fastest_overlap_pct"] = pd.to_numeric(chart_df.get("fastest_overlap_rate"), errors="coerce") * 100.0
        chart_df = chart_df.replace([math.inf, -math.inf], pd.NA).dropna(
            subset=["actual_distance_km", "detour_ratio_pct", "fastest_overlap_pct"],
            how="all",
        )

        if len(chart_df) > 0:
            top_chart_df = chart_df.sort_values("detour_ratio_pct", ascending=False).head(10)
            chart_cols = st.columns([1.25, 1, 1])
            with chart_cols[0]:
                st.markdown("#### 绕行比例排行")
                detour_chart = (
                    alt.Chart(top_chart_df)
                    .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                    .encode(
                        x=alt.X("detour_ratio_pct:Q", title="绕行比例(%)"),
                        y=alt.Y("order_label:N", sort="-x", title="车辆-订单"),
                        color=alt.Color("vehicle_id:N", title="车辆"),
                        tooltip=[
                            alt.Tooltip("vehicle_id:N", title="车辆"),
                            alt.Tooltip("order_id:N", title="订单"),
                            alt.Tooltip("detour_ratio_pct:Q", title="绕行比例(%)", format=".1f"),
                            alt.Tooltip("actual_distance_km:Q", title="实际距离(km)", format=".2f"),
                        ],
                    )
                )
                st.altair_chart(detour_chart, use_container_width=True)

            with chart_cols[1]:
                st.markdown("#### 实际距离分布")
                distance_chart = (
                    alt.Chart(chart_df.dropna(subset=["actual_distance_km"]))
                    .mark_bar(color="#2563eb")
                    .encode(
                        x=alt.X("actual_distance_km:Q", bin=alt.Bin(maxbins=12), title="实际距离(km)"),
                        y=alt.Y("count():Q", title="订单数"),
                        tooltip=[alt.Tooltip("count():Q", title="订单数")],
                    )
                )
                st.altair_chart(distance_chart, use_container_width=True)

            with chart_cols[2]:
                st.markdown("#### 路线重合率分布")
                overlap_chart = (
                    alt.Chart(chart_df.dropna(subset=["fastest_overlap_pct"]))
                    .mark_bar(color="#16a34a")
                    .encode(
                        x=alt.X("fastest_overlap_pct:Q", bin=alt.Bin(maxbins=10), title="重合率(%)"),
                        y=alt.Y("count():Q", title="订单数"),
                        tooltip=[alt.Tooltip("count():Q", title="订单数")],
                    )
                )
                st.altair_chart(overlap_chart, use_container_width=True)

        top_cols = [
            col for col in [
                "order_id",
                "vehicle_id",
                "detour_ratio",
                "fastest_overlap_rate",
                "actual_distance_m",
                "shortest_distance_m",
            ]
            if col in success_df.columns
        ]
        st.markdown("#### 绕行比例最高的3个订单")
        st.dataframe(success_df.sort_values("detour_ratio", ascending=False).head(3)[top_cols], width="stretch", hide_index=True)

    display_cols = [
        col for col in [
            "success",
            "order_id",
            "vehicle_id",
            "actual_distance_m",
            "shortest_distance_m",
            "fastest_distance_m",
            "detour_ratio",
            "fastest_overlap_rate",
            "error",
        ]
        if col in results.columns
    ]
    with st.expander("查看批量明细", expanded=False):
        st.dataframe(results[display_cols], width="stretch", hide_index=True)

def _format_ratio(value):
    if value is None or pd.isna(value):
        return "无"
    return f"{float(value) * 100:.1f}%"


def render_map_selection_mode(payload, speed_vehicle_ids):
    """地图选点模式。控制按钮由左侧控制台触发，主区域只保留地图与结果。"""
    if st.session_state.get("route_map_reset_requested"):
        st.session_state["route_map_reset_requested"] = False
        st.session_state.route_points = []
        st.session_state.route_result = None
        if 'map_view_center' in st.session_state:
            del st.session_state.map_view_center
        if 'map_view_zoom' in st.session_state:
            del st.session_state.map_view_zoom

    if st.session_state.get("route_map_calculate_requested"):
        st.session_state["route_map_calculate_requested"] = False
        can_calculate = len(st.session_state.route_points) == 2
        if can_calculate:
            origin = st.session_state.route_points[0]
            dest = st.session_state.route_points[1]
            with st.spinner("正在生成全日道路基准速度并计算最短/最快路线..."):
                route_result = plan_baseline_routes_between_points(
                    origin['lat'],
                    origin['lng'],
                    dest['lat'],
                    dest['lng'],
                    vehicle_ids=tuple(speed_vehicle_ids or []),
                    query_date=payload["start_time"].date(),
                )
            if route_result.get("success"):
                st.session_state.route_result = route_result
                st.success("路线计算完成。")
            else:
                st.error(f"路线计算失败: {route_result.get('error', '未知错误')}")
        else:
            st.warning("请先在地图上选择起点和终点。")

    click_payload = render_route_component_map(
        st.session_state.route_points,
        st.session_state.route_result,
        key=f"route_map_{len(st.session_state.route_points)}_{st.session_state.get('route_result') is not None}",
        boundary_geojson=_load_route_boundary_geojson(),
    )

    # 处理地图点击事件
    if isinstance(click_payload, dict) and "lat" in click_payload and "lng" in click_payload:
        clicked_lat = float(click_payload["lat"])
        clicked_lng = float(click_payload["lng"])

        # 检查是否是新的点击
        last_click = st.session_state.get('last_route_click', {})
        current_click = str(click_payload.get("nonce") or f"{clicked_lat}_{clicked_lng}")

        if last_click != current_click:
            st.session_state.last_route_click = current_click

            # 如果已有2个点，清空重新开始
            if len(st.session_state.route_points) >= 2:
                st.session_state.route_points = []
                st.session_state.route_result = None
                # 清除地图视图状态，重置为默认视图
                if 'map_view_center' in st.session_state:
                    del st.session_state.map_view_center
                if 'map_view_zoom' in st.session_state:
                    del st.session_state.map_view_zoom

            # 添加新点
            st.session_state.route_points.append({
                'lat': clicked_lat,
                'lng': clicked_lng
            })

            st.rerun()

    # 显示路线统计信息
    if st.session_state.route_result and st.session_state.route_result.get("success"):
        display_route_map_and_stats(None)


def display_route_map_and_stats(selected_od=None):
    """显示路线地图和统计信息"""
    st.markdown("---")
    st.markdown("### 路线统计")

    result = st.session_state.route_result
    shortest = result["shortest"]
    fastest = result["fastest"]

    # 指标卡片
    metric_cols = st.columns(4)
    metric_cols[0].metric("最短距离", f"{shortest['distance_m'] / 1000:.2f} km")
    metric_cols[1].metric("最快路线距离", f"{fastest['distance_m'] / 1000:.2f} km")
    metric_cols[2].metric("预计时间", f"{fastest['route_cost_s'] / 60:.1f} 分钟")
    metric_cols[3].metric("可靠速度边", result.get("speed_meta", {}).get("reliable_edges", 0))

    # 详细统计表
    summary_df = pd.DataFrame([
        {
            "路线": "最短距离",
            "道路边数": shortest["edge_count"],
            "距离(km)": round(shortest["distance_m"] / 1000, 3),
            "基准成本(分钟)": round(shortest["route_cost_s"] / 60, 2),
            "节点数": len(shortest["nodes"]),
        },
        {
            "路线": "基准最快",
            "道路边数": fastest["edge_count"],
            "距离(km)": round(fastest["distance_m"] / 1000, 3),
            "基准成本(分钟)": round(fastest["route_cost_s"] / 60, 2),
            "节点数": len(fastest["nodes"]),
        },
    ])
    st.dataframe(summary_df, width="stretch", hide_index=True)

    # 如果是从 OD 选择的，显示对比信息
    if selected_od is not None:
        st.markdown("#### 实际 vs 规划对比")
        compare_cols = st.columns(3)

        actual_distance = selected_od.get('OD_Dist_km', 0)
        planned_distance = fastest['distance_m'] / 1000
        distance_diff = actual_distance - planned_distance

        compare_cols[0].metric(
            "历史实际距离",
            f"{actual_distance:.2f} km",
            delta=f"{distance_diff:+.2f} km" if actual_distance > 0 else None
        )
        compare_cols[1].metric(
            "规划最快距离",
            f"{planned_distance:.2f} km"
        )

        if actual_distance > 0:
            detour_ratio = (distance_diff / planned_distance) * 100
            compare_cols[2].metric(
                "绕路率",
                f"{detour_ratio:+.1f}%",
                delta="可能绕路" if detour_ratio > 10 else "路线合理"
            )

    # 元信息
    speed_meta = result.get("speed_meta", {})
    st.caption(
        f"路网节点: {result.get('network', {}).get('nodes', 0)} | "
        f"路网边数: {result.get('network', {}).get('edges', 0)} | "
        f"速度样本边: {speed_meta.get('edge_rows', 0)} | "
        f"可靠速度边: {speed_meta.get('reliable_edges', 0)}"
    )
