#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math

import pandas as pd

from map_plotter import (
    CONFIG,
    apply_baseline_route_cost,
    build_edge_baseline_speed_cache,
    haversine_distance,
    nearest_road_node,
    plan_baseline_routes,
)


OD_COMPLETED_CACHE_PATH = "cache/od_endpoints_completed.parquet"
EDGE_SPEED_CACHE_PATH = "cache/edge_baseline_speed.parquet"


def write_dataframe_cache(df, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".parquet"):
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            csv_path = os.path.splitext(path)[0] + ".csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
            return csv_path
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def read_dataframe_cache(path):
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _node_lat_lng(graph, node):
    attrs = graph.nodes[node]
    lng = attrs.get("x", attrs.get("lon", attrs.get("lng")))
    lat = attrs.get("y", attrs.get("lat"))
    if lng is None or lat is None:
        raise ValueError(f"路网节点 {node} 缺少坐标。")
    return float(lat), float(lng)


def snap_point_to_graph(graph, lat, lng):
    node = nearest_road_node(graph, lng, lat)
    snapped_lat, snapped_lng = _node_lat_lng(graph, node)
    return {
        "lat": float(lat),
        "lng": float(lng),
        "node": node,
        "snapped_lat": snapped_lat,
        "snapped_lng": snapped_lng,
        "snap_distance_m": float(haversine_distance(float(lat), float(lng), snapped_lat, snapped_lng) * 1000.0),
    }


def _build_node_spatial_index(graph):
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None

    nodes = []
    coords = []
    lat_values = []
    for node, attrs in graph.nodes(data=True):
        lng = attrs.get("x", attrs.get("lon", attrs.get("lng")))
        lat = attrs.get("y", attrs.get("lat"))
        if lng is None or lat is None:
            continue
        lat = float(lat)
        lng = float(lng)
        nodes.append(node)
        lat_values.append(lat)
        coords.append((lng, lat))
    if not coords:
        return None
    scale = math.cos(math.radians(sum(lat_values) / len(lat_values)))
    scaled = [(lng * scale, lat) for lng, lat in coords]
    return {
        "tree": cKDTree(scaled),
        "nodes": nodes,
        "scale": scale,
    }


def _snap_point_to_graph_fast(graph, lat, lng, spatial_index=None):
    if not spatial_index:
        return snap_point_to_graph(graph, lat, lng)
    _dist, pos = spatial_index["tree"].query((float(lng) * spatial_index["scale"], float(lat)))
    node = spatial_index["nodes"][int(pos)]
    snapped_lat, snapped_lng = _node_lat_lng(graph, node)
    return {
        "lat": float(lat),
        "lng": float(lng),
        "node": node,
        "snapped_lat": snapped_lat,
        "snapped_lng": snapped_lng,
        "snap_distance_m": float(haversine_distance(float(lat), float(lng), snapped_lat, snapped_lng) * 1000.0),
    }


def _snap_points_to_graph_fast(graph, lats, lngs, spatial_index=None):
    if not spatial_index:
        return [_snap_point_to_graph_fast(graph, lat, lng, spatial_index=None) for lat, lng in zip(lats, lngs)]
    query_points = [
        (float(lng) * spatial_index["scale"], float(lat))
        for lat, lng in zip(lats, lngs)
    ]
    _distances, positions = spatial_index["tree"].query(query_points)
    snapped = []
    for lat, lng, pos in zip(lats, lngs, positions):
        node = spatial_index["nodes"][int(pos)]
        snapped_lat, snapped_lng = _node_lat_lng(graph, node)
        snapped.append(
            {
                "lat": float(lat),
                "lng": float(lng),
                "node": node,
                "snapped_lat": snapped_lat,
                "snapped_lng": snapped_lng,
                "snap_distance_m": float(haversine_distance(float(lat), float(lng), snapped_lat, snapped_lng) * 1000.0),
            }
        )
    return snapped


def _vehicle_column(df):
    for col in ["vehicle_id", "id", "O_TAXI_ID"]:
        if col in df.columns:
            return col
    return None


def _vehicle_id_token(value):
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value).strip()
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer() and text.replace(".", "", 1).isdigit():
            return str(int(number))
    except ValueError:
        pass
    return text


def _empty_corrected_df():
    return pd.DataFrame(
        columns=[
            "vehicle_id",
            "time",
            "matched_lat",
            "matched_lon",
            "matched_node",
            "edge_u",
            "edge_v",
            "edge_key",
        ]
    )


def _nearest_corrected_row(corrected, vehicle_id, event_time, tolerance):
    if corrected is None or len(corrected) == 0:
        return None
    vehicle_col = _vehicle_column(corrected)
    if vehicle_col is None or "time" not in corrected.columns:
        return None
    work = corrected[corrected[vehicle_col].astype(str) == str(vehicle_id)].copy()
    if len(work) == 0:
        return None
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    event_time = pd.Timestamp(event_time)
    work["_delta"] = (work["time"] - event_time).abs()
    work = work.dropna(subset=["_delta"]).sort_values("_delta")
    if len(work) == 0 or work.iloc[0]["_delta"] > tolerance:
        return None
    return work.iloc[0]


def _prepare_corrected_groups(corrected):
    if corrected is None or len(corrected) == 0:
        return {}
    vehicle_col = _vehicle_column(corrected)
    if vehicle_col is None or "time" not in corrected.columns:
        return {}
    work = corrected.copy()
    work["_vehicle_token"] = work[vehicle_col].map(_vehicle_id_token)
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    work = work.dropna(subset=["time"])
    groups = {}
    for vehicle_id, group in work.groupby("_vehicle_token", sort=False):
        groups[vehicle_id] = group.sort_values("time").reset_index(drop=True)
    return groups


def _nearest_corrected_row_from_groups(groups, vehicle_id, event_time, tolerance):
    work = groups.get(_vehicle_id_token(vehicle_id))
    if work is None or len(work) == 0:
        return None
    event_time = pd.Timestamp(event_time)
    deltas = (work["time"] - event_time).abs()
    if len(deltas) == 0:
        return None
    nearest_idx = deltas.idxmin()
    if pd.isna(deltas.loc[nearest_idx]) or deltas.loc[nearest_idx] > tolerance:
        return None
    return work.loc[nearest_idx]


def _fill_endpoint_from_corrected(output, row_index, prefix, od_row, corrected_row):
    raw_lat = float(od_row[f"{prefix}_lat"])
    raw_lng = float(od_row[f"{prefix}_lng"])
    matched_lat = float(corrected_row["matched_lat"])
    matched_lng = float(corrected_row["matched_lon"])
    output.at[row_index, f"{prefix}_corrected_lat"] = matched_lat
    output.at[row_index, f"{prefix}_corrected_lng"] = matched_lng
    output.at[row_index, f"{prefix}_matched_node"] = corrected_row.get("matched_node")
    output.at[row_index, f"{prefix}_edge_u"] = corrected_row.get("edge_u")
    output.at[row_index, f"{prefix}_edge_v"] = corrected_row.get("edge_v")
    output.at[row_index, f"{prefix}_edge_key"] = corrected_row.get("edge_key")
    output.at[row_index, f"{prefix}_snap_distance_m"] = float(haversine_distance(raw_lat, raw_lng, matched_lat, matched_lng) * 1000.0)
    output.at[row_index, f"{prefix}_endpoint_source"] = "corrected_track"
    output.at[row_index, f"{prefix}_endpoint_message"] = "已关联时间容差内校正轨迹点。"


def _fill_endpoint_from_nearest(output, row_index, prefix, od_row, graph, spatial_index=None):
    raw_lat = float(od_row[f"{prefix}_lat"])
    raw_lng = float(od_row[f"{prefix}_lng"])
    snapped = _snap_point_to_graph_fast(graph, raw_lat, raw_lng, spatial_index=spatial_index)
    output.at[row_index, f"{prefix}_corrected_lat"] = snapped["snapped_lat"]
    output.at[row_index, f"{prefix}_corrected_lng"] = snapped["snapped_lng"]
    output.at[row_index, f"{prefix}_matched_node"] = snapped["node"]
    output.at[row_index, f"{prefix}_edge_u"] = pd.NA
    output.at[row_index, f"{prefix}_edge_v"] = pd.NA
    output.at[row_index, f"{prefix}_edge_key"] = pd.NA
    output.at[row_index, f"{prefix}_snap_distance_m"] = snapped["snap_distance_m"]
    output.at[row_index, f"{prefix}_endpoint_source"] = "nearest_node"
    output.at[row_index, f"{prefix}_endpoint_message"] = "未找到时间容差内校正轨迹点，已吸附到最近路网节点。"


def _fill_nearest_fallbacks_batch(output, fallbacks, graph, spatial_index=None):
    if not fallbacks:
        return 0
    filled = 0
    for prefix in ["O", "D"]:
        current = [item for item in fallbacks if item["prefix"] == prefix]
        if not current:
            continue
        snapped_points = _snap_points_to_graph_fast(
            graph,
            [item["lat"] for item in current],
            [item["lng"] for item in current],
            spatial_index=spatial_index,
        )
        indices = [item["idx"] for item in current]
        output.loc[indices, f"{prefix}_corrected_lat"] = [item["snapped_lat"] for item in snapped_points]
        output.loc[indices, f"{prefix}_corrected_lng"] = [item["snapped_lng"] for item in snapped_points]
        output.loc[indices, f"{prefix}_matched_node"] = [item["node"] for item in snapped_points]
        output.loc[indices, f"{prefix}_edge_u"] = pd.NA
        output.loc[indices, f"{prefix}_edge_v"] = pd.NA
        output.loc[indices, f"{prefix}_edge_key"] = pd.NA
        output.loc[indices, f"{prefix}_snap_distance_m"] = [item["snap_distance_m"] for item in snapped_points]
        output.loc[indices, f"{prefix}_endpoint_source"] = "nearest_node"
        output.loc[indices, f"{prefix}_endpoint_message"] = "未找到时间容差内校正轨迹点，已吸附到最近路网节点。"
        filled += len(current)
    return filled


def complete_od_endpoints(od_df, corrected_track, graph, time_tolerance="5min", save_path=None):
    if od_df is None:
        od_df = pd.DataFrame()
    required = {"O_TAXI_ID", "O_time", "O_lat", "O_lng", "D_time", "D_lat", "D_lng"}
    missing = required - set(od_df.columns)
    if missing:
        raise ValueError(f"OD表缺少字段: {sorted(missing)}")
    if graph is None:
        raise ValueError("路网未加载，无法补全 OD 端点。")

    output = od_df.copy()
    if "O_TAXI_ID" in output.columns:
        output["O_TAXI_ID"] = output["O_TAXI_ID"].map(_vehicle_id_token)
    corrected = corrected_track.copy() if corrected_track is not None else _empty_corrected_df()
    corrected_groups = _prepare_corrected_groups(corrected)
    spatial_index = _build_node_spatial_index(graph)
    tolerance = pd.Timedelta(time_tolerance)
    corrected_matches = 0
    nearest_fallbacks = 0
    failures = 0
    nearest_fallback_rows = []

    for prefix in ["O", "D"]:
        for suffix in [
            "corrected_lat",
            "corrected_lng",
            "matched_node",
            "edge_u",
            "edge_v",
            "edge_key",
            "snap_distance_m",
            "endpoint_source",
            "endpoint_message",
        ]:
            col = f"{prefix}_{suffix}"
            if col not in output.columns:
                output[col] = pd.NA

    for idx, od_row in output.iterrows():
        vehicle_id = od_row["O_TAXI_ID"]
        for prefix, time_col in [("O", "O_time"), ("D", "D_time")]:
            try:
                matched = _nearest_corrected_row_from_groups(corrected_groups, vehicle_id, od_row[time_col], tolerance)
                if matched is not None and pd.notna(matched.get("matched_lat")) and pd.notna(matched.get("matched_lon")):
                    _fill_endpoint_from_corrected(output, idx, prefix, od_row, matched)
                    corrected_matches += 1
                else:
                    nearest_fallback_rows.append(
                        {
                            "idx": idx,
                            "prefix": prefix,
                            "lat": float(od_row[f"{prefix}_lat"]),
                            "lng": float(od_row[f"{prefix}_lng"]),
                        }
                    )
            except Exception as exc:
                failures += 1
                output.at[idx, f"{prefix}_endpoint_source"] = "failed"
                output.at[idx, f"{prefix}_endpoint_message"] = str(exc)

    try:
        nearest_fallbacks = _fill_nearest_fallbacks_batch(output, nearest_fallback_rows, graph, spatial_index=spatial_index)
    except Exception as exc:
        failures += len(nearest_fallback_rows)
        for item in nearest_fallback_rows:
            output.at[item["idx"], f"{item['prefix']}_endpoint_source"] = "failed"
            output.at[item["idx"], f"{item['prefix']}_endpoint_message"] = str(exc)

    meta = {
        "success": failures == 0,
        "rows": int(len(output)),
        "endpoints": int(len(output) * 2),
        "corrected_matches": int(corrected_matches),
        "nearest_fallbacks": int(nearest_fallbacks),
        "failures": int(failures),
    }
    if save_path:
        meta["cache_path"] = write_dataframe_cache(output, save_path)
    return output, meta


def load_or_build_speed_cache(matched_track, graph, min_samples=3, cache_path=EDGE_SPEED_CACHE_PATH):
    existing_paths = [cache_path]
    if cache_path.lower().endswith(".parquet"):
        existing_paths.append(os.path.splitext(cache_path)[0] + ".csv")
    for path in existing_paths:
        if os.path.exists(path):
            return read_dataframe_cache(path), {"success": True, "cache_path": path, "cache_hit": True}
    speed_stats, meta = build_edge_baseline_speed_cache(matched_track, graph, min_samples=min_samples)
    meta["cache_path"] = write_dataframe_cache(speed_stats, cache_path)
    meta["cache_hit"] = False
    return speed_stats, meta


def load_corrected_track_cache(cache_dir="cache/road_corrected", vehicle_ids=None):
    if not os.path.isdir(cache_dir):
        return _empty_corrected_df()
    wanted = {str(vehicle_id) for vehicle_id in vehicle_ids or [] if str(vehicle_id).strip()}
    frames = []
    for filename in sorted(os.listdir(cache_dir)):
        if not filename.endswith(".csv"):
            continue
        vehicle_token = filename.split("_", 1)[0]
        if wanted and vehicle_token not in wanted:
            continue
        path = os.path.join(cache_dir, filename)
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if len(frame) > 0:
            frames.append(frame)
    if not frames:
        return _empty_corrected_df()
    return pd.concat(frames, ignore_index=True)


def prepare_graph_route_costs(graph, speed_stats=None, highway_median_speed=None, min_samples=3):
    if speed_stats is None:
        speed_stats = pd.DataFrame(columns=["edge_u", "edge_v", "edge_key", "avg_speed", "sample_count"])
    return apply_baseline_route_cost(
        graph,
        speed_stats,
        highway_median_speed=highway_median_speed or {},
        default_speed_kph=CONFIG["DEFAULT_HIGHWAY_SPEED_KPH"],
        min_samples=min_samples,
    )


def _graph_has_route_cost(graph):
    for _, _, _, data in graph.edges(keys=True, data=True):
        try:
            if float(data.get("route_cost", 0.0) or 0.0) <= 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def plan_dual_routes_between_points(graph, origin_lat, origin_lng, dest_lat, dest_lng, speed_stats=None, highway_median_speed=None):
    if graph is None:
        return {"success": False, "error": "路网未加载。"}
    try:
        if not _graph_has_route_cost(graph):
            prepare_graph_route_costs(graph, speed_stats=speed_stats, highway_median_speed=highway_median_speed)
        origin = snap_point_to_graph(graph, origin_lat, origin_lng)
        destination = snap_point_to_graph(graph, dest_lat, dest_lng)
        result = plan_baseline_routes(graph, origin["node"], destination["node"])
        result.update({"origin": origin, "destination": destination})
        return result
    except Exception as exc:
        return {"success": False, "error": f"路线计算失败: {exc}"}


def route_coordinates_from_od_row(od_row):
    def endpoint(prefix):
        lat = od_row.get(f"{prefix}_corrected_lat")
        lng = od_row.get(f"{prefix}_corrected_lng")
        source = od_row.get(f"{prefix}_endpoint_source", "raw")
        if pd.isna(lat) or pd.isna(lng):
            lat = od_row.get(f"{prefix}_lat")
            lng = od_row.get(f"{prefix}_lng")
            source = "raw"
        return {"lat": float(lat), "lng": float(lng), "source": source}

    return {"origin": endpoint("O"), "destination": endpoint("D")}
