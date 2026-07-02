#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 08：历史订单三路线对比分析。

本模块只读取已经生成的缓存：
- 校正 OD 缓存
- 单车校正轨迹缓存

严禁在本阶段内重新执行 GPS 原始大表读取、HMM 匹配、路网校正或最近邻吸附。
三条路线的起终点节点必须直接来自校正 OD 缓存。
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_OD_CACHE = "cache/od_corrected.parquet"
DEFAULT_TRACK_CACHE_DIR = "cache/vehicle_corrected"
DEFAULT_GRAPH_PATH = "shenzhen_drive.pkl"
DEFAULT_OUTPUT_DIR = "output/stage08"
DEFAULT_SPEED_KPH = 30.0
OD_CACHE_FALLBACKS = [
    DEFAULT_OD_CACHE,
    "cache/od_endpoints_completed.parquet",
    "cache/od_endpoints_completed.csv",
]
TRACK_CACHE_FALLBACKS = [
    DEFAULT_TRACK_CACHE_DIR,
    "cache/road_corrected",
]


class Stage08DataError(RuntimeError):
    """阶段 08 缓存数据不满足处理条件。"""


@dataclass
class SingleOrderResult:
    """单订单处理结果，便于脚本和 Notebook 分段调试。"""

    success: bool
    order_id: str
    vehicle_id: str
    metrics: dict
    output_html: str | None = None
    error: str | None = None


def vehicle_id_token(value) -> str:
    """将 123.0 这类车辆 ID 规范成 123，避免缓存文件名匹配失败。"""
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


def _is_present(value) -> bool:
    return value is not None and not pd.isna(value)


def _first_present(row, names, default=None):
    for name in names:
        try:
            value = row.get(name, default)
        except AttributeError:
            value = row[name] if name in row else default
        if _is_present(value):
            return value
    return default


def normalize_order(order: pd.Series | dict) -> pd.Series:
    """
    将不同阶段缓存列名统一为阶段 08 的标准列名。

    支持规格列名：
    id, vehicle_id, pickup_time, dropoff_time, pickup_node, dropoff_node,
    pickup_matched_lat, pickup_matched_lon, dropoff_matched_lat, dropoff_matched_lon

    同时兼容当前项目缓存列名：
    O_TAXI_ID, O_time, D_time, O_matched_node, D_matched_node,
    O_corrected_lat, O_corrected_lng, D_corrected_lat, D_corrected_lng
    """
    row = pd.Series(order).copy()
    normalized = row.to_dict()
    normalized["id"] = _first_present(row, ["id", "order_id", "OD_ID"], row.name)
    normalized["vehicle_id"] = vehicle_id_token(_first_present(row, ["vehicle_id", "O_TAXI_ID", "taxi_id"]))
    normalized["pickup_time"] = pd.to_datetime(_first_present(row, ["pickup_time", "O_time"]), errors="coerce")
    normalized["dropoff_time"] = pd.to_datetime(_first_present(row, ["dropoff_time", "D_time"]), errors="coerce")
    normalized["pickup_node"] = _first_present(row, ["pickup_node", "O_matched_node", "O_node"])
    normalized["dropoff_node"] = _first_present(row, ["dropoff_node", "D_matched_node", "D_node"])
    normalized["pickup_matched_lat"] = _first_present(row, ["pickup_matched_lat", "O_corrected_lat", "O_matched_lat", "O_lat"])
    normalized["pickup_matched_lon"] = _first_present(row, ["pickup_matched_lon", "O_corrected_lng", "O_matched_lon", "O_lng"])
    normalized["dropoff_matched_lat"] = _first_present(row, ["dropoff_matched_lat", "D_corrected_lat", "D_matched_lat", "D_lat"])
    normalized["dropoff_matched_lon"] = _first_present(row, ["dropoff_matched_lon", "D_corrected_lng", "D_matched_lon", "D_lng"])
    return pd.Series(normalized)


def normalize_od_dataframe(od_df: pd.DataFrame) -> pd.DataFrame:
    """批量统一 OD 缓存字段，并确保时间列为 datetime。"""
    if od_df is None or len(od_df) == 0:
        return pd.DataFrame()
    source = od_df.copy()

    def series_from(names, default=pd.NA):
        for name in names:
            if name in source.columns:
                return source[name]
        return pd.Series([default] * len(source), index=source.index)

    if any(name in source.columns for name in ["id", "order_id", "OD_ID"]):
        id_series = series_from(["id", "order_id", "OD_ID"])
    else:
        id_series = pd.Series(source.index, index=source.index)

    normalized = source.copy()
    normalized["id"] = id_series
    normalized["vehicle_id"] = series_from(["vehicle_id", "O_TAXI_ID", "taxi_id"])
    normalized["pickup_time"] = series_from(["pickup_time", "O_time"])
    normalized["dropoff_time"] = series_from(["dropoff_time", "D_time"])
    normalized["pickup_node"] = series_from(["pickup_node", "O_matched_node", "O_node"])
    normalized["dropoff_node"] = series_from(["dropoff_node", "D_matched_node", "D_node"])
    normalized["pickup_matched_lat"] = series_from(["pickup_matched_lat", "O_corrected_lat", "O_matched_lat", "O_lat"])
    normalized["pickup_matched_lon"] = series_from(["pickup_matched_lon", "O_corrected_lng", "O_matched_lon", "O_lng"])
    normalized["dropoff_matched_lat"] = series_from(["dropoff_matched_lat", "D_corrected_lat", "D_matched_lat", "D_lat"])
    normalized["dropoff_matched_lon"] = series_from(["dropoff_matched_lon", "D_corrected_lng", "D_matched_lon", "D_lng"])
    normalized["pickup_time"] = pd.to_datetime(normalized["pickup_time"], errors="coerce")
    normalized["dropoff_time"] = pd.to_datetime(normalized["dropoff_time"], errors="coerce")
    normalized["vehicle_id"] = normalized["vehicle_id"].map(vehicle_id_token)
    return normalized


def read_dataframe_cache(path: str) -> pd.DataFrame:
    """只读取缓存文件，支持 parquet/csv。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"缓存文件不存在: {path}")
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    raise Stage08DataError(f"不支持的缓存格式: {path}")


def resolve_existing_path(path: str, fallbacks: list[str], label: str) -> str:
    """解析缓存路径；默认路径不存在时只在同类缓存候选中回退。"""
    if path and os.path.exists(path):
        return path
    for candidate in fallbacks:
        if candidate and os.path.exists(candidate):
            if path != candidate:
                LOGGER.warning("WARNING %s路径不存在，回退到缓存: %s", label, candidate)
                print(f"WARNING {label}路径不存在，回退到缓存: {candidate}")
            return candidate
    raise FileNotFoundError(f"未找到{label}缓存: {path}")


def load_od_cache(path: str) -> pd.DataFrame:
    """读取校正 OD 缓存并统一字段。"""
    LOGGER.info("使用校正OD缓存: %s", path)
    return normalize_od_dataframe(read_dataframe_cache(path))


def available_vehicle_tokens(track_cache_path: str) -> set[str]:
    """列出已有单车校正轨迹缓存的车辆 ID。"""
    if os.path.isfile(track_cache_path):
        base = os.path.basename(track_cache_path)
        return {vehicle_id_token(os.path.splitext(base)[0].split("_", 1)[0])}
    if not os.path.isdir(track_cache_path):
        return set()
    tokens = set()
    for filename in os.listdir(track_cache_path):
        lower = filename.lower()
        if not lower.endswith((".parquet", ".csv")):
            continue
        tokens.add(vehicle_id_token(os.path.splitext(filename)[0].split("_", 1)[0]))
    return {token for token in tokens if token}


def track_cache_time_windows(track_cache_path: str) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """读取每个车辆校正轨迹缓存的时间覆盖窗口。"""
    windows = {}
    if os.path.isfile(track_cache_path):
        paths = [track_cache_path]
    elif os.path.isdir(track_cache_path):
        paths = [
            os.path.join(track_cache_path, filename)
            for filename in sorted(os.listdir(track_cache_path))
            if filename.lower().endswith((".parquet", ".csv"))
        ]
    else:
        return windows

    for path in paths:
        token = vehicle_id_token(os.path.splitext(os.path.basename(path))[0].split("_", 1)[0])
        try:
            frame = read_dataframe_cache(path)
        except Exception as exc:
            LOGGER.warning("WARNING 车辆校正轨迹缓存读取失败，跳过时间窗口: %s error=%s", path, exc)
            continue
        if "time" not in frame.columns or len(frame) == 0:
            continue
        times = pd.to_datetime(frame["time"], errors="coerce").dropna()
        if len(times) == 0:
            continue
        start = pd.Timestamp(times.min())
        end = pd.Timestamp(times.max())
        if token in windows:
            old_start, old_end = windows[token]
            windows[token] = (min(old_start, start), max(old_end, end))
        else:
            windows[token] = (start, end)
    return windows


def select_valid_order(od_df: pd.DataFrame) -> pd.Series:
    """
    模块 A：选择一条有效订单。

    有效条件：
    - pickup_node 和 dropoff_node 均非空
    - dropoff_time > pickup_time
    """
    normalized = normalize_od_dataframe(od_df)
    if len(normalized) == 0:
        raise Stage08DataError("校正OD缓存为空，无法选择有效订单。")

    valid = normalized[
        normalized["pickup_node"].notna()
        & normalized["dropoff_node"].notna()
        & normalized["pickup_time"].notna()
        & normalized["dropoff_time"].notna()
        & (normalized["dropoff_time"] > normalized["pickup_time"])
    ].copy()

    if len(valid) == 0:
        raise Stage08DataError("未找到 pickup_node/dropoff_node 完整且时间有效的订单。")

    order = valid.iloc[0]
    LOGGER.info(
        "选中订单: order_id=%s vehicle_id=%s pickup_time=%s dropoff_time=%s pickup_node=%s dropoff_node=%s",
        order["id"],
        order["vehicle_id"],
        order["pickup_time"],
        order["dropoff_time"],
        order["pickup_node"],
        order["dropoff_node"],
    )
    print(
        "选中订单 "
        f"ID={order['id']} | 车辆={order['vehicle_id']} | "
        f"上车={order['pickup_time']} | 下车={order['dropoff_time']} | "
        f"节点={order['pickup_node']}->{order['dropoff_node']}"
    )
    return order


def _resolve_vehicle_cache_path(vehicle_id, track_cache_path: str) -> str:
    """根据车辆 ID 定位单车校正轨迹缓存。"""
    token = vehicle_id_token(vehicle_id)
    if os.path.isfile(track_cache_path):
        return track_cache_path
    if not os.path.isdir(track_cache_path):
        raise FileNotFoundError(f"车辆校正轨迹缓存目录不存在: {track_cache_path}")

    exact_candidates = [
        os.path.join(track_cache_path, f"{token}.parquet"),
        os.path.join(track_cache_path, f"{token}.csv"),
    ]
    for path in exact_candidates:
        if os.path.exists(path):
            return path

    prefix = f"{token}_"
    for filename in sorted(os.listdir(track_cache_path)):
        if filename.startswith(prefix) and filename.lower().endswith((".parquet", ".csv")):
            return os.path.join(track_cache_path, filename)

    raise FileNotFoundError(f"未找到车辆 {token} 的校正轨迹缓存: {track_cache_path}")


def _normalize_track_columns(track_df: pd.DataFrame) -> pd.DataFrame:
    """统一校正轨迹缓存中的经纬度字段名。"""
    work = track_df.copy()
    if "matched_lng" not in work.columns and "matched_lon" in work.columns:
        work["matched_lng"] = work["matched_lon"]
    if "lng" not in work.columns and "lon" in work.columns:
        work["lng"] = work["lon"]
    if "lng" not in work.columns and "raw_lon" in work.columns:
        work["lng"] = work["raw_lon"]
    if "lat" not in work.columns and "raw_lat" in work.columns:
        work["lat"] = work["raw_lat"]
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    return work


def extract_actual_points_from_track(order, track_cache_path: str) -> list[tuple[float, float]]:
    """从单车校正轨迹缓存中提取订单时间窗内的辅助散点。"""
    normalized_order = normalize_order(order)
    vehicle_cache = _resolve_vehicle_cache_path(normalized_order["vehicle_id"], track_cache_path)
    track_df = _normalize_track_columns(read_dataframe_cache(vehicle_cache))
    pickup_time = pd.Timestamp(normalized_order["pickup_time"])
    dropoff_time = pd.Timestamp(normalized_order["dropoff_time"])
    segment = track_df[track_df["time"].between(pickup_time, dropoff_time, inclusive="both")].copy()
    lat_col = "lat" if "lat" in segment.columns else "matched_lat"
    lng_col = "lng" if "lng" in segment.columns else "matched_lng"
    if lat_col not in segment.columns or lng_col not in segment.columns:
        return []
    points = []
    for row in segment[[lat_col, lng_col]].dropna().itertuples(index=False):
        points.append((float(row[0]), float(row[1])))
    return points


def extract_actual_edges_from_track(G, order, track_cache_path: str):
    """
    模块 B：截取历史实际路线并提取道路边序列。

    关键约束：
    - 只读取对应车辆的校正轨迹缓存。
    - 只删除连续重复边，保留非连续重复边。
    """
    normalized_order = normalize_order(order)
    vehicle_id = normalized_order["vehicle_id"]
    vehicle_cache = _resolve_vehicle_cache_path(vehicle_id, track_cache_path)
    LOGGER.info("使用车辆校正轨迹缓存: %s", vehicle_cache)

    track_df = _normalize_track_columns(read_dataframe_cache(vehicle_cache))
    pickup_time = pd.Timestamp(normalized_order["pickup_time"])
    dropoff_time = pd.Timestamp(normalized_order["dropoff_time"])
    segment = track_df[track_df["time"].between(pickup_time, dropoff_time, inclusive="both")].copy()
    raw_point_count = int(len(segment))

    if raw_point_count < 2:
        raise Stage08DataError(f"轨迹数据不完整: 截取点数 {raw_point_count} < 2")

    required = ["edge_u", "edge_v", "edge_key"]
    missing = [col for col in required if col not in segment.columns]
    if missing:
        raise Stage08DataError(f"车辆校正轨迹缓存缺少道路边字段: {missing}")

    edge_series = segment[required].dropna().copy()
    if len(edge_series) == 0:
        raise Stage08DataError("轨迹数据不完整: 截取区间内道路边序列为空")

    # 只压缩连续重复边。非连续重复边代表掉头、绕圈等真实行为，必须保留。
    edge_series = edge_series[edge_series.ne(edge_series.shift()).any(axis=1)].copy()
    if len(edge_series) == 0:
        raise Stage08DataError("轨迹数据不完整: 连续重复压缩后道路边序列为空")

    edge_series["edge_u"] = edge_series["edge_u"].map(_coerce_node_id)
    edge_series["edge_v"] = edge_series["edge_v"].map(_coerce_node_id)
    edge_series["edge_key"] = edge_series["edge_key"].map(_coerce_edge_key)
    edge_series = edge_series.reset_index(drop=True)

    LOGGER.info("实际轨迹点数=%s，连续去重后边数=%s", raw_point_count, len(edge_series))
    print(f"实际轨迹点数: {raw_point_count} | 连续去重后边数: {len(edge_series)}")
    return edge_series, raw_point_count


def _coerce_node_id(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value)
    try:
        number = float(text)
        if number.is_integer():
            return int(number)
    except ValueError:
        pass
    return value


def _coerce_edge_key(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value)
    try:
        number = float(text)
        if number.is_integer():
            return int(number)
    except ValueError:
        pass
    return value


def _edge_geometry_coords(G, u, v, data) -> list[tuple[float, float]]:
    """返回 folium 使用的 (lat, lon) 坐标序列。"""
    geometry = data.get("geometry") if data else None
    if geometry is not None and hasattr(geometry, "coords"):
        return [(float(lat), float(lon)) for lon, lat in geometry.coords]
    try:
        return [
            (float(G.nodes[u]["y"]), float(G.nodes[u]["x"])),
            (float(G.nodes[v]["y"]), float(G.nodes[v]["x"])),
        ]
    except KeyError as exc:
        raise Stage08DataError(f"路网节点缺少坐标: {exc}") from exc


def compute_actual_route_geometry(G, edge_series: pd.DataFrame):
    """
    模块 C：恢复历史实际路线的道路几何并累计距离。

    返回：
    actual_distance_m, actual_lines, meta
    """
    actual_distance_m = 0.0
    actual_lines = []
    missing_edges = []

    for row in edge_series.itertuples(index=False):
        u = _coerce_node_id(row.edge_u)
        v = _coerce_node_id(row.edge_v)
        key = _coerce_edge_key(row.edge_key)
        data = G.get_edge_data(u, v, key)
        if data is None:
            LOGGER.warning("WARNING 实际路线边在路网中不存在，已跳过: (%s, %s, %s)", u, v, key)
            missing_edges.append((u, v, key))
            continue
        actual_distance_m += float(data.get("length", 0.0) or 0.0)
        actual_lines.append(_edge_geometry_coords(G, u, v, data))

    meta = {
        "has_break": bool(missing_edges),
        "missing_edges": missing_edges,
        "used_edges": int(len(actual_lines)),
    }
    if missing_edges:
        print(f"WARNING 实际路线存在断裂，缺失边数: {len(missing_edges)}")
    return float(actual_distance_m), actual_lines, meta


def ensure_route_cost(G, default_speed_kph: float = DEFAULT_SPEED_KPH):
    """若路网缺 route_cost，则用 length/default_speed 生成静态通行成本。"""
    for _u, _v, _key, data in G.edges(keys=True, data=True):
        try:
            cost = float(data.get("route_cost", 0.0) or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        if cost > 0:
            continue
        length = float(data.get("length", 0.0) or 0.0)
        speed = float(data.get("baseline_speed_kph", default_speed_kph) or default_speed_kph)
        if speed <= 0:
            speed = default_speed_kph
        data["route_cost"] = max(length / speed * 3.6, 1e-6)


def _best_edge_for_step(G, u, v, weight: str):
    edge_dict = G.get_edge_data(u, v)
    if not edge_dict:
        return None, None

    def edge_weight(item):
        _key, data = item
        value = data.get(weight, data.get("length", 0.0))
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.inf

    key, data = min(edge_dict.items(), key=edge_weight)
    return key, data


def _make_linestring(coords):
    try:
        from shapely.geometry import LineString
    except Exception:
        return None
    return LineString([(lon, lat) for lat, lon in coords])


def _route_nodes_to_gdf(G, route_nodes: list, weight: str):
    rows = []
    index = []
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        key, data = _best_edge_for_step(G, u, v, weight)
        if data is None:
            LOGGER.warning("WARNING 规划路线边在路网中不存在，已跳过: (%s, %s)", u, v)
            continue
        coords = _edge_geometry_coords(G, u, v, data)
        rows.append(
            {
                "edge_u": u,
                "edge_v": v,
                "edge_key": key,
                "length": float(data.get("length", 0.0) or 0.0),
                "route_cost": float(data.get("route_cost", data.get("length", 0.0)) or 0.0),
                "geometry": data.get("geometry") or _make_linestring(coords),
                "coordinates": coords,
            }
        )
        index.append((u, v, key))

    frame = pd.DataFrame(rows)
    frame.index = pd.MultiIndex.from_tuples(index, names=["edge_u", "edge_v", "edge_key"])
    frame.attrs["route_nodes"] = list(route_nodes)
    frame.attrs["weight"] = weight
    frame.attrs["distance_m"] = float(frame["length"].sum()) if len(frame) else 0.0
    frame.attrs["route_cost_s"] = float(frame["route_cost"].sum()) if len(frame) else 0.0

    try:
        import geopandas as gpd

        return gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326")
    except Exception:
        return frame


def _safe_shortest_path(G, origin_node, dest_node, weight: str):
    try:
        return nx.shortest_path(G, origin_node, dest_node, weight=weight)
    except nx.NetworkXNoPath:
        LOGGER.warning("WARNING 规划路线不可达: origin=%s dest=%s weight=%s", origin_node, dest_node, weight)
        print(f"WARNING 规划路线不可达: {origin_node}->{dest_node}, weight={weight}")
        return None
    except nx.NodeNotFound as exc:
        LOGGER.warning("WARNING 规划路线节点不存在: %s", exc)
        print(f"WARNING 规划路线节点不存在: {exc}")
        return None


def get_planning_routes(G, origin_node, dest_node):
    """
    模块 D：基于同一校正 OD 端点节点计算两条规划路线。

    不进行坐标吸附，不使用直线替代不可达路线。
    """
    origin_node = _coerce_node_id(origin_node)
    dest_node = _coerce_node_id(dest_node)
    ensure_route_cost(G)

    shortest_nodes = _safe_shortest_path(G, origin_node, dest_node, "length")
    fastest_nodes = _safe_shortest_path(G, origin_node, dest_node, "route_cost")

    shortest_gdf = _route_nodes_to_gdf(G, shortest_nodes, "length") if shortest_nodes else None
    fastest_gdf = _route_nodes_to_gdf(G, fastest_nodes, "route_cost") if fastest_nodes else None
    return shortest_gdf, fastest_gdf


def _route_distance(route_gdf) -> float:
    if route_gdf is None:
        return math.nan
    if "length" not in route_gdf.columns or len(route_gdf) == 0:
        return math.nan
    return float(route_gdf["length"].sum())


def _edge_tuple_set_from_frame(frame: pd.DataFrame) -> set[tuple]:
    if frame is None or len(frame) == 0:
        return set()
    if isinstance(frame.index, pd.MultiIndex):
        return set(frame.index.tolist())
    return set(zip(frame["edge_u"], frame["edge_v"], frame["edge_key"]))


def _edge_length_map(frame: pd.DataFrame) -> dict[tuple, float]:
    if frame is None or len(frame) == 0:
        return {}
    result = {}
    for edge in _edge_tuple_set_from_frame(frame):
        try:
            result[edge] = float(frame.loc[edge, "length"])
        except Exception:
            subset = frame[(frame["edge_u"] == edge[0]) & (frame["edge_v"] == edge[1]) & (frame["edge_key"] == edge[2])]
            result[edge] = float(subset["length"].iloc[0]) if len(subset) else 0.0
    return result


def _length_weighted_overlap(actual_edges: pd.DataFrame, route_gdf: pd.DataFrame | None) -> float:
    if route_gdf is None or len(route_gdf) == 0:
        return math.nan
    actual_set = set(
        zip(
            actual_edges["edge_u"].map(_coerce_node_id),
            actual_edges["edge_v"].map(_coerce_node_id),
            actual_edges["edge_key"].map(_coerce_edge_key),
        )
    )
    route_lengths = _edge_length_map(route_gdf)
    denominator = sum(route_lengths.values())
    if denominator <= 0:
        return math.nan
    numerator = sum(length for edge, length in route_lengths.items() if edge in actual_set)
    return float(numerator / denominator)


def calculate_comparison_metrics(order, actual_edges, actual_dist, shortest_gdf, fastest_gdf):
    """模块 E：计算距离、耗时、绕行比例和道路边重合率。"""
    normalized_order = normalize_order(order)
    pickup_time = pd.Timestamp(normalized_order["pickup_time"])
    dropoff_time = pd.Timestamp(normalized_order["dropoff_time"])
    actual_duration_s = float((dropoff_time - pickup_time).total_seconds())

    shortest_dist = _route_distance(shortest_gdf)
    fastest_dist = _route_distance(fastest_gdf)
    fastest_cost_s = float(fastest_gdf["route_cost"].sum()) if fastest_gdf is not None and len(fastest_gdf) else math.nan
    shortest_cost_s = float(shortest_gdf["route_cost"].sum()) if shortest_gdf is not None and len(shortest_gdf) else math.nan

    distance_diff = float(actual_dist - shortest_dist) if not math.isnan(shortest_dist) else math.nan
    detour_ratio = distance_diff / shortest_dist if shortest_dist and not math.isnan(shortest_dist) else math.nan

    metrics = {
        "order_id": normalized_order["id"],
        "vehicle_id": normalized_order["vehicle_id"],
        "pickup_node": _coerce_node_id(normalized_order["pickup_node"]),
        "dropoff_node": _coerce_node_id(normalized_order["dropoff_node"]),
        "actual_point_edges": int(len(actual_edges)),
        "actual_duration_s": actual_duration_s,
        "actual_distance_m": float(actual_dist),
        "shortest_distance_m": shortest_dist,
        "fastest_distance_m": fastest_dist,
        "shortest_route_cost_s": shortest_cost_s,
        "fastest_route_cost_s": fastest_cost_s,
        "distance_diff_m": distance_diff,
        "detour_ratio": float(detour_ratio) if not math.isnan(detour_ratio) else math.nan,
        "shortest_overlap_rate": _length_weighted_overlap(actual_edges, shortest_gdf),
        "fastest_overlap_rate": _length_weighted_overlap(actual_edges, fastest_gdf),
        "shortest_edge_count": int(len(shortest_gdf)) if shortest_gdf is not None else 0,
        "fastest_edge_count": int(len(fastest_gdf)) if fastest_gdf is not None else 0,
    }
    return metrics


def _add_route_group(folium_module, fmap, name: str, lines: Iterable[list[tuple[float, float]]], color: str, weight: int, opacity: float):
    group = folium_module.FeatureGroup(name=name, show=True)
    added = 0
    for coords in lines:
        if coords and len(coords) >= 2:
            folium_module.PolyLine(coords, color=color, weight=weight, opacity=opacity).add_to(group)
            added += 1
    if added == 0:
        folium_module.Marker(
            location=fmap.location,
            tooltip=f"{name}不可用",
            icon=folium_module.Icon(color="lightgray", icon="info-sign"),
        ).add_to(group)
    group.add_to(fmap)


def _route_lines(route_gdf) -> list[list[tuple[float, float]]]:
    if route_gdf is None or len(route_gdf) == 0:
        return []
    if "coordinates" in route_gdf.columns:
        return route_gdf["coordinates"].tolist()
    lines = []
    for geom in route_gdf.get("geometry", []):
        if geom is not None and hasattr(geom, "coords"):
            lines.append([(float(lat), float(lon)) for lon, lat in geom.coords])
    return lines


def plot_three_routes_map(G, order, actual_lines, shortest_gdf, fastest_gdf, output_html, actual_points=None):
    """模块 F：绘制历史实际、最短距离、基准最快三路线对比地图。"""
    import folium

    normalized_order = normalize_order(order)
    center = [float(normalized_order["pickup_matched_lat"]), float(normalized_order["pickup_matched_lon"])]
    dropoff = [float(normalized_order["dropoff_matched_lat"]), float(normalized_order["dropoff_matched_lon"])]

    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
    _add_route_group(folium, fmap, "最短距离路线", _route_lines(shortest_gdf), "blue", 4, 0.85)
    _add_route_group(folium, fmap, "基准最快路线", _route_lines(fastest_gdf), "green", 4, 0.85)
    _add_route_group(folium, fmap, "历史实际路线", actual_lines, "red", 5, 0.8)

    points_group = folium.FeatureGroup(name="GPS散点辅助", show=False)
    for point in actual_points or []:
        folium.CircleMarker(
            location=[point[0], point[1]],
            radius=2,
            color="gray",
            fill=True,
            fill_opacity=0.5,
            opacity=0.5,
            tooltip="轨迹缓存散点",
        ).add_to(points_group)
    folium.CircleMarker(center, radius=5, color="gray", fill=True, fill_opacity=0.5, tooltip="校正上车点").add_to(points_group)
    folium.CircleMarker(dropoff, radius=5, color="gray", fill=True, fill_opacity=0.5, tooltip="校正下车点").add_to(points_group)
    points_group.add_to(fmap)

    folium.Marker(center, tooltip="上车点", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker(dropoff, tooltip="下车点", icon=folium.Icon(color="red", icon="flag")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    fmap.save(output_html)
    LOGGER.info("三路线对比地图已保存: %s", output_html)
    return output_html


def load_graph(graph_path: str = DEFAULT_GRAPH_PATH):
    """加载路网 pkl。仅加载路网，不读取 GPS 原始数据。"""
    LOGGER.info("加载路网文件: %s", graph_path)
    with open(graph_path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        for key in ["graph", "G", "network"]:
            if key in payload:
                return payload[key]
    return payload


def _order_output_id(order) -> str:
    normalized = normalize_order(order)
    token = str(normalized["id"])
    if token in ["", "None", "<NA>"]:
        token = f"{normalized['vehicle_id']}_{pd.Timestamp(normalized['pickup_time']).strftime('%Y%m%d%H%M%S')}"
    return "".join(ch if ch.isalnum() or ch in ["-", "_"] else "_" for ch in token)


def process_single_order(G, order, track_cache_path: str, output_dir: str = DEFAULT_OUTPUT_DIR, generate_map: bool = True) -> SingleOrderResult:
    """模块 G：单订单封装，返回指标字典并按需生成 HTML。"""
    normalized_order = normalize_order(order)
    order_id = str(normalized_order["id"])
    vehicle_id = normalized_order["vehicle_id"]
    LOGGER.info("处理订单: order_id=%s vehicle_id=%s", order_id, vehicle_id)
    print(f"\n处理订单: ID={order_id} | 车辆={vehicle_id}")

    try:
        actual_edges, raw_points = extract_actual_edges_from_track(G, normalized_order, track_cache_path)
        actual_dist, actual_lines, actual_meta = compute_actual_route_geometry(G, actual_edges)
        if actual_meta["has_break"]:
            LOGGER.warning("WARNING 订单 %s 实际路线存在路网断裂。", order_id)

        origin_node = int(normalized_order["pickup_node"])
        dest_node = int(normalized_order["dropoff_node"])
        shortest_gdf, fastest_gdf = get_planning_routes(G, origin_node, dest_node)
        if shortest_gdf is None:
            LOGGER.warning("WARNING 订单 %s 最短距离路线不可达。", order_id)
        if fastest_gdf is None:
            LOGGER.warning("WARNING 订单 %s 基准最快路线不可达。", order_id)

        metrics = calculate_comparison_metrics(normalized_order, actual_edges, actual_dist, shortest_gdf, fastest_gdf)
        metrics["raw_track_points"] = raw_points
        metrics["actual_used_edges"] = actual_meta["used_edges"]
        metrics["actual_has_break"] = actual_meta["has_break"]

        output_html = None
        if generate_map:
            actual_points = extract_actual_points_from_track(normalized_order, track_cache_path)
            output_html = os.path.join(output_dir, f"comparison_{_order_output_id(normalized_order)}.html")
            plot_three_routes_map(G, normalized_order, actual_lines, shortest_gdf, fastest_gdf, output_html, actual_points=actual_points)

        print_order_metrics(metrics)
        return SingleOrderResult(True, order_id, vehicle_id, metrics, output_html=output_html)
    except Exception as exc:
        LOGGER.warning("WARNING 订单处理失败: order_id=%s vehicle_id=%s error=%s", order_id, vehicle_id, exc)
        return SingleOrderResult(False, order_id, vehicle_id, {}, error=str(exc))


def print_order_metrics(metrics: dict):
    """打印检查点要求的核心指标。"""
    print(
        "距离与指标: "
        f"实际={metrics.get('actual_distance_m', math.nan):.2f}m | "
        f"最短={metrics.get('shortest_distance_m', math.nan):.2f}m | "
        f"最快={metrics.get('fastest_distance_m', math.nan):.2f}m | "
        f"实际耗时={metrics.get('actual_duration_s', math.nan):.1f}s | "
        f"绕行比例={metrics.get('detour_ratio', math.nan):.4f} | "
        f"最短重合率={metrics.get('shortest_overlap_rate', math.nan):.4f} | "
        f"最快重合率={metrics.get('fastest_overlap_rate', math.nan):.4f}"
    )


def select_valid_orders(od_df: pd.DataFrame, limit: int = 10, track_cache_path: str | None = None) -> pd.DataFrame:
    """批量选择 5~10 条有效订单。"""
    normalized = normalize_od_dataframe(od_df)
    valid = normalized[
        normalized["pickup_node"].notna()
        & normalized["dropoff_node"].notna()
        & normalized["pickup_time"].notna()
        & normalized["dropoff_time"].notna()
        & (normalized["dropoff_time"] > normalized["pickup_time"])
    ].copy()
    if track_cache_path:
        windows = track_cache_time_windows(track_cache_path)
        if windows:
            def covered(row):
                window = windows.get(vehicle_id_token(row["vehicle_id"]))
                if not window:
                    return False
                start, end = window
                return pd.Timestamp(row["pickup_time"]) >= start and pd.Timestamp(row["dropoff_time"]) <= end

            valid = valid[valid.apply(covered, axis=1)].copy()
        else:
            tokens = available_vehicle_tokens(track_cache_path)
            if tokens:
                valid = valid[valid["vehicle_id"].map(vehicle_id_token).isin(tokens)].copy()
    return valid.head(max(1, min(int(limit), 10))).reset_index(drop=True)


def summarize_batch_results(results: list[SingleOrderResult]) -> pd.DataFrame:
    """生成批量汇总 DataFrame 并打印统计预览。"""
    rows = []
    for result in results:
        if result.success:
            row = result.metrics.copy()
            row["success"] = True
            row["output_html"] = result.output_html
        else:
            row = {
                "order_id": result.order_id,
                "vehicle_id": result.vehicle_id,
                "success": False,
                "error": result.error,
            }
        rows.append(row)
    summary = pd.DataFrame(rows)
    print("\n批量汇总预览:")
    print(summary.head(20).to_string(index=False))
    success_df = summary[summary["success"] == True].copy() if "success" in summary.columns else pd.DataFrame()
    if len(success_df) > 0:
        print("\n批量统计:")
        print(f"平均绕行比例: {success_df['detour_ratio'].mean():.4f}")
        print("最快路线重合率分布:")
        print(success_df["fastest_overlap_rate"].describe().to_string())
        print("\n绕行比例最高的 3 个订单:")
        cols = [col for col in ["order_id", "vehicle_id", "detour_ratio", "actual_distance_m", "shortest_distance_m"] if col in success_df.columns]
        print(success_df.sort_values("detour_ratio", ascending=False).head(3)[cols].to_string(index=False))
    return summary


def run_stage08(
    od_cache_path: str = DEFAULT_OD_CACHE,
    track_cache_path: str = DEFAULT_TRACK_CACHE_DIR,
    graph_path: str = DEFAULT_GRAPH_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    batch_size: int = 10,
):
    """阶段 08 主流程：先跑单订单，再跑 5~10 条批量汇总。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    od_cache_path = resolve_existing_path(od_cache_path, OD_CACHE_FALLBACKS, "校正OD")
    track_cache_path = resolve_existing_path(track_cache_path, TRACK_CACHE_FALLBACKS, "车辆校正轨迹")
    print(f"使用校正OD缓存路径: {od_cache_path}")
    print(f"使用车辆校正轨迹缓存路径: {track_cache_path}")
    print(f"使用路网文件路径: {graph_path}")

    G = load_graph(graph_path)
    od_df = load_od_cache(od_cache_path)

    valid_orders = select_valid_orders(od_df, limit=batch_size, track_cache_path=track_cache_path)
    if len(valid_orders) == 0:
        raise Stage08DataError("没有同时具备校正OD端点和车辆校正轨迹缓存的订单。")

    order = select_valid_order(valid_orders)
    single = process_single_order(G, order, track_cache_path, output_dir=output_dir, generate_map=True)
    if single.success and single.output_html:
        print(f"单订单地图: {single.output_html}")
    elif not single.success:
        print(f"WARNING 单订单处理失败: {single.error}")

    if len(valid_orders) < 5:
        LOGGER.warning("WARNING 有效订单不足 5 条，当前仅处理 %s 条。", len(valid_orders))
    results = []
    for _, batch_order in valid_orders.iterrows():
        results.append(process_single_order(G, batch_order, track_cache_path, output_dir=output_dir, generate_map=True))
    summary = summarize_batch_results(results)
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "stage08_batch_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\n批量汇总结果已保存: {summary_path}")
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="阶段08：历史订单三路线对比分析")
    parser.add_argument("--od-cache", default=DEFAULT_OD_CACHE, help="校正OD缓存路径，默认 cache/od_corrected.parquet")
    parser.add_argument("--track-cache", default=DEFAULT_TRACK_CACHE_DIR, help="车辆校正轨迹缓存目录或单个缓存文件")
    parser.add_argument("--graph", default=DEFAULT_GRAPH_PATH, help="路网 pkl 文件路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="HTML和汇总结果输出目录")
    parser.add_argument("--batch-size", type=int, default=10, help="批量订单数量，建议 5~10")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        run_stage08(
            od_cache_path=args.od_cache,
            track_cache_path=args.track_cache,
            graph_path=args.graph,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    except Stage08DataError as exc:
        LOGGER.error("ERROR 阶段08无法继续: %s", exc)
        print(f"ERROR 阶段08无法继续: {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
