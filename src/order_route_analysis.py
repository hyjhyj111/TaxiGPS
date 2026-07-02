#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import pickle

import pandas as pd

from map_plotter import CONFIG, haversine_distance
from route_planning import (
    OD_COMPLETED_CACHE_PATH,
    complete_od_endpoints,
    load_corrected_track_cache,
    plan_dual_routes_between_points,
    read_dataframe_cache,
    route_coordinates_from_od_row,
    write_dataframe_cache,
)


BATCH_RESULTS_CACHE_PATH = "cache/order_route_batch_results.csv"
ORDER_REVIEW_CACHE_PATH = "cache/order_route_reviews.csv"
LOGGER = logging.getLogger(__name__)


def normalize_vehicle_id(value):
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


def _normalize_od_vehicle_column(df):
    if df is None or len(df) == 0 or "O_TAXI_ID" not in df.columns:
        return df
    normalized = df.copy()
    normalized["O_TAXI_ID"] = normalized["O_TAXI_ID"].map(normalize_vehicle_id)
    return normalized


def _cache_candidates(cache_path):
    candidates = [cache_path]
    root, ext = os.path.splitext(cache_path)
    if ext.lower() == ".parquet":
        candidates.append(root + ".csv")
    elif ext.lower() == ".csv":
        candidates.append(root + ".parquet")
    return candidates


def load_completed_od_cache(cache_path=OD_COMPLETED_CACHE_PATH):
    for path in _cache_candidates(cache_path):
        if not os.path.exists(path):
            continue
        try:
            df = read_dataframe_cache(path)
        except Exception as exc:
            return pd.DataFrame(), {
                "success": False,
                "source": "completed_od_cache",
                "cache_path": path,
                "error": f"校正OD缓存读取失败: {exc}",
            }
        for col in ["O_time", "D_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df = _normalize_od_vehicle_column(df)
        return {
            "data": df,
            "meta": {
                "success": True,
                "source": "completed_od_cache",
                "cache_path": path,
                "rows": int(len(df)),
            },
        }["data"], {
            "success": True,
            "source": "completed_od_cache",
            "cache_path": path,
            "rows": int(len(df)),
        }

    return pd.DataFrame(), {
        "success": False,
        "source": "completed_od_cache",
        "cache_path": cache_path,
        "error": f"未找到校正OD缓存: {cache_path}。系统可从已有OD缓存补建；历史OD查询不会自动读取原始OD大表。",
    }


def filter_completed_od_orders(od_df, start_time=None, end_time=None, vehicle_ids=None):
    if od_df is None or len(od_df) == 0:
        return pd.DataFrame()
    df = od_df.copy()
    for col in ["O_time", "D_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    start = pd.Timestamp(start_time) if start_time is not None else None
    end = pd.Timestamp(end_time) if end_time is not None else None
    if start is not None and "O_time" in df.columns:
        df = df[df["O_time"] >= start]
    if end is not None and "D_time" in df.columns:
        df = df[df["D_time"] <= end]
    wanted = {str(vehicle_id) for vehicle_id in vehicle_ids or [] if str(vehicle_id).strip()}
    if wanted and "O_TAXI_ID" in df.columns:
        wanted = {normalize_vehicle_id(vehicle_id) for vehicle_id in wanted}
        df["O_TAXI_ID"] = df["O_TAXI_ID"].map(normalize_vehicle_id)
        df = df[df["O_TAXI_ID"].isin(wanted)]
    return df.sort_values(["O_time", "D_time"]).reset_index(drop=True)


def load_existing_od_cache(source_od_cache_path=None):
    path = source_od_cache_path or CONFIG.get("OD_CACHE_PATH", "cache/od_cache.pkl")
    if not os.path.exists(path):
        return pd.DataFrame(), {
            "success": False,
            "source": "existing_od_cache",
            "cache_path": path,
            "error": f"未找到已有OD缓存: {path}。不会自动读取原始OD大表。",
        }
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        df = payload.get("data") if isinstance(payload, dict) else payload
        if df is None:
            df = pd.DataFrame()
        df = df.copy()
        for col in ["O_time", "D_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df = _normalize_od_vehicle_column(df)
        return df, {
            "success": True,
            "source": "existing_od_cache",
            "cache_path": path,
            "rows": int(len(df)),
        }
    except Exception as exc:
        return pd.DataFrame(), {
            "success": False,
            "source": "existing_od_cache",
            "cache_path": path,
            "error": f"已有OD缓存读取失败: {exc}",
        }


def build_completed_od_cache_from_existing_caches(
    graph,
    source_od_cache_path=None,
    corrected_cache_dir=None,
    output_path=OD_COMPLETED_CACHE_PATH,
    start_time=None,
    end_time=None,
    vehicle_ids=None,
    time_tolerance="5min",
):
    od_df, source_meta = load_existing_od_cache(source_od_cache_path)
    if not source_meta.get("success"):
        return pd.DataFrame(), source_meta
    filtered_od = filter_completed_od_orders(od_df, start_time=start_time, end_time=end_time, vehicle_ids=vehicle_ids)
    if len(filtered_od) == 0:
        return pd.DataFrame(), {
            "success": False,
            "source": "existing_od_cache",
            "cache_path": source_meta.get("cache_path"),
            "error": "当前筛选条件下已有OD缓存没有订单记录。",
            "rows": 0,
        }

    corrected = load_corrected_track_cache(
        cache_dir=corrected_cache_dir or CONFIG.get("ROAD_CORRECTED_CACHE_DIR", "cache/road_corrected"),
        vehicle_ids=vehicle_ids,
    )
    completed, meta = complete_od_endpoints(
        filtered_od,
        corrected,
        graph,
        time_tolerance=time_tolerance,
        save_path=output_path,
    )
    meta.update(
        {
            "source": "existing_od_cache",
            "source_cache_path": source_meta.get("cache_path"),
            "corrected_cache_dir": corrected_cache_dir or CONFIG.get("ROAD_CORRECTED_CACHE_DIR", "cache/road_corrected"),
            "input_rows": int(len(filtered_od)),
            "cache_path": meta.get("cache_path", output_path),
            "success": bool(meta.get("success")) and len(completed) > 0,
        }
    )
    return completed, meta


def ensure_completed_od_cache(
    graph,
    output_path=OD_COMPLETED_CACHE_PATH,
    force=False,
    source_od_cache_path=None,
    corrected_cache_dir=None,
):
    if not force:
        completed, meta = load_completed_od_cache(output_path)
        if meta.get("success") and len(completed) > 0:
            meta["ensured"] = True
            meta["rebuilt"] = False
            return completed, meta

    completed, meta = build_completed_od_cache_from_existing_caches(
        graph,
        source_od_cache_path=source_od_cache_path,
        corrected_cache_dir=corrected_cache_dir,
        output_path=output_path,
        start_time=None,
        end_time=None,
        vehicle_ids=None,
    )
    meta["ensured"] = bool(meta.get("success"))
    meta["rebuilt"] = bool(meta.get("success"))
    return completed, meta


def route_distance_m(points):
    total = 0.0
    for idx in range(1, len(points or [])):
        prev = points[idx - 1]
        current = points[idx]
        distance_km = haversine_distance(prev[0], prev[1], current[0], current[1])
        if pd.notna(distance_km):
            total += float(distance_km) * 1000.0
    return total


def _vehicle_id_from_order(od_row):
    return normalize_vehicle_id(od_row.get("O_TAXI_ID", od_row.get("vehicle_id", "")))


def _order_times(od_row):
    return pd.Timestamp(od_row["O_time"]), pd.Timestamp(od_row["D_time"])


def order_key_from_row(od_row):
    vehicle_id = _vehicle_id_from_order(od_row)
    start, end = _order_times(od_row)
    return f"{vehicle_id}|{start.isoformat()}|{end.isoformat()}"


def _series_get(od_row, key, default=None):
    try:
        value = od_row.get(key, default)
    except AttributeError:
        value = od_row[key] if key in od_row else default
    if pd.isna(value):
        return default
    return value


def select_validation_orders(od_df, min_count=5, max_count=10):
    if od_df is None or len(od_df) == 0:
        return pd.DataFrame()
    required = [
        "O_TAXI_ID",
        "O_time",
        "D_time",
        "O_corrected_lat",
        "O_corrected_lng",
        "D_corrected_lat",
        "D_corrected_lng",
    ]
    df = od_df.copy()
    for col in required:
        if col not in df.columns:
            return pd.DataFrame(columns=df.columns)
    df = _normalize_od_vehicle_column(df)
    for col in ["O_time", "D_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    complete = df.dropna(subset=required).copy()
    complete = complete[complete["D_time"] > complete["O_time"]]
    complete = complete.sort_values(["O_time", "D_time", "O_TAXI_ID"]).reset_index(drop=True)
    if len(complete) < min_count:
        return complete.head(0).reset_index(drop=True)
    limit = min(max_count, len(complete))
    return complete.head(limit).reset_index(drop=True)


def summarize_od_vehicle_candidates(od_df):
    if od_df is None or len(od_df) == 0 or "O_TAXI_ID" not in od_df.columns:
        return pd.DataFrame(columns=["vehicle_id", "order_count", "first_pickup", "last_dropoff"])
    df = _normalize_od_vehicle_column(od_df)
    for col in ["O_time", "D_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    grouped = (
        df.dropna(subset=["O_TAXI_ID"])
        .assign(vehicle_id=lambda item: item["O_TAXI_ID"].astype(str))
        .groupby("vehicle_id", as_index=False)
        .agg(
            order_count=("O_TAXI_ID", "size"),
            first_pickup=("O_time", "min"),
            last_dropoff=("D_time", "max"),
        )
        .sort_values(["order_count", "vehicle_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return grouped


def _edge_tuple(row):
    try:
        if pd.isna(row.get("edge_u")) or pd.isna(row.get("edge_v")):
            return None
        key = 0 if pd.isna(row.get("edge_key")) else row.get("edge_key")
        return (str(row.get("edge_u")), str(row.get("edge_v")), str(key))
    except TypeError:
        return None


def _edge_set_from_corrected_rows(rows):
    edges = []
    for _, row in rows.iterrows():
        item = _edge_tuple(row)
        if item and (not edges or edges[-1] != item):
            edges.append(item)
    return set(edges)


def _edge_set_from_route_summary(route_summary):
    edges = set()
    for edge in route_summary.get("edges", []) or []:
        key = 0 if edge.get("key") is None else edge.get("key")
        edges.add((str(edge.get("u")), str(edge.get("v")), str(key)))
    return edges


def slice_actual_route_from_corrected_rows(corrected_rows, od_row):
    if corrected_rows is None or len(corrected_rows) == 0:
        return {
            "success": False,
            "source": "corrected_track_cache",
            "error": "未找到该车辆的校正轨迹缓存。",
            "rows": pd.DataFrame(),
            "points": [],
        }

    rows = corrected_rows.copy()
    vehicle_id = _vehicle_id_from_order(od_row)
    vehicle_col = "vehicle_id" if "vehicle_id" in rows.columns else "id" if "id" in rows.columns else None
    if vehicle_col:
        rows = rows[rows[vehicle_col].astype(str) == vehicle_id].copy()
    if len(rows) == 0:
        return {
            "success": False,
            "source": "corrected_track_cache",
            "error": f"校正轨迹缓存中没有车辆 {vehicle_id} 的记录。",
            "rows": pd.DataFrame(),
            "points": [],
        }

    start, end = _order_times(od_row)
    rows["time"] = pd.to_datetime(rows["time"], errors="coerce")
    rows = rows[(rows["time"] >= start) & (rows["time"] <= end)].copy()
    if "sequence" in rows.columns:
        rows["sequence"] = pd.to_numeric(rows["sequence"], errors="coerce")
        rows = rows.sort_values(["time", "sequence"])
    else:
        rows = rows.sort_values("time")

    points = []
    for _, row in rows.dropna(subset=["matched_lat", "matched_lon"]).iterrows():
        point = [float(row["matched_lat"]), float(row["matched_lon"])]
        if points and points[-1] == point:
            continue
        points.append(point)

    duration_s = max(0.0, (end - start).total_seconds())
    return {
        "success": len(points) >= 2,
        "source": "corrected_track_cache",
        "rows": rows.reset_index(drop=True),
        "points": points,
        "distance_m": route_distance_m(points),
        "duration_s": float(duration_s),
        "edge_set": _edge_set_from_corrected_rows(rows),
        "error": "" if len(points) >= 2 else "当前订单时间窗内的校正轨迹点不足。",
    }


def compute_order_route_metrics(actual_route, route_result):
    shortest = route_result.get("shortest", {}) if route_result else {}
    fastest = route_result.get("fastest", {}) if route_result else {}
    actual_distance_m = float(actual_route.get("distance_m", 0.0) or 0.0)
    shortest_distance_m = float(shortest.get("distance_m", 0.0) or 0.0)
    fastest_distance_m = float(fastest.get("distance_m", 0.0) or 0.0)
    detour_ratio = (actual_distance_m / shortest_distance_m - 1.0) if shortest_distance_m > 0 else 0.0

    actual_edges = actual_route.get("edge_set", set()) or set()
    fastest_edges = _edge_set_from_route_summary(fastest)
    fastest_overlap_rate = (len(actual_edges & fastest_edges) / len(actual_edges)) if actual_edges else 0.0

    return {
        "actual_distance_m": actual_distance_m,
        "shortest_distance_m": shortest_distance_m,
        "fastest_distance_m": fastest_distance_m,
        "actual_duration_s": float(actual_route.get("duration_s", 0.0) or 0.0),
        "detour_ratio": float(detour_ratio),
        "fastest_overlap_rate": float(fastest_overlap_rate),
    }


def analyze_single_order_route(od_row, corrected_rows, graph, speed_stats=None, highway_median_speed=None):
    coords = route_coordinates_from_od_row(od_row)
    actual = slice_actual_route_from_corrected_rows(corrected_rows, od_row)
    if not actual.get("success"):
        return {"success": False, "error": actual.get("error", "历史实际路线不可用。"), "actual": actual}

    route_result = plan_dual_routes_between_points(
        graph,
        coords["origin"]["lat"],
        coords["origin"]["lng"],
        coords["destination"]["lat"],
        coords["destination"]["lng"],
        speed_stats=speed_stats,
        highway_median_speed=highway_median_speed,
    )
    if not route_result.get("success"):
        return {"success": False, "error": route_result.get("error", "路线计算失败。"), "actual": actual, "route_result": route_result}

    return {
        "success": True,
        "order": dict(od_row),
        "actual": actual,
        "route_result": route_result,
        "metrics": compute_order_route_metrics(actual, route_result),
    }


def analyze_order_metrics_dict(
    od_row,
    corrected_rows,
    graph,
    speed_stats=None,
    highway_median_speed=None,
    logger=None,
):
    logger = logger or LOGGER
    order_key = order_key_from_row(od_row)
    vehicle_id = _vehicle_id_from_order(od_row)
    start, end = _order_times(od_row)
    base = {
        "order_key": order_key,
        "vehicle_id": vehicle_id,
        "O_time": start.isoformat(),
        "D_time": end.isoformat(),
        "success": False,
        "error": "",
        "actual_distance_m": 0.0,
        "shortest_distance_m": 0.0,
        "fastest_distance_m": 0.0,
        "actual_duration_s": max(0.0, (end - start).total_seconds()),
        "detour_ratio": 0.0,
        "fastest_overlap_rate": 0.0,
        "actual_edge_count": 0,
        "shortest_edge_count": 0,
        "fastest_edge_count": 0,
    }
    try:
        analysis = analyze_single_order_route(
            od_row,
            corrected_rows,
            graph,
            speed_stats=speed_stats,
            highway_median_speed=highway_median_speed,
        )
        if not analysis.get("success"):
            base["error"] = analysis.get("error", "订单路线分析失败。")
            logger.warning("订单路线分析失败: order_key=%s error=%s", order_key, base["error"])
            return base

        metrics = analysis["metrics"]
        shortest = analysis["route_result"].get("shortest", {})
        fastest = analysis["route_result"].get("fastest", {})
        actual = analysis.get("actual", {})
        base.update(
            {
                "success": True,
                "actual_distance_m": float(metrics.get("actual_distance_m", 0.0) or 0.0),
                "shortest_distance_m": float(metrics.get("shortest_distance_m", 0.0) or 0.0),
                "fastest_distance_m": float(metrics.get("fastest_distance_m", 0.0) or 0.0),
                "actual_duration_s": float(metrics.get("actual_duration_s", 0.0) or 0.0),
                "detour_ratio": float(metrics.get("detour_ratio", 0.0) or 0.0),
                "fastest_overlap_rate": float(metrics.get("fastest_overlap_rate", 0.0) or 0.0),
                "actual_edge_count": int(len(actual.get("edge_set", []))),
                "shortest_edge_count": int(shortest.get("edge_count", 0) or 0),
                "fastest_edge_count": int(fastest.get("edge_count", 0) or 0),
            }
        )
        logger.info("订单路线分析完成: order_key=%s", order_key)
        return base
    except Exception as exc:
        base["error"] = str(exc)
        logger.exception("订单路线分析异常: order_key=%s", order_key)
        return base


def _read_optional_results(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return read_dataframe_cache(path)
    except Exception:
        return pd.read_csv(path)


def _dedupe_results(results):
    if results is None or len(results) == 0:
        return pd.DataFrame()
    df = results.copy()
    if "order_key" not in df.columns:
        return df
    return df.drop_duplicates(subset=["order_key"], keep="last").reset_index(drop=True)


def batch_analyze_orders(
    od_df,
    corrected_rows,
    graph,
    speed_stats=None,
    highway_median_speed=None,
    batch_size=20,
    output_path=BATCH_RESULTS_CACHE_PATH,
    resume=True,
    logger=None,
):
    logger = logger or LOGGER
    if od_df is None or len(od_df) == 0:
        return pd.DataFrame(), {
            "success": False,
            "processed": 0,
            "skipped_existing": 0,
            "output_path": output_path,
            "error": "没有可处理的订单。",
        }

    batch_size = max(1, int(batch_size or 1))
    orders = od_df.copy().reset_index(drop=True)
    orders["_order_key"] = [order_key_from_row(row) for _, row in orders.iterrows()]
    existing = _dedupe_results(_read_optional_results(output_path)) if resume else pd.DataFrame()
    existing_keys = set(existing["order_key"].astype(str)) if len(existing) and "order_key" in existing.columns else set()
    pending = orders[~orders["_order_key"].astype(str).isin(existing_keys)].copy()
    records = existing.to_dict("records") if len(existing) else []
    processed = 0

    for start_idx in range(0, len(pending), batch_size):
        batch = pending.iloc[start_idx:start_idx + batch_size]
        for _, row in batch.iterrows():
            clean_row = row.drop(labels=["_order_key"])
            records.append(
                analyze_order_metrics_dict(
                    clean_row,
                    corrected_rows,
                    graph,
                    speed_stats=speed_stats,
                    highway_median_speed=highway_median_speed,
                    logger=logger,
                )
            )
            processed += 1

        partial = _dedupe_results(pd.DataFrame(records))
        saved_path = write_dataframe_cache(partial, output_path)
        output_path = saved_path
        logger.info("批量订单分析已保存中间结果: processed=%s path=%s", processed, output_path)

    results = _dedupe_results(pd.DataFrame(records))
    if len(results) and output_path:
        output_path = write_dataframe_cache(results, output_path)
    return results, {
        "success": True,
        "processed": int(processed),
        "skipped_existing": int(len(existing_keys)),
        "total_orders": int(len(orders)),
        "result_rows": int(len(results)),
        "batch_size": int(batch_size),
        "output_path": output_path,
        "error": "",
    }


def _distribution_table(series, bins, labels, percent_multiplier=1.0):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) == 0:
        return []
    grouped = pd.cut(clean * percent_multiplier, bins=bins, labels=labels, include_lowest=True, right=False)
    counts = grouped.value_counts(sort=False)
    total = int(len(clean))
    return [
        {
            "区间": str(label),
            "订单数": int(count),
            "占比": round(float(count) / total, 4) if total else 0.0,
        }
        for label, count in counts.items()
    ]


def summarize_batch_results(results_df, detour_threshold=0.3):
    if results_df is None or len(results_df) == 0:
        return {
            "total_orders": 0,
            "success_orders": 0,
            "failed_orders": 0,
            "exception_count": 0,
            "distance_stats": {},
            "detour_distribution": [],
            "overlap_distribution": [],
        }

    df = results_df.copy()
    success = df[df["success"].astype(bool)] if "success" in df.columns else df
    distance_km = pd.to_numeric(success.get("actual_distance_m", pd.Series(dtype=float)), errors="coerce") / 1000.0
    distance_km = distance_km.dropna()
    distance_stats = {
        "mean_km": round(float(distance_km.mean()), 4) if len(distance_km) else 0.0,
        "median_km": round(float(distance_km.median()), 4) if len(distance_km) else 0.0,
        "std_km": round(float(distance_km.std(ddof=0)), 4) if len(distance_km) else 0.0,
        "min_km": round(float(distance_km.min()), 4) if len(distance_km) else 0.0,
        "max_km": round(float(distance_km.max()), 4) if len(distance_km) else 0.0,
    }
    detour = pd.to_numeric(success.get("detour_ratio", pd.Series(dtype=float)), errors="coerce")
    overlap = pd.to_numeric(success.get("fastest_overlap_rate", pd.Series(dtype=float)), errors="coerce")
    return {
        "total_orders": int(len(df)),
        "success_orders": int(len(success)),
        "failed_orders": int(len(df) - len(success)),
        "exception_count": int((detour > detour_threshold).sum()),
        "distance_stats": distance_stats,
        "distance_distribution": _distribution_table(
            distance_km,
            bins=[0, 2, 5, 10, 20, float("inf")],
            labels=["0-2km", "2-5km", "5-10km", "10-20km", "20km以上"],
        ),
        "detour_distribution": _distribution_table(
            detour,
            bins=[-float("inf"), 0, 10, 30, 50, float("inf")],
            labels=["低于最短", "0-10%", "10-30%", "30-50%", "50%以上"],
            percent_multiplier=100.0,
        ),
        "overlap_distribution": _distribution_table(
            overlap,
            bins=[0, 20, 40, 60, 80, 101],
            labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
            percent_multiplier=100.0,
        ),
    }


def exception_orders(results_df, detour_threshold=0.3):
    if results_df is None or len(results_df) == 0:
        return pd.DataFrame()
    df = results_df.copy()
    df["detour_ratio"] = pd.to_numeric(df.get("detour_ratio"), errors="coerce")
    df = df[(df.get("success", False).astype(bool)) & (df["detour_ratio"] > detour_threshold)]
    return df.sort_values("detour_ratio", ascending=False).reset_index(drop=True)


def update_order_review_mark(
    review_path,
    order_key,
    review_status,
    suggestion,
    reviewer="",
    reviewed_at=None,
):
    reviewed_at = reviewed_at or pd.Timestamp.now().isoformat()
    existing = _read_optional_results(review_path)
    if len(existing) == 0:
        existing = pd.DataFrame(columns=["order_key", "review_status", "suggestion", "reviewer", "reviewed_at"])
    existing = existing[existing["order_key"].astype(str) != str(order_key)].copy()
    row = pd.DataFrame(
        [
            {
                "order_key": str(order_key),
                "review_status": str(review_status),
                "suggestion": str(suggestion),
                "reviewer": str(reviewer),
                "reviewed_at": str(reviewed_at),
            }
        ]
    )
    updated = pd.concat([existing, row], ignore_index=True)
    write_dataframe_cache(updated, review_path)
    return updated


def completed_od_cache_status(cache_path=OD_COMPLETED_CACHE_PATH):
    df, meta = load_completed_od_cache(cache_path)
    return {
        "available": bool(meta.get("success")),
        "rows": int(len(df)),
        "path": meta.get("cache_path", cache_path),
        "error": meta.get("error", ""),
    }
