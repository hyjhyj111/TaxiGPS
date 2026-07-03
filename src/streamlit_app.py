#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
出租车GPS轨迹查询系统

单页集成：
1. 按车辆ID和时间范围查询轨迹，支持最多 10 辆车并行展示
2. 按分钟查询所有车辆或指定车辆位置
3. 标注上车点和下车点
4. 单车动画轨迹播放
5. 预留 ETA / 路网校正坐标输入
"""

from datetime import datetime, time
import json
import logging
import os
import sys
from textwrap import dedent

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heatmap_analysis import (  # noqa: E402
    aggregate_minute_range_fast,
    build_dynamic_heatmap,
    build_static_heatmap,
    cached_standardize_heatmap_source,
    compute_daily_operation_statistics,
    compute_order_statistics,
    export_statistics_bundle,
    recommend_dbscan_params,
    run_pickup_cluster_analysis,
)
from map_plotter import (  # noqa: E402
    CONFIG,
    load_minute_data,
    load_od_data,
    load_vehicle_trajectory,
    estimate_eta,
    estimate_eta_model,
    plot_animated_trajectory,
    plot_congestion_roads,
    plot_eta_route,
    plot_multi_vehicle_animated_trajectory,
    plot_minute_vehicles,
    plot_od_points,
    plot_road_corrected_trajectories,
    plot_vehicle_trajectories,
    run_hmm_speed_aggregation,
    plot_vehicle_trajectory,
    road_network_status,
)
from order_route_analysis import (  # noqa: E402
    filter_completed_od_orders,
    load_completed_od_cache,
    load_existing_od_cache,
    normalize_vehicle_id,
)
from stage08_order_route_comparison import normalize_od_dataframe  # noqa: E402


logger = logging.getLogger(__name__)

PAGE_TITLE = "出租车GPS轨迹查询系统"
PAGE_ICON = "🚕"
VIEW_OPTIONS = ["轨迹查询", "动画轨迹", "分钟位置", "OD点标注", "热力图与统计分析", "路线规划", "拥堵与ETA"]
NAVIGATION_GROUPS = [
    {
        "label": "地图",
        "items": [
            {"view": "轨迹查询", "icon": "📍"},
            {"view": "动画轨迹", "icon": "▶️"},
            {"view": "分钟位置", "icon": "⏱️"},
            {"view": "OD点标注", "icon": "🔵"},
        ],
        "divider_after": True,
    },
    {
        "label": "分析",
        "items": [
            {"view": "热力图与统计分析", "icon": "🔥"},
        ],
        "divider_after": True,
    },
    {
        "label": "路线",
        "items": [
            {"view": "路线规划", "icon": "🗺️"},
            {"view": "拥堵与ETA", "icon": "🚗"},
        ],
        "divider_after": False,
    },
]


def get_view_context(view_name):
    """Return the group label and description for a given view name."""
    for group in NAVIGATION_GROUPS:
        for item in group["items"]:
            if item["view"] == view_name:
                descriptions = {
                    "轨迹查询": "按车辆ID和时间范围查询行驶轨迹",
                    "动画轨迹": "按时间顺序播放车辆运动轨迹",
                    "分钟位置": "查看指定分钟的车辙分布",
                    "OD点标注": "展示上车点和下车点位置",
                    "热力图与统计分析": "静态/动态热力图、聚类与订单统计",
                    "路线规划": "地图选点计算最短距离与基准最快路线",
                    "拥堵与ETA": "路段拥堵颜色与历史均速ETA",
                }
                return {
                    "group": group["label"],
                    "description": descriptions.get(view_name, ""),
                }
    return {"group": "", "description": ""}


DEFAULTS = {
    "vehicle_id": "22223",
    "query_date": datetime(2023, 10, 12).date(),
    "start_time_of_day": "08:00",
    "end_time_of_day": "10:00",
    "minute_time_of_day": "09:30",
    "time_scale": 2.0,
    "trajectory_vehicle_ids": [],
    "ref_lat": float(CONFIG["MAP_CENTER"][0]),
    "ref_lng": float(CONFIG["MAP_CENTER"][1]),
    "active_view": "轨迹查询",
    "last_query": None,
    "heatmap_source": "pickup",
    "heatmap_enable_cluster": False,
    "heatmap_eps_km": 0.35,
    "heatmap_min_samples": 8,
    "heatmap_threshold_quantile": 0.92,
    "dynamic_source": "pickup",
    "dynamic_granularity": 15,
    "dynamic_smoothing": "EMA",
    "dynamic_ema_alpha": 0.55,
    "dynamic_wma_window": 3,
    "dynamic_threshold_quantile": 0.92,
    "pickup_cluster_eps_km": 0.35,
    "pickup_cluster_min_samples": 8,
    "pickup_cluster_threshold_quantile": 0.92,
    "heatmap_analysis_panel": "静态热力图",
    "static_heatmap_request": None,
    "dynamic_heatmap_request": None,
    "pickup_cluster_request": None,
    "order_stats_request": None,
    "operation_stats_request": None,
    "last_active_view": "轨迹查询",
    "road_correction_enabled": False,
    "congestion_bucket_minutes": 15,
    "eta_mode": "规则版",
    "eta_model_train_orders": 800,
    "hmm_aggregation_max_vehicles": 8,
    "hmm_aggregation_min_samples": 3,
    "hmm_aggregation_force_rebuild": False,
    "congestion_use_full_day": True,
    "congestion_rebuild_requested": False,
    "eta_origin_lat": float(CONFIG["MAP_CENTER"][0]),
    "eta_origin_lng": float(CONFIG["MAP_CENTER"][1]),
    "eta_dest_lat": 22.6008,
    "eta_dest_lng": 114.1010,
    "route_origin_lat": float(CONFIG["MAP_CENTER"][0]),
    "route_origin_lng": float(CONFIG["MAP_CENTER"][1]),
    "route_dest_lat": 22.6008,
    "route_dest_lng": 114.1010,
    "route_source_mode": "地图选点",
    "route_map_reset_requested": False,
    "route_map_calculate_requested": False,
    "route_od_index": 0,
    "history_od_candidate_options": [],
    "history_od_candidate_selected": [],
    "history_od_active_vehicle_ids": [],
    "history_od_candidate_meta": {},
}


def is_running_under_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def init_page():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
       """
       <style>
       :root {
           --bg: #ffffff;
           --surface: #f5f5f5;
           --surface-2: #fafafa;
           --border: #e0e0e0;
           --border-strong: #cfcfcf;
           --text: #212121;
           --muted: #616161;
           --soft: #9e9e9e;
            --accent: #2563eb;
            --accent-light: #dbeafe;
            --accent-soft: #eff6ff;
       }

       .stApp {
           background: var(--bg);
           color: var(--text);
           font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       }

       .block-container {
            padding-top: 0.6rem;
            padding-bottom: 0.6rem;
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
       }

       [data-testid="stSidebar"] {
            background: #f8f9fb;
            border-right: 1px solid #e5e7eb;
            min-width: 260px !important;
            max-width: 300px !important;
        }



/* ─── Query Controls ─── */


        /* Hide Streamlit page header to prevent content overlap */
        header[data-testid="stHeader"] {
            display: none;
        }

        /* ─── Main Content: Map Focus ─── */
        .main-map-area {
            width: 100%;
        }

        .main-map-area iframe {
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }

        /* ─── Feature Controls ─── */
        .feature-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.75rem;
        }

        .feature-controls {
            background: #f8f9fb;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.75rem;
        }

        .feature-controls .stButton button {
            border-radius: 8px !important;
            font-size: 0.85rem !important;
        }

        /* ─── Responsive ─── */
        @media (max-width: 768px) {
            [data-testid="stSidebar"] {
                min-width: 100% !important;
                max-width: 100% !important;
            }
            .block-container {
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
        }
        </style>
        """,
       unsafe_allow_html=True,
   )


def init_state():
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.get("active_view") in {"05 热力图与统计分析", "06 下一阶段"}:
        st.session_state["active_view"] = "热力图与统计分析"


def reset_to_defaults():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["last_query"] = None
    st.rerun()


@st.cache_data(show_spinner=False, ttl=300)
def cached_vehicle_trajectory(vehicle_id, start_time, end_time):
    return load_vehicle_trajectory(vehicle_id, start_time, end_time)


@st.cache_data(show_spinner=False, ttl=300)
def cached_minute_data(minute_time):
    return load_minute_data(minute_time)


@st.cache_data(show_spinner=False, ttl=300)
def cached_od_data(start_time, end_time, vehicle_id=None, vehicle_ids=None):
    return load_od_data(start_time, end_time, vehicle_id, vehicle_ids=vehicle_ids)


@st.cache_data(show_spinner=False, ttl=300)
def cached_source_recommendation(source_type, start_time, end_time, vehicle_ids=None):
    if source_type == "minute":
        points_df, meta = aggregate_minute_range_fast(start_time, end_time, vehicle_ids=vehicle_ids, precision=4, time_bucket_minutes=None)
        if points_df is None or len(points_df) == 0:
            return {"eps_km": 0.35, "min_samples": 8}, meta
        sample_df = points_df[["lat", "lng", "weight", "time_start"]].rename(columns={"time_start": "time"}).copy()
        return recommend_dbscan_params(sample_df.assign(point_count=1)), meta

    std_df, meta = cached_standardize_heatmap_source(source_type, start_time, end_time, vehicle_ids=vehicle_ids)
    if std_df is None or len(std_df) == 0:
        return {"eps_km": 0.35, "min_samples": 8}, meta
    return recommend_dbscan_params(std_df[["lat", "lng", "weight", "time"]].assign(point_count=1)), meta


@st.cache_data(show_spinner=False, ttl=300)
def cached_static_heatmap(
    source_type,
    start_time,
    end_time,
    vehicle_ids=None,
    enable_cluster=False,
    eps_km=0.35,
    min_samples=8,
    threshold_quantile=0.92,
):
    return build_static_heatmap(
        source_type=source_type,
        start_time=start_time,
        end_time=end_time,
        vehicle_ids=vehicle_ids,
        enable_cluster=enable_cluster,
        eps_km=eps_km,
        min_samples=min_samples,
        threshold_quantile=threshold_quantile,
    )


@st.cache_data(show_spinner=False, ttl=300)
def cached_dynamic_heatmap(
    source_type,
    start_time,
    end_time,
    requested_granularity=15,
    vehicle_ids=None,
    smoothing_method="EMA",
    ema_alpha=0.55,
    wma_window=3,
    threshold_quantile=0.92,
):
    return build_dynamic_heatmap(
        source_type=source_type,
        start_time=start_time,
        end_time=end_time,
        requested_granularity=requested_granularity,
        vehicle_ids=vehicle_ids,
        smoothing_method=smoothing_method,
        ema_alpha=ema_alpha,
        wma_window=wma_window,
        threshold_quantile=threshold_quantile,
    )


@st.cache_data(show_spinner=False, ttl=300)
def cached_order_statistics(start_time, end_time, vehicle_ids=None):
    return compute_order_statistics(start_time, end_time, vehicle_ids=vehicle_ids)


@st.cache_data(show_spinner=False, ttl=1800)
def cached_daily_operation_statistics(query_date, vehicle_ids=None):
    return compute_daily_operation_statistics(query_date, vehicle_ids=vehicle_ids)


@st.cache_data(show_spinner=False, ttl=300)
def cached_pickup_cluster_analysis(start_time, end_time, vehicle_ids=None, eps_km=0.35, min_samples=8, threshold_quantile=0.92):
    return run_pickup_cluster_analysis(
        start_time=start_time,
        end_time=end_time,
        vehicle_ids=vehicle_ids,
        eps_km=eps_km,
        min_samples=min_samples,
        threshold_quantile=threshold_quantile,
    )


@st.cache_data(show_spinner=False, ttl=300)
def cached_congestion_roads(vehicle_ids, start_time, end_time, bucket_minutes):
    return plot_congestion_roads(tuple(vehicle_ids or []), start_time, end_time, bucket_minutes=bucket_minutes)


@st.cache_data(show_spinner=False, ttl=300)
def cached_eta_result(origin_lat, origin_lng, dest_lat, dest_lng, vehicle_ids, start_time, end_time, bucket_minutes):
    return estimate_eta(
        origin_lat,
        origin_lng,
        dest_lat,
        dest_lng,
        vehicle_ids=tuple(vehicle_ids or []),
        start_time=start_time,
        end_time=end_time,
        bucket_minutes=bucket_minutes,
    )


@st.cache_data(show_spinner=False, ttl=600)
def cached_eta_model_result(origin_lat, origin_lng, dest_lat, dest_lng, vehicle_ids, start_time, end_time, bucket_minutes, max_train_orders):
    return estimate_eta_model(
        origin_lat,
        origin_lng,
        dest_lat,
        dest_lng,
        departure_time=start_time,
        vehicle_ids=tuple(vehicle_ids or []),
        start_time=start_time,
        end_time=end_time,
        bucket_minutes=bucket_minutes,
        max_train_orders=int(max_train_orders),
    )


@st.cache_data(show_spinner=False, ttl=300)
def cached_hmm_speed_aggregation(vehicle_ids, start_time, end_time, bucket_minutes, min_samples, force_rebuild):
    return run_hmm_speed_aggregation(
        tuple(vehicle_ids or []),
        start_time=start_time,
        end_time=end_time,
        bucket_minutes=int(bucket_minutes),
        min_samples=int(min_samples),
        force_rebuild=bool(force_rebuild),
    )


@st.cache_data(show_spinner=False, ttl=300)
def cached_export_bundle(query_date, order_stats, operation_stats, cluster_result=None):
    return export_statistics_bundle(query_date, order_stats, operation_stats, cluster_result=cluster_result)


@st.cache_data(show_spinner=False, ttl=600)
def get_available_vehicle_ids():
    if not os.path.isdir(CONFIG["VEHICLE_CACHE_DIR"]):
        return []

    vehicle_ids = []
    for filename in os.listdir(CONFIG["VEHICLE_CACHE_DIR"]):
        if filename.lower().endswith(".csv"):
            vehicle_ids.append(os.path.splitext(filename)[0])

    def sort_key(value):
        value = str(value)
        return (0, int(value)) if value.isdigit() else (1, value)

    return sorted(set(vehicle_ids), key=sort_key)


@st.cache_data(show_spinner=False, ttl=300)
def get_vehicle_time_range(vehicle_id):
    """获取车辆缓存轨迹的时间范围。"""
    if not vehicle_id:
        return None

    vehicle_file = os.path.join(CONFIG["VEHICLE_CACHE_DIR"], f"{vehicle_id}.csv")
    if not os.path.exists(vehicle_file):
        return None

    try:
        df = pd.read_csv(vehicle_file)
        if "time" not in df.columns:
            return None
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        min_time = df["time"].min()
        max_time = df["time"].max()
        if pd.isna(min_time) or pd.isna(max_time):
            return None
        return {
            "min_time": min_time,
            "max_time": max_time,
            "total_points": len(df),
            "date": min_time.date(),
        }
    except Exception:
        logger.exception("获取车辆时间范围失败")
        return None


def apply_vehicle_time_range(vehicle_id):
    """将车辆缓存的时间范围回填到查询控件。"""
    time_range = get_vehicle_time_range(vehicle_id)
    if not time_range:
        return False

    start_time = time_range["min_time"].time().replace(microsecond=0)
    end_time = time_range["max_time"].time().replace(microsecond=0)
    st.session_state["query_date"] = time_range["min_time"].date()
    st.session_state["start_time_of_day"] = start_time.strftime("%H:%M")
    st.session_state["end_time_of_day"] = end_time.strftime("%H:%M")
    st.session_state["minute_time_of_day"] = start_time.strftime("%H:%M")
    return True


def parse_time_field(value, fallback):
    if isinstance(value, time):
        return value
    if isinstance(value, str):
        parts = value.strip().split(":")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            hour = max(0, min(23, int(parts[0])))
            minute = max(0, min(59, int(parts[1])))
            return time(hour, minute)
    return fallback


def build_query_payload():
    trajectory_vehicle_ids = [
        str(vehicle_id).strip()
        for vehicle_id in st.session_state.get("trajectory_vehicle_ids", [])
        if str(vehicle_id).strip()
    ]

    start_time = datetime.combine(
        st.session_state["query_date"],
        parse_time_field(st.session_state["start_time_of_day"], time(8, 0)),
    )
    end_time = datetime.combine(
        st.session_state["query_date"],
        parse_time_field(st.session_state["end_time_of_day"], time(10, 0)),
    )
    minute_time = datetime.combine(
        st.session_state["query_date"],
        parse_time_field(st.session_state["minute_time_of_day"], time(9, 30)),
    )

    return {
        "vehicle_id": trajectory_vehicle_ids[0] if trajectory_vehicle_ids else "",
        "trajectory_vehicle_ids": trajectory_vehicle_ids,
        "history_od_vehicle_ids": [
            normalize_vehicle_id(vehicle_id)
            for vehicle_id in st.session_state.get("history_od_active_vehicle_ids", [])
            if normalize_vehicle_id(vehicle_id)
        ],
        "start_time": start_time,
        "end_time": end_time,
        "minute_time": minute_time,
        "time_scale": float(st.session_state["time_scale"]),
        "ref_lat": float(st.session_state["ref_lat"]),
        "ref_lng": float(st.session_state["ref_lng"]),
    }


def _history_od_day_range(query_date=None):
    day = query_date or st.session_state["query_date"]
    start_time = datetime.combine(day, time(0, 0))
    end_time = datetime.combine(day, time(23, 59, 59))
    return start_time, end_time


def query_history_od_vehicle_candidates(query_date=None):
    od_df, meta = load_completed_od_cache()
    source_label = "校正OD缓存"
    if not meta.get("success"):
        od_df, meta = load_existing_od_cache()
        source_label = "已有OD缓存"
    if not meta.get("success"):
        return pd.DataFrame(), {
            "success": False,
            "source_label": source_label,
            "error": meta.get("error", "未找到可用OD缓存。"),
        }
    day_start, day_end = _history_od_day_range(query_date)
    filtered = filter_completed_od_orders(od_df, start_time=day_start, end_time=day_end)
    normalized = normalize_od_dataframe(filtered)
    if len(normalized) > 0:
        normalized = normalized[
            normalized["vehicle_id"].notna()
            & normalized["pickup_node"].notna()
            & normalized["dropoff_node"].notna()
            & normalized["pickup_time"].notna()
            & normalized["dropoff_time"].notna()
            & (normalized["dropoff_time"] > normalized["pickup_time"])
        ].copy()

    vehicle_cache_dir = os.path.join("cache", "vehicles")
    cached_vehicle_ids = set()
    if os.path.isdir(vehicle_cache_dir):
        for filename in os.listdir(vehicle_cache_dir):
            lower = filename.lower()
            if lower.endswith((".csv", ".parquet")):
                cached_vehicle_ids.add(normalize_vehicle_id(os.path.splitext(filename)[0].split("_", 1)[0]))

    if len(normalized) > 0 and cached_vehicle_ids:
        normalized = normalized[normalized["vehicle_id"].map(normalize_vehicle_id).isin(cached_vehicle_ids)].copy()

    if len(normalized) == 0:
        summary = pd.DataFrame(columns=["vehicle_id", "order_count", "first_pickup", "last_dropoff"])
    else:
        summary = (
            normalized.groupby("vehicle_id", as_index=False)
            .agg(
                order_count=("id", "count"),
                first_pickup=("pickup_time", "min"),
                last_dropoff=("dropoff_time", "max"),
            )
            .sort_values(["order_count", "vehicle_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
    return summary, {
        "success": True,
        "source_label": source_label,
        "cache_path": meta.get("cache_path", ""),
        "orders": int(len(normalized)),
        "vehicles": int(len(summary)),
        "date": day_start.strftime("%Y-%m-%d"),
    }


def render_history_od_sidebar_tools():
    st.markdown("---")
    st.markdown("**历史OD查询**")

    if st.button("查询有OD车辆", key="history_od_candidate_query", use_container_width=True):
        candidates, meta = query_history_od_vehicle_candidates(st.session_state["query_date"])
        if meta.get("success"):
            options = candidates["vehicle_id"].map(normalize_vehicle_id).tolist() if len(candidates) else []
            st.session_state["history_od_candidate_options"] = options
            st.session_state["history_od_candidate_meta"] = meta
            st.session_state["history_od_candidate_table"] = candidates
            st.session_state["history_od_candidate_selected"] = options[: min(10, len(options))]
        else:
            st.session_state["history_od_candidate_options"] = []
            st.session_state["history_od_candidate_selected"] = []
            st.session_state["history_od_candidate_meta"] = meta

    meta = st.session_state.get("history_od_candidate_meta", {})
    if meta:
        if meta.get("success"):
            st.caption(
                f"{meta.get('source_label')} | 车辆 {meta.get('vehicles', 0)} | "
                f"订单 {meta.get('orders', 0)}"
            )
        else:
            st.warning(meta.get("error", "OD车辆候选查询失败。"))

    options = st.session_state.get("history_od_candidate_options", [])
    current_selected = [
        str(item)
        for item in st.session_state.get("history_od_candidate_selected", [])
        if str(item) in options
    ]
    if current_selected != st.session_state.get("history_od_candidate_selected", []):
        st.session_state["history_od_candidate_selected"] = current_selected
    selected = st.multiselect(
        "OD车辆候选",
        options=options,
        key="history_od_candidate_selected",
        placeholder="先点击查询有OD车辆",
    )

    if st.button("应用OD车辆", key="history_od_apply_vehicles", use_container_width=True, disabled=not selected):
        st.session_state["history_od_active_vehicle_ids"] = [
            normalize_vehicle_id(item)
            for item in selected[:10]
            if normalize_vehicle_id(item)
        ]
        st.session_state["route_points"] = []
        st.session_state["route_result"] = None
        st.session_state["last_query"] = build_query_payload()
        st.rerun()


def validate_payload(payload):
    errors = []
    trajectory_vehicle_ids = payload.get("trajectory_vehicle_ids", [])

    if len(trajectory_vehicle_ids) > 10:
        errors.append("轨迹查询最多同时选择 10 辆车。")

    for item in trajectory_vehicle_ids:
        if item and not str(item).isdigit():
            errors.append("轨迹查询车辆ID必须为数字。")
            break

    if payload["start_time"] >= payload["end_time"]:
        errors.append("开始时间必须早于结束时间。")

    if payload["time_scale"] < 1 or payload["time_scale"] > 5:
        errors.append("动画倍速必须在 1-5 倍之间。")

    return errors


def render_sidebar():
    """Render vertical navigation menu + compact query controls in sidebar."""
    with st.sidebar:
        # ── App Title ──
        st.markdown("🚕 **GPS轨迹系统**")
        st.markdown("---")

        active_view = st.radio(
            "功能选择",
            VIEW_OPTIONS,
            index=VIEW_OPTIONS.index(st.session_state.get("active_view", "轨迹查询")),
            key="nav_radio",
            label_visibility="collapsed",
        )

        if active_view != st.session_state.get("active_view"):
            st.session_state["active_view"] = active_view
            st.rerun()

        st.markdown("**查询条件**")

        available_vehicle_ids = get_available_vehicle_ids()
        current_trajectory_ids = [
            str(vehicle_id).strip()
            for vehicle_id in st.session_state.get("trajectory_vehicle_ids", [])
            if str(vehicle_id).strip() in available_vehicle_ids
        ]
        if current_trajectory_ids != st.session_state.get("trajectory_vehicle_ids", []):
            st.session_state["trajectory_vehicle_ids"] = current_trajectory_ids

        st.date_input("日期", key="query_date", label_visibility="collapsed")

        st.multiselect(
            "车辆ID",
            options=available_vehicle_ids,
            key="trajectory_vehicle_ids",
            help="最多同时支持10车查询",
            placeholder="选择车辆ID",
        )

        if active_view == "路线规划":
            st.markdown("---")
            st.markdown("**路线规划**")
            if st.session_state.get("route_source_mode") not in {"地图选点", "历史OD端点"}:
                st.session_state["route_source_mode"] = "地图选点"
            route_mode = st.radio(
                "起终点来源",
                ["地图选点", "历史OD端点"],
                key="route_source_mode",
            )
            if route_mode == "地图选点":
                route_cols = st.columns(2)
                with route_cols[0]:
                    if st.button("重新选点", key="route_sidebar_reset_points", use_container_width=True):
                        st.session_state["route_map_reset_requested"] = True
                        st.rerun()
                with route_cols[1]:
                    can_calculate_route = len(st.session_state.get("route_points", [])) == 2
                    if st.button(
                        "计算路线",
                        key="route_sidebar_calculate",
                        use_container_width=True,
                        type="primary" if can_calculate_route else "secondary",
                        disabled=not can_calculate_route,
                    ):
                        st.session_state["route_map_calculate_requested"] = True
                        st.rerun()
            else:
                render_history_od_sidebar_tools()

        if active_view == "拥堵与ETA":
            st.markdown("---")
            st.markdown("**拥堵与ETA**")
            st.selectbox(
                "聚合时间片",
                options=[5, 15, 30, 60],
                index=[5, 15, 30, 60].index(int(st.session_state.get("congestion_bucket_minutes", 15))),
                key="congestion_bucket_minutes",
                format_func=lambda value: f"{value} 分钟",
            )
            st.checkbox("道路拥堵使用当天全日", key="congestion_use_full_day")

        c1, c2 = st.columns(2)
        with c1:
            st.text_input("开始", key="start_time_of_day", placeholder="08:00", label_visibility="collapsed")
        with c2:
            st.text_input("结束", key="end_time_of_day", placeholder="10:00", label_visibility="collapsed")

        st.text_input("分钟查询", key="minute_time_of_day", placeholder="09:30", label_visibility="collapsed")

        if active_view == "热力图与统计分析":
            selected_heatmap_panel = st.session_state.get("heatmap_analysis_panel", "静态热力图")
            if selected_heatmap_panel == "静态热力图":
                st.markdown("---")
                st.markdown("**静态热力图**")
                st.selectbox(
                    "数据来源",
                    options=["minute", "pickup"],
                    key="heatmap_source",
                    format_func=_source_label,
                )
                st.checkbox("采用聚类热力图", key="heatmap_enable_cluster")
            elif selected_heatmap_panel == "动态热力图":
                st.markdown("---")
                st.markdown("**动态热力图**")
                st.selectbox(
                    "数据来源",
                    options=["minute", "pickup"],
                    key="dynamic_source",
                    format_func=_source_label,
                )
                st.selectbox(
                    "时间粒度",
                    options=[1, 15, 30, 60],
                    key="dynamic_granularity",
                    format_func=lambda value: f"{value} 分钟",
                )
                if st.button("生成动态热力图", key="sidebar_dynamic_heatmap_submit", use_container_width=True, type="primary"):
                    st.session_state["dynamic_heatmap_sidebar_submitted"] = True

        query_cols = st.columns(2)
        with query_cols[0]:
            submitted = st.button("执行查询", use_container_width=True, type="primary")
        with query_cols[1]:
            if st.button("重置", use_container_width=True):
                reset_to_defaults()

        st.markdown(
            '<div style="padding:0.3rem 0.5rem;">'
            '<span style="font-size:0.75rem;color:#9ca3af;">清空缓存</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.sidebar.button("清空缓存", key="clear_cache_btn", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    return submitted


def render_summary(payload):
    cols = st.columns(4)
    trajectory_vehicle_ids = payload.get("trajectory_vehicle_ids", [])
    if trajectory_vehicle_ids:
        if len(trajectory_vehicle_ids) == 1:
            trajectory_vehicle_value = trajectory_vehicle_ids[0]
        else:
            preview_ids = "、".join(trajectory_vehicle_ids[:3])
            trajectory_vehicle_value = f"{len(trajectory_vehicle_ids)}辆: {preview_ids}"
            if len(trajectory_vehicle_ids) > 3:
                trajectory_vehicle_value += " ..."
    else:
        trajectory_vehicle_value = "全部车辆"

    cards = [
        ("车辆ID", trajectory_vehicle_value),
        ("轨迹区间", f"{payload['start_time'].strftime('%H:%M')} - {payload['end_time'].strftime('%H:%M')}"),
        ("分钟查询", payload["minute_time"].strftime("%H:%M")),
        ("参考坐标", f"{payload['ref_lat']:.4f}, {payload['ref_lng']:.4f}"),
    ]
    for col, (label, value) in zip(cols, cards):
        with col:
             st.markdown(
                f"""
                <div class="summary-card">
                  <div class="label">{label}</div>
                  <div class="value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
    )
    st.caption("轨迹读取车辆缓存，分钟读取分钟缓存，OD 标注读取完成后的 OD 表。")


def render_html_map(html_path, height=700, loading_message="加载中..."):
    if html_path and os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=height, scrolling=True)
    else:
        st.error("地图生成失败，请检查数据缓存是否存在。")


def render_vehicle_status_panel(trajectory_frames):
    if not trajectory_frames:
        return

    st.markdown("#### 多车状态概览")
    rows = []
    for vehicle_id, df in trajectory_frames:
        if df is None or len(df) == 0:
            continue
        points = len(df)
        avg_speed = float(pd.to_numeric(df.get("speed", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()) if "speed" in df.columns else 0.0
        occupied_points = int((df["status"] == 1).sum()) if "status" in df.columns else 0
        occupied_ratio = occupied_points / points if points else 0.0
        current_status = "载客" if int(df.iloc[-1]["status"]) == 1 else "空载"
        start_time = df.iloc[0]["time"].strftime("%H:%M:%S") if "time" in df.columns and pd.notna(df.iloc[0]["time"]) else "--"
        end_time = df.iloc[-1]["time"].strftime("%H:%M:%S") if "time" in df.columns and pd.notna(df.iloc[-1]["time"]) else "--"
        rows.append(
            {
                "vehicle_id": vehicle_id,
                "points": points,
                "avg_speed": avg_speed,
                "occupied_ratio": occupied_ratio,
                "current_status": current_status,
                "start_time": start_time,
                "end_time": end_time,
            }
        )

    if not rows:
        return

    card_css = dedent(
        """
        <style>
        .multi-vehicle-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
            margin: 0.5rem 0 0.75rem 0;
        }
        .multi-vehicle-card {
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 14px;
            padding: 12px 14px;
        }
        .multi-vehicle-card .vehicle-id {
            font-size: 0.95rem;
            font-weight: 800;
            color: #212121;
            margin-bottom: 0.35rem;
        }
        .multi-vehicle-card .line {
            font-size: 0.84rem;
            color: #616161;
            margin-bottom: 0.2rem;
        }
        .multi-vehicle-card .line strong {
            color: #212121;
        }
        @media (max-width: 1200px) {
            .multi-vehicle-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 768px) {
            .multi-vehicle-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """
    ).strip()
    cards_html = '<div class="multi-vehicle-grid">'
    for row in rows:
        cards_html += dedent(
            f"""
            <div class="multi-vehicle-card">
              <div class="vehicle-id">车辆 {row['vehicle_id']}</div>
              <div class="line">轨迹点 <strong>{row['points']}</strong></div>
              <div class="line">平均速度 <strong>{row['avg_speed']:.1f} km/h</strong></div>
              <div class="line">载客率 <strong>{row['occupied_ratio'] * 100:.0f}%</strong></div>
              <div class="line">终点状态 <strong>{row['current_status']}</strong></div>
              <div class="line">时间 <strong>{row['start_time']} - {row['end_time']}</strong></div>
            </div>
            """
        ).strip()
    cards_html += "</div>"
    st.markdown(card_css + "\n" + cards_html, unsafe_allow_html=True)


def render_trajectory_view(payload):
    st.subheader("轨迹查询")
    trajectory_vehicle_ids = payload.get("trajectory_vehicle_ids", [])
    if not trajectory_vehicle_ids:
        st.info("请选择至少一辆车后查看轨迹。")
        return

    control_cols = st.columns([1, 2])
    with control_cols[0]:
        enable_road_correction = st.checkbox(
            "启用路网校正",
            key="road_correction_enabled",
            help="将 GPS 点吸附到深圳路网节点，并用最短路径拼接校正轨迹。建议一次选择 1-3 辆车和较短时间窗口。",
        )
    with control_cols[1]:
        status = road_network_status()
        if enable_road_correction and status["available"]:
            st.caption(f"路网文件: {status['path']}")
        elif enable_road_correction:
            st.warning("未找到路网文件。请将 shenzhen_drive.pkl 或 shenzhen_drive.graphml 放到项目根目录、data/ 或 cache/，或设置 TAXIGPS_ROAD_NETWORK_PATH。")
        else:
            st.caption("点击生成的地图可显示经纬度；打开路网校正后会同时展示原始轨迹与道路校正轨迹。")

    if enable_road_correction and len(trajectory_vehicle_ids) > CONFIG["MAX_ROAD_CORRECTION_VEHICLES"]:
        st.warning(f"路网校正示例将仅处理前 {CONFIG['MAX_ROAD_CORRECTION_VEHICLES']} 辆车，避免一次性处理全量车辆。")

    with st.spinner("正在读取车辆缓存并生成轨迹地图..."):
        trajectory_frames = []
        missing_vehicle_ids = []
        for vehicle_id in trajectory_vehicle_ids[:10]:
            df = cached_vehicle_trajectory(vehicle_id, payload["start_time"], payload["end_time"])
            if df is None or len(df) == 0:
                missing_vehicle_ids.append(vehicle_id)
                continue
            trajectory_frames.append((vehicle_id, df))

        if missing_vehicle_ids:
            st.warning("以下车辆在当前时间范围内没有可显示的轨迹数据: " + "、".join(missing_vehicle_ids))

        if not trajectory_frames:
            st.warning("未找到符合条件的轨迹数据。")
            return

        if enable_road_correction:
            html_path, _info = plot_road_corrected_trajectories(
                [vehicle_id for vehicle_id, _ in trajectory_frames],
                payload["start_time"],
                payload["end_time"],
                enable_correction=True,
            )
            render_html_map(html_path, height=760)
        elif len(trajectory_frames) == 1:
            render_html_map(
                plot_vehicle_trajectory(trajectory_frames[0][0], payload["start_time"], payload["end_time"]),
                height=700,
            )
        else:
            render_html_map(
                plot_vehicle_trajectories(
                    [vehicle_id for vehicle_id, _ in trajectory_frames],
                    payload["start_time"],
                    payload["end_time"],
                ),
                height=700,
            )


def render_minute_view(payload):
    st.subheader("分钟位置查询")
    with st.spinner("正在读取分钟缓存并生成位置地图..."):
        df = cached_minute_data(payload["minute_time"])
        if df is None or len(df) == 0:
            st.warning("该分钟没有车辆数据。")
            return

        display_df = df
        selected_vehicle_ids = payload.get("trajectory_vehicle_ids", [])
        if selected_vehicle_ids:
            if len(selected_vehicle_ids) == 1:
                display_df = df[df["vehicle_id"] == str(selected_vehicle_ids[0])]
                if len(display_df) == 0:
                    st.warning(f"车辆 {selected_vehicle_ids[0]} 在该分钟没有位置数据。")
                    time_range = get_vehicle_time_range(selected_vehicle_ids[0])
                    if time_range:
                        st.caption(
                            f"该车辆缓存轨迹时间范围为 {time_range['min_time'].strftime('%Y-%m-%d %H:%M:%S')} "
                            f"至 {time_range['max_time'].strftime('%Y-%m-%d %H:%M:%S')}，可尝试调整分钟查询。"
                        )
                    return
            else:
                display_df = df[df["vehicle_id"].isin(selected_vehicle_ids)]
                if len(display_df) == 0:
                    st.warning("所选车辆在该分钟没有位置数据。")
                    return

        render_html_map(
            plot_minute_vehicles(
                payload["minute_time"],
                vehicle_ids=selected_vehicle_ids if selected_vehicle_ids else None,
            ),
            height=700,
        )


def render_od_view(payload):
    st.subheader("OD 上下车点")
    with st.spinner("正在读取 OD 表并生成上下车点地图..."):
        selected_vehicle_ids = payload.get("trajectory_vehicle_ids", [])
        if selected_vehicle_ids:
            df = cached_od_data(payload["start_time"], payload["end_time"], None, tuple(selected_vehicle_ids))
        else:
            df = cached_od_data(payload["start_time"], payload["end_time"], None, None)
        if df is None or len(df) == 0:
            if selected_vehicle_ids:
                if len(selected_vehicle_ids) == 1:
                    st.warning(f"车辆 {selected_vehicle_ids[0]} 在当前时间范围内没有 OD 记录。")
                    time_range = get_vehicle_time_range(selected_vehicle_ids[0])
                else:
                    st.warning("所选车辆在当前时间范围内没有 OD 记录。")
                    time_range = None
                if time_range:
                    st.caption(
                        f"该车辆轨迹缓存范围为 {time_range['min_time'].strftime('%Y-%m-%d %H:%M:%S')} "
                        f"至 {time_range['max_time'].strftime('%Y-%m-%d %H:%M:%S')}，建议扩大时间范围后重试。"
                    )
            else:
                st.warning("当前时间范围内未找到 OD 数据。")
            return

        render_html_map(
            plot_od_points(
                payload["start_time"],
                payload["end_time"],
                None,
                selected_vehicle_ids if selected_vehicle_ids else None,
            ),
            height=700,
        )


def render_animation_view(payload):
    animation_vehicle_ids = payload.get("trajectory_vehicle_ids", [])

    if not animation_vehicle_ids:
        st.subheader("动画轨迹")
        st.info("请选择至少一辆车后播放动画轨迹。")
        return

    st.subheader("多车动画轨迹" if len(animation_vehicle_ids) > 1 else "单车动画轨迹")

    with st.spinner("正在生成动画轨迹..."):
        animation_frames = []
        missing_vehicle_ids = []
        for vehicle_id in animation_vehicle_ids[:10]:
            df = cached_vehicle_trajectory(vehicle_id, payload["start_time"], payload["end_time"])
            if df is None or len(df) < 2:
                missing_vehicle_ids.append(vehicle_id)
                continue
            animation_frames.append((vehicle_id, df))

        if missing_vehicle_ids:
            st.warning("以下车辆在当前时间范围内轨迹点不足，已跳过动画: " + "、".join(missing_vehicle_ids))

        if not animation_frames:
            st.warning("轨迹点不足，无法生成动画。")
            return

        if len(animation_frames) == 1:
            render_html_map(
                plot_animated_trajectory(
                    animation_frames[0][0],
                    payload["start_time"],
                    payload["end_time"],
                    payload["time_scale"],
                ),
                height=840,
            )
        else:
            render_html_map(
                plot_multi_vehicle_animated_trajectory(
                    [vehicle_id for vehicle_id, _ in animation_frames],
                    payload["start_time"],
                    payload["end_time"],
                    payload["time_scale"],
                ),
                height=940,
            )


def render_query_status(payload):
    """显示当前查询状态，帮助用户确认是否在正确范围内查询。"""
    trajectory_vehicle_ids = payload.get("trajectory_vehicle_ids", [])
    if len(trajectory_vehicle_ids) == 1:
        time_range = get_vehicle_time_range(trajectory_vehicle_ids[0])
        if time_range:
            st.info(
                f"当前车辆 {trajectory_vehicle_ids[0]} 的原始轨迹缓存可用范围为 "
                f"{time_range['min_time'].strftime('%Y-%m-%d %H:%M:%S')} 至 "
                f"{time_range['max_time'].strftime('%Y-%m-%d %H:%M:%S')}；"
                f"当前查询范围为 {payload['start_time'].strftime('%Y-%m-%d %H:%M:%S')} 至 "
                f"{payload['end_time'].strftime('%Y-%m-%d %H:%M:%S')}。"
            )
    elif len(trajectory_vehicle_ids) > 1:
        st.info(f"当前已选择 {len(trajectory_vehicle_ids)} 辆车进行轨迹并行展示。")
    else:
        st.caption("当前为全部车辆模式，结果将按时间范围查询并展示汇总预览。")


def _source_label(source_type):
    return "分钟缓存车辆位置" if source_type == "minute" else "OD 上车点"


def _normalize_vehicle_scope(vehicle_ids):
    return tuple(vehicle_ids or ())


def _resolve_heatmap_vehicle_scope(source_type, selected_vehicle_ids):
    if source_type == "minute":
        return ()
    return tuple(selected_vehicle_ids or ())


def render_static_heatmap_tab(payload):
    selected_vehicle_ids = _normalize_vehicle_scope(payload.get("trajectory_vehicle_ids", []))
    selected_source_type = st.session_state.get("heatmap_source", "pickup")
    enable_cluster = bool(st.session_state.get("heatmap_enable_cluster", False))
    request_vehicle_ids = _resolve_heatmap_vehicle_scope(selected_source_type, selected_vehicle_ids)

    request = {
        "source_type": selected_source_type,
        "enable_cluster": enable_cluster,
        "eps_km": float(st.session_state.get("heatmap_eps_km", DEFAULTS["heatmap_eps_km"])),
        "min_samples": int(st.session_state.get("heatmap_min_samples", DEFAULTS["heatmap_min_samples"])),
        "threshold_quantile": float(st.session_state.get("heatmap_threshold_quantile", DEFAULTS["heatmap_threshold_quantile"])),
        "start_time": payload["start_time"],
        "end_time": payload["end_time"],
        "vehicle_ids": request_vehicle_ids,
    }
    if st.session_state.get("static_heatmap_request") != request:
        st.session_state["static_heatmap_request"] = request

    with st.spinner("正在生成静态热力图..."):
        html_path, _info = cached_static_heatmap(
            source_type=request["source_type"],
            start_time=request["start_time"],
            end_time=request["end_time"],
            vehicle_ids=request["vehicle_ids"] or None,
            enable_cluster=request["enable_cluster"],
            eps_km=request["eps_km"],
            min_samples=request["min_samples"],
            threshold_quantile=request["threshold_quantile"],
        )

    if not html_path:
        st.warning("当前条件下没有可用于静态热力图的数据。")
        return

    render_html_map(html_path, height=820)


def render_dynamic_heatmap_tab(payload):
    selected_vehicle_ids = _normalize_vehicle_scope(payload.get("trajectory_vehicle_ids", []))
    submitted = bool(st.session_state.get("dynamic_heatmap_sidebar_submitted", False))
    st.session_state["dynamic_heatmap_sidebar_submitted"] = False

    if submitted:
        dynamic_source_type = st.session_state.get("dynamic_source", "pickup")
        request_vehicle_ids = _resolve_heatmap_vehicle_scope(dynamic_source_type, selected_vehicle_ids)
        st.session_state["dynamic_heatmap_request"] = {
            "source_type": dynamic_source_type,
            "requested_granularity": int(st.session_state.get("dynamic_granularity", DEFAULTS["dynamic_granularity"])),
            "vehicle_ids": request_vehicle_ids,
            "smoothing_method": st.session_state.get("dynamic_smoothing", DEFAULTS["dynamic_smoothing"]),
            "ema_alpha": float(st.session_state.get("dynamic_ema_alpha", DEFAULTS["dynamic_ema_alpha"])),
            "wma_window": int(st.session_state.get("dynamic_wma_window", DEFAULTS["dynamic_wma_window"])),
            "threshold_quantile": float(st.session_state.get("dynamic_threshold_quantile", DEFAULTS["dynamic_threshold_quantile"])),
            "start_time": payload["start_time"],
            "end_time": payload["end_time"],
        }

    request = st.session_state.get("dynamic_heatmap_request")
    if request and request["source_type"] == "minute" and request.get("vehicle_ids"):
        request = {**request, "vehicle_ids": ()}
        st.session_state["dynamic_heatmap_request"] = request
    if request is None:
        return

    with st.spinner("正在生成动态热力图..."):
        html_path, _info = cached_dynamic_heatmap(
            source_type=request["source_type"],
            start_time=request["start_time"],
            end_time=request["end_time"],
            requested_granularity=request["requested_granularity"],
            vehicle_ids=request["vehicle_ids"] or None,
            smoothing_method=request["smoothing_method"],
            ema_alpha=request["ema_alpha"],
            wma_window=request["wma_window"],
            threshold_quantile=request["threshold_quantile"],
        )

    if not html_path:
        st.warning("当前条件下没有可用于动态热力图的数据。")
        return

    render_html_map(html_path, height=820)

def render_order_statistics_tab(payload):
    st.markdown("#### 订单统计分析")
    selected_vehicle_ids = _normalize_vehicle_scope(payload.get("trajectory_vehicle_ids", []))
    with st.form("order_stats_form", clear_on_submit=False):
        submitted = st.form_submit_button("生成订单统计", use_container_width=True, type="primary")

    if submitted:
        st.session_state["order_stats_request"] = {
            "start_time": payload["start_time"],
            "end_time": payload["end_time"],
            "vehicle_ids": selected_vehicle_ids,
        }

    request = st.session_state["order_stats_request"]
    if request is None:
        st.info("当前仅显示订单统计配置。点击“生成订单统计”后才会开始计算。")
        return
    st.caption("切换功能模块时不会重新计算订单统计；只有点击“生成订单统计”才会更新结果。")
    with st.spinner("正在计算订单统计..."):
        stats = cached_order_statistics(request["start_time"], request["end_time"], request["vehicle_ids"] or None)

    summary = stats["summary"]
    cols = st.columns(4)
    metrics = [
        ("订单数", summary["order_count"]),
        ("平均里程", f"{summary['avg_distance_km']:.2f} km"),
        ("载客车辆峰值", summary["occupied_vehicle_peak"]),
        ("载客率峰值", f"{summary['occupancy_rate_peak'] * 100:.1f}%"),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)

    hourly_df = stats["hourly"].copy()
    if len(hourly_df):
        hourly_df["hour_label"] = pd.to_datetime(hourly_df["hour"]).dt.strftime("%H:%M")
        order_chart = (
            alt.Chart(hourly_df)
            .mark_line(point=True, color="#d97706")
            .encode(x=alt.X("hour_label:N", title="小时"), y=alt.Y("order_count:Q", title="订单数"))
            .properties(height=280, title="小时订单数趋势")
        )
        vehicle_chart = (
            alt.Chart(hourly_df)
            .mark_line(point=True, color="#2563eb")
            .encode(x=alt.X("hour_label:N", title="小时"), y=alt.Y("occupied_vehicles:Q", title="载客车辆数"))
            .properties(height=280, title="小时载客车辆数趋势")
        )
        occupancy_chart = (
            alt.Chart(hourly_df)
            .mark_area(line={"color": "#16a34a"}, color="#86efac", opacity=0.55)
            .encode(x=alt.X("hour_label:N", title="小时"), y=alt.Y("occupancy_rate:Q", title="载客率", axis=alt.Axis(format="%")))
            .properties(height=280, title="小时载客率趋势")
        )
        st.altair_chart(order_chart, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(vehicle_chart, use_container_width=True)
        with c2:
            st.altair_chart(occupancy_chart, use_container_width=True)
    else:
        st.info("当前时间范围内没有可视化的小时统计结果。")

    bucket_df = stats["distance_buckets"].copy()
    if len(bucket_df):
        bucket_chart = (
            alt.Chart(bucket_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#0f766e")
            .encode(x=alt.X("里程区间:N", title="里程区间"), y=alt.Y("订单数:Q", title="订单数"), tooltip=["里程区间", "订单数", alt.Tooltip("占比:Q", format=".2%")])
            .properties(height=260, title="订单里程结构")
        )
        st.altair_chart(bucket_chart, use_container_width=True)
        st.dataframe(bucket_df, use_container_width=True, hide_index=True)

    st.caption(f"统计口径: {stats.get('stat_definition', '--')} 分钟明细行数 {stats.get('minute_rows', 0)}，OD 行数 {stats.get('od_rows', 0)}。")


def render_operation_statistics_tab(payload):
    st.markdown("#### 车辆运营统计")
    selected_vehicle_ids = _normalize_vehicle_scope(payload.get("trajectory_vehicle_ids", []))
    with st.form("operation_stats_form", clear_on_submit=False):
        submitted = st.form_submit_button("生成车辆运营统计", use_container_width=True, type="primary")

    if submitted:
        st.session_state["operation_stats_request"] = {
            "query_date": payload["start_time"].date(),
            "vehicle_ids": selected_vehicle_ids,
            "start_time": payload["start_time"],
            "end_time": payload["end_time"],
        }

    request = st.session_state["operation_stats_request"]
    if request is None:
        st.info("当前仅显示车辆运营统计配置。点击“生成车辆运营统计”后才会开始计算。")
        return
    st.caption("切换功能模块时不会重新计算车辆运营统计；只有点击“生成车辆运营统计”才会更新结果。")
    with st.spinner("正在计算全天运营统计..."):
        operation_stats = cached_daily_operation_statistics(request["query_date"], request["vehicle_ids"] or None)

    cols = st.columns(4)
    metrics = [
        ("运营车辆数", operation_stats["vehicle_count"]),
        ("全天载客率", f"{operation_stats['occupancy_ratio'] * 100:.2f}%"),
        ("总里程估计", f"{operation_stats['total_distance_km']:.2f} km"),
        ("空载里程估计", f"{operation_stats['empty_distance_km']:.2f} km"),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)

    st.dataframe(operation_stats["summary_table"], use_container_width=True, hide_index=True)
    st.caption(f"统计口径: {operation_stats.get('stat_definition', '--')}")

    if st.button("导出统计结果为 CSV / Excel", use_container_width=True):
        order_stats = cached_order_statistics(request["start_time"], request["end_time"], request["vehicle_ids"] or None)
        cluster_result = cached_pickup_cluster_analysis(
            request["start_time"],
            request["end_time"],
            request["vehicle_ids"] or None,
            float(st.session_state["pickup_cluster_eps_km"]),
            int(st.session_state["pickup_cluster_min_samples"]),
            float(st.session_state["pickup_cluster_threshold_quantile"]),
        )
        export_info = cached_export_bundle(request["query_date"], order_stats, operation_stats, cluster_result)
        st.success(f"统计结果已导出到: {export_info['export_dir']}")
        st.caption(f"CSV: {' | '.join(export_info['csv_files'])}")
        st.caption(f"Excel: {export_info['xlsx_file']}")


def render_pickup_cluster_tab(payload):
    st.markdown("#### 上车点聚类分析")
    selected_vehicle_ids = _normalize_vehicle_scope(payload.get("trajectory_vehicle_ids", []))
    rec, _ = cached_source_recommendation("pickup", payload["start_time"], payload["end_time"], selected_vehicle_ids or None)

    with st.form("pickup_cluster_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("eps(km)", min_value=0.05, max_value=2.0, step=0.05, key="pickup_cluster_eps_km", format="%.2f")
        with c2:
            st.number_input("min_samples", min_value=2, max_value=50, step=1, key="pickup_cluster_min_samples")
        with c3:
            st.slider("热度阈值分位数", 0.50, 0.99, key="pickup_cluster_threshold_quantile", step=0.01)
        st.caption(
            f"参数建议: eps≈{rec['eps_km']} km, min_samples≈{rec['min_samples']}。聚类热力值定义为簇内上车订单累计权重，地图按聚类中心而非原始点渲染。"
        )
        submitted = st.form_submit_button("执行上车点聚类", use_container_width=True, type="primary")

    if submitted or st.session_state["pickup_cluster_request"] is None:
        st.session_state["pickup_cluster_request"] = {
            "start_time": payload["start_time"],
            "end_time": payload["end_time"],
            "vehicle_ids": selected_vehicle_ids,
            "eps_km": float(st.session_state["pickup_cluster_eps_km"]),
            "min_samples": int(st.session_state["pickup_cluster_min_samples"]),
            "threshold_quantile": float(st.session_state["pickup_cluster_threshold_quantile"]),
        }

    request = st.session_state["pickup_cluster_request"]
    st.caption("参数调整不会立即重算，点击“执行上车点聚类”后才会执行。")
    with st.spinner("正在执行上车点 DBSCAN 聚类..."):
        cluster_result = cached_pickup_cluster_analysis(
            request["start_time"],
            request["end_time"],
            request["vehicle_ids"] or None,
            request["eps_km"],
            request["min_samples"],
            request["threshold_quantile"],
        )

    clusters = cluster_result.get("clusters", pd.DataFrame())
    if len(clusters) == 0:
        st.warning("当前条件下未形成有效聚类簇，请尝试放宽时间范围或调大 eps。")
        return

    cols = st.columns(4)
    metrics = [
        ("聚类簇数", len(clusters)),
        ("最大热力值", f"{clusters['heat_value'].max():.0f}"),
        ("平均热力值", f"{clusters['heat_value'].mean():.1f}"),
        ("噪声点数", len(cluster_result.get('noise', pd.DataFrame()))),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)

    st.caption(
        f"来源标识: {cluster_result['meta'].get('source_label', '--')}；热力值定义: {cluster_result['meta'].get('cluster_heat_value_definition', '--')}；"
        f"地图输出 {cluster_result['meta'].get('output_path', '--')}。"
    )
    render_html_map(cluster_result.get("map_path"), height=760)
    st.dataframe(clusters, use_container_width=True, hide_index=True)


def render_heatmap_stats_view(payload):
    st.subheader("热力图与统计分析")
    selected_panel = st.radio(
        "分析功能",
        ["静态热力图", "动态热力图", "订单统计", "车辆运营", "上车点聚类"],
        key="heatmap_analysis_panel",
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_panel == "静态热力图":
        render_static_heatmap_tab(payload)
    elif selected_panel == "动态热力图":
        render_dynamic_heatmap_tab(payload)
    elif selected_panel == "订单统计":
        render_order_statistics_tab(payload)
    elif selected_panel == "车辆运营":
        render_operation_statistics_tab(payload)
    elif selected_panel == "上车点聚类":
        render_pickup_cluster_tab(payload)


def _eta_route_result_for_component(eta_result):
    if not eta_result or not eta_result.get("success"):
        return None
    alternatives = eta_result.get("alternatives") or []
    shortest = alternatives[0] if alternatives else {}
    fastest = alternatives[1] if len(alternatives) > 1 else shortest
    return {
        "success": True,
        "shortest": {
            "points": shortest.get("route_points", eta_result.get("route_points", [])),
            "distance_m": float(shortest.get("distance_km", eta_result.get("distance_km", 0.0))) * 1000.0,
            "route_cost_s": float(shortest.get("eta_minutes", eta_result.get("eta_minutes", 0.0))) * 60.0,
            "edge_count": int(shortest.get("edge_count", len(shortest.get("route_segments", [])))),
        },
        "fastest": {
            "points": fastest.get("route_points", eta_result.get("route_points", [])),
            "distance_m": float(fastest.get("distance_km", eta_result.get("distance_km", 0.0))) * 1000.0,
            "route_cost_s": float(fastest.get("eta_minutes", eta_result.get("eta_minutes", 0.0))) * 60.0,
            "edge_count": int(fastest.get("edge_count", len(fastest.get("route_segments", [])))),
        },
    }


@st.cache_data(show_spinner=False, ttl=3600)
def _load_shenzhen_boundary_geojson():
    for path in CONFIG.get("BOUNDARY_PATHS", []):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                logger.exception("深圳边界文件读取失败: %s", path)
                return None
    return None


def _render_eta_map_picker(eta_result=None):
    import importlib
    import route_planner

    render_route_component_map = importlib.reload(route_planner).render_route_component_map

    st.session_state.setdefault("eta_route_points", [])
    click_payload = render_route_component_map(
        st.session_state["eta_route_points"],
        _eta_route_result_for_component(eta_result),
        key=f"eta_route_map_{len(st.session_state['eta_route_points'])}_{bool(eta_result and eta_result.get('success'))}",
        height=620,
        boundary_geojson=_load_shenzhen_boundary_geojson(),
    )
    if isinstance(click_payload, dict) and "lat" in click_payload and "lng" in click_payload:
        clicked_lat = float(click_payload["lat"])
        clicked_lng = float(click_payload["lng"])
        current_click = str(click_payload.get("nonce") or f"{clicked_lat}_{clicked_lng}")
        if st.session_state.get("last_eta_route_click") != current_click:
            st.session_state["last_eta_route_click"] = current_click
            if len(st.session_state["eta_route_points"]) >= 2:
                st.session_state["eta_route_points"] = []
            st.session_state["eta_route_points"].append({"lat": clicked_lat, "lng": clicked_lng})
            if len(st.session_state["eta_route_points"]) >= 1:
                st.session_state["eta_origin_lat"] = st.session_state["eta_route_points"][0]["lat"]
                st.session_state["eta_origin_lng"] = st.session_state["eta_route_points"][0]["lng"]
            if len(st.session_state["eta_route_points"]) >= 2:
                st.session_state["eta_dest_lat"] = st.session_state["eta_route_points"][1]["lat"]
                st.session_state["eta_dest_lng"] = st.session_state["eta_route_points"][1]["lng"]
            st.rerun()


def _render_eta_explanation(eta_result):
    alternatives = eta_result.get("alternatives") or []
    if alternatives:
        rows = []
        for item in alternatives:
            rows.append(
                {
                    "路线方案": item.get("label"),
                    "距离(km)": round(float(item.get("distance_km", 0.0)), 3),
                    "ETA(分钟)": round(float(item.get("eta_minutes", 0.0)), 2),
                    "估算均速(km/h)": round(float(item.get("avg_speed_kmh", 0.0)), 2),
                    "道路边数": item.get("edge_count", 0),
                    "缺失速度边": item.get("missing_speed_edges", 0),
                    "路径模式": item.get("path_mode", ""),
                }
            )
        st.markdown("#### 路线方案对比")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    speed_sources = eta_result.get("speed_sources") or {}
    if speed_sources:
        source_df = pd.DataFrame(
            [{"速度来源": key, "路段数": value} for key, value in speed_sources.items()]
        ).sort_values("路段数", ascending=False)
        chart = (
            alt.Chart(source_df)
            .mark_bar(color="#2563eb")
            .encode(
                x=alt.X("路段数:Q", title="路段数"),
                y=alt.Y("速度来源:N", sort="-x", title="速度来源"),
                tooltip=["速度来源", "路段数"],
            )
        )
        st.markdown("#### ETA速度来源")
        st.altair_chart(chart, use_container_width=True)

    slow_segments = eta_result.get("slow_segments") or []
    if slow_segments:
        slow_df = pd.DataFrame(slow_segments)[["segment_key", "length_km", "speed_kmh", "eta_minutes", "speed_source"]].copy()
        slow_df = slow_df.rename(
            columns={
                "segment_key": "道路边",
                "length_km": "长度(km)",
                "speed_kmh": "速度(km/h)",
                "eta_minutes": "耗时(分钟)",
                "speed_source": "速度来源",
            }
        )
        st.markdown("#### 主要慢速路段")
        st.dataframe(slow_df.round(3), use_container_width=True, hide_index=True)



def _congestion_time_window(payload):
    if st.session_state.get("congestion_use_full_day", True):
        day = payload["start_time"].date()
        return datetime.combine(day, time(0, 0)), datetime.combine(day, time(23, 59, 59))
    return payload["start_time"], payload["end_time"]


def _congestion_vehicle_scope(selected_vehicle_ids):
    available_vehicle_ids = get_available_vehicle_ids()
    cleaned = [str(item).strip() for item in selected_vehicle_ids or [] if str(item).strip()]
    if cleaned:
        return cleaned[: int(st.session_state.get("hmm_aggregation_max_vehicles", 8))]
    return available_vehicle_ids[: int(st.session_state.get("hmm_aggregation_max_vehicles", 8))]


def render_congestion_roads_tab(payload, selected_vehicle_ids):
    bucket_minutes = int(st.session_state.get("congestion_bucket_minutes", 15))
    congestion_start, congestion_end = _congestion_time_window(payload)
    aggregation_vehicle_ids = _congestion_vehicle_scope(selected_vehicle_ids)

    with st.expander("道路速度缓存", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            st.number_input("车辆上限", min_value=1, max_value=50, step=1, key="hmm_aggregation_max_vehicles")
        with cols[1]:
            st.number_input("最小样本数", min_value=1, max_value=20, step=1, key="hmm_aggregation_min_samples")
        with cols[2]:
            st.checkbox("强制重建", key="hmm_aggregation_force_rebuild")
        if st.button("生成/更新拥堵速度缓存", key="congestion_rebuild_speed_cache", use_container_width=True, type="primary"):
            st.session_state["congestion_rebuild_requested"] = True

    if st.session_state.pop("congestion_rebuild_requested", False):
        if not aggregation_vehicle_ids:
            st.warning("没有可用于聚合的车辆缓存。")
        else:
            with st.spinner("正在执行HMM路网校正并聚合道路速度..."):
                result = cached_hmm_speed_aggregation(
                    aggregation_vehicle_ids,
                    congestion_start,
                    congestion_end,
                    bucket_minutes,
                    int(st.session_state.get("hmm_aggregation_min_samples", 3)),
                    bool(st.session_state.get("hmm_aggregation_force_rebuild", False)),
                )
            if result.get("success"):
                st.success(
                    f"速度缓存已更新：可靠道路边 {result.get('reliable_edges', 0)} 条，"
                    f"时间片记录 {result.get('time_slice_rows', 0)} 条。"
                )
                st.cache_data.clear()
            else:
                st.error(result.get("error", "拥堵速度缓存生成失败。"))

    with st.spinner("正在读取道路速度缓存并生成拥堵地图..."):
        congestion_html, congestion_meta = cached_congestion_roads(
            selected_vehicle_ids,
            congestion_start,
            congestion_end,
            bucket_minutes,
        )

    if not congestion_meta or not congestion_meta.get("success"):
        st.error((congestion_meta or {}).get("error", "拥堵道路生成失败。"))
        return

    render_html_map(congestion_html, height=800)

def render_eta_prediction_tab(payload, selected_vehicle_ids):
    bucket_minutes = int(st.session_state.get("congestion_bucket_minutes", 15))
    mode_cols = st.columns([1, 1])
    with mode_cols[0]:
        input_mode = st.radio("ETA输入方式", ["地图选点", "文本坐标"], horizontal=True, label_visibility="collapsed")
    with mode_cols[1]:
        eta_mode = st.radio("ETA模型", ["规则版", "模型版"], key="eta_mode", horizontal=True, label_visibility="collapsed")

    eta_result = None
    if eta_mode == "模型版":
        st.number_input("训练订单上限", min_value=100, max_value=3000, step=100, key="eta_model_train_orders")

    if input_mode == "文本坐标":
        eta_cols = st.columns(4)
        with eta_cols[0]:
            origin_lat = st.number_input("起点纬度", format="%.6f", key="eta_origin_lat")
        with eta_cols[1]:
            origin_lng = st.number_input("起点经度", format="%.6f", key="eta_origin_lng")
        with eta_cols[2]:
            dest_lat = st.number_input("终点纬度", format="%.6f", key="eta_dest_lat")
        with eta_cols[3]:
            dest_lng = st.number_input("终点经度", format="%.6f", key="eta_dest_lng")
    else:
        if st.button("重新选点", use_container_width=True):
            st.session_state["eta_route_points"] = []
            st.rerun()
        origin_lat = float(st.session_state.get("eta_origin_lat", CONFIG["MAP_CENTER"][0]))
        origin_lng = float(st.session_state.get("eta_origin_lng", CONFIG["MAP_CENTER"][1]))
        dest_lat = float(st.session_state.get("eta_dest_lat", 22.6008))
        dest_lng = float(st.session_state.get("eta_dest_lng", 114.1010))

    can_estimate = input_mode == "文本坐标" or len(st.session_state.get("eta_route_points", [])) >= 2
    if can_estimate:
        if eta_mode == "模型版":
            with st.spinner("正在训练OD耗时回归模型并预测ETA..."):
                eta_result = cached_eta_model_result(
                    origin_lat,
                    origin_lng,
                    dest_lat,
                    dest_lng,
                    selected_vehicle_ids,
                    payload["start_time"],
                    payload["end_time"],
                    bucket_minutes,
                    int(st.session_state.get("eta_model_train_orders", 800)),
                )
        else:
            with st.spinner("正在按路网边速度缓存估算ETA..."):
                eta_result = cached_eta_result(
                    origin_lat,
                    origin_lng,
                    dest_lat,
                    dest_lng,
                    selected_vehicle_ids,
                    payload["start_time"],
                    payload["end_time"],
                    bucket_minutes,
                )

    if input_mode == "地图选点":
        _render_eta_map_picker(eta_result)
        if not can_estimate:
            st.info("请在地图上依次点击起点和终点。")
            return

    if not eta_result or not eta_result.get("success"):
        st.error((eta_result or {}).get("error", "ETA 估算失败。"))
        return

    eta_metric_cols = st.columns(5)
    eta_metric_cols[0].metric("预测耗时", f"{eta_result['eta_minutes']:.1f} 分钟")
    eta_metric_cols[1].metric("路网距离", f"{eta_result.get('distance_km', 0):.2f} km")
    eta_metric_cols[2].metric("估算均速", f"{eta_result.get('avg_speed_kmh', 0):.1f} km/h")
    if eta_mode == "模型版":
        eta_metric_cols[3].metric("原始模型输出", f"{eta_result.get('raw_eta_minutes', eta_result['eta_minutes']):.1f} 分钟")
        eta_metric_cols[4].metric("训练/测试", f"{eta_result.get('train_rows', 0)} / {eta_result.get('test_rows', 0)}")
    else:
        eta_metric_cols[3].metric("速度回退边", eta_result.get("missing_speed_edges", 0))
        eta_metric_cols[4].metric("路径模式", eta_result.get("path_mode", "--"))
    st.caption(eta_result.get("method", ""))

    if input_mode == "文本坐标":
        eta_path = plot_eta_route(eta_result)
        render_html_map(eta_path, height=620)

    if eta_mode == "模型版":
        metrics = eta_result.get("model_metrics", {})
        if metrics:
            metric_df = pd.DataFrame([{"指标": key, "值": round(float(value), 4)} for key, value in metrics.items()])
            st.markdown("#### 模型指标")
            st.dataframe(metric_df, use_container_width=True, hide_index=True)
        sample_errors = eta_result.get("sample_errors") or []
        if sample_errors:
            st.markdown("#### 样例误差")
            st.dataframe(pd.DataFrame(sample_errors), use_container_width=True, hide_index=True)
        feature_values = eta_result.get("feature_values") or {}
        if feature_values:
            feature_labels = eta_result.get("feature_labels") or {}
            feature_df = pd.DataFrame(
                [
                    {"特征": feature_labels.get(key, key), "字段": key, "当前值": round(float(value), 4)}
                    for key, value in feature_values.items()
                ]
            )
            with st.expander("查看当前预测特征", expanded=False):
                st.dataframe(feature_df, use_container_width=True, hide_index=True)
    else:
        _render_eta_explanation(eta_result)


def render_cache_validation_tab(payload, selected_vehicle_ids):
    status = road_network_status()
    od_df, od_meta = load_completed_od_cache()
    speed_path = CONFIG.get("BASELINE_SPEED_CACHE_PATH", "cache/edge_baseline_speed.csv")
    road_cache_dir = CONFIG["ROAD_CORRECTED_CACHE_DIR"]
    road_files = []
    if os.path.isdir(road_cache_dir):
        road_files = [name for name in os.listdir(road_cache_dir) if name.lower().endswith(".csv")]
    speed_exists = os.path.exists(speed_path)
    rows = [
        {"缓存项": "路网文件", "状态": "可用" if status.get("available") else "缺失", "路径/说明": status.get("path") or status.get("error", "")},
        {"缓存项": "校正OD缓存", "状态": "可用" if od_meta.get("success") else "缺失", "路径/说明": od_meta.get("cache_path") or od_meta.get("error", "")},
        {"缓存项": "路网校正轨迹缓存", "状态": "可用" if road_files else "缺失", "路径/说明": f"{road_cache_dir} | 文件 {len(road_files)} 个"},
        {"缓存项": "道路速度缓存", "状态": "可用" if speed_exists else "缺失", "路径/说明": speed_path},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("拥堵与ETA页面只读取以上缓存；不会扫描原始GPS大表，也不会在查询时重新执行路网校正。")

    selected = selected_vehicle_ids or []
    if selected:
        details = []
        for vehicle_id in selected[: CONFIG["MAX_CONGESTION_VEHICLES"]]:
            matches = [name for name in road_files if name.startswith(f"{vehicle_id}_")]
            details.append({"车辆ID": vehicle_id, "路网校正缓存": "有" if matches else "无", "文件数": len(matches)})
        st.markdown("#### 已选车辆缓存覆盖")
        st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)


def render_congestion_eta_view(payload):
    st.subheader("拥堵与ETA")
    trajectory_vehicle_ids = payload.get("trajectory_vehicle_ids", [])
    selected_vehicle_ids = trajectory_vehicle_ids[: CONFIG["MAX_CONGESTION_VEHICLES"]]

    status = road_network_status()
    if status["available"]:
        st.caption(f"路网文件: {status['path']}")
    else:
        st.warning("未找到路网文件。请将 shenzhen_drive.pkl 或 shenzhen_drive.graphml 放到项目根目录、data/ 或 cache/，或设置 TAXIGPS_ROAD_NETWORK_PATH。")
        return

    tabs = st.tabs(["道路拥堵", "ETA预测", "缓存校验"])
    with tabs[0]:
        render_congestion_roads_tab(payload, selected_vehicle_ids)
    with tabs[1]:
        render_eta_prediction_tab(payload, selected_vehicle_ids)
    with tabs[2]:
        render_cache_validation_tab(payload, selected_vehicle_ids)




def main():
    try:
        init_page()
        init_state()

        st.session_state.setdefault("last_query", build_query_payload())

        submitted = render_sidebar()

        payload = build_query_payload()
        if submitted:
            errors = validate_payload(payload)
            if errors:
                for error in errors:
                    st.error(error)
                return
            st.session_state["last_query"] = payload
            st.session_state["static_heatmap_request"] = None
            st.session_state["dynamic_heatmap_request"] = None
            st.session_state["pickup_cluster_request"] = None
            st.session_state["order_stats_request"] = None
            st.session_state["operation_stats_request"] = None
            st.success("查询条件已更新。")

        active_payload = st.session_state.get("last_query") or payload
        active_view = st.session_state.get("active_view", "轨迹查询")

        previous_active_view = st.session_state.get("last_active_view")
        if active_view == "热力图与统计分析" and previous_active_view != "热力图与统计分析":
            st.session_state["heatmap_analysis_panel"] = "静态热力图"
        st.session_state["last_active_view"] = active_view

        if active_view == "轨迹查询":
            render_trajectory_view(active_payload)
        elif active_view == "动画轨迹":
            render_animation_view(active_payload)
        elif active_view == "分钟位置":
            render_minute_view(active_payload)
        elif active_view == "OD点标注":
            render_od_view(active_payload)
        elif active_view == "热力图与统计分析":
            render_heatmap_stats_view(active_payload)
        elif active_view == "路线规划":
            from route_planner import render_route_planning_view
            render_route_planning_view(active_payload)
        elif active_view == "拥堵与ETA":
            render_congestion_eta_view(active_payload)

    except Exception:
        logger.exception("页面主流程错误")
        st.error("页面加载失败，请稍后重试。")


if __name__ == "__main__":
    if is_running_under_streamlit():
        main()
    else:
        print("请使用以下命令启动 Streamlit 应用：")
        print("streamlit run src/streamlit_app.py")
