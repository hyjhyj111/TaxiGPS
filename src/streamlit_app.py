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
    plan_baseline_routes_between_points,
    plot_baseline_route_comparison,
    plot_animated_trajectory,
    plot_congestion_roads,
    plot_eta_route,
    plot_multi_vehicle_animated_trajectory,
    plot_minute_vehicles,
    plot_od_points,
    plot_road_corrected_trajectories,
    plot_vehicle_trajectories,
    plot_vehicle_trajectory,
    road_network_status,
)


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
                    "路线规划": "最短距离与基准最快路线规划",
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
    "eta_origin_lat": float(CONFIG["MAP_CENTER"][0]),
    "eta_origin_lng": float(CONFIG["MAP_CENTER"][1]),
    "eta_dest_lat": 22.6008,
    "eta_dest_lng": 114.1010,
    "route_origin_lat": float(CONFIG["MAP_CENTER"][0]),
    "route_origin_lng": float(CONFIG["MAP_CENTER"][1]),
    "route_dest_lat": 22.6008,
    "route_dest_lng": 114.1010,
    "route_source_mode": "手动坐标",
    "route_od_index": 0,
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


@st.cache_data(show_spinner=False, ttl=300)
def cached_baseline_route_result(origin_lat, origin_lng, dest_lat, dest_lng, vehicle_ids, query_date):
    return plan_baseline_routes_between_points(
        origin_lat,
        origin_lng,
        dest_lat,
        dest_lng,
        vehicle_ids=tuple(vehicle_ids or []),
        query_date=query_date,
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
        "start_time": start_time,
        "end_time": end_time,
        "minute_time": minute_time,
        "time_scale": float(st.session_state["time_scale"]),
        "ref_lat": float(st.session_state["ref_lat"]),
        "ref_lng": float(st.session_state["ref_lng"]),
    }


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

        c1, c2 = st.columns(2)
        with c1:
            st.text_input("开始", key="start_time_of_day", placeholder="08:00", label_visibility="collapsed")
        with c2:
            st.text_input("结束", key="end_time_of_day", placeholder="10:00", label_visibility="collapsed")

        st.text_input("分钟查询", key="minute_time_of_day", placeholder="09:30", label_visibility="collapsed")


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
    st.markdown("#### 静态热力图")
    selected_vehicle_ids = _normalize_vehicle_scope(payload.get("trajectory_vehicle_ids", []))

    with st.form("static_heatmap_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            selected_source_type = st.selectbox(
                "数据来源",
                options=["minute", "pickup"],
                key="heatmap_source",
                format_func=_source_label,
                help="分钟缓存表示车辆位置累计分布，OD 上车点表示乘客上车需求累计分布。",
            )
        with c2:
            enable_cluster = st.checkbox("启用聚类热力图", key="heatmap_enable_cluster")
        with c3:
            threshold_quantile = st.slider("热度阈值分位数", 0.50, 0.99, key="heatmap_threshold_quantile", step=0.01, help="用于裁剪过高权重，避免全图过热。")

        c6, c7 = st.columns(2)
        with c6:
            eps_km = st.number_input("聚类半径 eps(km)", min_value=0.05, max_value=2.0, step=0.05, key="heatmap_eps_km", format="%.2f")
        with c7:
            min_samples = st.number_input("最小样本数 min_samples", min_value=2, max_value=50, step=1, key="heatmap_min_samples")
        st.caption("聚类参数仅在启用聚类时生效。聚类后热力值定义为簇内空间聚合点权重之和。")
        submitted = st.form_submit_button("生成静态热力图", use_container_width=True, type="primary")

    if submitted or st.session_state["static_heatmap_request"] is None:
        request_vehicle_ids = _resolve_heatmap_vehicle_scope(selected_source_type, selected_vehicle_ids)
        st.session_state["static_heatmap_request"] = {
            "source_type": selected_source_type,
            "enable_cluster": bool(enable_cluster),
            "eps_km": float(eps_km),
            "min_samples": int(min_samples),
            "threshold_quantile": float(threshold_quantile),
            "start_time": payload["start_time"],
            "end_time": payload["end_time"],
            "vehicle_ids": request_vehicle_ids,
        }

    request = st.session_state["static_heatmap_request"]
    if request and request["source_type"] == "minute" and request.get("vehicle_ids"):
        request = {**request, "vehicle_ids": ()}
        st.session_state["static_heatmap_request"] = request
    st.caption("参数调整不会立即重算，点击“生成静态热力图”后才会执行。")
    rec, _ = cached_source_recommendation(
        request["source_type"],
        request["start_time"],
        request["end_time"],
        request["vehicle_ids"] or None,
    )
    st.caption(
        f"当前执行来源 {_source_label(request['source_type'])}；推荐 DBSCAN 参数 eps≈{rec['eps_km']} km, min_samples≈{rec['min_samples']}。"
    )
    if request["source_type"] == "minute":
        st.caption("分钟缓存车辆位置热力图固定按全部车辆统计，不受左侧车辆ID筛选影响。")
    else:
        st.caption("OD 上车点热力图会按当前车辆ID筛选结果执行。")

    with st.spinner("正在生成静态热力图..."):
        html_path, info = cached_static_heatmap(
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

    cols = st.columns(5)
    metrics = [
        ("原始点数", info.get("input_points", 0)),
        ("聚合热力点", info.get("heat_points", 0)),
        ("渲染点数", info.get("render_heat_points", info.get("heat_points", 0))),
        ("聚类簇数", info.get("cluster_count", 0) if request["enable_cluster"] else "未启用"),
        ("阈值上限", f"{info.get('threshold_cap', 0.0):.2f}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)

    filter_stats = info.get("filter_stats", {})
    st.caption(
        f"来源标识: {info.get('source_label', '--')}；边界过滤 {filter_stats.get('bounds_removed', 0)} 点，"
        f"漂移过滤 {filter_stats.get('drift_removed', 0)} 点，"
        f"渲染策略 {info.get('render_reduction_method', 'none')} / 上限 {info.get('render_point_limit', '--')}，"
        f"输出地图 {info.get('output_path', '--')}。"
    )
    render_html_map(html_path, height=760)


def render_dynamic_heatmap_tab(payload):
    st.markdown("#### 动态热力图")
    selected_vehicle_ids = _normalize_vehicle_scope(payload.get("trajectory_vehicle_ids", []))

    with st.form("dynamic_heatmap_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.selectbox("数据来源", ["minute", "pickup"], key="dynamic_source", format_func=_source_label)
        with c2:
            st.selectbox("时间粒度", [1, 15, 30, 60], key="dynamic_granularity", format_func=lambda x: f"{x} 分钟")
        with c3:
            st.selectbox("平滑算法", ["EMA", "WMA"], key="dynamic_smoothing")

        c4, c5, c6 = st.columns(3)
        with c4:
            st.slider("热度阈值分位数", 0.50, 0.99, key="dynamic_threshold_quantile", step=0.01)
        with c5:
            st.slider("EMA alpha", 0.10, 0.95, key="dynamic_ema_alpha", step=0.05)
        with c6:
            st.slider("WMA 窗口", 2, 8, key="dynamic_wma_window")
        st.caption("动态热力图时间片结构为 `[{time, points:[[lat, lon, weight], ...]}]`，分钟级请求过密时会自动提升到 15/30/60 分钟聚合。")
        submitted = st.form_submit_button("生成动态热力图", use_container_width=True, type="primary")

    if submitted:
        dynamic_source_type = st.session_state["dynamic_source"]
        request_vehicle_ids = _resolve_heatmap_vehicle_scope(dynamic_source_type, selected_vehicle_ids)
        st.session_state["dynamic_heatmap_request"] = {
            "source_type": dynamic_source_type,
            "requested_granularity": int(st.session_state["dynamic_granularity"]),
            "vehicle_ids": request_vehicle_ids,
            "smoothing_method": st.session_state["dynamic_smoothing"],
            "ema_alpha": float(st.session_state["dynamic_ema_alpha"]),
            "wma_window": int(st.session_state["dynamic_wma_window"]),
            "threshold_quantile": float(st.session_state["dynamic_threshold_quantile"]),
            "start_time": payload["start_time"],
            "end_time": payload["end_time"],
        }

    request = st.session_state["dynamic_heatmap_request"]
    if request and request["source_type"] == "minute" and request.get("vehicle_ids"):
        request = {**request, "vehicle_ids": ()}
        st.session_state["dynamic_heatmap_request"] = request
    if request is None:
        st.info("当前仅显示动态热力图配置。点击“生成动态热力图”后才会开始加载和渲染。")
        return
    st.caption("参数调整不会立即重算，点击“生成动态热力图”后才会执行。")
    if request["source_type"] == "minute":
        st.caption("分钟缓存车辆位置动态热力图固定按全部车辆统计，不受左侧车辆ID筛选影响。")
    else:
        st.caption("OD 上车点动态热力图会按当前车辆ID筛选结果执行。")
    with st.spinner("正在生成动态热力图..."):
        html_path, info = cached_dynamic_heatmap(
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

    granularity = info.get("granularity", {})
    animation_profile = info.get("animation_profile", {})
    cols = st.columns(4)
    metrics = [
        ("请求粒度", f"{granularity.get('requested_minutes', '--')} 分钟"),
        ("实际粒度", f"{granularity.get('actual_minutes', '--')} 分钟"),
        ("时间片数", granularity.get("estimated_slices", 0)),
        ("动画配置", f"{animation_profile.get('target_fps', 60)}fps / {animation_profile.get('transition_ms', '--')}ms"),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)

    if granularity.get("auto_adjusted"):
        st.info(
            f"为避免一次加载过多时间片，系统已将动态热力图从 {granularity.get('requested_minutes')} 分钟自动调整为 "
            f"{granularity.get('actual_minutes')} 分钟聚合。"
        )

    time_slices = info.get("time_slices", [])
    preview_slice = time_slices[0]["points"][:3] if time_slices else []
    st.caption(
        f"来源标识: {info.get('source_label', '--')}；平滑算法 {info.get('smoothing_method', '--')}；时间片数 {len(time_slices)}；首片样例 {preview_slice if preview_slice else '[]'}；"
        f"导出地图 {info.get('output_path', '--')}。"
    )
    render_html_map(html_path, height=760)


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


def render_congestion_eta_view(payload):
    st.subheader("拥堵道路与 ETA")
    trajectory_vehicle_ids = payload.get("trajectory_vehicle_ids", [])
    if not trajectory_vehicle_ids:
        st.info(f"请选择 1-{CONFIG['MAX_CONGESTION_VEHICLES']} 辆车作为 HMM 匹配和历史速度样本。")
        return

    if len(trajectory_vehicle_ids) > CONFIG["MAX_CONGESTION_VEHICLES"]:
        st.warning(f"为保持响应速度，拥堵道路示例仅使用前 {CONFIG['MAX_CONGESTION_VEHICLES']} 辆车；可缩短时间窗口来观察更细的拥堵变化。")
    selected_vehicle_ids = trajectory_vehicle_ids[: CONFIG["MAX_CONGESTION_VEHICLES"]]

    status = road_network_status()
    if status["available"]:
        st.caption(f"路网文件: {status['path']}")
    else:
        st.warning("未找到路网文件。请将 shenzhen_drive.pkl 或 shenzhen_drive.graphml 放到项目根目录、data/ 或 cache/，或设置 TAXIGPS_ROAD_NETWORK_PATH。")
        return

    control_cols = st.columns([1, 1, 2])
    with control_cols[0]:
        bucket_minutes = st.selectbox(
            "聚合时间片",
            options=[5, 15, 30, 60],
            index=[5, 15, 30, 60].index(int(st.session_state.get("congestion_bucket_minutes", 15))),
            key="congestion_bucket_minutes",
            help="按固定时间片统计每个匹配路段平均速度。",
        )
    with control_cols[1]:
        st.metric("样本车辆", len(selected_vehicle_ids))
    with control_cols[2]:
        st.caption("近似 HMM 使用最近道路节点作为发射近似，并用最短路连续性约束相邻 GPS 点转移；路段 key 使用道路节点对。")

    with st.spinner("正在执行近似 HMM 地图匹配并聚合路段速度..."):
        congestion_html, congestion_meta = cached_congestion_roads(
            selected_vehicle_ids,
            payload["start_time"],
            payload["end_time"],
            bucket_minutes,
        )

    if not congestion_meta or not congestion_meta.get("success"):
        st.error((congestion_meta or {}).get("error", "拥堵道路生成失败。"))
    else:
        metric_cols = st.columns(4)
        metric_cols[0].metric("时间片", congestion_meta.get("time_slices", 0))
        metric_cols[1].metric("路段记录", congestion_meta.get("segment_rows", 0))
        metric_cols[2].metric("聚合粒度", f"{congestion_meta.get('bucket_minutes', bucket_minutes)} 分钟")
        metric_cols[3].metric("匹配方法", "近似 HMM")
        if congestion_meta.get("matching"):
            st.dataframe(pd.DataFrame(congestion_meta["matching"]), use_container_width=True, hide_index=True)
        render_html_map(congestion_html, height=760)

    st.markdown("#### ETA 预测")
    eta_cols = st.columns(4)
    with eta_cols[0]:
        origin_lat = st.number_input("起点纬度", format="%.6f", key="eta_origin_lat")
    with eta_cols[1]:
        origin_lng = st.number_input("起点经度", format="%.6f", key="eta_origin_lng")
    with eta_cols[2]:
        dest_lat = st.number_input("终点纬度", format="%.6f", key="eta_dest_lat")
    with eta_cols[3]:
        dest_lng = st.number_input("终点经度", format="%.6f", key="eta_dest_lng")

    with st.spinner("正在按路网距离和历史平均速度估算 ETA..."):
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

    if not eta_result.get("success"):
        st.error(eta_result.get("error", "ETA 估算失败。"))
        return

    eta_metric_cols = st.columns(4)
    eta_metric_cols[0].metric("预计耗时", f"{eta_result['eta_minutes']:.1f} 分钟")
    eta_metric_cols[1].metric("路网距离", f"{eta_result['distance_km']:.2f} km")
    eta_metric_cols[2].metric("估算均速", f"{eta_result['avg_speed_kmh']:.1f} km/h")
    eta_metric_cols[3].metric("历史均速", f"{eta_result['historical_speed_kmh']:.1f} km/h")
    st.caption(eta_result.get("method", ""))

    eta_path = plot_eta_route(eta_result)
    render_html_map(eta_path, height=620)


def render_baseline_route_view(payload):
    st.subheader("最短距离与基准最快路线")
    selected_vehicle_ids = payload.get("trajectory_vehicle_ids", [])
    status = road_network_status()
    if status["available"]:
        st.caption(f"路网文件: {status['path']}")
    else:
        st.warning("未找到路网文件。请将 shenzhen_drive.pkl 或 shenzhen_drive.graphml 放到项目根目录、data/ 或 cache/，或设置 TAXIGPS_ROAD_NETWORK_PATH。")
        return

    if not selected_vehicle_ids:
        st.info("未选择车辆时，基准最快路线会使用道路类型默认速度；选择车辆后会从车辆缓存生成全日道路边平均速度样本。")
    elif len(selected_vehicle_ids) > CONFIG["MAX_CONGESTION_VEHICLES"]:
        st.warning(f"基准速度样本将使用前 {CONFIG['MAX_CONGESTION_VEHICLES']} 辆车，避免一次查询处理过多轨迹。")
    speed_vehicle_ids = selected_vehicle_ids[: CONFIG["MAX_CONGESTION_VEHICLES"]]

    mode = st.radio(
        "起终点来源",
        ["手动坐标", "历史OD端点"],
        horizontal=True,
        key="route_source_mode",
        help="历史OD端点来自当前查询窗口；地图中的连续选点面板可辅助读取坐标后填入手动坐标。",
    )

    origin_lat = float(st.session_state.get("route_origin_lat", CONFIG["MAP_CENTER"][0]))
    origin_lng = float(st.session_state.get("route_origin_lng", CONFIG["MAP_CENTER"][1]))
    dest_lat = float(st.session_state.get("route_dest_lat", 22.6008))
    dest_lng = float(st.session_state.get("route_dest_lng", 114.1010))
    od_row = None

    if mode == "历史OD端点":
        od_df = load_od_data(payload["start_time"], payload["end_time"], vehicle_ids=speed_vehicle_ids or None)
        if od_df is None or len(od_df) == 0:
            st.warning("当前查询窗口没有可用 OD 记录，请切换到手动坐标或调整时间/车辆。")
        else:
            max_options = min(50, len(od_df))
            od_options = list(range(max_options))
            selected_index = st.selectbox(
                "历史 OD 样例",
                od_options,
                key="route_od_index",
                format_func=lambda idx: (
                    f"{od_df.iloc[idx]['O_TAXI_ID']} | "
                    f"{od_df.iloc[idx]['O_time'].strftime('%H:%M:%S')} -> {od_df.iloc[idx]['D_time'].strftime('%H:%M:%S')} | "
                    f"{od_df.iloc[idx]['OD_Dist_km']:.2f} km"
                ),
            )
            od_row = od_df.iloc[int(selected_index)]
            origin_lat = float(od_row["O_lat"])
            origin_lng = float(od_row["O_lng"])
            dest_lat = float(od_row["D_lat"])
            dest_lng = float(od_row["D_lng"])
            st.caption("OD 原始坐标会保留；路线计算会额外吸附到最近路网节点。")
    else:
        route_cols = st.columns(4)
        with route_cols[0]:
            origin_lat = st.number_input("起点纬度", format="%.6f", key="route_origin_lat")
        with route_cols[1]:
            origin_lng = st.number_input("起点经度", format="%.6f", key="route_origin_lng")
        with route_cols[2]:
            dest_lat = st.number_input("终点纬度", format="%.6f", key="route_dest_lat")
        with route_cols[3]:
            dest_lng = st.number_input("终点经度", format="%.6f", key="route_dest_lng")

    with st.spinner("正在生成全日道路基准速度并计算最短/最快路线..."):
        route_result = cached_baseline_route_result(
            origin_lat,
            origin_lng,
            dest_lat,
            dest_lng,
            speed_vehicle_ids,
            payload["start_time"].date(),
        )

    if not route_result.get("success"):
        st.error(route_result.get("error", "路线计算失败。"))
        return

    shortest = route_result["shortest"]
    fastest = route_result["fastest"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("最短距离", f"{shortest['distance_m'] / 1000:.2f} km")
    metric_cols[1].metric("最快路线距离", f"{fastest['distance_m'] / 1000:.2f} km")
    metric_cols[2].metric("最快路线成本", f"{fastest['route_cost_s'] / 60:.1f} 分钟")
    metric_cols[3].metric("可靠速度边", route_result.get("speed_meta", {}).get("reliable_edges", 0))

    summary_df = pd.DataFrame(
        [
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
        ]
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    if od_row is not None:
        od_summary = pd.DataFrame(
            [
                {"字段": "车辆", "值": od_row["O_TAXI_ID"]},
                {"字段": "上车原始坐标", "值": f"{od_row['O_lat']:.6f}, {od_row['O_lng']:.6f}"},
                {"字段": "下车原始坐标", "值": f"{od_row['D_lat']:.6f}, {od_row['D_lng']:.6f}"},
                {"字段": "上车吸附节点", "值": route_result["origin"]["node"]},
                {"字段": "下车吸附节点", "值": route_result["destination"]["node"]},
            ]
        )
        st.dataframe(od_summary, use_container_width=True, hide_index=True)

    st.caption(route_result.get("method", ""))
    route_path = plot_baseline_route_comparison(route_result)
    render_html_map(route_path, height=700)


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
            render_baseline_route_view(active_payload)
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
