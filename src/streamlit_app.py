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
from textwrap import dedent

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_plotter import (  # noqa: E402
    CONFIG,
    load_minute_data,
    load_od_data,
    load_vehicle_trajectory,
    plot_animated_trajectory,
    plot_multi_vehicle_animated_trajectory,
    plot_minute_vehicles,
    plot_od_points,
    plot_vehicle_trajectories,
    plot_vehicle_trajectory,
)


logger = logging.getLogger(__name__)

PAGE_TITLE = "出租车GPS轨迹查询系统"
PAGE_ICON = "🚕"
VIEW_OPTIONS = ["轨迹查询", "动画轨迹", "分钟位置", "OD点标注"]


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
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 1.2rem;
        }

        [data-testid="stSidebar"] {
            background: #fafafa;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * {
            color: var(--text) !important;
        }

        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] .stDateInput input,
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stSelectbox div,
        [data-testid="stSidebar"] .stSlider {
            background: #ffffff !important;
        }

        [data-testid="stSidebar"] .stNumberInput input {
            color: var(--text) !important;
            text-align: center;
            font-variant-numeric: tabular-nums;
        }

        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stTimeInput input {
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
        }

        [data-testid="stSidebar"] .stTimeInput input {
            color: var(--text) !important;
            text-align: center;
            font-variant-numeric: tabular-nums;
        }

        [data-testid="stSidebar"] .stTextInput input {
            color: var(--text) !important;
            text-align: center;
            font-variant-numeric: tabular-nums;
        }

        .hero {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 22px 22px 18px 22px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 1.8rem;
            line-height: 1.15;
            color: var(--text);
        }

        .hero p {
            margin: 0;
            color: var(--muted);
            max-width: 62rem;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 0.75rem 0 1rem 0;
        }

        .summary-card {
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px 16px;
        }

        .summary-card .label {
            font-size: 0.82rem;
            color: var(--muted);
            margin-bottom: 0.25rem;
        }

        .summary-card .value {
            font-size: 1.06rem;
            font-weight: 700;
            color: var(--text);
            overflow-wrap: anywhere;
        }

        .subtle-panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px 16px;
        }

        .stRadio > div {
            gap: 0.35rem;
        }

        .stRadio label {
            border: 1px solid var(--border) !important;
            border-radius: 999px !important;
            padding: 0.45rem 0.85rem !important;
            background: #ffffff !important;
        }

        .stRadio label[data-checked="true"] {
            background: #f0f0f0 !important;
            border-color: var(--border-strong) !important;
        }

        hr {
            border-color: var(--border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


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

    if payload["time_scale"] <= 0:
        errors.append("动画速度必须大于 0。")

    return errors


def sidebar_form():
    st.sidebar.markdown("### 查询控制台")
    with st.sidebar.form("query_form", clear_on_submit=False):
        st.date_input("日期", key="query_date")

        available_vehicle_ids = get_available_vehicle_ids()
        current_trajectory_ids = [
            str(vehicle_id).strip()
            for vehicle_id in st.session_state.get("trajectory_vehicle_ids", [])
            if str(vehicle_id).strip() in available_vehicle_ids
        ]
        if current_trajectory_ids != st.session_state.get("trajectory_vehicle_ids", []):
            st.session_state["trajectory_vehicle_ids"] = current_trajectory_ids

        st.multiselect(
            "车辆ID",
            options=available_vehicle_ids,
            key="trajectory_vehicle_ids",
            help="最多同时支持10车查询",
        )

        st.markdown("#### 轨迹区间")
        st.text_input("开始时间", key="start_time_of_day", placeholder="08:00")
        st.text_input("结束时间", key="end_time_of_day", placeholder="10:00")

        st.markdown("#### 分钟查询")
        st.text_input("查询时间", key="minute_time_of_day", placeholder="09:30")

        st.markdown("#### ETA 预留坐标")
        c3, c4 = st.columns(2)
        with c3:
            st.number_input("参考纬度", format="%.6f", key="ref_lat")
        with c4:
            st.number_input("参考经度", format="%.6f", key="ref_lng")

        submitted = st.form_submit_button("执行查询", use_container_width=True, type="primary")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 系统说明")
    st.sidebar.markdown(
        """
        - 轨迹查询：读取车辆缓存，展示指定时间范围内的行驶轨迹
        - 分钟位置：读取分钟缓存，查看某一分钟的车辆分布
        - OD 点标注：展示上车点和下车点，并自动区分线路
        - 动画轨迹：在动画页面中按时间顺序播放单车运动，并显示速度变化
        """
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("重置为默认值", use_container_width=True):
        reset_to_defaults()
    if st.sidebar.button("清空缓存", use_container_width=True):
        st.cache_data.clear()
        st.success("缓存已清空")
    return submitted


def render_header():
    st.markdown(
        """
        <div class="hero">
          <h1>出租车GPS轨迹查询系统</h1>
          <p>面向出租车 GPS 数据的统一查询控制台。系统基于车辆缓存、分钟缓存与 OD 结果表提供轨迹检索、分钟位置、上下车点标注和单车动画展示，页面保持简洁、清晰且响应迅速。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
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

        combined_df = pd.concat([df.assign(vehicle_id=vehicle_id) for vehicle_id, df in trajectory_frames], ignore_index=True)
        cols = st.columns(4)
        metrics = [
            ("车辆数", len(trajectory_frames)),
            ("轨迹点", len(combined_df)),
            ("载客点", int((combined_df["status"] == 1).sum())),
            ("空载点", int((combined_df["status"] == 0).sum())),
        ]
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.metric(label, value)

        st.caption("轨迹颜色区分车辆，线型区分状态：实线表示载客，虚线表示空载。")
        render_vehicle_status_panel(trajectory_frames)
        if len(trajectory_frames) == 1:
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
        st.caption("点击地图上的车辆点，可展开查看该车后续轨迹。")

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
                st.caption(f"当前仅显示车辆 {selected_vehicle_ids[0]} 的分钟位置。")
            else:
                display_df = df[df["vehicle_id"].isin(selected_vehicle_ids)]
                if len(display_df) == 0:
                    st.warning("所选车辆在该分钟没有位置数据。")
                    return
                st.caption(f"当前显示所选 {len(selected_vehicle_ids)} 辆车的分钟位置。")
        else:
            st.caption("当前显示所有车辆的分钟位置，系统会自动按上限抽样以提升流畅度。")

        cols = st.columns(3)
        minute_metrics = [
            ("车辆总数", len(display_df)),
            ("载客车辆", int((display_df["status"] == 1).sum())),
            ("空载车辆", int((display_df["status"] == 0).sum())),
        ]
        for col, (label, value) in zip(cols, minute_metrics):
            with col:
                st.metric(label, value)

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

        cols = st.columns(4)
        metrics = [
            ("OD 订单数", len(df)),
            ("总行驶距离", f"{df['OD_Dist_km'].sum():.1f} km"),
            ("平均距离", f"{df['OD_Dist_km'].mean():.2f} km"),
            ("平均时长", f"{df['OD_Time_s'].mean():.0f} 秒"),
        ]
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.metric(label, value)

        st.caption("绿色为上车点，红色为下车点。点较多时会自动聚类并抽样连线。")
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

        combined_df = pd.concat([df.assign(vehicle_id=vehicle_id) for vehicle_id, df in animation_frames], ignore_index=True)
        st.caption("动画采用逐帧插值播放，并根据相邻轨迹点的时间差模拟速度变化。")
        if len(animation_frames) == 1:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("轨迹点", len(combined_df))
            with c2:
                st.metric("时间跨度", f"{(combined_df.iloc[-1]['time'] - combined_df.iloc[0]['time']).total_seconds() / 60:.0f} 分钟")
            with c3:
                st.metric("平均速度", f"{combined_df['speed'].mean():.1f} km/h")
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
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("车辆数", len(animation_frames))
            with c2:
                st.metric("轨迹点", len(combined_df))
            with c3:
                st.metric("平均速度", f"{combined_df['speed'].mean():.1f} km/h")
            with c4:
                st.metric("时间跨度", f"{(combined_df['time'].max() - combined_df['time'].min()).total_seconds() / 60:.0f} 分钟")
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
                f"当前车辆 {trajectory_vehicle_ids[0]} 的缓存轨迹范围为 "
                f"{time_range['min_time'].strftime('%Y-%m-%d %H:%M:%S')} 至 "
                f"{time_range['max_time'].strftime('%Y-%m-%d %H:%M:%S')}。"
            )
    elif len(trajectory_vehicle_ids) > 1:
        st.info(f"当前已选择 {len(trajectory_vehicle_ids)} 辆车进行轨迹并行展示。")
    else:
        st.caption("当前为全部车辆模式，结果将按时间范围查询并展示汇总预览。")


def main():
    try:
        init_page()
        init_state()

        st.session_state.setdefault("last_query", build_query_payload())

        render_header()
        submitted = sidebar_form()

        payload = build_query_payload()
        if submitted:
            errors = validate_payload(payload)
            if errors:
                for error in errors:
                    st.error(error)
                return
            st.session_state["last_query"] = payload
            st.success("查询条件已更新。")

        active_payload = st.session_state.get("last_query", payload)
        render_summary(active_payload)
        render_query_status(active_payload)

        active_view = st.radio(
            "功能切换",
            VIEW_OPTIONS,
            horizontal=True,
            key="active_view",
            label_visibility="collapsed",
        )

        if active_view == "轨迹查询":
            render_trajectory_view(active_payload)
        elif active_view == "动画轨迹":
            render_animation_view(active_payload)
        elif active_view == "分钟位置":
            render_minute_view(active_payload)
        elif active_view == "OD点标注":
            render_od_view(active_payload)

        st.markdown("---")
        st.caption("数据来源: cache/vehicles/ | cache/minutes/ | data/processed/od_table.csv")
    except Exception:
        logger.exception("页面主流程错误")
        st.error("页面加载失败，请稍后重试。")


if __name__ == "__main__":
    if is_running_under_streamlit():
        main()
    else:
        print("请使用以下命令启动 Streamlit 应用：")
        print("streamlit run src/streamlit_app.py")
