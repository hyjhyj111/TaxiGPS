#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
出租车GPS轨迹查询系统

单页集成：
1. 按车辆ID和时间范围查询轨迹
2. 按分钟查询所有车辆或指定车辆位置
3. 标注上车点和下车点
4. 单车动画轨迹播放
5. 预留 ETA / 路网校正坐标输入
"""

from datetime import datetime, time
import logging
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_plotter import (  # noqa: E402
    CONFIG,
    load_minute_data,
    load_od_data,
    load_vehicle_trajectory,
    plot_animated_trajectory,
    plot_minute_vehicles,
    plot_od_points,
    plot_vehicle_trajectory,
)


logger = logging.getLogger(__name__)

PAGE_TITLE = "出租车GPS轨迹查询系统"
PAGE_ICON = "🚕"
VIEW_OPTIONS = ["轨迹查询", "分钟位置", "OD点标注", "动画轨迹"]


DEFAULTS = {
    "vehicle_id": "22223",
    "query_date": datetime(2023, 10, 12).date(),
    "start_hour": 8,
    "start_minute": 0,
    "end_hour": 10,
    "end_minute": 0,
    "minute_hour": 9,
    "minute_minute": 30,
    "time_scale": 2.0,
    "display_limit": 1200,
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
        [data-testid="stSidebar"] .stTimeInput input,
        [data-testid="stSidebar"] .stSelectbox div,
        [data-testid="stSidebar"] .stSlider {
            background: #ffffff !important;
        }

        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stTimeInput input {
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
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

        .section-panel {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
        }

        .subtle-panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px 16px;
        }

        .view-selector {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 10px 12px;
            margin-bottom: 12px;
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
def cached_od_data(start_time, end_time, vehicle_id):
    return load_od_data(start_time, end_time, vehicle_id)


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
    st.session_state["start_hour"] = start_time.hour
    st.session_state["start_minute"] = start_time.minute
    st.session_state["end_hour"] = end_time.hour
    st.session_state["end_minute"] = end_time.minute
    st.session_state["minute_hour"] = start_time.hour
    st.session_state["minute_minute"] = start_time.minute
    return True


def build_query_payload():
    start_time = datetime.combine(
        st.session_state["query_date"],
        time(int(st.session_state["start_hour"]), int(st.session_state["start_minute"])),
    )
    end_time = datetime.combine(
        st.session_state["query_date"],
        time(int(st.session_state["end_hour"]), int(st.session_state["end_minute"])),
    )
    minute_time = datetime.combine(
        st.session_state["query_date"],
        time(int(st.session_state["minute_hour"]), int(st.session_state["minute_minute"])),
    )

    return {
        "vehicle_id": st.session_state["vehicle_id"].strip(),
        "start_time": start_time,
        "end_time": end_time,
        "minute_time": minute_time,
        "time_scale": float(st.session_state["time_scale"]),
        "display_limit": int(st.session_state["display_limit"]),
        "ref_lat": float(st.session_state["ref_lat"]),
        "ref_lng": float(st.session_state["ref_lng"]),
    }


def validate_payload(payload):
    errors = []
    vehicle_id = payload["vehicle_id"]

    if vehicle_id and not vehicle_id.isdigit():
        errors.append("车辆ID必须为数字。")

    if payload["start_time"] >= payload["end_time"]:
        errors.append("开始时间必须早于结束时间。")

    if payload["time_scale"] <= 0:
        errors.append("动画速度必须大于 0。")

    if payload["display_limit"] <= 0:
        errors.append("分钟显示数量必须大于 0。")

    return errors


def sidebar_form():
    st.sidebar.markdown("### 查询控制台")
    with st.sidebar.form("query_form", clear_on_submit=False):
        st.text_input("车辆 ID", key="vehicle_id", placeholder="留空可用于分钟查询/OD查询")
        st.date_input("日期", key="query_date")

        st.markdown("#### 轨迹时间")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("开始小时", min_value=0, max_value=23, key="start_hour", step=1)
            st.number_input("开始分钟", min_value=0, max_value=59, key="start_minute", step=1)
        with c2:
            st.number_input("结束小时", min_value=0, max_value=23, key="end_hour", step=1)
            st.number_input("结束分钟", min_value=0, max_value=59, key="end_minute", step=1)

        st.markdown("#### 分钟查询时间")
        c3, c4 = st.columns(2)
        with c3:
            st.number_input("分钟小时", min_value=0, max_value=23, key="minute_hour", step=1)
        with c4:
            st.number_input("分钟分钟", min_value=0, max_value=59, key="minute_minute", step=1)

        st.slider("动画速度", 0.5, 5.0, key="time_scale", step=0.1)
        st.slider("分钟地图最大车辆数", 100, 20000, key="display_limit", step=100)

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
        - 动画轨迹：按时间顺序播放单车运动，并显示速度变化
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 车辆时间范围")
    vehicle_id = st.session_state.get("vehicle_id", "").strip()
    if vehicle_id and vehicle_id.isdigit():
        time_range = get_vehicle_time_range(vehicle_id)
        if time_range:
            st.sidebar.markdown(
                f"""
                <div class="subtle-panel">
                    <div style="font-size:0.82rem;color:#616161;margin-bottom:0.25rem;">缓存轨迹范围</div>
                    <div style="font-size:0.98rem;font-weight:700;margin-bottom:0.2rem;">{time_range['min_time'].strftime('%Y-%m-%d %H:%M:%S')}</div>
                    <div style="font-size:0.98rem;font-weight:700;margin-bottom:0.45rem;">{time_range['max_time'].strftime('%Y-%m-%d %H:%M:%S')}</div>
                    <div style="font-size:0.82rem;color:#616161;">轨迹点数: {time_range['total_points']:,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.sidebar.caption("根据该范围调整轨迹起止时间，更容易命中缓存中的有效数据。")
            if st.sidebar.button("使用建议时间范围", use_container_width=True):
                if apply_vehicle_time_range(vehicle_id):
                    st.rerun()

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
    cards = [
        ("车辆 ID", payload["vehicle_id"] or "全部车辆"),
        ("轨迹时间", f"{payload['start_time'].strftime('%H:%M')} - {payload['end_time'].strftime('%H:%M')}"),
        ("分钟查询", payload["minute_time"].strftime("%H:%M")),
        ("动画速度", f"{payload['time_scale']:.1f}x"),
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
    st.caption(
        f"参考坐标: {payload['ref_lat']:.6f}, {payload['ref_lng']:.6f} | "
        "轨迹读取车辆缓存，分钟读取分钟缓存，OD 标注读取完成后的 OD 表。"
    )


def render_html_map(html_path, height=700):
    if html_path and os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=height, scrolling=True)
    else:
        st.error("地图生成失败，请检查数据缓存是否存在。")


def render_data_preview(df, title, max_rows=20):
    st.markdown(f"**{title}**")
    if df is None or len(df) == 0:
        st.caption("没有可预览的数据。")
        return

    preview_df = df.head(max_rows).copy()
    st.dataframe(preview_df, use_container_width=True, height=min(420, 42 + 28 * len(preview_df)))
    st.caption(f"共 {len(df):,} 条记录，当前仅预览前 {len(preview_df):,} 条。")


def render_trajectory_view(payload):
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    st.subheader("轨迹查询")
    if not payload["vehicle_id"]:
        st.info("请输入车辆 ID 后查看单车轨迹。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with st.spinner("正在读取车辆缓存并生成轨迹地图..."):
        df = cached_vehicle_trajectory(payload["vehicle_id"], payload["start_time"], payload["end_time"])
        if df is None or len(df) == 0:
            st.warning("未找到符合条件的轨迹数据。")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        cols = st.columns(4)
        metrics = [
            ("轨迹点", len(df)),
            ("载客点", int((df["status"] == 1).sum())),
            ("空载点", int((df["status"] == 0).sum())),
            ("平均速度", f"{df['speed'].mean():.1f} km/h"),
        ]
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.metric(label, value)

        st.caption("轨迹按状态自动着色：红色表示载客，蓝色表示空载。")
        render_html_map(plot_vehicle_trajectory(payload["vehicle_id"], payload["start_time"], payload["end_time"]), height=700)
    st.markdown("</div>", unsafe_allow_html=True)


def render_minute_view(payload):
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    st.subheader("分钟位置查询")
    with st.spinner("正在读取分钟缓存并生成位置地图..."):
        df = cached_minute_data(payload["minute_time"])
        if df is None or len(df) == 0:
            st.warning("该分钟没有车辆数据。")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        df_view = df
        if payload["vehicle_id"]:
            vehicle_df = df[df["vehicle_id"] == str(payload["vehicle_id"])]
            if len(vehicle_df) == 0:
                st.warning(f"车辆 {payload['vehicle_id']} 在该分钟没有位置数据。")
                time_range = get_vehicle_time_range(payload["vehicle_id"])
                if time_range:
                    st.caption(
                        f"该车辆缓存轨迹时间范围为 {time_range['min_time'].strftime('%Y-%m-%d %H:%M:%S')} "
                        f"至 {time_range['max_time'].strftime('%Y-%m-%d %H:%M:%S')}，可尝试调整分钟查询时间。"
                    )
                st.markdown("</div>", unsafe_allow_html=True)
                return
            df_view = vehicle_df
            st.caption(f"当前仅显示车辆 {payload['vehicle_id']} 的分钟位置。")
        else:
            st.caption("当前显示所有车辆的分钟位置，系统会自动按上限抽样以提升流畅度。")

        cols = st.columns(3)
        minute_metrics = [
            ("车辆总数", len(df)),
            ("载客车辆", int((df["status"] == 1).sum())),
            ("空载车辆", int((df["status"] == 0).sum())),
        ]
        for col, (label, value) in zip(cols, minute_metrics):
            with col:
                st.metric(label, value)

        render_data_preview(df_view, "分钟数据预览", max_rows=20)
        render_html_map(
            plot_minute_vehicles(
                payload["minute_time"],
                payload["vehicle_id"] if payload["vehicle_id"] else None,
                display_limit=payload["display_limit"],
            ),
            height=700,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_od_view(payload):
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    st.subheader("OD 上下车点")
    with st.spinner("正在读取 OD 表并生成上下车点地图..."):
        df = cached_od_data(payload["start_time"], payload["end_time"], payload["vehicle_id"] or None)
        if df is None or len(df) == 0:
            if payload["vehicle_id"]:
                st.warning(f"车辆 {payload['vehicle_id']} 在当前时间范围内没有 OD 记录。")
                time_range = get_vehicle_time_range(payload["vehicle_id"])
                if time_range:
                    st.caption(
                        f"该车辆轨迹缓存范围为 {time_range['min_time'].strftime('%Y-%m-%d %H:%M:%S')} "
                        f"至 {time_range['max_time'].strftime('%Y-%m-%d %H:%M:%S')}，建议扩大时间范围后重试。"
                    )
            else:
                st.warning("当前时间范围内未找到 OD 数据。")
            st.markdown("</div>", unsafe_allow_html=True)
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
        render_data_preview(df, "OD 数据预览", max_rows=20)
        render_html_map(
            plot_od_points(payload["start_time"], payload["end_time"], payload["vehicle_id"] or None),
            height=700,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_animation_view(payload):
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    st.subheader("单车动画轨迹")
    if not payload["vehicle_id"]:
        st.info("请输入车辆 ID 后播放动画轨迹。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with st.spinner("正在生成动画轨迹..."):
        df = cached_vehicle_trajectory(payload["vehicle_id"], payload["start_time"], payload["end_time"])
        if df is None or len(df) < 2:
            st.warning("轨迹点不足，无法生成动画。")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        st.caption(
            "动画会根据相邻轨迹点的时间差自动调节播放节奏，并在当前位置显示速度与状态。"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("轨迹点", len(df))
        with c2:
            st.metric("时间跨度", f"{(df.iloc[-1]['time'] - df.iloc[0]['time']).total_seconds() / 60:.0f} 分钟")
        with c3:
            st.metric("平均速度", f"{df['speed'].mean():.1f} km/h")

        render_html_map(
            plot_animated_trajectory(
                payload["vehicle_id"],
                payload["start_time"],
                payload["end_time"],
                payload["time_scale"],
            ),
            height=780,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_query_status(payload):
    """显示当前查询状态，帮助用户确认是否在正确范围内查询。"""
    vehicle_id = payload["vehicle_id"]
    if vehicle_id:
        time_range = get_vehicle_time_range(vehicle_id)
        if time_range:
            st.info(
                f"当前车辆 {vehicle_id} 的缓存轨迹范围为 "
                f"{time_range['min_time'].strftime('%Y-%m-%d %H:%M:%S')} 至 "
                f"{time_range['max_time'].strftime('%Y-%m-%d %H:%M:%S')}。"
            )
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

        st.markdown('<div class="view-selector">', unsafe_allow_html=True)
        active_view = st.radio(
            "功能切换",
            VIEW_OPTIONS,
            horizontal=True,
            key="active_view",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if active_view == "轨迹查询":
            render_trajectory_view(active_payload)
        elif active_view == "分钟位置":
            render_minute_view(active_payload)
        elif active_view == "OD点标注":
            render_od_view(active_payload)
        elif active_view == "动画轨迹":
            render_animation_view(active_payload)

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
