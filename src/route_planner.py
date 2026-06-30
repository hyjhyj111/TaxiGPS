#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线规划模块
支持地图点击选择起终点，计算最短距离和基准最快路线
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from map_plotter import (
    CONFIG,
    plan_baseline_routes_between_points,
    build_map,
    road_network_status,
)
import pandas as pd


def create_interactive_route_map(points, route_result=None):
    """
    创建交互式路线地图

    Args:
        points: 已选择的点列表 [{'lat': ..., 'lng': ...}, ...]
        route_result: 路线计算结果（可选）

    Returns:
        folium.Map 对象
    """
    # 确定地图中心
    if route_result and route_result.get("success"):
        # 使用路线结果中的点
        shortest_points = route_result.get("shortest", {}).get("points", [])
        fastest_points = route_result.get("fastest", {}).get("points", [])
        all_points = shortest_points + fastest_points
        if all_points:
            map_df = pd.DataFrame({
                "lati": [p[0] for p in all_points],
                "long": [p[1] for p in all_points]
            })
            m = build_map(map_df)
        else:
            m = folium.Map(
                location=CONFIG["MAP_CENTER"],
                zoom_start=12,
                tiles='OpenStreetMap'
            )
    elif points:
        # 使用已选择的点
        center_lat = sum(p['lat'] for p in points) / len(points)
        center_lng = sum(p['lng'] for p in points) / len(points)
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=13,
            tiles='OpenStreetMap'
        )
    else:
        # 使用默认中心
        m = folium.Map(
            location=CONFIG["MAP_CENTER"],
            zoom_start=12,
            tiles='OpenStreetMap'
        )

    # 添加已选择的点标记
    if len(points) >= 1:
        # 起点（绿色）
        folium.Marker(
            [points[0]['lat'], points[0]['lng']],
            popup=f'起点<br>{points[0]["lat"]:.6f}, {points[0]["lng"]:.6f}',
            tooltip='起点',
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)

    if len(points) >= 2:
        # 终点（红色）
        folium.Marker(
            [points[1]['lat'], points[1]['lng']],
            popup=f'终点<br>{points[1]["lat"]:.6f}, {points[1]["lng"]:.6f}',
            tooltip='终点',
            icon=folium.Icon(color='red', icon='flag')
        ).add_to(m)

        # 连线（虚线）
        folium.PolyLine(
            [[points[0]['lat'], points[0]['lng']],
             [points[1]['lat'], points[1]['lng']]],
            color='#94a3b8',
            weight=2,
            dash_array='5,5',
            opacity=0.6
        ).add_to(m)

    # 如果有路线结果，绘制路线
    if route_result and route_result.get("success"):
        shortest = route_result.get("shortest", {})
        fastest = route_result.get("fastest", {})

        # 最短距离路线（蓝色）
        if shortest.get("points"):
            folium.PolyLine(
                shortest["points"],
                color="#2563eb",
                weight=6,
                opacity=0.86,
                tooltip=f"最短距离路线 {shortest['distance_m'] / 1000:.2f} km",
            ).add_to(m)

        # 基准最快路线（绿色）
        if fastest.get("points"):
            folium.PolyLine(
                fastest["points"],
                color="#16a34a",
                weight=5,
                opacity=0.9,
                dash_array="8,6" if fastest["points"] == shortest["points"] else None,
                tooltip=f"基准最快路线 {fastest['route_cost_s'] / 60:.1f} min",
            ).add_to(m)

    return m


def render_route_planning_view(payload):
    """渲染路线规划视图"""
    st.subheader("🗺️ 路线规划")

    # 检查路网状态
    status = road_network_status()
    if not status["available"]:
        st.warning("未找到路网文件。请将 shenzhen_drive.pkl 或 shenzhen_drive.graphml 放到项目根目录、data/ 或 cache/，或设置 TAXIGPS_ROAD_NETWORK_PATH。")
        return

    st.caption(f"路网文件: {status['path']}")

    # 获取选中的车辆ID（用于生成道路速度缓存）
    selected_vehicle_ids = payload.get("trajectory_vehicle_ids", [])
    if not selected_vehicle_ids:
        st.info("💡 提示：未选择车辆时，基准最快路线会使用道路类型默认速度；选择车辆后会从车辆缓存生成全日道路边平均速度样本。")
    elif len(selected_vehicle_ids) > CONFIG["MAX_CONGESTION_VEHICLES"]:
        st.warning(f"基准速度样本将使用前 {CONFIG['MAX_CONGESTION_VEHICLES']} 辆车，避免一次查询处理过多轨迹。")

    speed_vehicle_ids = selected_vehicle_ids[:CONFIG["MAX_CONGESTION_VEHICLES"]]

    # 初始化状态
    if 'route_points' not in st.session_state:
        st.session_state.route_points = []
    if 'route_result' not in st.session_state:
        st.session_state.route_result = None

    # 操作按钮
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🔄 重新选点", use_container_width=True):
            st.session_state.route_points = []
            st.session_state.route_result = None
            st.rerun()

    with col2:
        can_calculate = len(st.session_state.route_points) == 2
        if st.button(
            "✅ 计算路线",
            use_container_width=True,
            type="primary" if can_calculate else "secondary",
            disabled=not can_calculate
        ):
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
                    st.success("✅ 路线计算完成！")
                    st.rerun()
                else:
                    st.error(f"路线计算失败: {route_result.get('error', '未知错误')}")

    with col3:
        points_count = len(st.session_state.route_points)
        if points_count == 0:
            st.info("📍 请点击地图选择起点")
        elif points_count == 1:
            st.info("📍 已选择起点，请点击地图选择终点")
        else:
            st.success("✅ 已选择起点和终点，点击「计算路线」按钮")

    # 创建交互式地图
    route_map = create_interactive_route_map(
        st.session_state.route_points,
        st.session_state.route_result
    )

    # 渲染地图并捕获点击事件
    map_data = st_folium(
        route_map,
        width=None,
        height=600,
        key=f"route_map_{len(st.session_state.route_points)}_{st.session_state.get('route_result') is not None}"
    )

    # 处理地图点击事件
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lng = map_data["last_clicked"]["lng"]

        # 检查是否是新的点击
        last_click = st.session_state.get('last_route_click', {})
        current_click = f"{clicked_lat}_{clicked_lng}"

        if last_click != current_click:
            st.session_state.last_route_click = current_click

            # 如果已有2个点，清空重新开始
            if len(st.session_state.route_points) >= 2:
                st.session_state.route_points = []
                st.session_state.route_result = None

            # 添加新点
            st.session_state.route_points.append({
                'lat': clicked_lat,
                'lng': clicked_lng
            })

            st.rerun()

    # 显示路线统计信息
    if st.session_state.route_result and st.session_state.route_result.get("success"):
        st.markdown("---")
        st.markdown("### 📊 路线统计")

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
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # 元信息
        speed_meta = result.get("speed_meta", {})
        st.caption(
            f"路网节点: {result.get('network', {}).get('nodes', 0)} | "
            f"路网边数: {result.get('network', {}).get('edges', 0)} | "
            f"速度样本边: {speed_meta.get('edge_rows', 0)} | "
            f"可靠速度边: {speed_meta.get('reliable_edges', 0)}"
        )
