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
    # 确定地图中心和缩放级别
    # 优先使用已保存的视图状态，避免选点时地图跳动
    if 'map_view_center' in st.session_state and 'map_view_zoom' in st.session_state:
        # 使用保存的视图状态（用户当前看到的位置）
        map_center = st.session_state.map_view_center
        map_zoom = st.session_state.map_view_zoom
    elif route_result and route_result.get("success"):
        # 有路线结果时，使用路线的边界来确定视图
        shortest_points = route_result.get("shortest", {}).get("points", [])
        fastest_points = route_result.get("fastest", {}).get("points", [])
        all_points = shortest_points + fastest_points
        if all_points:
            map_df = pd.DataFrame({
                "lati": [p[0] for p in all_points],
                "long": [p[1] for p in all_points]
            })
            # 计算合适的中心点和缩放级别
            map_center = [map_df["lati"].mean(), map_df["long"].mean()]
            map_zoom = 13
        else:
            map_center = CONFIG["MAP_CENTER"]
            map_zoom = 12
    elif points:
        # 有选点但没有保存的视图状态时，才使用选点的中心
        # 这只在第一次选点时发生
        center_lat = sum(p['lat'] for p in points) / len(points)
        center_lng = sum(p['lng'] for p in points) / len(points)
        map_center = [center_lat, center_lng]
        map_zoom = 13
    else:
        # 默认视图（没有选点，没有路线，没有保存的视图）
        map_center = CONFIG["MAP_CENTER"]
        map_zoom = 12

    # 创建地图
    m = folium.Map(
        location=map_center,
        zoom_start=map_zoom,
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

        # 添加图例
        legend_html = f"""
        <div style="position: fixed;
                    top: 10px;
                    right: 10px;
                    z-index: 9999;
                    background: white;
                    padding: 15px 20px;
                    border: 2px solid #ccc;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                    min-width: 200px;">
            <div style="font-weight: bold;
                        margin-bottom: 12px;
                        font-size: 16px;
                        color: #333;
                        border-bottom: 2px solid #e5e7eb;
                        padding-bottom: 8px;">
                📍 路线图例
            </div>

            <div style="margin-bottom: 10px;
                        display: flex;
                        align-items: center;">
                <div style="width: 40px;
                            height: 5px;
                            background-color: #2563eb;
                            margin-right: 10px;
                            border-radius: 2px;"></div>
                <span style="color: #1e40af; font-weight: 600;">最短距离路线</span>
            </div>

            <div style="margin-bottom: 10px;
                        font-size: 12px;
                        color: #6b7280;
                        margin-left: 50px;">
                距离：{shortest['distance_m'] / 1000:.2f} km<br>
                道路边：{shortest['edge_count']} 条
            </div>

            <div style="margin-bottom: 10px;
                        display: flex;
                        align-items: center;">
                <div style="width: 40px;
                            height: 5px;
                            background-color: #16a34a;
                            margin-right: 10px;
                            border-radius: 2px;"></div>
                <span style="color: #15803d; font-weight: 600;">基准最快路线</span>
            </div>

            <div style="font-size: 12px;
                        color: #6b7280;
                        margin-left: 50px;">
                距离：{fastest['distance_m'] / 1000:.2f} km<br>
                时间：{fastest['route_cost_s'] / 60:.1f} 分钟<br>
                道路边：{fastest['edge_count']} 条
            </div>

            {'<div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #e5e7eb; font-size: 11px; color: #9ca3af;">⚠️ 两条路线相同</div>' if fastest['points'] == shortest['points'] else ''}
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

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
    if 'route_source_mode' not in st.session_state:
        st.session_state.route_source_mode = "地图选点"

    # 起终点来源选择
    st.markdown("### 📍 选择起终点")

    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        if st.button("🗺️ 地图选点", use_container_width=True,
                     type="primary" if st.session_state.route_source_mode == "地图选点" else "secondary"):
            st.session_state.route_source_mode = "地图选点"
            st.session_state.route_points = []
            st.session_state.route_result = None
            st.rerun()

    with mode_col2:
        if st.button("📋 历史OD端点", use_container_width=True,
                     type="primary" if st.session_state.route_source_mode == "历史OD端点" else "secondary"):
            st.session_state.route_source_mode = "历史OD端点"
            st.session_state.route_points = []
            st.session_state.route_result = None
            st.rerun()

    # 根据模式显示不同的界面
    if st.session_state.route_source_mode == "历史OD端点":
        render_od_selection_mode(payload, speed_vehicle_ids)
    else:
        render_map_selection_mode(payload, speed_vehicle_ids)


def render_od_selection_mode(payload, speed_vehicle_ids):
    """历史 OD 端点选择模式"""
    from map_plotter import load_od_data

    st.info("💡 从历史订单中选择起终点，系统会使用订单的上下车点计算路线。")

    # 加载 OD 数据
    with st.spinner("正在加载历史 OD 数据..."):
        od_df = load_od_data(
            payload["start_time"],
            payload["end_time"],
            vehicle_ids=speed_vehicle_ids or None
        )

    if od_df is None or len(od_df) == 0:
        st.warning("当前时间范围内没有可用的 OD 记录，请调整时间范围或车辆选择。")
        return

    st.success(f"✅ 找到 {len(od_df)} 条 OD 记录")

    # 选择订单
    max_options = min(100, len(od_df))
    od_options = list(range(max_options))

    selected_index = st.selectbox(
        "选择历史订单",
        od_options,
        format_func=lambda idx: (
            f"订单 {idx+1}: 车辆 {od_df.iloc[idx]['O_TAXI_ID']} | "
            f"{od_df.iloc[idx]['O_time'].strftime('%H:%M:%S')} → "
            f"{od_df.iloc[idx]['D_time'].strftime('%H:%M:%S')} | "
            f"{od_df.iloc[idx].get('OD_Dist_km', 0):.2f} km"
        ),
        key="route_od_selector"
    )

    if selected_index is not None:
        selected_od = od_df.iloc[int(selected_index)]

        # 显示订单详情
        st.markdown("#### 📊 订单详情")
        detail_cols = st.columns(4)
        detail_cols[0].metric("车辆ID", selected_od['O_TAXI_ID'])
        detail_cols[1].metric("上车时间", selected_od['O_time'].strftime('%H:%M:%S'))
        detail_cols[2].metric("下车时间", selected_od['D_time'].strftime('%H:%M:%S'))
        detail_cols[3].metric("行驶距离", f"{selected_od.get('OD_Dist_km', 0):.2f} km")

        coord_cols = st.columns(2)
        with coord_cols[0]:
            st.caption("**上车点坐标**")
            st.text(f"纬度: {selected_od['O_lat']:.6f}")
            st.text(f"经度: {selected_od['O_lng']:.6f}")

        with coord_cols[1]:
            st.caption("**下车点坐标**")
            st.text(f"纬度: {selected_od['D_lat']:.6f}")
            st.text(f"经度: {selected_od['D_lng']:.6f}")

        # 计算路线按钮
        if st.button("✅ 计算路线", type="primary", use_container_width=True):
            with st.spinner("正在生成全日道路基准速度并计算最短/最快路线..."):
                route_result = plan_baseline_routes_between_points(
                    selected_od['O_lat'],
                    selected_od['O_lng'],
                    selected_od['D_lat'],
                    selected_od['D_lng'],
                    vehicle_ids=tuple(speed_vehicle_ids or []),
                    query_date=payload["start_time"].date(),
                )

            if route_result.get("success"):
                st.session_state.route_result = route_result
                # 保存起终点用于地图显示
                st.session_state.route_points = [
                    {'lat': selected_od['O_lat'], 'lng': selected_od['O_lng']},
                    {'lat': selected_od['D_lat'], 'lng': selected_od['D_lng']}
                ]
                st.success("✅ 路线计算完成！")
                st.rerun()
            else:
                st.error(f"路线计算失败: {route_result.get('error', '未知错误')}")

        # 显示地图和统计信息
        if st.session_state.route_result and st.session_state.route_result.get("success"):
            display_route_map_and_stats(selected_od)


def render_map_selection_mode(payload, speed_vehicle_ids):
    """地图选点模式"""
    st.info("💡 点击地图依次选择起点和终点")

    # 操作按钮
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🔄 重新选点", width="stretch"):
            st.session_state.route_points = []
            st.session_state.route_result = None
            # 清除地图视图状态，重置为默认视图
            if 'map_view_center' in st.session_state:
                del st.session_state.map_view_center
            if 'map_view_zoom' in st.session_state:
                del st.session_state.map_view_zoom
            st.rerun()

    with col2:
        can_calculate = len(st.session_state.route_points) == 2
        if st.button(
            "✅ 计算路线",
            width="stretch",
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
    # 使用返回的 map_data 中的 center 和 zoom 来保存当前视图
    map_data = st_folium(
        route_map,
        width=None,
        height=600,
        key=f"route_map_{len(st.session_state.route_points)}_{st.session_state.get('route_result') is not None}",
        returned_objects=["last_clicked"]
    )

    # 保存当前地图的视图状态（从 map_data 中获取）
    if map_data:
        # 如果地图返回了中心点和缩放级别，保存它们
        if map_data.get("center"):
            st.session_state.map_view_center = [map_data["center"]["lat"], map_data["center"]["lng"]]
        if map_data.get("zoom"):
            st.session_state.map_view_zoom = map_data["zoom"]

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
    st.dataframe(summary_df, width="stretch", hide_index=True)

    # 如果是从 OD 选择的，显示对比信息
    if selected_od is not None:
        st.markdown("#### 🔍 实际 vs 规划对比")
        compare_cols = st.columns(3)

        actual_distance = selected_od.get('OD_Dist_km', 0)
        planned_distance = fastest['distance_m'] / 1000
        distance_diff = actual_distance - planned_distance

        compare_cols[0].metric(
            "实际行驶距离",
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
