#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式路线选择器
支持点击地图直接更新 Streamlit session state
"""

import streamlit as st
import streamlit.components.v1 as components
import folium
from folium.plugins import MousePosition
import json
import os


def create_interactive_route_picker(center_lat=22.52847, center_lng=114.05454, zoom=13):
    """
    创建一个交互式路线选择器地图
    用户点击地图后，坐标会自动更新到 session state

    Args:
        center_lat: 地图中心纬度
        center_lng: 地图中心经度
        zoom: 缩放级别

    Returns:
        None (直接更新 session_state)
    """

    # 初始化 session state
    if 'picker_origin_lat' not in st.session_state:
        st.session_state.picker_origin_lat = None
    if 'picker_origin_lng' not in st.session_state:
        st.session_state.picker_origin_lng = None
    if 'picker_dest_lat' not in st.session_state:
        st.session_state.picker_dest_lat = None
    if 'picker_dest_lng' not in st.session_state:
        st.session_state.picker_dest_lng = None
    if 'picker_step' not in st.session_state:
        st.session_state.picker_step = 0  # 0: 选择起点, 1: 选择终点, 2: 完成

    # 创建地图
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom,
        tiles='OpenStreetMap',
        control_scale=True,
    )

    # 添加鼠标位置显示
    MousePosition().add_to(m)

    # 如果已有起点，添加标记
    if st.session_state.picker_origin_lat is not None:
        folium.Marker(
            [st.session_state.picker_origin_lat, st.session_state.picker_origin_lng],
            popup='起点',
            tooltip='起点',
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)

    # 如果已有终点，添加标记
    if st.session_state.picker_dest_lat is not None:
        folium.Marker(
            [st.session_state.picker_dest_lat, st.session_state.picker_dest_lng],
            popup='终点',
            tooltip='终点',
            icon=folium.Icon(color='red', icon='flag')
        ).add_to(m)

        # 添加连线
        if st.session_state.picker_origin_lat is not None:
            folium.PolyLine(
                [
                    [st.session_state.picker_origin_lat, st.session_state.picker_origin_lng],
                    [st.session_state.picker_dest_lat, st.session_state.picker_dest_lng]
                ],
                color='#64748b',
                weight=3,
                dash_array='6,6',
                opacity=0.8
            ).add_to(m)

    # 添加交互式 JavaScript
    click_handler_js = """
    <script>
    function setupMapClickHandler() {
        // 等待 Leaflet 地图加载
        var checkExist = setInterval(function() {
            if (typeof map_object !== 'undefined') {
                clearInterval(checkExist);

                map_object.on('click', function(e) {
                    var lat = e.latlng.lat.toFixed(6);
                    var lng = e.latlng.lng.toFixed(6);

                    // 发送数据到 Streamlit
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: {
                            lat: parseFloat(lat),
                            lng: parseFloat(lng)
                        }
                    }, '*');
                });
            }
        }, 100);
    }

    // 页面加载完成后设置
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupMapClickHandler);
    } else {
        setupMapClickHandler();
    }
    </script>
    """

    # 保存地图到临时文件
    map_html = m._repr_html_()

    # 创建带提示面板的 HTML
    step_text = "点击地图选择起点"
    if st.session_state.picker_step == 1:
        step_text = "点击地图选择终点"
    elif st.session_state.picker_step == 2:
        step_text = "已完成选择，可以重新选择"

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }}
            #instruction-panel {{
                position: fixed;
                top: 16px;
                left: 16px;
                z-index: 1000;
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid rgba(148, 163, 184, 0.45);
                border-radius: 12px;
                padding: 12px 16px;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.14);
                color: #0f172a;
                font-size: 13px;
                font-weight: 600;
                max-width: 280px;
            }}
            #instruction-panel .step {{
                color: #3b82f6;
                margin-bottom: 4px;
            }}
            #instruction-panel .hint {{
                font-size: 11px;
                color: #64748b;
                font-weight: 400;
                line-height: 1.4;
            }}
        </style>
    </head>
    <body>
        <div id="instruction-panel">
            <div class="step">{step_text}</div>
            <div class="hint">点击地图上任意位置设置坐标</div>
        </div>
        {map_html}
        {click_handler_js}
    </body>
    </html>
    """

    # 使用 iframe 组件显示
    clicked_data = components.html(full_html, height=600, scrolling=False)

    return clicked_data


def render_route_picker_with_inputs():
    """
    渲染完整的路线选择器界面（地图 + 输入框 + 控制按钮）
    """
    st.markdown("### 交互式路线选择")

    # 控制按钮
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("📍 点击选起点", use_container_width=True):
            st.session_state.picker_step = 0
            st.rerun()
    with col2:
        if st.button("🏁 点击选终点", use_container_width=True):
            st.session_state.picker_step = 1
            st.rerun()
    with col3:
        if st.button("🔄 重置选择", use_container_width=True):
            st.session_state.picker_origin_lat = None
            st.session_state.picker_origin_lng = None
            st.session_state.picker_dest_lat = None
            st.session_state.picker_dest_lng = None
            st.session_state.picker_step = 0
            st.rerun()

    # 显示交互式地图
    clicked = create_interactive_route_picker()

    # 处理点击事件
    if clicked is not None:
        if st.session_state.picker_step == 0:
            st.session_state.picker_origin_lat = clicked['lat']
            st.session_state.picker_origin_lng = clicked['lng']
            st.session_state.picker_step = 1
            st.rerun()
        elif st.session_state.picker_step == 1:
            st.session_state.picker_dest_lat = clicked['lat']
            st.session_state.picker_dest_lng = clicked['lng']
            st.session_state.picker_step = 2
            st.rerun()

    # 显示当前选择的坐标
    st.markdown("---")
    st.markdown("#### 已选择的坐标")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**起点坐标**")
        if st.session_state.picker_origin_lat is not None:
            st.success(f"纬度: {st.session_state.picker_origin_lat:.6f}")
            st.success(f"经度: {st.session_state.picker_origin_lng:.6f}")
        else:
            st.info("尚未选择起点")

    with col2:
        st.markdown("**终点坐标**")
        if st.session_state.picker_dest_lat is not None:
            st.success(f"纬度: {st.session_state.picker_dest_lat:.6f}")
            st.success(f"经度: {st.session_state.picker_dest_lng:.6f}")
        else:
            st.info("尚未选择终点")

    # 应用到路线规划的按钮
    if st.session_state.picker_origin_lat is not None and st.session_state.picker_dest_lat is not None:
        if st.button("✅ 应用到路线规划", type="primary", use_container_width=True):
            st.session_state.route_origin_lat = st.session_state.picker_origin_lat
            st.session_state.route_origin_lng = st.session_state.picker_origin_lng
            st.session_state.route_dest_lat = st.session_state.picker_dest_lat
            st.session_state.route_dest_lng = st.session_state.picker_dest_lng
            st.success("✓ 坐标已应用到路线规划！")
            st.balloons()
