import os
import sys
import unittest
from unittest.mock import patch


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class RoutePlannerImportTest(unittest.TestCase):
    def test_route_planner_imports_without_streamlit_folium(self):
        real_import = __import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "streamlit_folium":
                raise ModuleNotFoundError("No module named 'streamlit_folium'")
            return real_import(name, globals, locals, fromlist, level)

        sys.modules.pop("route_planner", None)
        sys.modules.pop("streamlit_folium", None)
        with patch("builtins.__import__", side_effect=guarded_import):
            import route_planner

        self.assertTrue(hasattr(route_planner, "render_route_planning_view"))


    def test_eta_model_feature_set_matches_requested_feature_groups(self):
        sys.modules.pop("map_plotter", None)
        import map_plotter

        required = {
            "hour",
            "weekday",
            "is_peak",
            "pickup_lat",
            "pickup_lng",
            "dropoff_lat",
            "dropoff_lng",
            "pickup_region_id",
            "dropoff_region_id",
            "region_pair_id",
            "straight_distance_km",
            "order_distance_km",
            "network_distance_km",
            "shortest_route_km",
            "fastest_route_cost_min",
            "historical_region_speed",
        }
        self.assertTrue(required.issubset(set(map_plotter.ETA_MODEL_FEATURES)))
        self.assertIn("订单载客公里数(km)", set(map_plotter.ETA_MODEL_FEATURE_LABELS.values()))
        self.assertNotIn("duration_s", map_plotter.ETA_MODEL_FEATURES)

    def test_route_click_component_assets_exist(self):
        sys.modules.pop("route_planner", None)
        import route_planner

        self.assertTrue(route_planner.ROUTE_CLICK_COMPONENT_DIR.exists())
        self.assertTrue((route_planner.ROUTE_CLICK_COMPONENT_DIR / "index.html").exists())

    def test_route_click_component_stabilizes_leaflet_layout_after_resize(self):
        sys.modules.pop("route_planner", None)
        import route_planner

        html = (route_planner.ROUTE_CLICK_COMPONENT_DIR / "index.html").read_text(encoding="utf-8")

        self.assertIn("__taxigpsRouteRefreshLayout", html)
        self.assertIn("ResizeObserver", html)
        self.assertIn("map.invalidateSize({ pan: false })", html)
        self.assertIn("[0, 80, 180, 360, 720, 1200]", html)

    def test_route_click_component_inlines_critical_leaflet_layout_css(self):
        sys.modules.pop("route_planner", None)
        import route_planner

        html = (route_planner.ROUTE_CLICK_COMPONENT_DIR / "index.html").read_text(encoding="utf-8")

        self.assertIn("Critical Leaflet layout fallback", html)
        self.assertIn(".leaflet-tile", html)
        self.assertIn(".leaflet-pane", html)
        self.assertIn("position: absolute", html)

    def test_route_click_component_draws_actual_route_above_planned_routes(self):
        sys.modules.pop("route_planner", None)
        import route_planner

        html = (route_planner.ROUTE_CLICK_COMPONENT_DIR / "index.html").read_text(encoding="utf-8")

        fastest_index = html.index("fastestLine = L.polyline")
        actual_index = html.index("actualLine = L.polyline")
        self.assertGreater(actual_index, fastest_index)
        self.assertNotIn("dashArray: \"10,6\"", html)
        self.assertIn("lineCap: \"round\"", html)

    def test_route_planner_history_od_ui_is_simplified_and_professional(self):
        source_path = os.path.join(SRC_DIR, "route_planner.py")
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()

        for symbol in ["💡", "📍", "🗺️", "📋", "🔄", "✅", "📊", "🔍", "⚠️"]:
            self.assertNotIn(symbol, source)
        for removed_text in ["生成当前筛选范围校正OD缓存", "订单详情", "上车时间", "下车时间", "行驶距离", "分析订单路线"]:
            self.assertNotIn(removed_text, source)
        self.assertIn("render_history_od_stage08_from_sidebar", source)

    def test_route_planner_exposes_left_sidebar_driven_history_od_ui(self):
        source_path = os.path.join(SRC_DIR, "route_planner.py")
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("render_history_od_stage08_from_sidebar", source)
        self.assertIn("左侧控制台", source)
        self.assertIn("display_stage08_batch_results", source)
        self.assertIn("绕行比例最高订单", source)

    def test_route_planner_mode_switch_lives_in_sidebar(self):
        route_source_path = os.path.join(SRC_DIR, "route_planner.py")
        app_source_path = os.path.join(SRC_DIR, "streamlit_app.py")
        with open(route_source_path, "r", encoding="utf-8") as f:
            route_source = f.read()
        with open(app_source_path, "r", encoding="utf-8") as f:
            app_source = f.read()

        self.assertNotIn('st.button("地图选点"', route_source)
        self.assertNotIn('st.button("历史OD端点"', route_source)
        self.assertNotIn('st.button("重新选点"', route_source)
        self.assertNotIn('st.button("计算路线"', route_source)
        self.assertIn('key="route_source_mode"', app_source)
        self.assertIn('key="route_sidebar_reset_points"', app_source)
        self.assertIn('key="route_sidebar_calculate"', app_source)
        self.assertIn('render_history_od_sidebar_tools()', app_source)

    def test_route_planner_exposes_stage08_frontend_without_script_entry(self):
        source_path = os.path.join(SRC_DIR, "route_planner.py")
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("历史订单分析", source)
        self.assertIn("render_history_od_stage08_from_sidebar", source)
        self.assertIn("extract_actual_edges_from_track", source)
        self.assertIn("calculate_comparison_metrics", source)
        self.assertNotIn("run_stage08.py", source)


if __name__ == "__main__":
    unittest.main()
