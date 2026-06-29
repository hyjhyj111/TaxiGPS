import os
import inspect
import sys
import unittest
from datetime import datetime


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit_app


class StreamlitNavigationTest(unittest.TestCase):
    def test_navigation_groups_cover_view_options_in_display_order(self):
        grouped_views = [
            item["view"]
            for group in streamlit_app.NAVIGATION_GROUPS
            for item in group["items"]
        ]

        self.assertEqual(grouped_views, streamlit_app.VIEW_OPTIONS)
        self.assertEqual(grouped_views[0], "轨迹查询")
        self.assertIn("路线规划", grouped_views)
        self.assertIn("拥堵与ETA", grouped_views)

    def test_map_modules_are_grouped_together(self):
        map_group = next(
            group for group in streamlit_app.NAVIGATION_GROUPS if group["label"] == "地图"
        )
        map_views = [item["view"] for item in map_group["items"]]

        self.assertEqual(map_views, ["轨迹查询", "动画轨迹", "分钟位置", "OD点标注"])

    def test_view_context_describes_selected_module(self):
        context = streamlit_app.get_view_context("路线规划")

        self.assertEqual(context["group"], "路线")
        self.assertIn("最短距离", context["description"])

    def test_every_view_has_context(self):
        """Every VIEW_OPTIONS should have a description lookup."""
        for view in streamlit_app.VIEW_OPTIONS:
            context = streamlit_app.get_view_context(view)
            self.assertIn("group", context)
            self.assertIn("description", context)
            self.assertTrue(len(context["group"]) > 0)

    def test_map_first_views_do_not_render_metric_previews_above_maps(self):
        for fn in [
            streamlit_app.render_trajectory_view,
            streamlit_app.render_animation_view,
            streamlit_app.render_minute_view,
            streamlit_app.render_od_view,
        ]:
            source = inspect.getsource(fn)
            self.assertNotIn("st.metric", source, fn.__name__)
            self.assertNotIn("render_vehicle_status_panel", source, fn.__name__)

    def test_validate_payload_limits_animation_speed_to_one_to_five(self):
        payload = {
            "trajectory_vehicle_ids": ["22225"],
            "start_time": datetime(2023, 10, 12, 8, 0),
            "end_time": datetime(2023, 10, 12, 9, 0),
            "time_scale": 10.0,
        }

        errors = streamlit_app.validate_payload(payload)

        self.assertIn("动画倍速必须在 1-5 倍之间。", errors)


if __name__ == "__main__":
    unittest.main()
