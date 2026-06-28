import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import heatmap_analysis as ha


class DummyMap:
    def __init__(self):
        self.saved_path = None

    def save(self, output_path):
        self.saved_path = output_path


class StaticHeatmapRenderLimitTest(unittest.TestCase):
    def test_static_heatmap_caps_render_points_for_large_minute_maps(self):
        points = pd.DataFrame(
            {
                "lat": np.linspace(22.45, 22.85, 2000),
                "lng": np.linspace(113.80, 114.55, 2000),
                "weight": np.linspace(1.0, 2000.0, 2000),
            }
        )
        meta = {
            "source_label": "分钟缓存车辆位置",
            "filter_stats": {"output_points": len(points), "bounds_removed": 0, "drift_removed": 0},
        }
        captured = {}
        old_limit = ha.ANALYSIS_CONFIG.get("MAX_STATIC_HEATMAP_POINTS")
        ha.ANALYSIS_CONFIG["MAX_STATIC_HEATMAP_POINTS"] = 800

        class RecordingHeatMap:
            def __init__(self, heat_points, **kwargs):
                captured["point_count"] = len(heat_points)

            def add_to(self, target):
                return self

        try:
            with patch.object(ha, "aggregate_minute_range_fast", return_value=(points, meta)), \
                patch.object(ha, "_filter_points_to_shenzhen_boundary", return_value=(points, 0)), \
                patch.object(ha, "_build_generic_map", return_value=DummyMap()), \
                patch.object(ha, "HeatMap", RecordingHeatMap), \
                patch.object(ha, "add_map_layers", lambda m: None):
                output_path, result_meta = ha.build_static_heatmap(
                    "minute",
                    "2023-10-12 08:00:00",
                    "2023-10-12 10:00:00",
                    save_path="/private/tmp/static_heatmap_limit_test.html",
                )
        finally:
            if old_limit is None:
                ha.ANALYSIS_CONFIG.pop("MAX_STATIC_HEATMAP_POINTS", None)
            else:
                ha.ANALYSIS_CONFIG["MAX_STATIC_HEATMAP_POINTS"] = old_limit

        self.assertEqual(output_path, "/private/tmp/static_heatmap_limit_test.html")
        self.assertLessEqual(captured["point_count"], 800)
        self.assertEqual(result_meta["heat_points"], 2000)
        self.assertEqual(result_meta["render_heat_points"], 800)


if __name__ == "__main__":
    unittest.main()
