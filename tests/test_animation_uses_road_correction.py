import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import map_plotter


class AnimationRoadCorrectionTest(unittest.TestCase):
    def _raw_vehicle_df(self):
        return pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 08:00:00", "2023-10-12 08:05:00"]),
                "long": [114.0000, 114.0200],
                "lati": [22.0000, 22.0200],
                "status": [0, 1],
                "speed": [20.0, 21.0],
            }
        )

    def _corrected_slice(self, vehicle_id, start_time=None, end_time=None, graph=None, network_meta=None, force_rebuild=False):
        del start_time, end_time, graph, network_meta, force_rebuild
        rows = pd.DataFrame(
            {
                "vehicle_id": [str(vehicle_id), str(vehicle_id), str(vehicle_id)],
                "time": pd.to_datetime(["2023-10-12 08:00:00", "2023-10-12 08:02:30", "2023-10-12 08:05:00"]),
                "status": [0, 0, 1],
                "speed": [20.0, 20.5, 21.0],
                "raw_lon": [114.0, 114.01, 114.02],
                "raw_lat": [22.0, 22.01, 22.02],
                "matched_lon": [114.1000, 114.1100, 114.1200],
                "matched_lat": [22.1000, 22.1150, 22.1200],
                "matched_node": [10, 11, 12],
                "edge_u": [10, 10, 11],
                "edge_v": [10, 11, 12],
                "edge_key": [0, 0, 0],
                "path_mode": ["same_node", "directed", "directed"],
                "sequence": [0, 1, 2],
            }
        )
        return {
            "rows": rows,
            "points": [[22.1, 114.1], [22.115, 114.11], [22.12, 114.12]],
            "meta": {"success": True, "cache_hit": True},
        }

    def test_single_vehicle_animation_prefers_road_corrected_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp:
            output_path = temp.name
        self.addCleanup(lambda: os.path.exists(output_path) and os.remove(output_path))

        with patch.object(map_plotter, "load_vehicle_trajectory", return_value=self._raw_vehicle_df()), patch.object(
            map_plotter, "load_road_network", return_value=(object(), {"path": "/tmp/road.pkl"})
        ), patch.object(map_plotter, "load_road_corrected_vehicle_slice", side_effect=self._corrected_slice):
            result_path = map_plotter.plot_animated_trajectory(
                "22223",
                start_time="2023-10-12 08:00:00",
                end_time="2023-10-12 08:05:00",
                save_path=output_path,
            )

        self.assertEqual(result_path, output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()

        self.assertIn('"coordinates": [114.1, 22.1]', html)
        self.assertIn('"coordinates": [114.11, 22.115]', html)
        self.assertNotIn('"coordinates": [114.0, 22.0]', html)

    def test_multi_vehicle_animation_prefers_road_corrected_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp:
            output_path = temp.name
        self.addCleanup(lambda: os.path.exists(output_path) and os.remove(output_path))

        with patch.object(map_plotter, "load_vehicle_trajectory", return_value=self._raw_vehicle_df()), patch.object(
            map_plotter, "load_road_network", return_value=(object(), {"path": "/tmp/road.pkl"})
        ), patch.object(map_plotter, "load_road_corrected_vehicle_slice", side_effect=self._corrected_slice):
            result_path = map_plotter.plot_multi_vehicle_animated_trajectory(
                ["22223", "22224"],
                start_time="2023-10-12 08:00:00",
                end_time="2023-10-12 08:05:00",
                save_path=output_path,
            )

        self.assertEqual(result_path, output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()

        self.assertIn('"lat": 22.1, "lng": 114.1', html)
        self.assertIn('"lat": 22.115, "lng": 114.11', html)
        self.assertNotIn('"lat": 22.0, "lng": 114.0', html)

    def test_animation_dataframe_spreads_duplicate_corrected_times(self):
        rows = pd.DataFrame(
            {
                "vehicle_id": ["22223", "22223", "22223"],
                "time": pd.to_datetime(["2023-10-12 08:05:00", "2023-10-12 08:05:00", "2023-10-12 08:05:00"]),
                "status": [0, 0, 1],
                "speed": [20.0, 20.0, 21.0],
                "raw_lon": [114.0, 114.01, 114.02],
                "raw_lat": [22.0, 22.01, 22.02],
                "matched_lon": [114.1000, 114.1100, 114.1200],
                "matched_lat": [22.1000, 22.1150, 22.1200],
                "matched_node": [10, 11, 12],
                "edge_u": [10, 10, 11],
                "edge_v": [10, 11, 12],
                "edge_key": [0, 0, 0],
                "path_mode": ["directed", "directed", "directed"],
                "sequence": [0, 1, 2],
            }
        )

        df = map_plotter._corrected_rows_to_animation_df(rows, fallback_start_time="2023-10-12 08:00:00")

        self.assertEqual(df.iloc[0]["time"], pd.Timestamp("2023-10-12 08:00:00"))
        self.assertEqual(df.iloc[-1]["time"], pd.Timestamp("2023-10-12 08:05:00"))
        self.assertTrue(df["time"].is_monotonic_increasing)
        self.assertEqual(df["time"].nunique(), len(df))

    def test_single_vehicle_animation_html_caps_playback_segment_duration(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 08:00:00", "2023-10-12 08:00:01"]),
                "long": [114.0000, 114.0200],
                "lati": [22.0000, 22.0000],
                "status": [0, 0],
                "speed": [20.0, 20.0],
                "speed_kmh": [0.0, 120.0],
            }
        )

        html = map_plotter._build_animation_html("22223", df, 1.0)

        self.assertIn("var speedBaseMultiplier = 1.0;", html)
        self.assertIn("maxPlaybackSpeedKmh", html)
        self.assertIn("observedDuration", html)
        self.assertIn("Math.max(rawDuration, observedDuration, minimumDuration)", html)
        self.assertIn("speedAwareProgress(rawT, segment.startDisplaySpeed, segment.endDisplaySpeed)", html)

    def test_single_vehicle_animation_payload_prefers_observed_speed(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 08:00:00", "2023-10-12 08:00:01"]),
                "long": [114.0000, 114.0200],
                "lati": [22.0000, 22.0000],
                "status": [0, 0],
                "speed": [12.0, 18.0],
                "speed_kmh": [0.0, 120.0],
            }
        )

        html = map_plotter._build_animation_html("22223", df, 1.0)

        self.assertIn('"speed": 12.0', html)
        self.assertIn('"speed": 18.0', html)
        self.assertNotIn('"speed": 120.0', html)
        self.assertIn("lerp(segment.startDisplaySpeed, segment.endDisplaySpeed", html)

    def test_single_vehicle_animation_html_avoids_per_frame_path_rebuilds(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 08:00:00", "2023-10-12 08:00:01", "2023-10-12 08:00:02"]),
                "long": [114.0000, 114.0100, 114.0200],
                "lati": [22.0000, 22.0050, 22.0000],
                "status": [0, 0, 1],
                "speed": [12.0, 18.0, 14.0],
            }
        )

        html = map_plotter._build_animation_html("22223", df, 1.0)

        self.assertIn("function updateTravelLineEndpoint", html)
        self.assertIn("traveledLine.addLatLng", html)
        self.assertIn("travelLineState", html)
        self.assertNotIn("pathLatLngs.slice(0, segment.startIndex + 1).concat", html)

    def test_single_vehicle_animation_html_instruments_fps_and_throttles_status_updates(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 08:00:00", "2023-10-12 08:00:01"]),
                "long": [114.0000, 114.0200],
                "lati": [22.0000, 22.0000],
                "status": [0, 0],
                "speed": [12.0, 18.0],
            }
        )

        html = map_plotter._build_animation_html("22223", df, 1.0)

        self.assertIn("var targetFrameRate = 60;", html)
        self.assertIn("window.__taxigpsAnimationStats", html)
        self.assertIn("function recordFrameSample", html)
        self.assertIn("var statusUpdateIntervalMs = 100;", html)
        self.assertIn("maybeUpdateInfoByPosition", html)
        self.assertIn('min="1"', html)
        self.assertIn('max="5"', html)

    def test_multi_vehicle_animation_html_caps_playback_segment_duration(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 08:00:00", "2023-10-12 08:00:01"]),
                "long": [114.0000, 114.0200],
                "lati": [22.0000, 22.0000],
                "status": [0, 0],
                "speed": [20.0, 20.0],
                "speed_kmh": [0.0, 120.0],
            }
        )

        html = map_plotter._build_multi_animation_html(
            [("22223", df, "#2563eb")],
            1.0,
            {"minLat": 22.0, "maxLat": 22.0, "minLng": 114.0, "maxLng": 114.02},
            int(df["time"].min().timestamp() * 1000),
            int(df["time"].max().timestamp() * 1000),
        )

        self.assertIn("var speedBaseMultiplier = 1.0;", html)
        self.assertIn("maxPlaybackSpeedKmh", html)
        self.assertIn("observedDuration", html)
        self.assertIn("Math.max(rawDuration, observedDuration, minimumDuration)", html)
        self.assertIn("speedAwareProgress(rawT, segment.startDisplaySpeed, segment.endDisplaySpeed)", html)

    def test_multi_vehicle_animation_payload_prefers_observed_speed(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 08:00:00", "2023-10-12 08:00:01"]),
                "long": [114.0000, 114.0200],
                "lati": [22.0000, 22.0000],
                "status": [0, 0],
                "speed": [12.0, 18.0],
                "speed_kmh": [0.0, 120.0],
            }
        )

        html = map_plotter._build_multi_animation_html(
            [("22223", df, "#2563eb")],
            1.0,
            {"minLat": 22.0, "maxLat": 22.0, "minLng": 114.0, "maxLng": 114.02},
            int(df["time"].min().timestamp() * 1000),
            int(df["time"].max().timestamp() * 1000),
        )

        self.assertIn('"speed": 12.0', html)
        self.assertIn('"speed": 18.0', html)
        self.assertNotIn('"speed": 120.0', html)
        self.assertIn("lerp(segment.startDisplaySpeed, segment.endDisplaySpeed", html)

    def test_single_vehicle_animation_display_speed_uses_driving_reference_when_moving(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 08:00:00", "2023-10-12 08:00:01"]),
                "long": [114.0000, 114.0100],
                "lati": [22.0000, 22.0000],
                "status": [0, 0],
                "speed": [0.0, 0.86],
                "speed_kmh": [0.0, 34.0],
                "distance_km": [0.0, 1.0],
            }
        )

        html = map_plotter._build_animation_html("22223", df, 1.0)

        self.assertNotIn('"speed": 0.86', html)
        self.assertIn('"speed": 34.0', html)

    def test_single_vehicle_animation_display_speed_keeps_true_stops_at_zero(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 08:00:00", "2023-10-12 08:00:01"]),
                "long": [114.0000, 114.0000],
                "lati": [22.0000, 22.0000],
                "status": [0, 0],
                "speed": [0.0, 0.0],
                "speed_kmh": [0.0, 0.0],
                "distance_km": [0.0, 0.0],
            }
        )

        html = map_plotter._build_animation_html("22223", df, 1.0)

        self.assertIn('"speed": 0.0', html)
        self.assertNotIn('"speed": 15.0', html)

    def test_multi_vehicle_animation_html_avoids_per_frame_path_rebuilds(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 08:00:00", "2023-10-12 08:00:01", "2023-10-12 08:00:02"]),
                "long": [114.0000, 114.0100, 114.0200],
                "lati": [22.0000, 22.0050, 22.0000],
                "status": [0, 0, 1],
                "speed": [12.0, 18.0, 14.0],
            }
        )

        html = map_plotter._build_multi_animation_html(
            [("22223", df, "#2563eb")],
            1.0,
            {"minLat": 22.0, "maxLat": 22.005, "minLng": 114.0, "maxLng": 114.02},
            int(df["time"].min().timestamp() * 1000),
            int(df["time"].max().timestamp() * 1000),
        )

        self.assertIn("function updateTravelLineEndpoint", html)
        self.assertIn("line.addLatLng", html)
        self.assertIn("travelLineState", html)
        self.assertNotIn("vehicle.pathLatLngs.slice(0, segment.startIndex + 1).concat", html)

    def test_multi_vehicle_animation_html_instruments_fps_and_throttles_status_updates(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 08:00:00", "2023-10-12 08:00:01"]),
                "long": [114.0000, 114.0200],
                "lati": [22.0000, 22.0000],
                "status": [0, 0],
                "speed": [12.0, 18.0],
            }
        )

        html = map_plotter._build_multi_animation_html(
            [("22223", df, "#2563eb")],
            1.0,
            {"minLat": 22.0, "maxLat": 22.0, "minLng": 114.0, "maxLng": 114.02},
            int(df["time"].min().timestamp() * 1000),
            int(df["time"].max().timestamp() * 1000),
        )

        self.assertIn("var targetFrameRate = 60;", html)
        self.assertIn("window.__taxigpsAnimationStats", html)
        self.assertIn("function recordFrameSample", html)
        self.assertIn("var statusUpdateIntervalMs = 100;", html)
        self.assertIn("maybeUpdateVehicleCard", html)
        self.assertIn('min="1"', html)
        self.assertIn('max="5"', html)

    def test_animation_time_scale_is_clamped_to_one_to_five(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 08:00:00", "2023-10-12 08:00:01"]),
                "long": [114.0000, 114.0200],
                "lati": [22.0000, 22.0000],
                "status": [0, 0],
                "speed": [12.0, 18.0],
            }
        )

        html = map_plotter._build_animation_html("22223", df, 10.0)

        self.assertIn('value="5.0"', html)
        self.assertIn("var speedMultiplier = 5.0 * speedBaseMultiplier;", html)


if __name__ == "__main__":
    unittest.main()
