import os
import shutil
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


class RoadCorrectionCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="taxigps-road-cache-")
        self.original_cache_dir = map_plotter.CONFIG.get("ROAD_CORRECTED_CACHE_DIR")
        map_plotter.CONFIG["ROAD_CORRECTED_CACHE_DIR"] = self.temp_dir

    def tearDown(self):
        if self.original_cache_dir is None:
            map_plotter.CONFIG.pop("ROAD_CORRECTED_CACHE_DIR", None)
        else:
            map_plotter.CONFIG["ROAD_CORRECTED_CACHE_DIR"] = self.original_cache_dir
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_query_interval_uses_cache_only_when_requested_range_is_covered(self):
        def vehicle_window(vehicle_id, start_time=None, end_time=None, log_result=True):
            del vehicle_id, log_result
            start = pd.Timestamp(start_time)
            end = pd.Timestamp(end_time)
            return pd.DataFrame(
                {
                    "time": [start, end],
                    "long": [114.0, 114.01],
                    "lati": [22.0, 22.01],
                    "status": [0, 1],
                    "speed": [20.0, 22.0],
                }
            )

        def correction_for_window(df, graph=None, network_meta=None, max_points=None):
            del graph, network_meta, max_points
            start = pd.Timestamp(df.iloc[0]["time"])
            end = pd.Timestamp(df.iloc[-1]["time"])
            corrected_rows = pd.DataFrame(
                {
                    "vehicle_id": ["22223", "22223"],
                    "time": [start, end],
                    "status": [0, 1],
                    "speed": [20.0, 22.0],
                    "raw_lon": [114.0, 114.01],
                    "raw_lat": [22.0, 22.01],
                    "matched_lon": [114.001, 114.011],
                    "matched_lat": [22.001, 22.011],
                    "matched_node": [100, 101],
                    "edge_u": [100, 100],
                    "edge_v": [100, 101],
                    "edge_key": [0, 0],
                    "path_mode": ["same_node", "directed"],
                    "sequence": [0, 1],
                }
            )
            return {
                "points": [[22.001, 114.001], [22.011, 114.011]],
                "matched_nodes": [100, 101],
                "path_rows": corrected_rows,
                "meta": {"success": True, "raw_points": 2, "corrected_points": 2},
            }

        with patch.object(map_plotter, "load_vehicle_trajectory", side_effect=vehicle_window), patch.object(
            map_plotter, "correct_trajectory_with_road_network", side_effect=correction_for_window
        ) as correction_mock:
            first = map_plotter.load_road_corrected_vehicle_slice(
                "22223",
                start_time="2023-10-12 08:00:00",
                end_time="2023-10-12 09:00:00",
                graph=object(),
                network_meta={"path": "/tmp/shenzhen_drive.pkl"},
            )
            covered = map_plotter.load_road_corrected_vehicle_slice(
                "22223",
                start_time="2023-10-12 08:10:00",
                end_time="2023-10-12 08:50:00",
                graph=object(),
                network_meta={"path": "/tmp/shenzhen_drive.pkl"},
            )
            partial = map_plotter.load_road_corrected_vehicle_slice(
                "22223",
                start_time="2023-10-12 08:30:00",
                end_time="2023-10-12 09:30:00",
                graph=object(),
                network_meta={"path": "/tmp/shenzhen_drive.pkl"},
            )

        self.assertFalse(first["meta"]["cache_hit"])
        self.assertEqual(first["meta"]["processed_intervals"], 1)
        self.assertTrue(covered["meta"]["cache_hit"])
        self.assertEqual(covered["meta"]["processed_intervals"], 0)
        self.assertFalse(partial["meta"]["cache_hit"])
        self.assertEqual(partial["meta"]["processed_intervals"], 1)
        self.assertEqual(correction_mock.call_count, 2)
        self.assertTrue(os.path.exists(partial["meta"]["cache_path"]))
        self.assertTrue(os.path.exists(partial["meta"]["coverage_path"]))

    def test_road_correction_expands_shortest_path_edge_geometry(self):
        import networkx as nx
        from shapely.geometry import LineString

        graph = nx.MultiDiGraph()
        graph.graph["crs"] = "epsg:4326"
        graph.add_node(1, x=114.000, y=22.000)
        graph.add_node(2, x=114.020, y=22.000)
        graph.add_edge(
            1,
            2,
            key=0,
            length=3000.0,
            geometry=LineString([(114.000, 22.000), (114.010, 22.020), (114.020, 22.000)]),
        )
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 12:00:00", "2023-10-12 12:05:00"]),
                "long": [114.0001, 114.0199],
                "lati": [22.0001, 22.0001],
                "status": [0, 1],
                "speed": [20.0, 20.0],
                "vehicle_id": ["22223", "22223"],
            }
        )

        result = map_plotter.correct_trajectory_with_road_network(df, graph=graph, network_meta={"path": "/tmp/road.pkl"})

        self.assertIn([22.02, 114.01], result["points"])

    def test_road_correction_interpolates_times_across_edge_geometry(self):
        import networkx as nx
        from shapely.geometry import LineString

        graph = nx.MultiDiGraph()
        graph.graph["crs"] = "epsg:4326"
        graph.add_node(1, x=114.000, y=22.000)
        graph.add_node(2, x=114.020, y=22.000)
        graph.add_edge(
            1,
            2,
            key=0,
            length=3000.0,
            geometry=LineString([(114.000, 22.000), (114.010, 22.020), (114.020, 22.000)]),
        )
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 12:00:00", "2023-10-12 12:05:00"]),
                "long": [114.0001, 114.0199],
                "lati": [22.0001, 22.0001],
                "status": [0, 1],
                "speed": [20.0, 20.0],
                "vehicle_id": ["22223", "22223"],
            }
        )

        result = map_plotter.correct_trajectory_with_road_network(df, graph=graph, network_meta={"path": "/tmp/road.pkl"})
        rows = result["path_rows"]

        self.assertEqual(rows.iloc[0]["time"], pd.Timestamp("2023-10-12 12:00:00"))
        self.assertEqual(rows.iloc[-1]["time"], pd.Timestamp("2023-10-12 12:05:00"))
        self.assertTrue(rows["time"].is_monotonic_increasing)
        self.assertGreater(rows["time"].nunique(), 2)

    def test_road_correction_interpolates_speed_across_edge_geometry(self):
        import networkx as nx
        from shapely.geometry import LineString

        graph = nx.MultiDiGraph()
        graph.graph["crs"] = "epsg:4326"
        graph.add_node(1, x=114.000, y=22.000)
        graph.add_node(2, x=114.020, y=22.000)
        graph.add_edge(
            1,
            2,
            key=0,
            length=3000.0,
            geometry=LineString([(114.000, 22.000), (114.010, 22.020), (114.020, 22.000)]),
        )
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2023-10-12 12:00:00", "2023-10-12 12:05:00"]),
                "long": [114.0001, 114.0199],
                "lati": [22.0001, 22.0001],
                "status": [0, 1],
                "speed": [10.0, 30.0],
                "vehicle_id": ["22223", "22223"],
            }
        )

        result = map_plotter.correct_trajectory_with_road_network(df, graph=graph, network_meta={"path": "/tmp/road.pkl"})
        speeds = result["path_rows"]["speed"].round(3).tolist()

        self.assertEqual(speeds[0], 10.0)
        self.assertEqual(speeds[-1], 30.0)
        self.assertGreater(len(set(speeds)), 2)
        self.assertTrue(result["path_rows"]["speed"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
