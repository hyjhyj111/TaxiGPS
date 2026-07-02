import inspect
import os
import sys
import tempfile
import unittest

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class Stage08OrderRoutesTest(unittest.TestCase):
    def build_graph(self):
        import networkx as nx

        graph = nx.MultiDiGraph()
        graph.graph["crs"] = "epsg:4326"
        graph.add_node(1, x=114.000, y=22.000)
        graph.add_node(2, x=114.010, y=22.000)
        graph.add_node(3, x=114.020, y=22.000)
        graph.add_node(4, x=114.020, y=22.020)
        graph.add_edge(1, 2, key=0, length=1000.0, route_cost=500.0)
        graph.add_edge(2, 3, key=0, length=1000.0, route_cost=500.0)
        graph.add_edge(1, 4, key=0, length=4000.0, route_cost=100.0)
        graph.add_edge(4, 3, key=0, length=4000.0, route_cost=100.0)
        return graph

    def order(self):
        return pd.Series(
            {
                "id": "order-1",
                "vehicle_id": 123.0,
                "pickup_time": "2023-10-12 08:00:00",
                "dropoff_time": "2023-10-12 08:10:00",
                "pickup_node": 1,
                "dropoff_node": 3,
                "pickup_matched_lat": 22.000,
                "pickup_matched_lon": 114.000,
                "dropoff_matched_lat": 22.000,
                "dropoff_matched_lon": 114.020,
            }
        )

    def write_track_cache(self, temp_dir):
        track = pd.DataFrame(
            [
                {"time": "2023-10-12 07:59:00", "edge_u": 4, "edge_v": 3, "edge_key": 0, "matched_lat": 22.020, "matched_lng": 114.020},
                {"time": "2023-10-12 08:00:00", "edge_u": 1, "edge_v": 2, "edge_key": 0, "matched_lat": 22.000, "matched_lng": 114.000},
                {"time": "2023-10-12 08:01:00", "edge_u": 1, "edge_v": 2, "edge_key": 0, "matched_lat": 22.000, "matched_lng": 114.005},
                {"time": "2023-10-12 08:03:00", "edge_u": 2, "edge_v": 3, "edge_key": 0, "matched_lat": 22.000, "matched_lng": 114.010},
                {"time": "2023-10-12 08:06:00", "edge_u": 1, "edge_v": 2, "edge_key": 0, "matched_lat": 22.000, "matched_lng": 114.015},
                {"time": "2023-10-12 08:12:00", "edge_u": 4, "edge_v": 3, "edge_key": 0, "matched_lat": 22.020, "matched_lng": 114.020},
            ]
        )
        path = os.path.join(temp_dir, "123.parquet")
        track.to_parquet(path, index=False)
        return path

    def test_select_valid_order_filters_missing_nodes_and_bad_times(self):
        from stage08_order_route_comparison import select_valid_order

        od_df = pd.DataFrame(
            [
                {**self.order().to_dict(), "id": "missing-node", "pickup_node": pd.NA},
                {**self.order().to_dict(), "id": "bad-time", "dropoff_time": "2023-10-12 07:59:00"},
                self.order().to_dict(),
            ]
        )

        selected = select_valid_order(od_df)

        self.assertEqual(selected["id"], "order-1")
        self.assertEqual(int(selected["pickup_node"]), 1)
        self.assertEqual(int(selected["dropoff_node"]), 3)

    def test_select_valid_orders_prefers_orders_covered_by_track_cache_window(self):
        from stage08_order_route_comparison import select_valid_orders

        outside = self.order().copy()
        outside["id"] = "outside"
        outside["pickup_time"] = "2023-10-12 07:00:00"
        outside["dropoff_time"] = "2023-10-12 07:10:00"
        inside = self.order().copy()
        inside["id"] = "inside"

        with tempfile.TemporaryDirectory() as temp_dir:
            self.write_track_cache(temp_dir)

            selected = select_valid_orders(pd.DataFrame([outside.to_dict(), inside.to_dict()]), limit=10, track_cache_path=temp_dir)

        self.assertEqual(selected["id"].tolist(), ["inside"])

    def test_extract_actual_edges_removes_only_consecutive_duplicates(self):
        from stage08_order_route_comparison import extract_actual_edges_from_track

        with tempfile.TemporaryDirectory() as temp_dir:
            self.write_track_cache(temp_dir)

            edge_series, point_count = extract_actual_edges_from_track(self.build_graph(), self.order(), temp_dir)

        self.assertEqual(point_count, 4)
        self.assertEqual(
            list(edge_series.itertuples(index=False, name=None)),
            [(1, 2, 0), (2, 3, 0), (1, 2, 0)],
        )

    def test_extract_actual_edges_does_not_use_global_drop_duplicates(self):
        from stage08_order_route_comparison import extract_actual_edges_from_track

        source = inspect.getsource(extract_actual_edges_from_track)

        self.assertIn("shift()", source)
        self.assertNotIn("drop_duplicates", source)

    def test_compute_actual_route_geometry_counts_repeated_nonconsecutive_edges(self):
        from stage08_order_route_comparison import compute_actual_route_geometry

        edge_series = pd.DataFrame(
            [
                {"edge_u": 1, "edge_v": 2, "edge_key": 0},
                {"edge_u": 2, "edge_v": 3, "edge_key": 0},
                {"edge_u": 1, "edge_v": 2, "edge_key": 0},
            ]
        )

        distance_m, lines, meta = compute_actual_route_geometry(self.build_graph(), edge_series)

        self.assertEqual(distance_m, 3000.0)
        self.assertEqual(len(lines), 3)
        self.assertFalse(meta["has_break"])

    def test_get_planning_routes_uses_supplied_nodes_and_route_cost(self):
        from stage08_order_route_comparison import get_planning_routes

        shortest_gdf, fastest_gdf = get_planning_routes(self.build_graph(), 1, 3)

        self.assertEqual(shortest_gdf.attrs["route_nodes"], [1, 2, 3])
        self.assertEqual(fastest_gdf.attrs["route_nodes"], [1, 4, 3])
        self.assertEqual(float(shortest_gdf["length"].sum()), 2000.0)
        self.assertEqual(float(fastest_gdf["length"].sum()), 8000.0)

    def test_calculate_comparison_metrics_uses_length_weighted_overlap(self):
        from stage08_order_route_comparison import (
            calculate_comparison_metrics,
            compute_actual_route_geometry,
            get_planning_routes,
        )

        graph = self.build_graph()
        actual_edges = pd.DataFrame(
            [
                {"edge_u": 1, "edge_v": 2, "edge_key": 0},
                {"edge_u": 2, "edge_v": 3, "edge_key": 0},
                {"edge_u": 1, "edge_v": 2, "edge_key": 0},
            ]
        )
        actual_dist, _lines, _meta = compute_actual_route_geometry(graph, actual_edges)
        shortest_gdf, fastest_gdf = get_planning_routes(graph, 1, 3)

        metrics = calculate_comparison_metrics(self.order(), actual_edges, actual_dist, shortest_gdf, fastest_gdf)

        self.assertEqual(metrics["actual_duration_s"], 600.0)
        self.assertEqual(metrics["actual_distance_m"], 3000.0)
        self.assertEqual(metrics["shortest_distance_m"], 2000.0)
        self.assertAlmostEqual(metrics["detour_ratio"], 0.5)
        self.assertAlmostEqual(metrics["shortest_overlap_rate"], 1.0)
        self.assertAlmostEqual(metrics["fastest_overlap_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
