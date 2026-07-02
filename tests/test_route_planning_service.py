import os
import sys
import tempfile
import unittest

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class RoutePlanningServiceTest(unittest.TestCase):
    def build_graph(self):
        import networkx as nx

        graph = nx.MultiDiGraph()
        graph.graph["crs"] = "epsg:4326"
        graph.add_node(1, x=114.000, y=22.000)
        graph.add_node(2, x=114.010, y=22.000)
        graph.add_node(3, x=114.000, y=22.020)
        graph.add_node(4, x=114.020, y=22.020)
        graph.add_edge(1, 2, key=0, length=1000.0, highway="residential")
        graph.add_edge(2, 4, key=0, length=1000.0, highway="residential")
        graph.add_edge(1, 3, key=0, length=2000.0, highway="motorway")
        graph.add_edge(3, 4, key=0, length=2000.0, highway="motorway")
        return graph

    def test_complete_od_endpoints_prefers_corrected_rows(self):
        from route_planning import complete_od_endpoints

        od_df = pd.DataFrame(
            [
                {
                    "O_TAXI_ID": "22223",
                    "O_time": "2023-10-12 08:00:02",
                    "O_lat": 22.0001,
                    "O_lng": 114.0001,
                    "D_time": "2023-10-12 08:10:01",
                    "D_lat": 22.0201,
                    "D_lng": 114.0201,
                }
            ]
        )
        corrected = pd.DataFrame(
            [
                {
                    "vehicle_id": "22223",
                    "time": "2023-10-12 08:00:00",
                    "matched_lat": 22.0000,
                    "matched_lon": 114.0000,
                    "matched_node": 1,
                    "edge_u": 1,
                    "edge_v": 2,
                    "edge_key": 0,
                },
                {
                    "vehicle_id": "22223",
                    "time": "2023-10-12 08:10:00",
                    "matched_lat": 22.0200,
                    "matched_lon": 114.0200,
                    "matched_node": 4,
                    "edge_u": 3,
                    "edge_v": 4,
                    "edge_key": 0,
                },
            ]
        )

        completed, meta = complete_od_endpoints(od_df, corrected, self.build_graph(), time_tolerance="5s")

        row = completed.iloc[0]
        self.assertEqual(meta["corrected_matches"], 2)
        self.assertEqual(row["O_endpoint_source"], "corrected_track")
        self.assertEqual(row["D_endpoint_source"], "corrected_track")
        self.assertEqual(row["O_matched_node"], 1)
        self.assertEqual(row["D_matched_node"], 4)
        self.assertEqual(row["O_edge_u"], 1)
        self.assertEqual(row["D_edge_v"], 4)
        self.assertAlmostEqual(row["O_corrected_lat"], 22.0000)
        self.assertAlmostEqual(row["D_corrected_lng"], 114.0200)

    def test_complete_od_endpoints_falls_back_to_nearest_node(self):
        from route_planning import complete_od_endpoints

        od_df = pd.DataFrame(
            [
                {
                    "O_TAXI_ID": "22223",
                    "O_time": "2023-10-12 08:00:00",
                    "O_lat": 22.0002,
                    "O_lng": 114.0002,
                    "D_time": "2023-10-12 08:10:00",
                    "D_lat": 22.0199,
                    "D_lng": 114.0199,
                }
            ]
        )

        completed, meta = complete_od_endpoints(od_df, pd.DataFrame(), self.build_graph(), time_tolerance="1s")

        row = completed.iloc[0]
        self.assertEqual(meta["nearest_fallbacks"], 2)
        self.assertEqual(row["O_endpoint_source"], "nearest_node")
        self.assertEqual(row["D_endpoint_source"], "nearest_node")
        self.assertEqual(row["O_matched_node"], 1)
        self.assertEqual(row["D_matched_node"], 4)
        self.assertGreater(row["O_snap_distance_m"], 0)
        self.assertGreater(row["D_snap_distance_m"], 0)

    def test_prepare_graph_route_costs_assigns_positive_costs_to_all_edges(self):
        from route_planning import prepare_graph_route_costs

        graph = self.build_graph()
        weak_stats = pd.DataFrame(
            [
                {"edge_u": 1, "edge_v": 2, "edge_key": 0, "avg_speed": 8.0, "sample_count": 1},
                {"edge_u": 1, "edge_v": 3, "edge_key": 0, "avg_speed": 80.0, "sample_count": 3},
            ]
        )

        meta = prepare_graph_route_costs(graph, weak_stats, highway_median_speed={"motorway": 80.0}, min_samples=3)

        self.assertEqual(meta["updated_edges"], 4)
        for _, _, _, data in graph.edges(keys=True, data=True):
            self.assertIn("route_cost", data)
            self.assertGreater(data["route_cost"], 0)
            self.assertGreater(data["baseline_speed_kph"], 0)

    def test_plan_dual_routes_between_points_returns_snapped_metadata(self):
        from route_planning import plan_dual_routes_between_points, prepare_graph_route_costs

        graph = self.build_graph()
        speed_stats = pd.DataFrame(
            [
                {"edge_u": 1, "edge_v": 2, "edge_key": 0, "avg_speed": 10.0, "sample_count": 3},
                {"edge_u": 2, "edge_v": 4, "edge_key": 0, "avg_speed": 10.0, "sample_count": 3},
                {"edge_u": 1, "edge_v": 3, "edge_key": 0, "avg_speed": 80.0, "sample_count": 3},
                {"edge_u": 3, "edge_v": 4, "edge_key": 0, "avg_speed": 80.0, "sample_count": 3},
            ]
        )
        prepare_graph_route_costs(graph, speed_stats, min_samples=3)

        result = plan_dual_routes_between_points(graph, 22.0001, 114.0001, 22.0201, 114.0201)

        self.assertTrue(result["success"])
        self.assertEqual(result["origin"]["node"], 1)
        self.assertEqual(result["destination"]["node"], 4)
        self.assertEqual(result["shortest"]["nodes"], [1, 2, 4])
        self.assertEqual(result["fastest"]["nodes"], [1, 3, 4])
        self.assertIn("snap_distance_m", result["origin"])
        self.assertIn("snap_distance_m", result["destination"])

    def test_write_dataframe_cache_prefers_parquet_but_falls_back_to_csv(self):
        from route_planning import read_dataframe_cache, write_dataframe_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = os.path.join(temp_dir, "sample.parquet")
            written_path = write_dataframe_cache(pd.DataFrame({"value": [1, 2]}), base_path)
            loaded = read_dataframe_cache(written_path)

        self.assertIn(os.path.splitext(written_path)[1], [".parquet", ".csv"])
        self.assertEqual(loaded["value"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
