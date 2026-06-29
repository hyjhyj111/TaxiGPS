import os
import sys
import unittest

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from map_plotter import apply_baseline_route_cost, build_edge_baseline_speed_cache, plan_baseline_routes


class BaselineRoutePlanningTest(unittest.TestCase):
    def build_graph(self):
        import networkx as nx

        graph = nx.MultiDiGraph()
        graph.add_node(1, x=114.000, y=22.000)
        graph.add_node(2, x=114.010, y=22.000)
        graph.add_node(3, x=114.000, y=22.020)
        graph.add_node(4, x=114.020, y=22.020)
        graph.add_edge(1, 2, key=0, length=1000.0, highway="residential")
        graph.add_edge(2, 4, key=0, length=1000.0, highway="residential")
        graph.add_edge(1, 3, key=0, length=2000.0, highway="motorway")
        graph.add_edge(3, 4, key=0, length=2000.0, highway="motorway")
        return graph

    def test_static_route_cost_covers_all_edges_and_uses_observed_speed(self):
        graph = self.build_graph()
        matched_track = pd.DataFrame(
            [
                {"id": "slow", "edge_u": 1, "edge_v": 2, "edge_key": 0, "speed": 10},
                {"id": "slow", "edge_u": 1, "edge_v": 2, "edge_key": 0, "speed": 12},
                {"id": "slow", "edge_u": 2, "edge_v": 4, "edge_key": 0, "speed": 10},
                {"id": "slow", "edge_u": 2, "edge_v": 4, "edge_key": 0, "speed": 12},
                {"id": "fast", "edge_u": 1, "edge_v": 3, "edge_key": 0, "speed": 80},
                {"id": "fast", "edge_u": 1, "edge_v": 3, "edge_key": 0, "speed": 82},
                {"id": "fast", "edge_u": 3, "edge_v": 4, "edge_key": 0, "speed": 80},
                {"id": "fast", "edge_u": 3, "edge_v": 4, "edge_key": 0, "speed": 82},
            ]
        )

        speed_cache, meta = build_edge_baseline_speed_cache(matched_track, graph, min_samples=2)
        apply_baseline_route_cost(graph, speed_cache, meta["highway_median_speed"])

        for _, _, _, data in graph.edges(keys=True, data=True):
            self.assertIn("baseline_speed_kph", data)
            self.assertIn("route_cost", data)
            self.assertGreater(data["baseline_speed_kph"], 0)
            self.assertGreater(data["route_cost"], 0)

    def test_shortest_distance_and_baseline_fastest_use_different_weights(self):
        graph = self.build_graph()
        matched_track = pd.DataFrame(
            [
                {"id": "slow", "edge_u": 1, "edge_v": 2, "edge_key": 0, "speed": 10},
                {"id": "slow", "edge_u": 1, "edge_v": 2, "edge_key": 0, "speed": 10},
                {"id": "slow", "edge_u": 2, "edge_v": 4, "edge_key": 0, "speed": 10},
                {"id": "slow", "edge_u": 2, "edge_v": 4, "edge_key": 0, "speed": 10},
                {"id": "fast", "edge_u": 1, "edge_v": 3, "edge_key": 0, "speed": 80},
                {"id": "fast", "edge_u": 1, "edge_v": 3, "edge_key": 0, "speed": 80},
                {"id": "fast", "edge_u": 3, "edge_v": 4, "edge_key": 0, "speed": 80},
                {"id": "fast", "edge_u": 3, "edge_v": 4, "edge_key": 0, "speed": 80},
            ]
        )
        speed_cache, meta = build_edge_baseline_speed_cache(matched_track, graph, min_samples=2)
        apply_baseline_route_cost(graph, speed_cache, meta["highway_median_speed"])

        result = plan_baseline_routes(graph, 1, 4)

        self.assertTrue(result["success"])
        self.assertEqual(result["shortest"]["nodes"], [1, 2, 4])
        self.assertEqual(result["fastest"]["nodes"], [1, 3, 4])
        self.assertLess(result["shortest"]["distance_m"], result["fastest"]["distance_m"])
        self.assertLess(result["fastest"]["route_cost_s"], result["shortest"]["route_cost_s"])


if __name__ == "__main__":
    unittest.main()
