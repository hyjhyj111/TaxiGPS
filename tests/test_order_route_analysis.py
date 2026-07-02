import os
import pickle
import sys
import tempfile
import unittest

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class OrderRouteAnalysisTest(unittest.TestCase):
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

    def order_row(self):
        return pd.Series(
            {
                "O_TAXI_ID": "22223",
                "O_time": "2023-10-12 08:00:00",
                "D_time": "2023-10-12 08:10:00",
                "O_lat": 22.0000,
                "O_lng": 114.0000,
                "D_lat": 22.0200,
                "D_lng": 114.0200,
                "O_corrected_lat": 22.0000,
                "O_corrected_lng": 114.0000,
                "D_corrected_lat": 22.0200,
                "D_corrected_lng": 114.0200,
            }
        )

    def corrected_rows(self):
        return pd.DataFrame(
            [
                {"vehicle_id": "22223", "time": "2023-10-12 08:00:00", "matched_lat": 22.000, "matched_lon": 114.000, "matched_node": 1, "edge_u": 1, "edge_v": 2, "edge_key": 0, "sequence": 0},
                {"vehicle_id": "22223", "time": "2023-10-12 08:03:00", "matched_lat": 22.000, "matched_lon": 114.010, "matched_node": 2, "edge_u": 1, "edge_v": 2, "edge_key": 0, "sequence": 1},
                {"vehicle_id": "22223", "time": "2023-10-12 08:10:00", "matched_lat": 22.020, "matched_lon": 114.020, "matched_node": 4, "edge_u": 2, "edge_v": 4, "edge_key": 0, "sequence": 2},
            ]
        )

    def order_row_for(self, vehicle_id, minute_offset):
        row = self.order_row().copy()
        start = pd.Timestamp("2023-10-12 08:00:00") + pd.Timedelta(minutes=minute_offset)
        row["O_TAXI_ID"] = str(vehicle_id)
        row["O_time"] = start
        row["D_time"] = start + pd.Timedelta(minutes=10)
        return row

    def corrected_rows_for_orders(self, orders):
        rows = []
        for _, order in pd.DataFrame(orders).iterrows():
            vehicle_id = str(order["O_TAXI_ID"])
            start = pd.Timestamp(order["O_time"])
            rows.extend(
                [
                    {"vehicle_id": vehicle_id, "time": start, "matched_lat": 22.000, "matched_lon": 114.000, "matched_node": 1, "edge_u": 1, "edge_v": 2, "edge_key": 0, "sequence": 0},
                    {"vehicle_id": vehicle_id, "time": start + pd.Timedelta(minutes=3), "matched_lat": 22.000, "matched_lon": 114.010, "matched_node": 2, "edge_u": 1, "edge_v": 2, "edge_key": 0, "sequence": 1},
                    {"vehicle_id": vehicle_id, "time": start + pd.Timedelta(minutes=10), "matched_lat": 22.020, "matched_lon": 114.020, "matched_node": 4, "edge_u": 2, "edge_v": 4, "edge_key": 0, "sequence": 2},
                ]
            )
        return pd.DataFrame(rows)

    def test_load_completed_od_cache_strictly_requires_completed_cache(self):
        from order_route_analysis import load_completed_od_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = os.path.join(temp_dir, "od_endpoints_completed.parquet")
            df, meta = load_completed_od_cache(missing_path)

        self.assertEqual(len(df), 0)
        self.assertFalse(meta["success"])
        self.assertEqual(meta["source"], "completed_od_cache")
        self.assertIn("未找到校正OD缓存", meta["error"])

    def test_load_completed_od_cache_reads_csv_fallback_without_raw_od(self):
        from order_route_analysis import load_completed_od_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = os.path.join(temp_dir, "od_endpoints_completed.parquet")
            csv_path = os.path.join(temp_dir, "od_endpoints_completed.csv")
            pd.DataFrame([self.order_row()]).to_csv(csv_path, index=False)

            df, meta = load_completed_od_cache(parquet_path)

        self.assertTrue(meta["success"])
        self.assertEqual(meta["cache_path"], csv_path)
        self.assertEqual(len(df), 1)
        self.assertIn("O_corrected_lat", df.columns)

    def test_slice_actual_route_from_corrected_cache_uses_existing_rows(self):
        from order_route_analysis import slice_actual_route_from_corrected_rows

        actual = slice_actual_route_from_corrected_rows(self.corrected_rows(), self.order_row())

        self.assertTrue(actual["success"])
        self.assertEqual(actual["source"], "corrected_track_cache")
        self.assertEqual(len(actual["points"]), 3)
        self.assertEqual(actual["duration_s"], 600.0)
        self.assertGreater(actual["distance_m"], 0)

    def test_analyze_single_order_compares_actual_shortest_and_fastest(self):
        from order_route_analysis import analyze_single_order_route
        from route_planning import prepare_graph_route_costs

        graph = self.build_graph()
        speed_stats = pd.DataFrame(
            [
                {"edge_u": 1, "edge_v": 2, "edge_key": 0, "avg_speed": 10.0, "sample_count": 3},
                {"edge_u": 2, "edge_v": 4, "edge_key": 0, "avg_speed": 10.0, "sample_count": 3},
                {"edge_u": 1, "edge_v": 3, "edge_key": 0, "avg_speed": 80.0, "sample_count": 3},
                {"edge_u": 3, "edge_v": 4, "edge_key": 0, "avg_speed": 80.0, "sample_count": 3},
            ]
        )
        prepare_graph_route_costs(graph, speed_stats)

        result = analyze_single_order_route(self.order_row(), self.corrected_rows(), graph)

        self.assertTrue(result["success"])
        self.assertEqual(result["route_result"]["shortest"]["nodes"], [1, 2, 4])
        self.assertEqual(result["route_result"]["fastest"]["nodes"], [1, 3, 4])
        self.assertAlmostEqual(result["metrics"]["actual_duration_s"], 600.0)
        self.assertGreater(result["metrics"]["actual_distance_m"], 0)
        self.assertGreaterEqual(result["metrics"]["detour_ratio"], 0)
        self.assertEqual(result["metrics"]["fastest_overlap_rate"], 0.0)

    def test_build_completed_od_cache_uses_existing_od_and_corrected_caches(self):
        from order_route_analysis import build_completed_od_cache_from_existing_caches

        with tempfile.TemporaryDirectory() as temp_dir:
            source_od_path = os.path.join(temp_dir, "od_cache.pkl")
            corrected_dir = os.path.join(temp_dir, "road_corrected")
            output_path = os.path.join(temp_dir, "od_endpoints_completed.csv")
            os.makedirs(corrected_dir, exist_ok=True)

            od_df = pd.DataFrame([self.order_row().drop(labels=["O_corrected_lat", "O_corrected_lng", "D_corrected_lat", "D_corrected_lng"])])
            with open(source_od_path, "wb") as f:
                pickle.dump({"data": od_df}, f)
            self.corrected_rows().to_csv(os.path.join(corrected_dir, "22223_token.csv"), index=False)

            completed, meta = build_completed_od_cache_from_existing_caches(
                self.build_graph(),
                source_od_cache_path=source_od_path,
                corrected_cache_dir=corrected_dir,
                output_path=output_path,
                vehicle_ids=["22223"],
                start_time="2023-10-12 08:00:00",
                end_time="2023-10-12 08:30:00",
            )

        self.assertTrue(meta["success"])
        self.assertEqual(meta["source"], "existing_od_cache")
        self.assertTrue(meta["cache_path"].endswith(".csv"))
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed.iloc[0]["O_endpoint_source"], "corrected_track")

    def test_select_validation_orders_requires_five_to_ten_complete_orders(self):
        from order_route_analysis import select_validation_orders

        too_few = pd.DataFrame([self.order_row_for(f"2222{idx}", idx * 15) for idx in range(4)])
        enough = pd.DataFrame([self.order_row_for(f"2223{idx}", idx * 15) for idx in range(12)])

        self.assertEqual(len(select_validation_orders(too_few)), 0)
        self.assertEqual(len(select_validation_orders(enough)), 10)

    def test_summarize_od_vehicle_candidates_returns_order_counts(self):
        from order_route_analysis import summarize_od_vehicle_candidates

        orders = pd.DataFrame(
            [
                self.order_row_for(22223.0, 0),
                self.order_row_for("22223", 15),
                self.order_row_for(22224.0, 30),
            ]
        )

        summary = summarize_od_vehicle_candidates(orders)

        self.assertEqual(summary.iloc[0]["vehicle_id"], "22223")
        self.assertEqual(summary.iloc[0]["order_count"], 2)
        self.assertEqual(summary.iloc[1]["vehicle_id"], "22224")
        self.assertEqual(summary.iloc[1]["order_count"], 1)
        self.assertNotIn(".", "".join(summary["vehicle_id"].tolist()))

    def test_ensure_completed_od_cache_builds_when_missing(self):
        from order_route_analysis import ensure_completed_od_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            source_od_path = os.path.join(temp_dir, "od_cache.pkl")
            corrected_dir = os.path.join(temp_dir, "road_corrected")
            output_path = os.path.join(temp_dir, "od_endpoints_completed.csv")
            os.makedirs(corrected_dir, exist_ok=True)

            od_df = pd.DataFrame([self.order_row().drop(labels=["O_corrected_lat", "O_corrected_lng", "D_corrected_lat", "D_corrected_lng"])])
            with open(source_od_path, "wb") as f:
                pickle.dump({"data": od_df}, f)
            self.corrected_rows().to_csv(os.path.join(corrected_dir, "22223_token.csv"), index=False)

            completed, meta = ensure_completed_od_cache(
                self.build_graph(),
                output_path=output_path,
                source_od_cache_path=source_od_path,
                corrected_cache_dir=corrected_dir,
            )

        self.assertTrue(meta["success"])
        self.assertTrue(meta["rebuilt"])
        self.assertEqual(len(completed), 1)

    def test_analyze_order_metrics_dict_returns_flat_batch_record(self):
        from order_route_analysis import analyze_order_metrics_dict

        result = analyze_order_metrics_dict(
            self.order_row(),
            self.corrected_rows(),
            self.build_graph(),
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["order_key"], "22223|2023-10-12T08:00:00|2023-10-12T08:10:00")
        self.assertIn("actual_distance_m", result)
        self.assertIn("detour_ratio", result)
        self.assertIn("fastest_overlap_rate", result)
        self.assertNotIn("rows", result)

    def test_batch_analyze_orders_persists_batches_and_resumes(self):
        from order_route_analysis import batch_analyze_orders, order_key_from_row

        orders = pd.DataFrame(
            [
                self.order_row_for("22223", 0),
                self.order_row_for("22224", 15),
                self.order_row_for("22225", 30),
            ]
        )
        corrected = self.corrected_rows_for_orders(orders)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "batch_results.csv")
            existing = pd.DataFrame(
                [
                    {
                        "order_key": order_key_from_row(orders.iloc[0]),
                        "success": True,
                        "vehicle_id": "22223",
                        "actual_distance_m": 1.0,
                        "detour_ratio": 0.0,
                        "fastest_overlap_rate": 0.0,
                    }
                ]
            )
            existing.to_csv(output_path, index=False)

            results, meta = batch_analyze_orders(
                orders,
                corrected,
                self.build_graph(),
                batch_size=1,
                output_path=output_path,
                resume=True,
            )

            saved = pd.read_csv(output_path)

        self.assertTrue(meta["success"])
        self.assertEqual(meta["skipped_existing"], 1)
        self.assertEqual(meta["processed"], 2)
        self.assertEqual(len(results), 3)
        self.assertEqual(len(saved), 3)
        self.assertEqual(set(saved["order_key"]), {order_key_from_row(row) for _, row in orders.iterrows()})

    def test_summarize_batch_results_builds_distance_detour_and_overlap_bins(self):
        from order_route_analysis import summarize_batch_results

        results = pd.DataFrame(
            [
                {"success": True, "actual_distance_m": 1000.0, "detour_ratio": -0.02, "fastest_overlap_rate": 0.10},
                {"success": True, "actual_distance_m": 2000.0, "detour_ratio": 0.15, "fastest_overlap_rate": 0.55},
                {"success": True, "actual_distance_m": 3000.0, "detour_ratio": 0.45, "fastest_overlap_rate": 0.95},
                {"success": False, "actual_distance_m": 0.0, "detour_ratio": 0.0, "fastest_overlap_rate": 0.0},
            ]
        )

        summary = summarize_batch_results(results, detour_threshold=0.3)

        self.assertEqual(summary["total_orders"], 4)
        self.assertEqual(summary["success_orders"], 3)
        self.assertAlmostEqual(summary["distance_stats"]["mean_km"], 2.0)
        self.assertEqual(summary["exception_count"], 1)
        self.assertGreaterEqual(len(summary["detour_distribution"]), 3)
        self.assertGreaterEqual(len(summary["overlap_distribution"]), 3)

    def test_update_order_review_mark_upserts_manual_review(self):
        from order_route_analysis import update_order_review_mark

        with tempfile.TemporaryDirectory() as temp_dir:
            review_path = os.path.join(temp_dir, "reviews.csv")
            first = update_order_review_mark(
                review_path,
                order_key="order-1",
                review_status="需复核",
                suggestion="检查轨迹漂移",
                reviewer="tester",
                reviewed_at="2026-07-02T10:00:00",
            )
            second = update_order_review_mark(
                review_path,
                order_key="order-1",
                review_status="已确认异常",
                suggestion="疑似绕行",
                reviewer="tester",
                reviewed_at="2026-07-02T10:05:00",
            )

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(second.iloc[0]["review_status"], "已确认异常")
        self.assertEqual(second.iloc[0]["suggestion"], "疑似绕行")


if __name__ == "__main__":
    unittest.main()
