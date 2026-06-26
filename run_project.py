#!/usr/bin/env python3
"""Unified command line entry for the Taxi GPS training project."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import pandas as pd

from taxigps.maps import write_dashboard
from taxigps.network import eta_from_points, write_eta_report
from taxigps.pipeline import PipelineConfig, iter_vehicle_ids, run_pipeline
from taxigps.webapp import generate_frontend, serve_directory


ROOT = Path(__file__).resolve().parent


def cmd_pipeline(args: argparse.Namespace) -> None:
    config = PipelineConfig(
        input_csv=Path(args.input),
        output_dir=Path(args.output),
        date=args.date,
        limit_rows=None if args.full else args.limit_rows,
        max_vehicle_cache=args.max_vehicle_cache,
        heatmap_freq=args.heatmap_freq,
    )
    summary = run_pipeline(config)
    clean = pd.read_csv(config.output_dir / "cleaned" / "taxi_clean.csv", parse_dates=["time"])
    od = pd.read_csv(config.output_dir / "od_orders.csv", parse_dates=["start_time", "end_time"])
    write_dashboard(clean, od, config.output_dir / "maps" / "dashboard.html", ROOT / "深圳市.json")

    eta_report = None
    if not od.empty and (ROOT / "shenzhen_drive.graphml").exists():
        first = od.iloc[0]
        eta_report = eta_from_points(
            ROOT / "shenzhen_drive.graphml",
            (float(first["start_lng"]), float(first["start_lat"])),
            (float(first["end_lng"]), float(first["end_lat"])),
            speed_kmh=max(float(first["avg_speed_kmh"]), 10.0),
        )
        write_eta_report(eta_report, config.output_dir / "stats" / "eta_sample.json")

    print(json.dumps({"pipeline": summary, "eta": eta_report}, ensure_ascii=False, indent=2))


def cmd_query_vehicle(args: argparse.Namespace) -> None:
    vehicle_path = Path(args.output) / "cache" / "vehicles" / f"{args.vehicle_id}.csv"
    if not vehicle_path.exists():
        available = ", ".join(iter_vehicle_ids(Path(args.output) / "cache"))
        raise SystemExit(f"vehicle cache not found: {vehicle_path}. Available: {available}")
    df = pd.read_csv(vehicle_path, parse_dates=["time"])
    if args.start:
        df = df.loc[df["time"] >= pd.Timestamp(args.start)]
    if args.end:
        df = df.loc[df["time"] <= pd.Timestamp(args.end)]
    print(df.head(args.limit).to_csv(index=False))


def cmd_query_minute(args: argparse.Namespace) -> None:
    minute_path = Path(args.output) / "cache" / "minute_positions.csv"
    df = pd.read_csv(minute_path, parse_dates=["minute", "time"])
    target = pd.Timestamp(args.minute)
    df = df.loc[df["minute"].eq(target)]
    if args.vehicle_id:
        df = df.loc[df["id"].astype(str).eq(str(args.vehicle_id))]
    print(df.head(args.limit).to_csv(index=False))


def cmd_webapp(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    web_dir = Path(args.web_dir)
    index = generate_frontend(output_dir, web_dir, ROOT)
    print(f"Frontend generated: {index}")
    if args.serve:
        serve_directory(web_dir, args.host, args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Taxi GPS data analysis and visualization project")
    sub = parser.add_subparsers(dest="command", required=True)

    pipeline = sub.add_parser("pipeline", help="run cleaning, OD, caches, stats, dashboard, ETA sample")
    pipeline.add_argument("--input", default=str(ROOT / "TaxiData.csv"))
    pipeline.add_argument("--output", default=str(ROOT / "outputs"))
    pipeline.add_argument("--date", default="2013-10-22")
    pipeline.add_argument("--limit-rows", type=int, default=300_000, help="fast demo row limit")
    pipeline.add_argument("--full", action="store_true", help="process all rows; requires enough memory/time")
    pipeline.add_argument("--max-vehicle-cache", type=int, default=20)
    pipeline.add_argument("--heatmap-freq", default="15min")
    pipeline.set_defaults(func=cmd_pipeline)

    qv = sub.add_parser("query-vehicle", help="query cached trajectory for one taxi")
    qv.add_argument("vehicle_id")
    qv.add_argument("--output", default=str(ROOT / "outputs"))
    qv.add_argument("--start")
    qv.add_argument("--end")
    qv.add_argument("--limit", type=int, default=20)
    qv.set_defaults(func=cmd_query_vehicle)

    qm = sub.add_parser("query-minute", help="query positions at one minute")
    qm.add_argument("minute", help="example: 2013-10-22 08:30:00")
    qm.add_argument("--vehicle-id")
    qm.add_argument("--output", default=str(ROOT / "outputs"))
    qm.add_argument("--limit", type=int, default=20)
    qm.set_defaults(func=cmd_query_minute)

    webapp = sub.add_parser("webapp", help="generate and optionally serve the frontend application")
    webapp.add_argument("--output", default=str(ROOT / "outputs"))
    webapp.add_argument("--web-dir", default=str(ROOT / "outputs" / "web"))
    webapp.add_argument("--serve", action="store_true")
    webapp.add_argument("--host", default="127.0.0.1")
    webapp.add_argument("--port", type=int, default=8765)
    webapp.set_defaults(func=cmd_webapp)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
