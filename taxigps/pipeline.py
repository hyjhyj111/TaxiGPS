"""Data cleaning, OD extraction, caches, and statistics."""

from __future__ import annotations

import json
import io
import os
import warnings
from contextlib import redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .geo import haversine_km, in_shenzhen_bounds


COLUMNS = ["id", "time", "lng", "lat", "status", "speed"]


@dataclass(frozen=True)
class PipelineConfig:
    input_csv: Path
    output_dir: Path
    date: str = "2013-10-22"
    limit_rows: int | None = 500_000
    max_vehicle_cache: int = 20
    heatmap_freq: str = "15min"


def read_raw(config: PipelineConfig) -> pd.DataFrame:
    kwargs: dict[str, object] = {
        "header": None,
        "names": COLUMNS,
        "usecols": range(6),
    }
    if config.limit_rows:
        kwargs["nrows"] = config.limit_rows
    return pd.read_csv(config.input_csv, **kwargs)


def normalize_time(values: pd.Series, date: str) -> pd.Series:
    text = values.astype(str).str.strip()
    has_date = text.str.contains(r"\d{4}-\d{1,2}-\d{1,2}", regex=True, na=False)
    text = text.where(has_date, date + " " + text)
    return pd.to_datetime(text, errors="coerce")


def _choose_duplicate(group: pd.DataFrame) -> pd.Series:
    status_mode = group["status"].mode(dropna=True)
    if not status_mode.empty:
        candidates = group[group["status"] == status_mode.iloc[0]]
        return candidates.iloc[0]
    return group.iloc[0]


def clean_data(raw: pd.DataFrame, date: str) -> tuple[pd.DataFrame, dict[str, int]]:
    df = raw.copy()
    before = len(df)
    df["time"] = normalize_time(df["time"], date)
    for col in ["id", "lng", "lat", "status", "speed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=COLUMNS)
    df["id"] = df["id"].astype("int64")
    df["status"] = df["status"].astype("int8")
    valid = (
        df["status"].isin([0, 1])
        & df["speed"].between(0, 140)
        & df.apply(lambda row: in_shenzhen_bounds(float(row["lng"]), float(row["lat"])), axis=1)
    )
    df = df.loc[valid].copy()

    df = df.sort_values(["id", "time"]).reset_index(drop=True)
    duplicate_rows = int(df.duplicated(["id", "time"], keep=False).sum())
    if duplicate_rows:
        df = (
            df.groupby(["id", "time"], as_index=False, group_keys=False)
            .apply(_choose_duplicate, include_groups=False)
            .reset_index(drop=True)
        )

    df = df.sort_values(["id", "time"]).reset_index(drop=True)
    previous = df.groupby("id")[["status", "time"]].shift(1)
    following = df.groupby("id")[["status", "time"]].shift(-1)
    abnormal = (
        previous["status"].notna()
        & following["status"].notna()
        & (df["status"] != previous["status"])
        & (df["status"] != following["status"])
        & ((following["time"] - previous["time"]).dt.total_seconds() < 60)
    )
    abnormal_rows = int(abnormal.sum())
    df = df.loc[~abnormal].sort_values(["id", "time"]).reset_index(drop=True)

    summary = {
        "raw_rows": int(before),
        "clean_rows": int(len(df)),
        "dropped_rows": int(before - len(df)),
        "duplicate_rows_seen": duplicate_rows,
        "abnormal_status_rows_dropped": abnormal_rows,
        "vehicle_count": int(df["id"].nunique()),
    }
    return df, summary


def add_segment_distance(df: pd.DataFrame) -> pd.DataFrame:
    data = df.sort_values(["id", "time"]).copy()
    prev = data.groupby("id")[["lng", "lat", "time", "status"]].shift(1)
    same_vehicle = prev["lng"].notna()
    dist = np.zeros(len(data), dtype=float)
    rows = data.loc[same_vehicle, ["lng", "lat"]].to_numpy()
    prev_rows = prev.loc[same_vehicle, ["lng", "lat"]].to_numpy()
    dist[same_vehicle.to_numpy()] = [
        haversine_km(p[0], p[1], r[0], r[1]) for p, r in zip(prev_rows, rows)
    ]
    data["segment_km"] = dist
    data["occupied_segment_km"] = np.where(data["status"].eq(1), data["segment_km"], 0.0)
    return data


def extract_od(clean: pd.DataFrame) -> pd.DataFrame:
    data = clean.sort_values(["id", "time"]).copy()
    data["prev_status"] = data.groupby("id")["status"].shift(1)
    changes = data.loc[
        data["prev_status"].notna() & (data["status"] != data["prev_status"])
    ].copy()
    records: list[dict[str, object]] = []
    for taxi_id, group in changes.groupby("id", sort=False):
        rows = group.sort_values("time").to_dict("records")
        i = 0
        while i < len(rows) - 1:
            start = rows[i]
            end = rows[i + 1]
            if start["status"] == 1 and end["status"] == 0:
                duration = (end["time"] - start["time"]).total_seconds()
                distance = haversine_km(start["lng"], start["lat"], end["lng"], end["lat"])
                avg_speed = distance / duration * 3600 if duration > 0 else np.nan
                if 60 <= duration <= 6 * 3600 and 0 < distance <= 100 and avg_speed <= 140:
                    records.append(
                        {
                            "id": int(taxi_id),
                            "start_time": start["time"],
                            "end_time": end["time"],
                            "start_lng": float(start["lng"]),
                            "start_lat": float(start["lat"]),
                            "end_lng": float(end["lng"]),
                            "end_lat": float(end["lat"]),
                            "duration_s": int(duration),
                            "distance_km": round(float(distance), 4),
                            "avg_speed_kmh": round(float(avg_speed), 2),
                        }
                    )
                i += 2
            else:
                i += 1
    return pd.DataFrame.from_records(records)


def build_caches(clean: pd.DataFrame, od: pd.DataFrame, output_dir: Path, max_vehicles: int) -> None:
    cache_dir = output_dir / "cache"
    vehicle_dir = cache_dir / "vehicles"
    vehicle_dir.mkdir(parents=True, exist_ok=True)
    for taxi_id, group in clean.groupby("id", sort=False):
        if len(list(vehicle_dir.glob("*.csv"))) >= max_vehicles:
            break
        group.to_csv(vehicle_dir / f"{taxi_id}.csv", index=False)

    minute = (
        clean.assign(minute=clean["time"].dt.floor("min"))
        .sort_values(["id", "time"])
        .groupby(["minute", "id"], as_index=False)
        .tail(1)
        .sort_values(["minute", "id"])
    )
    minute.to_csv(cache_dir / "minute_positions.csv", index=False)
    od.to_csv(cache_dir / "od_cache.csv", index=False)


def _trip_category(distance_km: float) -> str:
    if distance_km < 4:
        return "short"
    if distance_km <= 8:
        return "middle"
    return "long"


def compute_pickup_hotspots(od: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, str]:
    if od.empty:
        return pd.DataFrame(), "empty"
    try:
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
        warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
        with redirect_stderr(io.StringIO()):
            from sklearn.cluster import DBSCAN

        frames: list[pd.DataFrame] = []
        eps_km = 1.5
        min_samples = 3
        earth_radius_km = 6371.0088
        grouped = od.assign(time_slice=od["start_time"].dt.floor(freq)).groupby("time_slice")
        for time_slice, group in grouped:
            coords = group[["start_lat", "start_lng"]].to_numpy()
            if len(coords) < min_samples:
                continue
            with redirect_stderr(io.StringIO()):
                labels = DBSCAN(
                    eps=eps_km / earth_radius_km,
                    min_samples=min_samples,
                    metric="haversine",
                    n_jobs=1,
                ).fit_predict(np.radians(coords))
            clustered = group.assign(cluster=labels)
            clustered = clustered.loc[clustered["cluster"] >= 0]
            if clustered.empty:
                continue
            centers = (
                clustered.groupby("cluster", as_index=False)
                .agg(
                    lat=("start_lat", "mean"),
                    lng=("start_lng", "mean"),
                    count=("id", "count"),
                )
                .assign(time_slice=time_slice)
            )
            frames.append(centers[["time_slice", "lat", "lng", "count", "cluster"]])
        if frames:
            return pd.concat(frames, ignore_index=True).sort_values(["time_slice", "count"], ascending=[True, False]), "dbscan"
    except Exception:
        pass

    grid = (
        od.assign(
            lng=(od["start_lng"] / 0.005).round() * 0.005,
            lat=(od["start_lat"] / 0.005).round() * 0.005,
            time_slice=od["start_time"].dt.floor(freq),
        )
        .groupby(["time_slice", "lng", "lat"], as_index=False)
        .agg(count=("id", "count"))
        .sort_values(["time_slice", "count"], ascending=[True, False])
    )
    grid["cluster"] = -1
    return grid[["time_slice", "lat", "lng", "count", "cluster"]], "grid_fallback"


def compute_statistics(clean: pd.DataFrame, od: pd.DataFrame, output_dir: Path, freq: str) -> dict[str, object]:
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    clean_dist = add_segment_distance(clean)

    hourly = (
        od.assign(hour=od["start_time"].dt.hour)
        .groupby("hour", as_index=False)
        .agg(order_count=("id", "count"), avg_distance_km=("distance_km", "mean"), avg_duration_s=("duration_s", "mean"))
    )
    hourly.to_csv(stats_dir / "hourly_orders.csv", index=False)

    categories = (
        od.assign(category=od["distance_km"].map(_trip_category))
        .groupby("category", as_index=False)
        .agg(order_count=("id", "count"), avg_distance_km=("distance_km", "mean"))
    )
    categories.to_csv(stats_dir / "trip_distance_categories.csv", index=False)

    fleet = (
        clean_dist.groupby("id", as_index=False)
        .agg(
            gps_points=("id", "count"),
            total_km=("segment_km", "sum"),
            occupied_km=("occupied_segment_km", "sum"),
            occupied_points=("status", "sum"),
        )
    )
    fleet["empty_km"] = fleet["total_km"] - fleet["occupied_km"]
    fleet["occupancy_rate"] = fleet["occupied_points"] / fleet["gps_points"]
    fleet.to_csv(stats_dir / "vehicle_operation_stats.csv", index=False)

    pickup_hotspots, hotspot_method = compute_pickup_hotspots(od, freq)
    pickup_hotspots.to_csv(stats_dir / "pickup_hotspots.csv", index=False)

    summary = {
        "orders": int(len(od)),
        "hourly_rows": int(len(hourly)),
        "fleet_rows": int(len(fleet)),
        "hotspot_rows": int(len(pickup_hotspots)),
        "hotspot_method": hotspot_method,
        "avg_order_distance_km": float(od["distance_km"].mean()) if not od.empty else 0.0,
        "avg_order_duration_min": float(od["duration_s"].mean() / 60) if not od.empty else 0.0,
    }
    with (stats_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def save_core_outputs(clean: pd.DataFrame, od: pd.DataFrame, summary: dict[str, int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cleaned").mkdir(exist_ok=True)
    clean.to_csv(output_dir / "cleaned" / "taxi_clean.csv", index=False)
    od.to_csv(output_dir / "od_orders.csv", index=False)
    with (output_dir / "cleaning_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_raw(config)
    clean, cleaning_summary = clean_data(raw, config.date)
    od = extract_od(clean)
    save_core_outputs(clean, od, cleaning_summary, config.output_dir)
    build_caches(clean, od, config.output_dir, config.max_vehicle_cache)
    stats_summary = compute_statistics(clean, od, config.output_dir, config.heatmap_freq)
    return {"cleaning": cleaning_summary, "stats": stats_summary}


def iter_vehicle_ids(cache_dir: Path) -> Iterable[str]:
    vehicle_dir = cache_dir / "vehicles"
    if vehicle_dir.exists():
        for path in sorted(vehicle_dir.glob("*.csv")):
            yield path.stem
