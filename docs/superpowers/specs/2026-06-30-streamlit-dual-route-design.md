# Streamlit Dual Route Planning Design

## Goal

Build the requested OD endpoint completion, baseline road speed cache, and dual-route display as an integrated Streamlit feature in the existing TaxiGPS application.

## Scope

The feature stays inside the existing Streamlit app. It does not add a separate Flask or FastAPI service in this phase. The route computation remains reusable Python service logic so a future HTTP API can wrap it without changing the routing algorithm.

## Architecture

The implementation uses three layers:

1. Data preparation utilities generate reusable cache files:
   - `cache/od_endpoints_completed.parquet` when parquet support is available, otherwise `cache/od_endpoints_completed.csv`.
   - `cache/edge_baseline_speed.parquet` when parquet support is available, otherwise the existing `cache/edge_baseline_speed.csv`.
2. Routing service functions load the same directed Shenzhen road network, ensure every `(u, v, key)` edge has positive `route_cost`, snap input coordinates to road nodes, and compute both `length` and `route_cost` shortest paths.
3. The Streamlit "路线规划" page lets users choose manual coordinates, click two points on a Leaflet map, or select historical OD endpoints. It draws the shortest route in blue and the baseline fastest route in green.

## Data Rules

OD completion preserves original coordinates and adds corrected endpoint fields. For each pickup/dropoff endpoint it first searches corrected trajectory rows for the same vehicle near the endpoint time. If that fails, it snaps the raw endpoint coordinate to the nearest road node. Both paths record matched node, edge triplet when available, corrected coordinate, snap distance, source method, and a message.

Baseline speed generation aggregates valid corrected speeds in the range `1..120 km/h` by `(edge_u, edge_v, edge_key)`. Edges with fewer than `min_samples=3` samples use the same-highway median speed, then the default highway speed table, then a global fallback speed. Every directed edge receives a positive `route_cost` in seconds.

## UI Behavior

In manual mode, the page shows coordinate inputs and a route map. Consecutive clicks fill origin then destination and immediately trigger a rerun using the selected points. In historical OD mode, the selector reads completed endpoint fields when present and falls back to original OD coordinates when needed.

The route summary shows snapped nodes, distance, static cost, edge count, speed cache counts, and OD endpoint completion details. If a click is outside the network or no directed path exists, the page shows a clear error and keeps the app running.

## Testing

Tests cover OD endpoint completion by corrected rows and fallback snapping, speed cache fallback and positive route cost, route summary geometry expansion, and Streamlit helper selection of completed OD coordinates. Existing route tests remain valid.
