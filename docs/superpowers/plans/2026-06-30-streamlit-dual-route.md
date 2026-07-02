# Streamlit Dual Route Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Streamlit-integrated OD endpoint completion, baseline route-cost cache, and click-to-dual-route display.

**Architecture:** Add a focused `src/route_planning.py` service module that reuses existing road-network helpers from `src/map_plotter.py`. Keep UI changes in `src/streamlit_app.py` and leave the current Streamlit navigation intact.

**Tech Stack:** Python, pandas, networkx, osmnx when installed, Streamlit, Folium/Leaflet.

---

## File Structure

- Create `src/route_planning.py`: OD endpoint completion, speed cache read/write, route-cost preparation, coordinate snapping summaries, route planning helpers.
- Modify `src/map_plotter.py`: keep existing route rendering compatible with route results containing endpoint completion metadata.
- Modify `src/streamlit_app.py`: use completed OD endpoint cache in "路线规划" and surface route/cache summaries.
- Add `tests/test_route_planning_service.py`: focused unit tests for new service behavior.
- Update `README.md`: document the Streamlit integrated workflow and cache outputs.

## Task 1: OD Endpoint Completion Service

**Files:**
- Create: `src/route_planning.py`
- Test: `tests/test_route_planning_service.py`

- [ ] Write tests for completion from corrected trajectory rows and fallback nearest-node snapping.
- [ ] Run `python -m unittest tests.test_route_planning_service.RoutePlanningServiceTest.test_complete_od_endpoints_prefers_corrected_rows -v`; expect failure because the module does not exist.
- [ ] Implement `complete_od_endpoints`, `write_dataframe_cache`, `read_dataframe_cache`, and distance helpers.
- [ ] Run the new OD completion tests; expect pass.

## Task 2: Baseline Speed Cache and Route Cost Preparation

**Files:**
- Modify: `src/route_planning.py`
- Test: `tests/test_route_planning_service.py`

- [ ] Write tests proving every edge receives positive `route_cost` and weak samples fall back to highway/default speeds.
- [ ] Run the targeted tests; expect failure before implementation.
- [ ] Implement `load_or_build_speed_cache` and `prepare_graph_route_costs`, delegating aggregation and edge mutation to existing `map_plotter` helpers.
- [ ] Run the targeted tests; expect pass.

## Task 3: Route Planning Result Shape

**Files:**
- Modify: `src/route_planning.py`
- Test: `tests/test_route_planning_service.py`

- [ ] Write tests for `plan_dual_routes_between_points` returning snapped endpoint metadata, two route summaries, and clear errors for impossible paths.
- [ ] Run the targeted tests; expect failure before implementation.
- [ ] Implement `snap_point_to_graph` and `plan_dual_routes_between_points`.
- [ ] Run the targeted tests; expect pass.

## Task 4: Streamlit Route Page Integration

**Files:**
- Modify: `src/streamlit_app.py`
- Test: `tests/test_baseline_route_planning.py`

- [ ] Add tests or extend existing route tests for choosing completed OD endpoint fields over raw fields.
- [ ] Run the targeted tests; expect failure before implementation.
- [ ] Import the new service helpers and update `render_baseline_route_view` to load completed OD endpoints, use corrected coordinates when available, and display endpoint source/snap summaries.
- [ ] Run route planning tests; expect pass.

## Task 5: Documentation and Verification

**Files:**
- Modify: `README.md`

- [ ] Update route-planning documentation with cache filenames and Streamlit click workflow.
- [ ] Run `python -m unittest tests.test_route_planning_service tests.test_baseline_route_planning -v`.
- [ ] Run `python -m unittest discover tests -v` if the focused tests pass.
- [ ] Start Streamlit with `python -m streamlit run src/streamlit_app.py` when dependencies are available and report the local URL.
