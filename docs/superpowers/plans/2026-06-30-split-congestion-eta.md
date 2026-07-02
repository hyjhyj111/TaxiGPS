# Split Congestion And ETA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the combined "拥堵与ETA" Streamlit page into separate "道路拥堵" and "ETA预测" navigation modules, with data summaries rendered below each map.

**Architecture:** Reuse existing cached computation functions and map rendering helpers. Split only `src/streamlit_app.py` UI functions and update navigation tests/documentation.

**Tech Stack:** Python, Streamlit, unittest, Folium-generated HTML maps.

---

## File Structure

- Modify `tests/test_streamlit_navigation.py`: require two separate route modules and verify map rendering happens before data summaries in both functions.
- Modify `src/streamlit_app.py`: replace `render_congestion_eta_view` with `render_congestion_view` and `render_eta_view`; update `VIEW_OPTIONS`, `NAVIGATION_GROUPS`, `get_view_context`, and main dispatch.
- Modify `README.md`: rename combined user instructions into separate congestion and ETA entries.

## Task 1: Navigation Split

- [ ] Add tests expecting `道路拥堵` and `ETA预测` in `VIEW_OPTIONS`, and no `拥堵与ETA`.
- [ ] Run `python -m unittest tests.test_streamlit_navigation -v`; expect failure.
- [ ] Update navigation constants and context descriptions.
- [ ] Run `python -m unittest tests.test_streamlit_navigation -v`; expect pass.

## Task 2: UI Rendering Split

- [ ] Add tests inspecting source order: `render_html_map` appears before metrics/dataframe in both new render functions.
- [ ] Run `python -m unittest tests.test_streamlit_navigation -v`; expect failure.
- [ ] Split the combined render function into two functions and dispatch them separately.
- [ ] Run `python -m unittest tests.test_streamlit_navigation -v`; expect pass.

## Task 3: Documentation And Verification

- [ ] Update README feature list and usage steps.
- [ ] Run `python -m unittest discover tests -v`.
- [ ] Verify Streamlit can still run with `python -m streamlit run src/streamlit_app.py --server.port 8501 --server.headless true`.
