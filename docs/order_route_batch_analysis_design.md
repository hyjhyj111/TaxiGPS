# 订单路线批量分析系统设计与使用说明

## 目标

在路线规划模块的历史 OD 端点功能中，将单订单三路线对比扩展为批量汇总分析。系统只读取校正 OD 缓存、校正车辆轨迹缓存、道路速度缓存和路网文件，不自动读取原始 OD 大表。

## 数据输入

- 校正 OD 缓存：`cache/od_endpoints_completed.parquet`，缺失时回退读取同名 `.csv`。
- 校正车辆轨迹缓存：`cache/road_corrected/` 下的车辆缓存文件。
- 路网文件：`shenzhen_drive.pkl` 或 `shenzhen_drive.graphml`。
- 道路速度缓存：`cache/edge_baseline_speed.csv`，批量开始时生成或复用。

## 核心模块

### 单订单函数

`order_route_analysis.analyze_order_metrics_dict(od_row, corrected_rows, graph, speed_stats=None, highway_median_speed=None, logger=None)`

输入为一条校正 OD、已加载的校正轨迹缓存、已加载路网和可选速度统计。输出为扁平字典，包含：

- `order_key`
- `vehicle_id`
- `success`
- `error`
- `actual_distance_m`
- `shortest_distance_m`
- `fastest_distance_m`
- `actual_duration_s`
- `detour_ratio`
- `fastest_overlap_rate`
- `actual_edge_count`
- `shortest_edge_count`
- `fastest_edge_count`

### 批量处理函数

`order_route_analysis.batch_analyze_orders(...)`

支持 `batch_size`、`output_path` 和 `resume`。每个批次完成后立即写入结果文件，默认路径为：

`cache/order_route_batch_results.csv`

当 `resume=True` 时，函数会读取已有结果并按 `order_key` 跳过已完成订单，实现断点续处理。

### 统计分析函数

`order_route_analysis.summarize_batch_results(results_df, detour_threshold=0.3)`

输出实际距离统计、实际距离分布、绕行比例分布、路线重合率分布和异常订单数量。Streamlit 页面会将这些结果展示为指标、表格和柱状图。

### 异常复核函数

`order_route_analysis.exception_orders(results_df, detour_threshold=0.3)` 筛选异常订单。

`order_route_analysis.update_order_review_mark(...)` 保存人工复核状态，默认路径为：

`cache/order_route_reviews.csv`

## Streamlit 使用流程

1. 进入“路线规划”。
2. 选择“历史OD端点”。
3. 如果当前车辆没有 OD，使用左侧“历史OD查询”中的“查询有OD车辆”生成候选车辆列表。
4. 在“OD车辆候选”中选择多辆车，可点击“应用OD车辆”仅应用到历史 OD 查询条件。
5. 如果校正 OD 缓存缺失，历史 OD 页面会自动基于已有 OD 缓存和校正轨迹缓存补建一次，不读取原始 OD 大表。
6. 在“单订单分析”页签中验证单条订单三路线显示。
7. 在“批量汇总分析”页签中设置批次大小、处理上限、异常阈值和断点续处理。
8. 点击“开始批量分析”。
9. 在页面下方查看距离分布、绕行比例分布、路线重合率分布和批量结果表。
10. 在“异常订单复核”页签中加载异常订单三路线地图，填写检查结果和处理建议。

## 缓存策略

- 批量任务启动时一次性加载路网。
- 批量任务启动时一次性加载本批次涉及车辆的校正轨迹缓存。
- 批量任务启动时生成或复用道路速度缓存，并将速度统计传入每条订单分析。
- 每个批次完成后持续保存结果，避免异常中断后重复计算。

## 异常处理

- 单订单失败不会终止整批任务。
- 每条失败订单会写入 `success=False` 和 `error`。
- 批量任务可在下次以 `resume=True` 跳过已有 `order_key`。
