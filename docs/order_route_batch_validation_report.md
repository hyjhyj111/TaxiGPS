# 订单路线批量分析验证与性能报告

## 验证范围

本次验证覆盖以下功能：

- 5至10条数据完整订单的验证样本选择。
- 单订单分析函数输出标准化指标字典。
- 批量分批处理、中间结果保存和断点续处理。
- 实际距离、绕行比例和路线重合率的统计分布。
- 异常订单筛选、地图复核入口和人工标记保存。

## 自动化验证结果

已新增并执行以下测试：

- `test_analyze_order_metrics_dict_returns_flat_batch_record`
- `test_batch_analyze_orders_persists_batches_and_resumes`
- `test_summarize_batch_results_builds_distance_detour_and_overlap_bins`
- `test_update_order_review_mark_upserts_manual_review`
- `test_route_planner_exposes_batch_summary_and_exception_review_ui`

验证结论：

- 单订单处理逻辑已封装为可批量调用的扁平字典接口。
- 批量处理会按批次保存中间结果。
- 断点续处理会跳过已有 `order_key`。
- 统计分析可输出距离、绕行比例和重合率分布。
- 人工复核记录支持按 `order_key` 更新。

## 页面验证流程

1. 打开 Streamlit 应用。
2. 进入“路线规划”。
3. 切换到“历史OD端点”。
4. 在“单订单分析”页签选择候选路线并点击“计算路线”。
5. 在“批量汇总分析”页签确认验证样本数量为5至10条。
6. 设置批次大小、处理上限和异常阈值，点击“开始批量分析”。
7. 检查页面指标、柱状图和批量结果表。
8. 在“异常订单复核”页签选择异常订单，点击“加载异常路线”。
9. 检查三路线叠加地图，填写检查结果和处理建议，点击“保存复核结果”。

## 关键输出文件

- 批量分析结果：`cache/order_route_batch_results.csv`
- 人工复核结果：`cache/order_route_reviews.csv`
- 校正 OD 缓存：`cache/od_endpoints_completed.parquet` 或 `.csv`
- 道路速度缓存：`cache/edge_baseline_speed.csv`

## 性能策略

- 路网只在批量任务启动时加载一次。
- 校正轨迹缓存按本批次涉及车辆一次性读取。
- 道路速度缓存按本批次任务生成或复用一次。
- 每个批次结束立即保存结果，降低中断后重算成本。
- 默认处理上限为50条，用户可按机器性能调整。

## 风险与注意事项

- 若校正 OD 缓存缺失，页面会尝试从已有 OD 缓存和校正轨迹缓存补建，仍不会自动读取原始大表。
- 若某订单时间窗内校正轨迹点不足，该订单会标记为失败并保留错误信息。
- 异常订单地图复核会按需重新计算该订单三路线，用于展示完整几何。
