# Taxi GPS 数据分析与可视化实训项目

本项目根据 `roadmap/`、`实训V_项目介绍.docx` 和 `数据简单处理.docx` 整合实现出租车 GPS 数据处理主线：

数据清洗 -> OD 提取 -> 车辆缓存/分钟缓存 -> 轨迹查询 -> 统计分析/热力点 -> 地图展示 -> 路网匹配与 ETA 示例。

## 运行环境

项目使用根目录 `.venv` 作为 Python 环境。首次使用时先补齐 pip 和依赖：

```bash
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r requirements.txt
```

如果导入 Matplotlib 时提示缓存目录不可写，可使用项目内缓存目录：

```bash
mkdir -p .cache/matplotlib
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python run_project.py pipeline
```

默认只处理前 300000 行，便于演示和快速验收。全量处理：

```bash
.venv/bin/python run_project.py pipeline --full
```

## 常用命令

运行完整流程：

```bash
.venv/bin/python run_project.py pipeline --limit-rows 300000
```

查询某辆车轨迹缓存：

```bash
.venv/bin/python run_project.py query-vehicle 22223 --start "2013-10-22 08:00:00" --end "2013-10-22 09:00:00"
```

查询某一分钟车辆位置：

```bash
.venv/bin/python run_project.py query-minute "2013-10-22 08:30:00"
```

生成前端页面：

```bash
.venv/bin/python run_project.py webapp
```

启动前端服务：

```bash
.venv/bin/python run_project.py webapp --serve --port 8765
```

打开地址：

```text
http://127.0.0.1:8765/
```

## 输出目录

运行后生成：

- `outputs/cleaned/taxi_clean.csv`：清洗后的 GPS 明细
- `outputs/od_orders.csv`：OD 订单表
- `outputs/cache/vehicles/*.csv`：车辆轨迹缓存
- `outputs/cache/minute_positions.csv`：分钟位置缓存
- `outputs/cache/od_cache.csv`：OD 缓存
- `outputs/stats/hourly_orders.csv`：小时订单统计
- `outputs/stats/trip_distance_categories.csv`：短/中/长途统计
- `outputs/stats/vehicle_operation_stats.csv`：车辆里程、载客率统计
- `outputs/stats/pickup_hotspots.csv`：上车点热点，优先使用 DBSCAN，失败时回退网格聚合
- `outputs/stats/eta_sample.json`：路网最近节点 + 最短路 ETA 示例
- `outputs/maps/dashboard.html`：可打开的地图与统计仪表盘
- `outputs/web/index.html`：多页面前端应用入口
- `outputs/web/data.json`：前端汇总数据

## 字段口径

原始 CSV 无表头，统一为：

`id,time,lng,lat,status,speed`

其中 `status=1` 表示载客，`status=0` 表示空客。原始时间只有 `HH:MM:SS` 时，默认补充日期 `2013-10-22`。

## 验收说明

项目已在根目录 `.venv` 中验证，包含 `pandas`、`numpy`、`folium`、`networkx`、`scikit-learn`、`geopy`、`matplotlib` 等依赖。路网 ETA 当前使用标准库解析 GraphML，执行最近节点匹配与最短路径估计；上车热点优先使用 `scikit-learn` DBSCAN。
