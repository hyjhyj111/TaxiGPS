# 出租车 GPS 轨迹查询系统

出租车 GPS 轨迹查询系统是一个面向真实出租车时空数据的分析与可视化平台，支持轨迹查询、分钟位置查看、OD 点标注、动画轨迹演示，以及热力图与统计分析。

## 功能概览

### 轨迹查询
- 按车辆 ID 和时间范围查询轨迹
- 最多支持 10 辆车并行展示
- 按运营状态自动着色：红色为载客，蓝色为空载
- 优先读取车辆缓存，避免直接扫描原始大表

### 分钟位置
- 查看指定分钟的车辆分布
- 支持按车辆 ID 筛选
- 点击车辆点后可展开该车后续轨迹预览

### OD 点标注
- 展示上车点、下车点及连线关系
- 点位较多时自动聚类，减少拥挤
- 支持抽样展示 OD 路径

### 动画轨迹
- 支持单车和多车轨迹回放
- 支持播放速度调节
- 轨迹按时间顺序平滑播放

### 路网校正与地图选点
- 支持加载深圳道路网络 `shenzhen_drive.pkl` 或 `shenzhen_drive.graphml`
- 支持将 GPS 点匹配到最近道路节点，并用最短路径拼接校正轨迹
- 支持原始轨迹与路网校正轨迹同图对比
- 支持在地图上点击选点并显示经纬度
- 路网校正示例默认限制为 1-3 辆车和短时间窗口，避免一次处理全量车辆

### 热力图与统计分析
- 支持分钟缓存车辆位置静态热力图
- 支持 OD 上车点静态热力图
- 支持 `1 / 15 / 30 / 60` 分钟动态热力图
- 支持 `EMA / WMA` 平滑算法
- 支持 DBSCAN 上车点聚类分析
- 支持小时订单统计、里程区间统计、车辆运营统计
- 支持 CSV / Excel 统计导出

## 当前实现说明

### 热力图口径
- 分钟缓存车辆位置静态热力图固定按全部车辆统计，不受左侧车辆 ID 筛选影响
- 分钟缓存车辆位置动态热力图固定按全部车辆统计，不受左侧车辆 ID 筛选影响
- OD 上车点热力图仍按当前车辆筛选条件执行

### 动态热力图
- 已从 `HeatMapWithTime` 切换为自定义 `Leaflet.heat + requestAnimationFrame` 播放器
- 支持时间轴拖动、逐帧插值和平滑播放
- 已加入自适应点数预算控制，用于降低掉帧、闪烁和卡顿

### 静态热力图
- 默认显示范围收紧为深圳展示范围
- 已加入深圳行政边界内点位过滤，减少边界外热力点干扰

### 车辆运营统计
- 当前采用 OD 快速统计口径
- 页面中“总里程估计 / 空载里程估计”为估算值，不是逐轨迹积分值

## 运行前提

- Python `3.13+`
- 已准备车辆缓存、分钟缓存和 OD 结果表
- 已安装 `streamlit`、`folium`、`pandas`、`numpy`、`scikit-learn`、`altair`
- 如需路网校正，需额外安装 `networkx`；如使用 `.graphml` 路网文件，还需安装 `osmnx`

## 项目结构

```text
TaxiGPS/
├── cache/
│   ├── vehicles/              # 车辆轨迹缓存
│   └── minutes/               # 分钟位置缓存
├── data/
│   ├── raw/                   # 原始数据
│   ├── cleaned/               # 清洗后数据
│   └── processed/             # 处理后数据（含 OD 表）
├── exports/
│   └── heatmap_stats/         # 统计导出结果
├── logs/
│   ├── map_query.log          # 地图查询日志
│   └── startup_log.txt        # 启动日志
├── pages/
│   └── maps/                  # 生成的地图 HTML
├── src/
│   ├── streamlit_app.py       # Streamlit 主应用
│   ├── map_plotter.py         # 地图绘制与基础查询
│   └── heatmap_analysis.py    # 热力图、聚类与统计分析
└── README.md
```

## 技术栈

- Python
- Streamlit
- Folium / Leaflet
- Pandas
- NumPy
- scikit-learn
- Altair

## 安装依赖

```bash
pip install streamlit folium pandas numpy scikit-learn altair networkx osmnx
```

## 启动方式

```bash
cd TaxiGPS
python3 -m streamlit run src/streamlit_app.py
```

默认访问地址：
- `http://localhost:8501`

## 使用说明

1. 在左侧选择日期、时间范围和车辆 ID。
2. 点击“执行查询”后切换顶部功能模块。
3. 轨迹查询优先读取车辆缓存，分钟位置优先读取分钟缓存。
4. 在“热力图与统计分析”中可切换静态热力图、动态热力图、订单统计、车辆运营、上车点聚类。
5. 当热力图数据源为“分钟缓存车辆位置”时，系统固定按全部车辆统计。
6. 当热力图数据源为“OD 上车点”时，系统按当前车辆筛选执行。
7. 在“轨迹查询”中可打开“启用路网校正”，对所选前 1-3 辆车展示原始轨迹与校正轨迹。
8. 地图支持点击选点，左下角面板会显示点击位置的经纬度。

## 数据准备

清洗后的轨迹数据至少应包含以下字段：

- `id`：车辆 ID
- `time`：时间戳，格式 `YYYY-MM-DD HH:MM:SS`
- `long`：经度
- `lati`：纬度
- `status`：状态，`0=空载`，`1=载客`
- `speed`：速度，单位 `km/h`

建议处理流程：

1. 运行 `data_cleaning.py` 完成原始数据清洗。
2. 运行 `clean_od_extraction.py` 提取 OD 上下车点。
3. 首次启动系统时自动构建车辆缓存与分钟缓存。
4. 将深圳边界文件放入 `data/深圳市.json`。
5. 如需路网校正，将 `shenzhen_drive.pkl` 优先放入项目根目录、`data/` 或 `cache/`；也可使用 `TAXIGPS_ROAD_NETWORK_PATH` 指定自定义路径。

## 注意事项

- 地图 HTML 输出目录为 `pages/maps/`
- 统计导出目录为 `exports/heatmap_stats/`
- 分钟热力图和静态热力图都带有边界过滤与漂移过滤
- 动态热力图首次加载后可通过底部时间轴直接拖动查看任意时间片
- 路网校正优先读取 pkl 文件；`nearest_nodes` 使用经度、纬度顺序，最短路径失败会被记录并跳过对应片段

## 更新记录

详细变更请查看 [logs/changelog.txt](logs/changelog.txt)。

## 许可证

MIT License
