# 出租车GPS轨迹查询系统

基于真实出租车GPS数据构建的专业时空分析平台，支持轨迹查询、实时定位、OD分析与动画回放。

## 🚀 功能特性

### 📍 轨迹查询
- 按车辆ID查询单车行驶轨迹
- 轨迹按状态自动着色：红色表示载客，蓝色表示空载
- 支持自定义时间范围查询

### 📊 分钟位置查询
- 查看某一分钟所有车辆的位置分布
- 支持按车辆ID筛选
- 自动按上限抽样以提升流畅度

### 🔄 OD点标注
- 展示上车点（绿色）和下车点（红色）
- 点较多时自动聚类
- 抽样显示OD连线

### 🎬 动画轨迹
- 按时间顺序播放单车运动
- 显示速度变化
- 支持播放速度调节

## 📁 项目结构

```
TaxiGPS/
├── cache/           # 缓存目录
│   ├── vehicles/    # 车辆轨迹缓存（按车辆ID存储）
│   └── minutes/     # 分钟位置缓存（按时间分片）
├── data/            # 数据目录
│   ├── raw/         # 原始数据
│   ├── cleaned/     # 清洗后数据
│   └── processed/   # 处理后数据（OD表等）
├── docs/            # 文档目录
├── logs/            # 日志文件
│   ├── changelog.txt    # 版本变更记录
│   ├── map_query.log    # 地图查询日志
│   └── startup_log.txt  # 启动日志
├── pages/           # 可视化页面
│   └── maps/        # 生成的地图HTML文件
└── src/             # 源代码
    ├── streamlit_app.py  # Streamlit交互主应用
    └── map_plotter.py    # 地图绘制核心模块
```

## 🛠️ 技术栈

- **Python 3.13+** - 编程语言
- **Streamlit** - Web应用框架
- **Folium/Leaflet** - 地图可视化
- **Pandas** - 数据处理
- **NumPy** - 数值计算

## 📦 安装依赖

```bash
pip install streamlit folium pandas numpy
```

## 🚗 快速开始

### 启动应用

```bash
cd TaxiGPS
python -m streamlit run src/streamlit_app.py
```

### 访问地址

- 本地访问：http://localhost:8501
- 网络访问：http://your-ip:8501

## 📖 使用说明

1. **输入车辆ID**：在侧边栏输入框中输入车辆ID（如 22223）
2. **选择日期**：选择查询日期
3. **设置时间范围**：设置轨迹开始时间和结束时间
4. **执行查询**：点击"执行查询"按钮
5. **切换视图**：通过顶部标签页切换不同功能模块

## 🗂️ 数据准备

### 数据格式要求

清洗后的数据需包含以下字段：
- `id` - 车辆ID
- `time` - 时间戳（格式：YYYY-MM-DD HH:MM:SS）
- `long` - 经度
- `lati` - 纬度
- `status` - 状态（0=空载，1=载客）
- `speed` - 速度（km/h）

### 数据处理流程

1. **数据清洗**：运行 `data_cleaning.py` 清洗原始数据
2. **OD提取**：运行 `clean_od_extraction.py` 提取上下车点
3. **缓存构建**：系统首次运行时自动构建车辆和分钟缓存

## 📝 更新日志

详细变更记录请查看 [logs/changelog.txt](logs/changelog.txt)

## 📄 许可证

MIT License

## 📧 联系方式

如有问题或建议，请通过Issue反馈。