#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 08 运行入口。

示例：
python run_stage08.py --od-cache cache/od_corrected.parquet --track-cache cache/vehicle_corrected --batch-size 10
"""

import os
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from stage08_order_route_comparison import main  # noqa: E402


if __name__ == "__main__":
    main()
