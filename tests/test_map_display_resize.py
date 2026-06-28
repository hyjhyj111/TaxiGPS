import os
import sys
import unittest

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from map_plotter import add_map_layers, build_map


class MapDisplayResizeTest(unittest.TestCase):
    def test_generated_map_invalidates_leaflet_size_after_streamlit_layout_settles(self):
        output_path = "/private/tmp/map_resize_fix_test.html"
        df = pd.DataFrame(
            {
                "lati": [22.60, 22.66],
                "long": [114.00, 114.14],
            }
        )
        m = build_map(df)
        add_map_layers(m)
        m.save(output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()

        self.assertIn("__taxigpsInvalidateMapSize", html)
        self.assertIn(".invalidateSize", html)
        self.assertIn("ResizeObserver", html)


if __name__ == "__main__":
    unittest.main()
