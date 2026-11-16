"""pytest 根级配置：把 src/ 注入 import 路径（FZMotion CSV 适配的可选包）。

主代码在仓库根目录 pipeline/，由 pytest.ini 的 pythonpath=. 覆盖。
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
