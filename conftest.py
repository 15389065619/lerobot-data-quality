"""pytest 根级配置：把 src/ 注入 import 路径。

主代码在仓库根目录 pipeline/（由 pytest.ini 的 pythonpath=. 覆盖）；
FZMotion CSV 适配是可选独立能力，放在 src/lerobot_data_quality/ 下。
这里只做路径注入——不修改 pytest.ini，不影响既有测试的导入行为。
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
