"""加载器集合：各类非 Parquet 数据源的适配入口。

rosbag_loader 顶层只依赖 numpy/pandas（rosbags 在实际解析时才懒加载），
因此这里直接导出是安全的，未装 rosbags 也能 import。
"""

from .fzmotion_csv_loader import load_fzmotion_csv, load_fzmotion_csv_dir
from .rosbag_loader import RosbagAdapter, RosbagAdapterError, build_frame_table

__all__ = [
    "load_fzmotion_csv",
    "load_fzmotion_csv_dir",
    "RosbagAdapter",
    "RosbagAdapterError",
    "build_frame_table",
]
