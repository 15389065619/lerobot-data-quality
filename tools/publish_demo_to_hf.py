"""把演示数据集发布到 HuggingFace Hub（可选能力）。

先决条件：
    pip install -e ".[hf]"        # 安装 huggingface_hub
    huggingface-cli login         # 或设置环境变量 HF_TOKEN

用法：
    python tools/publish_demo_to_hf.py --repo_id <你的账号>/lerobot-data-quality-demo
    python tools/publish_demo_to_hf.py --repo_id ... --skip-generate   # 复用已有 demo_dataset
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)

README_TEMPLATE = """# {repo_id}

`lerobot-data-quality` 的演示数据集（LeRobot v3.0 布局），由 `tools/make_demo_dataset.py`
生成，内含 13 处注入异常（NaN、action 跳变、帧数异常、黑屏、重复 episode 等）。

可用它体验完整流水线：

```bash
pip install lerobot-data-quality
lerobot-data-quality --dataset_path . --output_dir ../outputs
```

期望输出：保留 24 / 修复 2 / 复核 3 / 剔除 7（清洗产物 29 episode / 2393 帧）。
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="上传演示数据集到 HuggingFace Hub")
    ap.add_argument("--repo_id", required=True, help="形如 user/name")
    ap.add_argument("--dataset_dir", default="demo_dataset", help="待上传目录")
    ap.add_argument("--skip-generate", action="store_true", help="跳过生成，直接上传已有目录")
    ap.add_argument("--private", action="store_true", help="创建私有数据集")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dataset_dir = Path(args.dataset_dir)

    if not args.skip_generate:
        LOG.info("生成演示数据集 → %s", dataset_dir)
        subprocess.run(
            [sys.executable, "tools/make_demo_dataset.py", "--output", str(dataset_dir)],
            check=True,
        )
    if not dataset_dir.exists():
        LOG.error("目录不存在：%s（去掉 --skip-generate 可自动生成）", dataset_dir)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        LOG.error("缺少 huggingface_hub，请先执行：pip install -e \".[hf]\"")
        return 1

    api = HfApi()
    repo_url = api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    LOG.info("仓库就绪：%s", repo_url)

    # 写一份 README，让 Hub 页面自带使用说明
    (dataset_dir / "README.md").write_text(README_TEMPLATE.format(repo_id=args.repo_id), encoding="utf-8")

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(dataset_dir),
        commit_message="upload lerobot-data-quality demo dataset",
    )
    LOG.info("上传完成：https://huggingface.co/datasets/%s", args.repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
