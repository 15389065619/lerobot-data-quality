# 实测记录

所有数字均来自本机真实运行（Windows 11 / Python 3.11 / 同一台机器）。

## 0. 环境与数据集指纹（复现基线）

| 项 | 值 |
|---|---|
| 代码版本 | `1933192`（本文全部数字都在该提交上测得） |
| Python | 3.11.7 |
| 依赖 | pandas 2.2.3 / pyarrow 26.0.0 / numpy 2.2.3 / matplotlib 3.11.2 |
| 平台 | Windows 11，纯 CPU（无 GPU） |
| 演示数据集 | `tools/make_demo_dataset.py` 生成，36 episode / 3008 帧 / 2 相机 / 13 处注入异常 |
| 点云数据集 | HuggingFace `rishabhrj11/gym-xarm-pointcloud`，v3.0 / 50 episode / 18374 帧 / fps 50 / 4 相机 |

复现命令（仓库根目录执行；前两步会重建本文用到的数据集）：

```bash
pip install -e .[dev]
python tools/make_demo_dataset.py                      # 注入缺陷的 demo_dataset
python main.py --dataset_path demo_dataset --output_dir outputs
python -m pytest tests -q --cov --cov-branch           # 覆盖率门槛见 pyproject

# 点云数据集（需联网；HF 直连不通时用镜像，下载到 data/gym_xarm_pointcloud）
set HF_ENDPOINT=https://hf-mirror.com        # PowerShell；bash 用 export
python -c "from huggingface_hub import snapshot_download; snapshot_download('rishabhrj11/gym-xarm-pointcloud', repo_type='dataset', local_dir='data/gym_xarm_pointcloud')"
python main.py --dataset_path data/gym_xarm_pointcloud --output_dir outputs_pc_full
```

## 1. demo_dataset（36 episode / 3008 帧 / 2 相机）

- 13 处注入异常全部命中，**0 误报**（自适应阈值下正常轨迹不触发跳变告警）
- 处置：保留 24 / 修复 2（边界裁帧、时间轴重建）/ 复核 2 / 剔除 8
- 修复验证：裁帧后 NaN 归零，重建的时间轴与 `frame_index/fps` 完全一致
- 清洗后 28 episode / 2388 帧，`length==rows`、视频窗口==行数、index 连续、stats 对齐
- 复核回灌：override 后 ep7 剔除 / ep23 保留，报告标注「人工覆盖 2 条」
- 归因：bob 批次异常率 44.4% vs alice 27.8%，直接定位到人

## 2. 真实点云数据集 gym_xarm_pointcloud

数据集：HuggingFace `rishabhrj11/gym-xarm-pointcloud`（LeRobot v3.0 原生点云，50 episode / 18374 帧 / 4 相机 224×224 / 点云 512×6）。

```bash
python main.py --dataset_path data/gym_xarm_pointcloud --output_dir outputs_pc_full
```

- 全流程 **8 步跑通**；`observation.environment_state` 被识别为重型二维特征（512×6），流式读取不占主内存
- 数值质检：50/50 episode 正常；点云质检：18374/18374 帧正常（0 NaN / 0 离群 / 0 重复 / 0 越界 / 0 颜色异常）
- 点云自适应阈值实测：场景尺度 1.138 m，离群半径 0.0615 m（1-NN 中位数 0.0246 m × 2.5），工作空间 `[0.87,-0.21,0.43] ~ [1.89,0.78,0.97]`
- 修复链路走 `noop`（数据本身干净），产物为完整 `cleaned_dataset`（含 4 路视频）

### 2.1 真实数据故障注入（人工注入 8 类坏帧后重跑点云质检 + 清洗）

| 注入 | 命中规则 | 处置 |
|---|---|---|
| 1 点 NaN | `pc_non_finite` | repair |
| 25% 重复点 | `pc_duplicate` | repair |
| 60 点 @8 m | `pc_outliers` + `pc_out_of_bounds` | repair |
| 30 点 @100 m | `pc_out_of_bounds` | repair |
| 20 点负颜色 | `pc_color_invalid` | repair |
| 整体 +0.08 m | `pc_temporal_jump`（2 帧） | review |
| 只留 8 点 | `pc_empty_or_sparse` | drop → hold-last-good |
| 全部同一点 | `pc_degenerate` | drop → hold-last-good |

结果：8 条规则全部命中；`repair 5 / drop 2 / keep 1802`；移除 221 点、补回 221 点；清洗后**行数不变（1809）、每帧仍 512 点、全部有限、`ARROW:extension:name` 保留**，修复点最大坐标 1.8555 m < 工作空间上界 1.89 m。

### 2.2 behavior1k_task0042（14 GB，200 episode / 128 万帧 / 9 路相机）：编号空间错位导致的全量误判

这是把一个**工具缺陷**当成**数据缺陷**的典型案例，也是本轮最重要的修复。

数据集来自官方仓库，`meta/episodes` 用全局 episode ID（420010, 420020, …, 步长 10，51 处缺口），而 `data/*.parquet` 里是 0..199，**两侧交集为 0**。定位全部落空后，逐 episode 检查把每一条都判成「视频帧数不一致 + 任务缺失」：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| episode 判定 | 0 正常 / **200 error** | 195 正常 / 5 warning |
| `frame_mismatch` | 200（100%） | **0** |
| `missing_task` | 200（100%） | **0** |
| 处置建议 | 剔除 200 个 / **128 万帧** | 保留 195 / 复核 5 / 剔除 0 |
| 健康度 | 严重 | 轻微 |

**判定数据本身没坏**的三条独立证据（均 200/200 精确成立）：

1. `meta.length` == 数据实际行数（逐条相等，合计 1,279,960 帧）
2. 视频时间窗帧数 == 数据行数（9 路相机取 RGB 三路核对）
3. `dataset_from_index` == length 的累计和

修复后的匹配逻辑与结论：

- 加载层按顺序建立映射 `0→420010, 1→420020, …`，两重证据一致率均 **1.0**，映射被接受（`mode=remapped`）
- 本次真正命中的问题只有 **5 个 episode 的 `action_jump`**（转人工复核），以及一个**真实**的 `meta_index_inconsistent`（`index` 列每个 episode 从 0 重启，max=9431 而 rows=133255）
- 全量 200 episode 流式跑完约 7 分钟；加载阶段多花 1.3 s 做单列扫描（`episode_index` 一列，约 10 MB）

顺带修掉的两个度量缺陷：

| 问题 | 表现 | 修法 |
|---|---|---|
| `stats_stale` 用 `\|recorded_mean\|` 作分母 | 均值接近 0 的维度把微小漂移放大成 **913 倍** | 分母改用该维自身量级 `max(\|mean\|, std)` → 同数据变为 0.99 |
| 子集扫描比全量 stats | `--max_episodes 20` 必然偏差，被报成「stats 过期」 | 部分扫描时标记为不可比并写入 `skipped_checks`，只有全量运行才判定 |

### 2.3 本地 17 个数据集全量审计（v3.0 / v2.1 混合，共 4108 episode）

把仓库里全部数据集跑一遍，目的不是看「能不能跑」，而是看**规则在真实分布上是否还成立**。

| 数据集 | episode | 健康度 | 主要结论 |
|---|---:|---|---|
| behavior1k_task0042 (14 GB) | 200 | 轻微 | 编号错位自动对齐；195 keep / 5 info / 0 drop |
| libero_10_image (12.8 GB) | 379 | 轻微 | 378 keep；ep290 仅比自适应下限短 3.8% |
| humanoid_everyday_h1 (v2.1) | 1418 | 轻微 | **info 声明 4885 ep / 1.66M 帧，实际 1418 ep / 452K 行** |
| kuavo4_hotel_services (v2.1) | 484 | 严重 | **两路腕部相机视频目录为空，前视仅 156/484** |
| gym_xarm_pc | 5 | 严重 | **info 声明 50 ep / 18374 帧，实际 5 ep / 1809 行** |
| chingmu_subset (120 Hz) | 20 | 严重 | 6 个片段帧数超出 120 Hz 动捕的分布区间 |
| pusht | 206 | 中等 | 17 条 action_jump，超阈 1.06~1.28 倍 |
| w01_v3 / xarm_lift_medium / gym_xarm_pointcloud / aloha 系 | 1151 | 干净 | 无结构性脏数据 |

这一轮暴露的三处工具缺陷（已修，均有回归测试）：

| 缺陷 | 表现 | 修法 |
|---|---|---|
| `video_missing` 分支不可达 | `video_location()` 内部先做 `is_file()` 过滤，文件缺失永远走 `video_unlocatable` → 被记成「帧数不一致」。kuavo4 484/484 全部误标，会把人引去查帧率和编码 | 增加 `require_exists` 参数；缺文件单独成规则 `video_missing`（error），定位失败降 info |
| v2.1 套用 v3 的 index 契约 | humanoid（每集归零）被判 error 并置顶阻断；kuavo4（全局偏移）却因 index 被重排而放过——同一版本两种结论 | 契约按 `codebase_version` 选择；v2 只要求 episode 内连续不重复，起点不限 |
| info 计数与实物不符时静默 | gym_xarm_pc 丢了 45/50 个 episode，报告只说「正常 0 / 异常 5」 | Schema 增加 `total_episodes` 与 `total_frames` 两条对账断言 |
| `action_jump` 命中集中在阈值边缘 | 54 条命中里超阈中位数仅 1.10~1.63 倍，humanoid 14/14 等于 `abs_floor` | 按超阈倍数分级：<1.5 降 info，默认行为不再新增剔除 |

## 3. 并发加速（同一数据集，workers=1 vs workers=8）
| 阶段 | workers=1 | workers=8 | 加速比 |
|---|---|---|---|
| 全流程总耗时 | 190.1 s | 64.7 s | 2.9× |
| 其中点云扫描 | ~170 s | 52 s | 3.3× |

点云扫描占单进程耗时的 89%，因此并行重点放在 `(parquet 文件, row group)` 分片粒度。8 项产物在归一化时间戳与输出路径后与串行**逐字节一致**，`tests/test_parallel.py` 把这条不变量钉死。

## 4. 容错与断点续传（故障注入 + 四连跑）

把 `gym_xarm_pointcloud` 复制一份，人为损坏其中一个 parquet 分片（`file-001.parquet` 写成 82 字节垃圾）：

| 轮次 | 命令 | 耗时 | 结果 |
|---|---|---|---|
| run1 | 冷跑 | 35.3 s | 不中断；23 个 episode 标 `episode_read_error` → 剔除；好分片 9380 帧照常质检 |
| run2 | 同命令重跑 | **1.6 s** | manifest 整体短路（22× 加速） |
| run3 | `--force` 强制重检 | 7.7 s | 点云帧级缓存命中，扫描阶段 27 s → 0 s |
| run4 | 修复该分片后 | 40.0 s | 指纹变化自动重扫该分片，18374 帧全部恢复，episode 剔除自动撤销 |

## 5. 测试与 CI

- **105 个单测**：数值/清洗/增量/流式/阈值 39 个 + 点云 8 个 + 并发一致性 2 个 + 容错续传 8 个 + 配置 8 个 + FZMotion 适配 5 个 + ROS Bag 适配 10 个 + 多模态对齐 6 个 + 预检/版本/策略/Dashboard 19 个
- 覆盖率 **65.35%**（branch=true，实测基线），CI 在 Python 3.12 上以 `fail_under=64` 强制门槛，只升不降
- GitHub Actions 在 **Python 3.10 / 3.11 / 3.12** 三个版本 × 干净 Linux 环境全部通过
- 打包链路本地预演（2026-11-04）：`python -m build` 产出 wheel 181 KB + sdist 177 KB，`twine check` 双产物 PASSED；在全新 venv 中安装 wheel 后 5 个 console 脚本全部生成、`lerobot-data-quality --help` 退出码 0、`import lerobot_data_quality / pipeline / main` 正常。**OIDC 上传这一步需要 PyPI 侧配置好 Trusted Publisher 后由 CI 完成**，本地无法替代

## 6. 已知边界（如实说明）

- 单机运行，适配 ≤ 数十 GB / 数千 episode；更大规模可按 parquet 文件分片并行（规则函数已无状态，扩展点在 main 的批处理循环）
- 视频重建用 libx264 重编码（crf20），画质有轻微损失；官方 CLI 场景由其自行处理编码
- 视频内容检测（黑屏）默认只抽首相机、限 400 次探针，可通过 config 调整
- `--overrides` 只改变处置动作，不改数据内容（如「补标任务」需在上游完成后重跑质检）
- `lerobot-edit-dataset` 分支按官方 CLI 契约实现但未在本机实测（未安装 lerobot），不可用时自动回退并在报告标注 `method: manual`
- 点云只做几何/颜色/时序质量，不做语义级检查（物体缺失、遮挡、标注错误）
- 固定点数契约下，`drop` 帧用 hold-last-good 占位：能保证训练张量对齐，但占位帧不是真实观测，全部记录在 `pointcloud_cleaning_notes.json`
- 离群半径/工作空间是**整数据集自适应**；多机器人、多场景混采的数据集建议在 config 里用 `workspace.mode=fixed` 明确包围盒
- 时序规则只看质心平移，不检测纯旋转（需要位姿/坐标系参考才能判断）

## 7. Metrics & Precision

> 统计对象：`outputs*/review_queue.csv` 历史异常记录；脚本：`tools/eval_precision.py`。
> 当前状态：已生成 100 条待标注样本（`eval_annotation.csv`），人工结论待回填。

### 7.1 抽样与标注口径

- 71 个历史 `review_queue.csv` 共 2476 条记录，按 `(dataset, episode_index, reasons, frame_count, detail)` 去重后 1339 条；
- 抽样先保证每条规则至少 1 条，其余随机补足，`--seed 42` 可复现；
- 标注只填 `human_label`：`true_positive` / `false_positive`；拿不准的保持 `unknown`，统计时跳过；
- 精确率 = TP / (TP + FP)，按规则统计；记录级精确率每条记录只计一次；
- 召回率需要全量标注（`--full-labels`，列为 `dataset, episode_index, rule_code, present`），否则显示 `-`。

### 7.2 精确率结果（待回填）

| 规则 | 抽样数 | 已标注 | TP | FP | 精确率 |
|---|---|---|---|---|---|
| frame_mismatch | _待填_ | | | | |
| action_jump | _待填_ | | | | |
| missing_task | _待填_ | | | | |
| episode_read_error | _待填_ | | | | |
| frame_count_abnormal | _待填_ | | | | |
| 其他（nan_state / duplicate / blank_video / timestamp 等） | | | | | |
| **记录级总体** | 100 | | | | |

### 7.3 召回率（需全量标注）

_待填：完成全量标注后，用 7.4 的命令生成 `recall` 列。未做全量标注时，本节明确写「未评估」，不臆造数字。_

### 7.4 复现命令

```bash
# 1) 抽样（默认扫描仓库根目录下所有 review_queue.csv）
python tools/eval_precision.py sample --n 100 --out eval_annotation.csv

# 2) 人工填写 human_label 后统计精确率
python tools/eval_precision.py report --annotations eval_annotation.csv

# 3) 可选：有全量标注时同时算召回率，并导出 Markdown 表格回填本章
python tools/eval_precision.py report --annotations eval_annotation.csv \
    --full-labels full_labels.csv --markdown-out docs/metrics_table.md
```

## 8. 单机读取性能与流式架构

测试数据：`data/stress/libero_10_image` 中最大的 5 个 data parquet，压缩体积 497.7 MB，
共 3,994 行，含两路图像 binary（与线上校验负载一致）。每种方式在独立子进程中运行，
用 psutil 每 10 ms 采样 RSS 峰值；`--repeat 2` 取热缓存后的最快一次。

| 方式 | 行数 | 耗时 (s) | 峰值 RSS (MB) | 增量 RSS (MB) | 吞吐 (MB/s) |
|---|---|---|---|---|---|
| `pandas.read_parquet(list)` | 3,994 | 0.94 | 2,694 | 2,616 | 531 |
| `duckdb.read_parquet(...).fetch_arrow_table()` | 3,994 | 0.52 | 1,509 | 1,478 | 954 |
| 流式 `ParquetFile.iter_batches(256)` | 3,994 | 0.46 | 732 | 691 | 1,090 |

复现：

```bash
python tools/benchmark_reader.py --dataset data/stress --n-files 5 --repeat 2
```

### 8.1 读法

- pandas 会把 5 个文件拼成一个 DataFrame，增量内存约为压缩体积的 **5.3 倍**（2.6 GB / 0.5 GB）。
  按同样倍率线性外推，14 GB 数据需要 ~74 GB 内存，24 GB 单机必然 OOM；TB 级更无从谈起。
- DuckDB 不经过 Python 对象，但 `fetch_arrow_table()` 仍把结果完整物化，增量约为 **3.0 倍**
  （1.5 GB / 0.5 GB）。它适合“读进来马上聚合/过滤”的查询，不适合“只扫一遍做校验”。
- 流式只保留当前 row group / batch，增量约 **1.4 倍**（0.7 GB），且不随文件数线性增长：
  内存上限由「最大 row group × 解码中间量」决定，而不是数据集总大小。本组数据每个文件只有
  1 个 row group 且含 2 路图像，因此单批解码仍有约 0.7 GB；纯数值列或更小 row group 会更低。
- 流式在这组数据上反而最快（1,090 MB/s），因为它只做 Arrow 解码、不把每行图像转成 Python
  `bytes`，也没有 pandas 拼接/索引开销。

### 8.2 为什么选择流式架构

1. **内存不随数据集增长**：TB 级数据无法整体放进单机内存，流式把峰值压在「一个 row group」
   量级；边缘设备（8–16 GB）也能跑完整个校验流程。
2. **校验任务本身就是单遍扫描**：frame_mismatch、action_jump、timestamp 等规则只需要顺序
   累积统计量，不需要随机访问或全量 join；`iter_batches` 天然匹配。
3. **可裁剪列**：校验只需要 `observation.state/action/timestamp/frame_index/episode_index`
   等少量列，`iter_batches(columns=...)` 可以跳过图像/视频列；本章为了与 pandas 公平对比
   才读取全部列，线上路径的实际内存更低。
4. **代价可控**：流式牺牲了随机访问和“读完即拿 DataFrame”的便利，换来常数级内存；对
   需要反复交互分析的场景仍可先用 DuckDB 聚合，再按需物化小结果。

> **本节的两组测量。** 「清洗让数据变好」这种话最容易自证，所以同一份数据量了两次：
> §9 是流水线的真实默认行为（只剔除 `drop`，`review` 原样留在产物里），
> §10 是把复核队列里那条单帧 ±18 的碰撞跳变人工判为 `drop` 后重跑导出的结果。
> 两次之差，就是「人工复核回灌」这一步的边际价值。
> 指标定义、归因方法与适用边界见 §9.5，请与结论一起读。

<!-- ACTION_DIST:BEGIN -->
## 9. 清洗前后动作分布对比（下游收益代理指标）

> 原始：`demo_dataset`（36 ep / 3,008 帧） → 清洗：`cleaned_dataset`（29 ep / 2,393 帧）
> 中间基线：同一份原始数据，只取清洗后保留的 episode —— 用来把「剔除」和「修复」两种效应拆开。
> 跳变阈值：`|Δ|∞ > 1.000`（`action_jump.abs_floor` 按 fps 折算）；口径：逐维总体方差（ddof=0）跨维平均。
> 复现：`python tools/analyze_action_distribution.py --raw demo_dataset --cleaned <cleaned_dataset> --report <quality_report.json> --update-eval docs/EVAL.md`

### 9.1 总量指标

| 指标 | 原始（全部） | 原始（仅保留集） | 清洗后 | 总变化 |
|---|---|---|---|---|
| episode 数 | 36 | 29 | 29 | -7 |
| 帧数 | 3,008 | 2,396 | 2,393 | -20.4% |
| 动作维度 | 7 | 7 | 7 | — |
| **逐维方差均值** | 0.4426 | 0.4317 | 0.4319 | **-2.4%** |
| 动作标准差 | 0.6653 | 0.6570 | 0.6572 | -1.2% |
| 离差平方和（SSE） | 9,320 | 7,240 | 7,234 | -22.4% |
| 最大跳变 max\|Δ\|∞ | 34.018 (ep31) | 34.004 | 34.004 | -0.0% |
| 每集最大跳变中位数 | 0.1709 | 0.1722 | 0.1722 | +0.7% |
| 每集最大跳变 P90 | 0.2031 | 0.2018 | 0.2018 | -0.7% |
| 超阈帧数 | 6 | 3 | 3 | -3 |
| 均值（跨维平均） | 0.0034 | -7.71e-04 | -0.0012 | -0.0047 |
| 归一化跳变 max\|Δ\|/std | 51.131 | 51.756 | 51.744 | +1.2% |

### 9.2 变化归因：剔除 vs 修复

删样本一定会改变方差，所以「方差下降 X%」必须先拆开，否则等于把删数据说成提纯。SSE 可跨 episode 相加，是这里唯一能做加法分解的量：

| 效应 | 定义 | 方差变化 | 说明 |
|---|---|---|---|
| 选择（剔除） | 原始全部 → 原始仅保留集 | -2.5% | 剔除 7 个 episode、612 帧（占帧数 20.346%），带走 22.262% 的离差平方和 |
| 修复 | 原始仅保留集 → 清洗后 | +0.0% | 边界裁帧 3 帧、时间戳重建；不改动作幅值 |
| **合计** | 原始全部 → 清洗后 | **-2.4%** | 两项之和（方差对样本构成非线性，故为近似） |

均值漂移：逐维均值差的最大值 0.0344（按原始 std 归一化 0.0517），其中由裁剪/重建贡献 9.07e-04。

### 9.3 被剔除 episode 的尾部贡献

| episode | 帧数 | 帧数占比 | 方差 | max\|Δ\|∞ | SSE 占比 | 剔除原因 |
|---|---|---|---|---|---|---|
| 31 | 84 | 2.793% | 1.384 | 34.018 | 9.101% | nan_state / action_jump |
| 17 | 104 | 3.457% | 0.5054 | 0.1978 | 4.421% | blank_video |
| 13 | 95 | 3.158% | 0.3585 | 0.1697 | 2.741% | duplicate_episode |
| 3 | 112 | 3.723% | 0.2335 | 0.1551 | 2.249% | nan_state |
| 27 | 76 | 2.527% | 0.2743 | 0.1812 | 1.841% | nan_action / action_jump |
| 19 | 77 | 2.560% | 0.2682 | 0.1330 | 1.799% | video_span_mismatch / frame_mismatch |
| 9 | 64 | 2.128% | 1.86e-14 | 5.61e-07 | 0.1100% | static_episode |
| **合计** | 612 | 20.346% | — | — | **22.262%** | — |

### 9.4 结论

- 剔除 7 个 episode（612 帧）后，整体动作方差由 0.4426 降至 0.4319，**-2.4%**；其中选择效应 -2.5%、修复效应 +0.0%（两项之和与合计略有出入，方差对样本构成非线性，见 §9.2）。
- 归一化跳变（max|Δ|/std）由 51.131 变为 51.744（+1.2%，未收窄）；清洗后单帧最大位移仍为 34.004，超过阈值 34.004 倍 —— 极值所在的 episode 不在本次剔除集合内（多为 review 未裁决，仍留在产物里）。
- 超过 `|Δ|∞ > 1.000` 的帧：6 → 3（涉及 2 → 1 个 episode）。
- 均值漂移 0.0344（0.0517 个 std，可忽略，动作中心未被移动）。

### 9.5 边界（这段不能删）

- **没有训练模型。** 本节给的是代理指标：异常跳变帧减少、归一化跳变幅度收窄、对应梯度被放大的通道变少。这不等于「精度提升」，也不等于「loss 曲线更平滑」；要那条结论必须做清洗前后的训练对照（同 seed、同 step、同评测集）。
- **方差下降不等于数据变好。** 剔除高方差 episode 必然降低整体方差——若这些 episode 是真实的高动态任务（而非碰撞/断连），降方差反而是信息损失。判断依据只能是 §9.3 的剔除原因，而不是方差这个数字本身。
- 逐维方差对量纲敏感；跨机器人/跨任务混采的数据集不能直接和本表比较。
- 跳变阈值取配置 `abs_floor`（按 fps 折算），不做量纲自适应：像素动作（如 pusht 的 0–512）上该阈值没有物理意义，重跑时用 `--jump-threshold` 显式指定。
- 修复阶段只裁剪边界帧与重建时间戳，不动动作幅值，因此「修复效应」理论上应接近 0；若它明显非 0，说明清洗改写了动作内容，需要单独解释。

> 互校：`stats.json` 里 action 各维 std 的均值为 0.6516，本脚本逐维 std 实算均值 0.6531（差 +0.2%）。两者应一致；不一致说明 stats 未随清洗重算，训练侧的归一化会用错分布。

<!-- ACTION_DIST:END -->

<!-- ACTION_DIST_REVIEW:BEGIN -->
## 10. 复核回灌后的再测量：把跳变 episode 判为 drop

> 原始：`demo_dataset`（36 ep / 3,008 帧） → 清洗：`cleaned_dataset`（28 ep / 2,335 帧）
> 中间基线：同一份原始数据，只取清洗后保留的 episode —— 用来把「剔除」和「修复」两种效应拆开。
> 跳变阈值：`|Δ|∞ > 1.000`（`action_jump.abs_floor` 按 fps 折算）；口径：逐维总体方差（ddof=0）跨维平均。
> 复现：`python tools/analyze_action_distribution.py --raw demo_dataset --cleaned <cleaned_dataset> --report <quality_report.json> --update-eval docs/EVAL.md --section-id ACTION_DIST_REVIEW --section-title "## 10. 复核回灌后的再测量：把跳变 episode 判为 drop"`

### 10.1 总量指标

| 指标 | 原始（全部） | 原始（仅保留集） | 清洗后 | 总变化 |
|---|---|---|---|---|
| episode 数 | 36 | 28 | 28 | -8 |
| 帧数 | 3,008 | 2,338 | 2,335 | -22.4% |
| 动作维度 | 7 | 7 | 7 | — |
| **逐维方差均值** | 0.4426 | 0.3984 | 0.3985 | **-10.0%** |
| 动作标准差 | 0.6653 | 0.6312 | 0.6313 | -5.1% |
| 离差平方和（SSE） | 9,320 | 6,520 | 6,514 | -30.1% |
| 最大跳变 max\|Δ\|∞ | 34.018 (ep31) | 0.2121 | 0.2121 | -99.4% |
| 每集最大跳变中位数 | 0.1709 | 0.1709 | 0.1709 | +0.0% |
| 每集最大跳变 P90 | 0.2031 | 0.1946 | 0.1946 | -4.2% |
| 超阈帧数 | 6 | 0 | 0 | -6 |
| 均值（跨维平均） | 0.0034 | 5.74e-04 | 1.07e-04 | -0.0033 |
| 归一化跳变 max\|Δ\|/std | 51.131 | 0.3360 | 0.3359 | -99.3% |

### 10.2 变化归因：剔除 vs 修复

删样本一定会改变方差，所以「方差下降 X%」必须先拆开，否则等于把删数据说成提纯。SSE 可跨 episode 相加，是这里唯一能做加法分解的量：

| 效应 | 定义 | 方差变化 | 说明 |
|---|---|---|---|
| 选择（剔除） | 原始全部 → 原始仅保留集 | -10.0% | 剔除 8 个 episode、670 帧（占帧数 22.274%），带走 29.973% 的离差平方和 |
| 修复 | 原始仅保留集 → 清洗后 | +0.0% | 边界裁帧 3 帧、时间戳重建；不改动作幅值 |
| **合计** | 原始全部 → 清洗后 | **-10.0%** | 两项之和（方差对样本构成非线性，故为近似） |

均值漂移：逐维均值差的最大值 0.0453（按原始 std 归一化 0.0681），其中由裁剪/重建贡献 9.29e-04。

### 10.3 被剔除 episode 的尾部贡献

| episode | 帧数 | 帧数占比 | 方差 | max\|Δ\|∞ | SSE 占比 | 剔除原因 |
|---|---|---|---|---|---|---|
| 31 | 84 | 2.793% | 1.384 | 34.018 | 9.101% | nan_state / action_jump |
| 7 | 58 | 1.928% | 1.640 | 34.004 | 7.711% | action_jump |
| 17 | 104 | 3.457% | 0.5054 | 0.1978 | 4.421% | blank_video |
| 13 | 95 | 3.158% | 0.3585 | 0.1697 | 2.741% | duplicate_episode |
| 3 | 112 | 3.723% | 0.2335 | 0.1551 | 2.249% | nan_state |
| 27 | 76 | 2.527% | 0.2743 | 0.1812 | 1.841% | nan_action / action_jump |
| 19 | 77 | 2.560% | 0.2682 | 0.1330 | 1.799% | video_span_mismatch / frame_mismatch |
| 9 | 64 | 2.128% | 1.86e-14 | 5.61e-07 | 0.1100% | static_episode |
| **合计** | 670 | 22.274% | — | — | **29.973%** | — |

### 10.4 结论

- 剔除 8 个 episode（670 帧）后，整体动作方差由 0.4426 降至 0.3985，**-10.0%**；其中选择效应 -10.0%、修复效应 +0.0%（两项之和与合计略有出入，方差对样本构成非线性，见 §10.2）。
- 归一化跳变（max|Δ|/std）由 51.131 收窄到 0.3359（-99.3%）；单帧最大位移从 34.018 降到 0.2121，已回到阈值以内。
- 超过 `|Δ|∞ > 1.000` 的帧：6 → 0（涉及 2 → 0 个 episode）。
- 均值漂移 0.0453（0.0681 个 std，可忽略，动作中心未被移动）。

### 10.5 边界（这段不能删）

- **没有训练模型。** 本节给的是代理指标：异常跳变帧减少、归一化跳变幅度收窄、对应梯度被放大的通道变少。这不等于「精度提升」，也不等于「loss 曲线更平滑」；要那条结论必须做清洗前后的训练对照（同 seed、同 step、同评测集）。
- **方差下降不等于数据变好。** 剔除高方差 episode 必然降低整体方差——若这些 episode 是真实的高动态任务（而非碰撞/断连），降方差反而是信息损失。判断依据只能是 §10.3 的剔除原因，而不是方差这个数字本身。
- 逐维方差对量纲敏感；跨机器人/跨任务混采的数据集不能直接和本表比较。
- 跳变阈值取配置 `abs_floor`（按 fps 折算），不做量纲自适应：像素动作（如 pusht 的 0–512）上该阈值没有物理意义，重跑时用 `--jump-threshold` 显式指定。
- 修复阶段只裁剪边界帧与重建时间戳，不动动作幅值，因此「修复效应」理论上应接近 0；若它明显非 0，说明清洗改写了动作内容，需要单独解释。

> 互校：`stats.json` 里 action 各维 std 的均值为 0.6295，本脚本逐维 std 实算均值 0.6309（差 +0.2%）。两者应一致；不一致说明 stats 未随清洗重算，训练侧的归一化会用错分布。

<!-- ACTION_DIST_REVIEW:END -->
