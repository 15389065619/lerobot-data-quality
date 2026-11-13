# LeRobot v3 数据质检与清洗 Pipeline — 源码级拆解

> 目标：不只说"做了什么"，而是说清"代码为什么必须这么写、换一种写法会坏在哪"。
> 阅读方式：先看第 1 节的数据流，再按第 2～11 节顺序读；每节末尾都有「设计取舍」。

---

## 1. 全局数据流

```
main.py
 ├─ [0] QualityManifest.load / _manifest_fresh        增量快路径，可能直接 return 0
 ├─ [1] load_dataset            -> LeRobotDatasetView  只读内存视图
 ├─ [2] validate_dataset        -> ValidationResult    逐 episode 结论 + 数据集级问题
 ├─ [3] policy.decide           -> dict[int, Decision] 结论翻译成处置动作
 ├─ [4] reporter.build_report   -> quality_report.json 结构化报告
 ├─ [5] cleaner.clean_dataset   -> CleaningResult      删除/裁帧/修时间轴/修 index
 ├─ [6] pointcloud.validate     -> PointCloudValidation
 ├─ [7] pointcloud 清洗          -> 修复点云，行数不变
 └─ [8] visualizer              -> plots/
```

核心分层原则：**每一层只产出数据结构，不替下一层做决定。**

| 层 | 输入 | 输出 | 禁止做的事 |
|---|---|---|---|
| loader | 磁盘目录 | `LeRobotDatasetView` | 不做任何质量判断、不写盘 |
| validator | view + config | `ValidationResult` | 不改数据 |
| policy | result + overrides | `Decision` | 不执行清洗 |
| reporter | result + decisions | JSON | 不依赖清洗是否成功 |
| cleaner | decisions | 新数据集 | 不重新判断对错 |
| pointcloud | 数据 + 特征列 | 修补后的点云 | 不增删行 |

这样任何一层都能单独重跑，且清洗失败时报告仍然可用。

---

## 2. main.py — 编排与失败隔离

### 2.1 参数分类

`build_parser()` 的参数分四类：

1. 数据/输出：`--dataset_path`、`--output_dir`、`--max_episodes`
2. 策略：`--config`、`--overrides`、`--on-review`
3. 阈值快捷覆盖：`--min-frames`、`--max-frames`、`--action-jump-threshold`
4. 性能/降级开关：`--skip-clean`、`--skip-pointcloud`、`--streaming`、`--workers`、`--incremental`

### 2.2 两条硬约束

```python
if args.streaming and not args.skip_clean:
    LOG.warning("--streaming 模式暂不支持清洗（需要整表重写），本次自动跳过 cleaner")
    args.skip_clean = True

if args.incremental and args.max_episodes is not None:
    LOG.warning("--max_episodes 与增量模式不兼容，本次按全量处理")
    incremental = False
```

为什么必须在入口处理，而不是让下游自己判断：

- cleaner 需要整张逐帧表做 `take / set_column / 重排`，流式模式下 `frames` 是空的。让它跑到一半再崩，不如入口降级并给出可读原因。
- 增量 manifest 记录的是"全量数据集签名"，只跑前 N 集时增量复用会得出错误结论。

### 2.3 第 0 步：增量快路径

```python
manifest = QualityManifest.load(manifest_path)
incremental = args.incremental and not args.force

if incremental and manifest.has_results:
    scan = load_dataset(dataset_path, load_frames=False)
    scan_config = _build_config(args, scan.info.get("robot_type"))
    if _manifest_fresh(manifest, scan, scan_config, output_dir,
                       require_cleaning=not args.skip_clean):
        return 0
```

`load_dataset(..., load_frames=False)` 只读 meta，不读 parquet——增量预检本身要便宜，否则"检查是否要重跑"的成本接近重跑。

`_manifest_fresh()` 同时要求五件事：

| 条件 | 为什么 |
|---|---|
| `config_digest` 相同 | 阈值/严重度变了，旧结论没有意义 |
| `quality_report.json` 存在 | 用户拿不到报告等于没跑 |
| 所有 episode 签名未变 | 数据没变才敢复用 |
| 清洗产物目录存在（未跳过清洗时） | 只复用 JSON 但目录被删是骗人 |
| `verification.ok is not False` | 上次清洗明确失败则不复用 |

注意最后一条用的是 `is not False` 而不是 `is True`：空 `verification` 视为可信，只有明确失败才拒绝。这是对旧版本 manifest 的宽容设计。

增量预检整体包在 `try/except` 里，失败只 warning 后走全量——**增量是优化，不是正确性依赖**。

### 2.4 阶段级异常边界

| 阶段 | 失败后的行为 | 返回码 |
|---|---|---|
| loader | 写 `failure_loader.json` | 2 |
| validator | 写 `failure_validator.json` | 3 |
| policy | 写 `failure_policy.json` | 4 |
| reporter | 写 `failure_reporter.json` | 5 |
| cleaner | `keep_going=True`，继续点云/可视化 | — |
| pointcloud | `keep_going=True` | — |
| visualizer | `keep_going=True` | — |

`_write_failure()` 把 `stage / error_type / error / traceback / recoverable` 写进 `intermediate/failure_<stage>.json`，连写失败都只 debug 不抛。

这是"报告优先"原则：cleaner 崩了不能把已经算好的质检报告一起陪葬。

### 2.5 点云阶段的数据源选择

```python
pc_source_root = Path(cleaning_dict["output_path"]) if cleaning_ok else dataset_path
pc_patch_in_place = bool(cleaning_dict and cleaning_dict.get("output_path"))
```

为什么点云要作用在清洗产物上：

- 点云清洗的契约是"只改点、不改行"；
- 如果先修点云再删 episode，行号会整体错位；
- 先删完 episode，`(episode_index, frame_index)` 就固定了。

`pc_patch_in_place` 决定 cleaner 是原地补丁还是独立复制：有清洗产物时直接改它，避免再拷一份几十 GB 的数据。

### 2.6 配置优先级

```
DEFAULT_CONFIG  <  配置文件 defaults  <  配置文件 robots[robot_type]  <  CLI overrides
```

`_build_config()` 里有一个容易忽略的细节：

```python
if args.min_frames is not None or args.max_frames is not None:
    overrides["frame_count"] = {"mode": "fixed", ...}
```

CLI 的 `--min-frames` 不只是改数字，而是**把模式切成 fixed**。否则自适应模式下的 `floor/ceiling` 只是安全边界，用户填 100 仍会被 median±MAD 夹住，看起来"参数没生效"。

### 2.7 设计取舍

- 用"阶段级 try/except"而不是全局 try：能精确知道哪一步坏了。
- 中间产物全部落 `intermediate/`：调试时可以只重跑某一阶段。
- 点云与可视化失败不影响退出码：它们是增值功能，不是数据可用性的前提。

---

## 3. loader.py — 只读视图

### 3.1 `LeRobotDatasetView` 字段

它不是"数据集对象"，而是"已读快照 + 一组延迟读取方法"：

| 字段 | 作用 |
|---|---|
| `info / stats / features` | meta 原样保留，报告和清洗都要用 |
| `episodes` | episode 级元数据（length、视频窗口、来源文件） |
| `frames` | 长表，可能为空（流式/只读元数据） |
| `video_index` | `{camera: {(chunk, file): path}}` |
| `data_files` | 流式扫描顺序 |
| `heavy_features` | 检测到但不读入内存的点云/embedding 列 |
| `corrupted_files` | 读失败的文件，供 validator 标 `episode_read_error` |
| `truncated_to` | `--max_episodes` 生效时的真实截断数 |

`frames` 允许为空，是整套流式设计的地基。所有属性都写成"有 frames 用 frames，没有就退回 meta"：

```python
@property
def num_frames(self):
    if len(self.frames):
        return int(len(self.frames))
    return int(self.frames_by_episode().sum())

@property
def column_names(self):
    if len(self.frames):
        return [c for c in self.frames.columns if c != SOURCE_COL]
    return list(dict.fromkeys([*self.features.keys(), *CORE_FRAME_COLUMNS]))
```

这样流式模式下 reporter / policy 不用改代码，只是拿到聚合值。

### 3.2 `SOURCE_COL`：为什么每行要记来源文件

`_read_frames()` 给每行打 `__source_file` 标记。

不这么写会怎样：v3 一个 parquet 混了多个 episode，删 episode 后若按 episode 重写，会把原文件拆成很多小文件，meta 的 `data/file_index` 全部要重算；pandas 往返还会破坏 `fixed_size_list` 的 schema。保留来源文件，就能"原文件 → 原文件"整块过滤。

### 3.3 重型特征阈值

```python
HEAVY_FEATURE_MIN_VALUES = 1000
```

`heavy_feature_keys()` 判断 `shape=[N, D]` 且 `N*D >= 1000`。512×6=3072 → 不读。

原因：pandas 读二维 `Array2D` 会变成"每行一个 ndarray"的 object 列，Python 对象开销可达原始数据的数十倍；主质检流程用不到点云，点云阶段自己流式读。

### 3.4 毒数据防冲击

三处独立的容错：

1. loader：单文件读失败 → 记 `corrupted_files` 继续；只有所有文件都读不出来才抛 `DatasetFormatError`
2. validator：meta 里有、frames 里没有的 episode → `episode_read_error`
3. pointcloud scanner：单文件/单 row group 失败 → 记 `corrupted` 继续

真实数据集的常见故障是"个别 mp4/parquet 截断"，不是整盘坏。一个坏文件不该让 999 个好 episode 无法质检。

### 3.5 探测式枚举

`_discover_data_files()` 按 `info.json` 模板顺序探测：

```python
for ci in range(chunks):
    for fi in range(files_per_chunk):
        path = root / tmpl.format(chunk_index=ci, file_index=fi, episode_index=fi)
        if path.is_file():
            found.append(path)
        elif fi > 0 and not _has_next(...):
            break
```

而不是 `rglob("*.parquet")` 全盘扫描。大数据集目录可能有几十万文件，全盘 glob 在机械盘上比读数据还慢；模板探测复杂度是 O(实际文件数 + chunk 数)。

### 3.6 `_read_tasks` 的三套兼容

依次尝试：

1. `tasks.parquet` 标准列 `task_index/task`
2. index 是任务文本、列里只有 `task_index` 的变体
3. `tasks.jsonl`

不是过度兼容，而是真实导出工具存在这些版本差异；少一种，`missing_task` 就会把整个数据集误判成"任务缺失"。

### 3.7 `iter_frame_chunks`

流式模式的分片迭代器：

```python
for path in dataset.data_files:
    table = pq.read_table(path, columns=wanted,
                          filters=[("episode_index", "in", sorted(keep)])
    yield table.to_pandas()
```

只读当前文件需要的列，内存峰值约等于单个 parquet 文件；`filters` 让 pyarrow 在 row group 级别裁剪，不把整文件解压进内存。

### 3.8 设计取舍

- 不做内存缓存：数据集动辄几十 GB，缓存等于把问题推后。
- `frames` 可空 + 属性自动降级：一份代码同时支持三种模式（全量、流式、仅 meta）。
- 读取容错但不静默：所有跳过的文件都进 `corrupted_files`，最终写进报告。

---

## 4. config.py — 阈值分层

### 4.1 合并顺序

```
DEFAULT_CONFIG
  <- 配置文件 defaults
  <- 配置文件 robots[robot_type]
  <- CLI overrides
```

`load_config()` 最终返回：

```python
{
  "defaults": effective,      # cfg_get 只读这里
  "severity": {...},          # severity_of 只读顶层
  "version": ...,
  "robot_type": ...
}
```

### 4.2 为什么 severity 不放进 defaults

severity 是**业务处置语义**，不是数值参数：

- 把它放进 defaults 会被 robot 段的深层 merge 意外覆盖；
- 报告里要单独快照一份，便于审计"为什么这条被判 error"。

同一个 `nan_action`，A 机器人可以是 error、B 机器人是 warning，但阈值仍共用。

---

## 5. validator 包

### 5.1 types.py — 数据结构

`EpisodeResult` 是"理由 + 证据"的最小单元：

```python
def add(self, code, **evidence):
    if code not in self.reasons:
        self.reasons.append(code)
    self.details.setdefault(code, {}).update(evidence)
```

- `code` 决定严重度（config 的 `severity` 表）
- `evidence` 决定能不能修（例如 `nan_state` 里带 `trimmable/head_frames/tail_frames`）

如果只记录"有 NaN"，下游 policy 只能整条删除；记录边界信息后，才能翻译成"裁头尾"。

`trim_plan()` 取 `nan_state` 与 `nan_action` 中**最大的 head/tail**——两个字段的坏帧边界可能不同，取并集才不会裁完还残留坏帧。

`ValidationResult.severity_of()`：

```python
levels = [severity_of(config, code) for code in res.reasons]
return max(levels, key=SEVERITY_ORDER.get)
```

一条 episode 多个问题时取最严重级别。处置语义上，只要有一个 error 就不能用，不能因为有 info 就降级。

### 5.2 context.py — 数据集级上下文

#### 帧数区间：median ± k·MAD

```python
mad = median(|counts - median|)
sigma = 1.4826 * mad
low  = max(floor, median - k_low * sigma)
high = min(ceiling, median + k_high * sigma)
```

细节：

1. **1.4826**：把 MAD 换算成正态分布下的 σ 估计，量纲才对得上。
2. **k_low=3 / k_high=8 不对称**：短 episode 基本是废数据（学不到时序）；长 episode 可能只是任务长或没切分，上界应宽松。
3. **`floor/ceiling` 再夹一次**：MAD=0（所有 episode 帧数相同）时 σ=0，区间退化成单点，必须靠 floor/ceiling 兜底。

#### action 阈值：按"每集最大 |Δ|"而非逐帧

```python
peaks = [max(|diff(arr)|, axis=0) for each episode]
if len(peaks) >= min_episodes_for_peak:
    med = median(peaks, axis=0)
    mad = median(|peaks - med|, axis=0)
    thr = max(abs_floor, med + k * 1.4826 * max(mad, margin * med))
```

真实 action 常是"保持 N 帧 → 跳一步"的步进指令，逐帧 |Δ| 分布是双峰（大量 0 + 少量大跳）。对逐帧 Δ 求 median 会被 0 拉成接近 0，任何正常运动都判异常。

改成"每集峰值"后，阈值描述的是**这个数据集里正常 episode 的运动包络**，只抓明显超出同类 episode 的。

两个退化处理：

- `len(peaks) < 8`：n 太小时 MAD 退化成极差 → 退回"逐帧非零 Δ"，并剔除 0（`col[col > 0]`）。
- `mad == 0`：阈值正好落在典型峰值上，float32 舍入噪声就会误报 → `spread = max(mad, 5% * med)`。

#### `_align_dims`

不同 episode 的 action 宽度不一致（混合机器人、合并数据集）时，`np.vstack` 会抛 `ValueError` 让整次质检中断。这里按**多数维度**保留，少数派交给 episode 级规则单独裁决。

#### 重复指纹

```python
quantized = round(nan_to_num(action), 3).astype(float32)
sha1(quantized.tobytes())[:16] -> 首次出现的 episode
```

量化到 3 位小数容忍浮点存储差异；用 `setdefault` 保留首次出现的 episode，后续重复项都指回同一条原始轨迹。

#### stats 时效性

比较 `stats.json` 的 mean 与实测 state 均值，相对偏差超 5% → `stats_stale`。训练归一化会整体偏移，所以这是**数据集级**问题，不是某个 episode 的问题。

#### index 连续性

判据：

```python
count == rows and min == 0 and max == rows - 1
```

N 个互不相同的整数如果铺满这个范围，必然就是 0..N-1，所以流式模式只统计四个标量就够了。

### 5.3 rules.py — 逐条规则

#### `check_finite`：裁剪判定

```python
head = 从前往后连续坏帧数
tail = 从后往前连续坏帧数
boundary_only = (head + tail) == bad.size
trimmable = boundary_only
            and max(head, tail) / n <= 0.25
            and (head + tail) < n
```

四个条件缺一不可：

- `boundary_only`：坏帧必须全部集中在首尾；中间夹一个 NaN，裁头裁尾都救不了。
- `max_ratio <= 0.25`：最多砍 1/4，否则保留下来的数据也不够训练。
- `head + tail < n`：不能裁空。
- `boundary_trim.enabled`：业务可关。

满足时才写 `trimmable=True` 与 `head_frames/tail_frames`，这是 policy → cleaner 的唯一通道。

#### `check_action_jump`：离散维必须放过

```python
nunique[d] <= discrete_max_unique(3) -> deltas[:, d] = 0
```

夹爪是 ±1 的二值指令，开合本身就是 2.0 的"跳变"，按连续维阈值必然误报。

`deltas = nan_to_num(deltas, nan=0, posinf=inf)`：

- Inf 要保留成 Inf 才会被判超阈值；
- NaN 不能参与 `argmax`。

#### `check_static`

```python
std = mean(std(finite, axis=0))
```

对每一维求 std 再平均，而不是把矩阵拉平求 std——后者会被某一维的大幅值主导，掩盖其他维的空转。

#### 时间戳的两种约定

`_check_timestamp_drift()`：

```python
if abs(ts[0]) <= tol:
    return   # episode 相对时间轴，绝对位置不可比
```

LeRobot v3 有两种约定：

- `timestamp` 从 episode 起点 0 开始：视频窗是**文件内绝对时间**，两者没有可比的绝对位置 → 必须跳过位置检查，否则多个 episode 拼一个 mp4 时整条误报。
- `timestamp` 是文件内全局时间：直接与 `from/to` 比较。

`_check_video_span()` 单独查"时长对不对"，和"位置对不对"互不重叠：帧数相等 ≠ 时间对齐，`from/to` 写错时帧数校验照样通过，但取出来的是另一段时间的画面。

`_check_frame_index_gap()` 要求 `frame_index == arange(n)`。视频按 `frame_index` 切窗口，断裂会导致画面与 state 错位。

#### `check_frame_mismatch`：先判截断，再判数量

```python
if from_ts >= 0 and to_ts > from_ts:
    start, end = round(from_ts * fps), round(to_ts * fps)
    if end > total + tol:      -> video_truncated
    elif abs(seg - n) > tol:   -> count_mismatch
```

顺序很重要：`end > total` 说明视频根本没录完，此时报"数量不一致"会误导清洗方向；先报截断，人工才知道要重录。

#### `check_blank_video`

内部一次 ffmpeg 抽多帧：

```python
ffmpeg -ss start -t dur -i path -vf "fps=2,scale=64:48" -f rawvideo -pix_fmt gray -
```

一次调用抽 2fps 的 64×48 灰度帧，算每帧亮度 std；`bad_ratio >= 0.6` 才判黑屏。逐帧起 subprocess 会把视频探针变成分钟级。

### 5.4 validator/__init__.py — 两阶段调度

#### 阶段一：纯 CPU 规则

- `DatasetContext.build()` 先算全局阈值与指纹
- 逐 episode（或并行）跑 `_validate_non_video`
- `prior/only` 复用：未变更 episode 直接 `from_dict`
- **但必须重算 duplicate**，因为重复是全局性质，别的 episode 增删会改变判定
- 单 episode 异常 → `episode_read_error`，不中断整体
- meta 有但没读到的 episode → `missing_eps` 兜底

#### 阶段二：视频探针

```python
want_blank = blank_enabled and len(pending_video) < max_probes   # 默认 400
```

为什么限 400：每个 episode 一次 ffmpeg 调用，400 次已是分钟级；不限制则 2 万 episode 要跑几小时。

`_run_video_checks` 用 `ThreadPoolExecutor`，默认 2 路：

- 视频探针是 IO/进程等待型，不是 CPU 型，所以用线程不是进程；
- 机械盘上并发解码会来回寻道，2 路是保守上限；
- `probes.py` 里每个视频文件有 `_KEY_LOCKS`，同一文件只允许一个线程探针。

#### 并行结果的确定性

```python
for ep_index in sorted(rechecked):
    pending_video.append(...)
results = dict(sorted(results.items()))
```

Future 完成顺序不确定，但 blank 探针名额是**按 episode 升序分配**的。不排序的话同样数据跑两次，哪些 episode 被抽样会不同，报告不可复现。

#### 阈值漂移导致增量失效

```python
if prior is not None and not _thresholds_equal(
    validation.resolved_thresholds, manifest.resolved_thresholds()
):
    validation = validate_dataset(dataset, config, ...)   # 回退全量
```

即使 episode 签名没变，**其他 episode 增删也会改变 median/MAD**，从而改变阈值。阈值一变，旧结论必须作废。

### 5.5 streaming.py — 两遍扫描

Pass 1 累计（必须全量）：

| 累计对象 | 为什么必须在第一遍 |
|---|---|
| 每 episode 真实行数 | 帧数区间需要全量分布 |
| action 阈值 | 需要跨 episode 的峰值分布 |
| stats 时效性 | 需要 state 均值 |
| index 连续性 | 只需 rows/count/min/max/NaN |
| 重复指纹 | 必须覆盖全量才能识别跨文件重复 |

一个容易看漏的优化：

```python
pass1_columns = [c for c in PASS1_COLUMNS
                 if c in columns and (c != STATE_KEY or state_sample)]
```

不需要判 stats 时效性时，state 整列都不读——50GB 级别少读一半列宽差距明显。

Pass 2 直接复用内存模式的 `_validate_non_video`，保证规则和严重度完全一致；同时用 `_FieldStats` 累计 min/max/mean/NaN/Inf，产物字段与 reporter 内存版一致。

与内存模式**唯一的语义差异**：action 阈值来自采样（默认前 200 个 episode）。样本足够大时 median/MAD 已稳定，且 `resolved_thresholds` 会记录 `basis=...`。

---

## 5.6 parallel.py — 并发契约

`run_pool()` 顶部注释就是设计文档：

1. **worker 是纯函数**：只读输入、返回可 pickle 结果，绝不写盘 → 多进程写冲突在设计上不存在。
2. **主进程统一落盘**：结果回收用 `as_completed` 的 Future 返回值，不需要 `Manager().Queue()` 转发层。
3. **共享上下文只分发一次**：配置/阈值/指纹经 `initializer/initargs` 注入子进程全局变量，不随每个任务重复 pickle。
4. **结果与顺序无关**：调用方按业务 key 归并并排序，保证并发与串行逐位一致。
5. **Windows spawn 兼容**：worker 必须是模块顶层函数。

```python
workers = max(1, min(int(workers), len(job_list), os.cpu_count() or 1))
```

避免起 64 个进程跑 10 个任务，白付 spawn 成本。

---

## 6. policy.py — 处置优先级

`_decide_one()` 的顺序不是随意的：

```
1. 无 reasons                                   -> keep
2. 有 trim_plan 且 reasons ⊆ {nan_state, nan_action} -> repair/trim
3. 所有 reasons ∈ REPAIR_ACTIONS                 -> repair
4. severity == error                            -> drop
5. severity == info                             -> keep
6. severity == warning                          -> review（或 --on-review drop）
```

逐步解释：

1. **裁剪优先**：边界坏帧最常见、最值得救，且是唯一"修完仍保留大部分数据"的动作。
2. **纯可修复项**：`timestamp_not_monotonic → timestamps`、`meta_index_inconsistent → index`。必须 `all(...)`，混一个不可修项就不能整体标 repair。
3. **error 才 drop**：`nan_action` 中段、`blank_video`、`static_episode` 都无法插值或没有信息量，留着只会污染训练。
4. **info 只记录**：`timestamp_step_deviation` 是掉帧抖动，不影响可用性。
5. **warning 交人**：`action_jump` 可能是真实动作突变，机器无法判断。

overrides 回灌时做三件事：

- 校验取值合法；
- 切 `source=override`；
- **如果改成非 repair，清空 repair/trim**。

最后一条很关键：否则人工把某条改成 `keep`，cleaner 还会按旧 trim 方案去裁帧。

`summary` 中 `kept_after_cleaning = keep + repair + review`：review 默认不删，所以算"保留"。

---

## 7. reporter.py — 报告数字的来源

### 7.1 `remaining_frames` 为什么扣 trim

```python
drop_frames   = drop 的 episode 帧数之和
trim_frames   = 所有 repair 的 head + tail
remaining_frames = total_frames - drop_frames - trim_frames
```

只扣 drop 的话，报告说"剩余 10000 帧"，清洗产物实际只有 9800 帧，下游会以为清洗丢数据了。

### 7.2 `frames_by_episode` 的三条来源

1. 内存模式：`dataset.frames.groupby("episode_index").size()`
2. 流式模式：`validation.aggregates["frames_by_episode"]`
3. 仅 meta：`dataset.frames_by_episode()`（meta 的 length 列）

三条路径产出同样字段，报告不需要知道当前模式。

### 7.3 为什么报告分两次写

```
build_report -> write_report -> JSON 落盘
   ... cleaner 跑完 ...
attach_cleaning_result -> 重读 JSON、改 cleaning、再写回
attach_artifacts -> 最后补可视化/复核队列路径
```

原因是失败隔离：报告必须**在 cleaner 之前**落盘。如果等所有阶段跑完再一次性写，cleaner 崩了就没有报告。代价是两次 JSON I/O，可接受。

### 7.4 归因分析

`attribute_by_metadata()` 只挑：

- dtype 为 object/unicode/string；
- `2 <= nunique <= 20`；
- 排除 `episode_index/length/tasks/dataset_*_index/data/*/videos/*/stats/*`。

限制 `nunique <= 20` 的原因：高基数列（时间戳、UUID）分组后每组 1 条，异常率没有统计意义。

### 7.5 中文字宽

`_display_width()` 把 `east_asian_width in "WF"` 的字符算 2 格。不这么写，终端表格中文列会全部错位。

---

## 8. cleaner 包

### 8.1 入口的三道闸

```python
out_root = output_dir / CLEANED_DIRNAME
assert_safe_output(dataset.root, out_root)
```

`assert_safe_output()` 拒绝三种配置："输出等于源"、"输出位于源内部"，以及反向的"输出包含源"——`publish_dir` 发布时会把最终路径整体改名成 `.prev-*` 再 `rmtree`，输出目录是源的祖先时，被删掉的就是源数据集本身。

`_parse_dispositions()` 把 `Decision` 归一成四个集合：

```
dropped / trim_plan / rebuild_ts / repair_index
```

`repair_index=True` 的条件是 `repair == "index"` 且没有裁帧/时间轴修复；同时 `_has_index_issue(dataset_issues)` 也会置位——全局 index 断裂是数据集级问题，即使没有任何 episode 需要修也要重建。

### 8.2 四条分支

| 条件 | 路径 | 原因 |
|---|---|---|
| 无删除、无修复、无 index 问题 | `copy_untouched` | 仍产出结构完整副本并重算 stats |
| 无删除、无修复、有 index 问题 | `reindex_untouched` | 只改 index，不重编码视频 |
| 有删除、无修复 | `try_cli` | 官方 CLI 只支持 delete_episodes |
| 有任何 repair | `manual_clean` | CLI 不能修补，必须走内置 |

`try_cli` 先跑 `--help` 检查 `"delete_episodes" in help_text`。不做探测，老版本 CLI 会直接报"未知操作"，再回退重试，浪费时间且日志混乱。

### 8.3 `manual_clean` 的执行顺序

```
1. _collect_video_windows   （裁帧时按 fps 收窄 from/to）
2. _rewrite_frames          （重写 data parquet + 累计 stats）
3. _rebuild_videos          （按相机重建 mp4）
4. rewrite_episodes_meta    （新 episode_index / length / from_index / 视频窗口）
5. copy_tasks
6. rewrite_info             （total_episodes / total_frames / splits）
7. rewrite_stats_streaming  （用第 2 步累计的统计量）
```

顺序不能换：meta 的 `length` 和 `dataset_from_index` 依赖 `_rewrite_frames` 的返回值；视频窗口依赖 `kept_lengths`。

### 8.4 `_rewrite_frames` 的关键操作

```python
mask = isin(ep_values, kept)
sub  = table.take(positions)          # 只保留 kept 行
new_eps = ep_index_map[old_eps]       # 重编号 0..K-1
```

裁帧用**行内序号**而不是 `frame_index`：

```python
keep_row[i] = head <= row_pos[i] < total - tail
```

因为 `frame_index` 本身可能就是脏的（这正是要修的问题之一）；用行序更可靠。

然后：

```python
frame_idx = running_frames[ep]++     # 跨文件连续
index     = global_index++           # 全局连续
timestamp = frame_idx / fps          # 仅对 rebuild_ts 的 episode
```

`running_frames` 跨文件保留，因为 v3 一个 episode 可能横跨多个 parquet；只用文件内序号会在文件边界断掉。

### 8.5 `_rebuild_videos` 的两条路径

**整文件复制**判定：

```python
whole_file_kept and window_covers_file and 没有收窄的 to
```

即"源文件里的 episode 全保留、窗口从 0 开始、没有被裁帧收窄" → `shutil.copy2`。这是大多数情况，避免无谓重编码。

**否则切片拼接**：

1. 每个 episode 用 `ffmpeg -ss from -t dur -c:v libx264 -crf 20` 抽成独立片段；
2. `-f concat -c copy` 无损拼接；
3. 单个片段失败 → 记录 warning 继续；
4. 全部失败 → 原样复制源文件 + warning。

最后一步是有意的降级：宁可视频里残留已删除 episode 的画面，也不让整个清洗失败。`warnings` 会如实告诉用户"画面会残留脏 episode"。

### 8.6 meta 重建

`rewrite_episodes_meta()`：

- 读所有 `meta/episodes/**/*.parquet` → `concat_tables`；
- 按 `kept` 过滤、按新 `ep_index_map` 排序；
- 重写 `episode_index/length/dataset_from_index/dataset_to_index/data/*`；
- 每个相机重写 `videos/<cam>/chunk_index/file_index/from_timestamp/to_timestamp`。

缺失相机信息时写 `file=0, from=-1, to=-1`，而不是删列——LeRobot 的 schema 要求这些列存在。

`rewrite_info()` 把 `total_episodes/total_frames/total_tasks/splits` 改成实际值；不改的话下游读 info 会以为还有被删的 episode。

### 8.7 产出物自检

```python
ok = num_episodes == expected and (expected_frames is None or num_frames == expected)
contiguous = episode_indices == list(range(num_episodes))
```

**重新用 loader 加载产出物**，而不是相信自己写的数。这是唯一能发现"meta 和 parquet 对不上"的方式。

校验失败不抛异常，只 `warnings.append` + `verification.ok=False`，由 main 在收尾时提示。

### 8.8 stats 重算

`stats_streaming` 用 sum / sumsq 增量累计：

```
mean = sum / n
std  = sqrt(sumsq / n - mean^2)
```

不把整列读进内存求分位数；分位数沿用旧值，因为影响归一化的主要是 mean/std/min/max。

---

## 9. pointcloud 包

### 9.1 为什么"只改点、不改行"

点云是逐帧模态，删一行会让 `state/action/视频` 全部错位；而 LeRobot 的 `Array2D` 特征声明了固定 `[N, D]`，修完还必须保持 N 点。

所以处置只有三种：

- `repair`：过滤/去重/裁剪/重采样，**行数不变、点数不变**
- `drop`：不是删行，而是用同 episode 最近的好帧**占位**（hold-last-good）
- `review`：只进复核队列，不改数据

### 9.2 自适应上下文

```python
min_points   = max(16, ceil(N * 0.02))
radius       = median(每帧 1-NN 距离) * 2.5
workspace    = 样本 0.5%/99.5% 分位数 ± 10% 对角线
color_scale  = 255 if max(color) > 1.5 else 1
```

四个阈值都不是硬编码坐标：

- `min_points` 按总点数比例；512 点云至少 11 点，取 max(16, ...)
- `outlier_radius` 按局部密度：先算"每个点到最近邻距离"的中位数，乘 2.5。不同场景点密度差十倍时，绝对半径必然失效
- `workspace` 用分位数而不是 min/max，避免单个飞点把包围盒撑爆
- `color_scale` 用 `max > 1.5` 判 0~1 还是 0~255

### 9.3 8 条规则的执行顺序

```
空帧 -> 直接 sparse，返回
NaN/Inf -> 记录，但继续用 finite 子集算
有效点 < min_points -> sparse，提前返回
bbox 对角线 ≈ 0 -> degenerate，提前返回
重复点比例 / 离群比例 / 越界比例 / 颜色越界
```

两个提前返回是有意的：点太少或几何退化时，后续比例统计没有意义（分母太小，任何噪声都是 100%）。

`neighbor_counts()` 按行分块：

```python
block = max(1, 4_000_000 // n)
```

N×N 距离矩阵是 O(N²) 内存；分块把峰值内存压到常量级。

`apply_temporal_rule()`：

- 先剔除不可靠帧（sparse/degenerate/NaN/outlier/out_of_bounds），它们的质心本身不可信；
- `threshold = max(0.05, median(jump) * 6)`；
- 只标**后一帧**（`frames[i+1]`），因为跳变是"从 i 到 i+1"的事件；
- 只在扫描归并后统一跑，因为跨帧依赖。

### 9.4 扫描与缓存

`scan()` 逐 batch 读，只保留 `FrameResult`，不保留点云。

`scan_parallel()` 按 `(文件, row group)` 切任务，worker 纯函数：

```python
return {"file", "results", "corrupted"}
```

读取失败返回 `corrupted` 而不是抛异常——避免一个坏 row group 拖垮整个进程池。

`ScanCache` 命中条件：

```python
envelope = {version, config_digest, feature_key, resolved}
key = sha1({path, size, mtime_ns})
```

`resolved` 里是自适应阈值（半径、包围盒、颜色量程）。**数据分布变了，阈值就变，旧结论必须失效**——不能只比 config digest。文件指纹用 stat 而非内容哈希，因为点云 parquet 动辄 GB，全量哈希比重扫还慢。

### 9.5 处置

```
无 reasons                          -> keep
sparse / degenerate                -> drop
有不可修复项（temporal_jump 等）    -> review
坏点比例 > 35%                      -> drop
否则                                -> repair
```

坏点比例取 `max(non_finite_ratio, outlier_ratio + duplicate_ratio)`。不把三者相加：NaN 与离群点可能是同一批点，相加会重复计算。

### 9.6 修复算法

```
filter: 去掉 NaN/Inf + 统计滤波去离群
bounds: 裁到工作空间
dedupe: 按坐标四舍五入去重
color:  clip 到 [0, color_scale]
补/裁到 expected_points
```

补点：

```python
rng  = default_rng(ep * 100000 + fi)       # 固定种子，可复现
idx  = arange(target) % kept               # 循环取已有好点
pts  = pts[idx[order]]                     # 随机打乱
pts[:, :3] += normal(0, 0.005 * scale)     # 抖动，避免完全重复
```

为什么循环取点 + 抖动：不能凭空生成几何，只能在已有好点附近采样；抖动半径用 `outlier_nn_median`（局部点间距）而非绝对坐标，保证不同尺度场景一致。

写回时：

- 修复帧 → 写新点，并更新 `last_good[ep]`
- drop 帧 → 用 `last_good[ep]`；没有好帧就保留原值 + warning
- 其他帧 → 原样写，若全 finite 也更新 `last_good`

`last_good` 是 **per episode**，不能跨 episode 借，否则会把别的场景的点云塞进来。

最后 `_rewrite_pointcloud_stats()` 用 sum/sumsq 增量算 mean/std，分位数沿用旧值——精确分位数需要全量排序，而影响归一化的是 mean/std/min/max。

---

## 10. manifest.py — 增量指纹

### 10.1 episode 签名记了什么

```
episode_index + length + from_index + to_index
+ data 文件 stat
+ 每个相机的视频文件 stat + from/to 窗口
```

`_stat()`：

```python
{"path", "exists", "size", "mtime_ns"}
+ 小文件（< 1MB）额外 sha256 内容哈希
```

为什么小文件才做内容哈希：meta 文件小，被篡改/损坏时 stat 可能不变；大数据文件全量哈希成本接近重跑。

### 10.2 清洗签名

`cleaning_signature()` 把"处置 + repair/trim + 每 episode 签名"整体哈希。

`reusable_cleaning()` 还额外要求**产物目录仍存在且 verification 未失败**——签名一致但目录被删，复用就是空指针。

---

## 11. visualizer 与辅助脚本

- `visualizer.py`：按 `--plots-sample` 采样 episode，输出 state/action 曲线、视频帧拼图到 `plots/`。失败被视为非致命。
- `precheck.py`：不跑完整 pipeline 的轻量预检（meta/文件数/体积），用于拿到数据集先判断值不值得跑。
- `tuning.py`：阈值调参辅助，基于历史报告分布给出建议。
- `generate_report.py`：把 `quality_report.json` 转成人类可读摘要。
- `tools/dashboard_server.py`：本地 dashboard，读取报告 JSON 做可视化浏览。

这些脚本都只读中间产物，不参与数据正确性链路。

---

## 12. 中间产物契约

| 文件 | 产出阶段 | 关键字段 |
|---|---|---|
| `intermediate/01_dataset_summary.json` | loader | episodes/frames/schema/video_features |
| `intermediate/02_validation_result.json` | validator | results/reasons/details/resolved_thresholds |
| `intermediate/03_decisions.json` | policy | 每条 episode 的 disposition/repair/trim/source |
| `quality_report.json` | reporter | summary/checks/episodes/attribution/cleaning |
| `review_queue.csv` | policy | 待人工复核项 |
| `pointcloud_scan_cache.json` | pointcloud | 帧级扫描结果 + envelope + 文件指纹 |
| `pointcloud_review_queue.csv` | pointcloud | 点云待复核项 |
| `intermediate/failure_*.json` | 任一阶段 | stage/error/type/traceback/recoverable |
| `quality_manifest.json` | 收尾 | 配置摘要/episode 签名/清洗签名/verification |

约定：**JSON 只追加不覆盖语义**（同一路径分阶段重写时保留其它键），失败文件独立命名，避免"一次失败抹掉上次成功记录"。

---

## 13. 四个反复出现的设计模式

**1. 只读 + 派生**
所有写操作落在 `output_dir`，源数据集严格只读。`assert_safe_output`、`SOURCE_COL`、manifest 放 output，都是同一原则的不同落点。

**2. 自适应阈值 + 绝对下限兜底**
帧数、action、点云半径、工作空间全部由数据分布推导，配置里的 `floor/abs_floor` 只在下界兜底。不同机器人单位不同，绝对阈值必然误报。

**3. 分级处置，能修不删**
`keep/repair/review/drop` 四档：裁剪优先于删除、修复优先于剔除、warning 交人而不是机器删。删数据不可逆，宁可多一步人工。

**4. 每层都可单独失败**
loader 单文件坏 → 标记；validator 单 episode 崩 → 标记；视频探针失败 → 跳过；清洗失败 → 报告仍在；点云失败 → 主链路继续。所有降级都写进 `warnings` / `skipped_checks` / `failure_*.json`，**不静默**。

---

## 14. 建议阅读顺序

1. `pipeline/loader.py::load_dataset` → 看 `LeRobotDatasetView` 的属性降级
2. `pipeline/validator/context.py::_thresholds_from_arrays` → 理解阈值怎么来
3. `pipeline/validator/__init__.py::_validate_non_video` → 看规则的调用顺序
4. `pipeline/validator/rules.py::check_finite` → 看"可修性"怎么被编码进证据
5. `pipeline/policy.py::_decide_one` → 看证据怎么变成动作
6. `pipeline/cleaner/manual.py::_rewrite_frames` → 看动作怎么落到 parquet
7. `pipeline/pointcloud/rules.py::evaluate_frame` → 看另一套逐帧契约

---

## 15. 常见误解

| 误解 | 事实 |
|---|---|
| "阈值是拍脑袋的固定值" | 除 `floor/ceiling/abs_floor` 外全部由数据集分布推导 |
| "warning 会被自动删" | 默认 `review`，只有 `--on-review drop` 才删 |
| "清洗会改源数据" | 源数据只读，产物在 `outputs/cleaned` |
| "点云清洗会删掉坏帧" | 不删行，用同 episode 的 last-good 帧占位 |
| "增量模式会漏检" | 阈值变化会触发全量回退；重复检测每次重算 |
| "流式模式结果和内存模式不同" | 规则实现复用，只有 action 阈值来自采样并在报告里注明 |
