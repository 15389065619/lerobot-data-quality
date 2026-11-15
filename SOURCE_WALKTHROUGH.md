# LeRobot v3 数据质检与清洗 Pipeline — 源码级拆解

> 目标：不只说"做了什么"，而是说清"代码为什么必须这么写、换一种写法会坏在哪"。
> 阅读方式：先看第 1 节的数据流，再按第 2～11 节顺序读；每节末尾都有「设计取舍」。
> 已同步到 `f66ae09` 之后的全部修复：schema 契约审计、episode 编号对账、meta 重复行、基线白名单、
> 帧数区间等长兜底、跨 parquet 裁帧、时间轴修复作用域、清洗产物归属校验、点云分片跳过语义等。

---

## 1. 全局数据流

```
main.py
 ├─ [0] QualityManifest.load / _manifest_fresh        增量快路径，可能直接 return 0
 ├─ [1] load_dataset            -> LeRobotDatasetView  只读内存视图 + schema 契约审计
 ├─ [2] validate_dataset        -> ValidationResult    逐 episode 结论 + 数据集级问题
 │      baseline.apply_whitelist                      摘除已确认误报（不改阈值）
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
| schema | view 的元数据 + 极值 | `schema_contract` 列表 | 不读整表 |
| validator | view + config | `ValidationResult` | 不改数据 |
| baseline | result + 白名单 | 摘除误报后的 result | 不改阈值、不删证据 |
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

### 2.4.1 基线白名单：插在 validator 与 policy 之间

```python
baseline_mod.apply_whitelist(validation, dataset_path.name, config)
if validation.baseline_ignored:
    save_intermediate(output_dir, "02b_baseline_ignored.json", validation.baseline_ignored)
```

位置不能挪：

- **必须在 policy 之前**：白名单是"确认过的误报"，摘除后不应再参与处置决策，否则人工标过的误报仍会被判 review，复核队列永远清不掉。
- **必须在 reporter 之前**：健康度按摘除后的 `reasons` 计算。
- **不能放在 validator 内部**：`ValidationResult` 是规则输出的纯结果，白名单属于"用户对结果的裁定"，混进 validator 会让缓存（manifest 复用的 episode 结论）与裁定耦合。

被摘除的告警不是丢弃，而是移到 `ignored_baseline` 并落 `02b_baseline_ignored.json`，报告里仍能追溯"哪条被忽略"。

`--accept-baseline <fingerprint>` 走的是另一条路径：`_accept_baseline()` 直接改写配置文件，不跑 pipeline。


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

### 3.8 schema 契约审计（schema.py）

`load_dataset()` 里调用 `audit_schema(...)`，结果写进 `view.schema_contract`。它**不读整表**，只用 meta 元数据 + 逐文件扫描出的两列极值（`idx_min / idx_max / idx_nunique`），所以几十 GB 也是秒级。

存在理由（[schema.py](file:///g:/Lerobot/pipeline/schema.py#L1-L9) 顶部注释）：很多"数据集能跑起来"并不等于 Schema 合规。`index` 列退化成"每集从 0 开始"时，全量读取不报错、吞吐正常，只在按 episode 过滤或开 `delta_timestamps` 时**静默取错帧**。

判据分三档：

1. **v2.x（每集独立 parquet）**：要求 `idx_max - idx_min + 1 == rows` 且 `nunique == rows`。起点可以是 0 也可以是全局偏移，两者都合法。
2. **v3.0（全局绝对帧号）**：`expect_from = [0, cumsum(rows)[:-1]]`，即每集 index 起点必须等于前序 episode 的累计行数。
   - 起点全为 0 → 单独报"取值范围仅 0..span-1 却承载 total 行"；
   - 起点序列正确但集内重复/断号互相抵消 → 用 `idx_nunique != rows` 补一刀。
3. **单 episode 数据集**：没有"前序累计"可比，但集内连续性与唯一性**单集即可证伪**，不能一并放过，否则 `--max_episodes 1` 时 index 损坏完全静默。

两个报错可读性细节：

- `_sample_episodes()` 用 summary 里的真实 `episode_index`，不用 DataFrame 位置。位置在 remap 或乱序后会指向别的 episode，把排查引偏。
- `_index_mismatch_evidence()` 报**首个**起点不符的 episode。只列前 3 个时，如果错位发生在后面，输出会变成"实际 [a,b,c]，应为 [a,b,c]"——两边一模一样，读者无从下手（so101_bomb ep 81 实测）。

`idx_nunique` 在"episode 跨多个 parquet"时记 `NaN`（合并全量集合代价太高），schema 侧遇到 NaN 就跳过唯一性判据，而不是猜一个值。

splits 也在这里校验：区间长度必须等于 `total_episodes`，否则按 split 加载会以为 episode 数就是区间长度。

### 3.9 episode 编号对账（`_reconcile_episode_ids`）

真实数据集里常见两侧编号体系不同：meta 用全局 ID（0, 2, 4…），data 用 0..N-1。不做对账，视频窗口和 task 会整体错配。

流程：

```
两侧集合相同            -> native，不映射
数量不一致              -> ambiguous（无法一一映射）
data 编号非升序          -> ambiguous（顺序配对不可信）
逐位置校验 length / 视频窗 -> agreement < 0.99 则 ambiguous
全部通过                -> remapped，返回 {data_id: meta_id}
```

关键设计：

- **三道前置检查都比"能不能映射"更保守**。数量不等或物理顺序乱序时，顺序配对的前提已经不成立，任何映射都是猜。
- **`agreement = min(length_agreement, video_agreement)`**，不是平均。任一侧证据不成立，说明顺序假设整体不成立；取平均会让一侧的高一致率掩盖另一侧的错配。
- **证据不足时宁可 ambiguous**：错配会把视频窗口和 task 全部错配，比不映射更危险（[loader.py](file:///g:/Lerobot/pipeline/loader.py#L1101-L1113)）。
- `length_arr` 只在 `len(lengths) == len(rows_arr)` 时启用。长度不等直接用会广播错位，得到一串假的一致率。
- `_data_episode_summary()` 的 episode 顺序按**物理行序首次出现**排序（`np.argsort(first_pos)`）。`np.unique` 默认升序，会把乱序的 data 伪装成"顺序正常"，让 `data_order_sorted` 失去唯一的兜底证据。

video 证据的算法是 `round((to_ts - from_ts) * fps) == rows`：窗口时长换算成帧数必须等于该集行数。这比单看 `length` 强，因为它同时验证了视频侧和帧表侧。

**ambiguous 的下游后果**（`LeRobotDatasetView.locator_available`，[loader.py](file:///g:/Lerobot/pipeline/loader.py#L103-L109)）：

```python
return str(self.id_reconciliation.get("mode", "native")) != "ambiguous"
```

一旦编号空间对不上且无法映射，`check_alignment` / `check_task` 直接 return，`frame_mismatch` 与 `missing_task` 写进 `skipped_checks` 并注明原因。

为什么不报 error：此时 meta 里根本没有该 episode 的视频窗口，定位必然失败。把它判成"帧数不符/任务缺失"会让**整批数据被误删**；正确语义是"不可评估"。


### 3.10 meta 重复行冲突（`_meta_duplicate_conflicts`）

`meta/episodes/**/*.parquet` 里同一个 `episode_index` 出现多行，本身可以安全 `drop_duplicates`。但**内容冲突**时不能静默去重：

```python
columns = ["length"] + [videos/*/from_timestamp, videos/*/to_timestamp]
若任一列在同一 episode 的多行间 nunique > 1 -> conflicts
```

典型来源是多次导出叠加。静默保留首行会让后续只看到"帧数不符"，把真正的原因（重复导出）掩盖掉——清洗方向会完全错。

结果经 `DatasetIssue("meta_episode_duplicate")`（severity=error）进报告，reporter 的 `_recommendations` 也会给出对应处置建议。

### 3.11 过滤顺序与 parquet 下推守卫

两处顺序性修复，都属于"看起来等价、实际会静默丢数据"的类型。

**类型归一必须在白名单过滤之前**（[loader.py](file:///g:/Lerobot/pipeline/loader.py#L946-L958)）：

```python
for col in ("episode_index", "frame_index", "task_index", "index"):
    out[col] = pd.to_numeric(out[col], errors="coerce")   # ← 先归一
if keep is not None:
    out = out[out["episode_index"].isin(keep)]            # ← 再过滤
```

字符串型 `episode_index` 与整数 `keep` 集合比较**永远不命中**，先过滤会把所有帧静默丢光，`--max_episodes` 下看起来就是"数据集是空的"。

**下推过滤只对整数列安全**：

```python
if keep is not None and _episode_index_is_integer(path):
    pq.read_table(path, filters=[("episode_index", "in", sorted(keep))])
```

`_episode_index_is_integer()` 用 `pq.read_schema` 看字段类型，非整数就退回全量读取。pyarrow 对字符串列做整数下推不报错，只是返回空集合。

### 3.12 meta 缺列兜底

`_normalize_episodes_meta()`：

- 缺 `episode_index` 列 → 告警后原样返回，不抛；
- `episode_index` 非法行丢弃并告警；
- `length` 用 `to_numeric(...).fillna(0)`，**缺列时不进这个分支**，避免 `KeyError`。

这些都是"meta 是人手/工具生成的，不可信"的体现：元数据缺列应当降级，而不是让整个数据集无法质检。

### 3.13 设计取舍

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
  "baseline_whitelist": [...],# baseline 只读顶层
  "version": ...,
  "robot_type": ...
}
```

### 4.2 为什么 severity 不放进 defaults

severity 是**业务处置语义**，不是数值参数：

- 把它放进 defaults 会被 robot 段的深层 merge 意外覆盖；
- 报告里要单独快照一份，便于审计"为什么这条被判 error"。

同一个 `nan_action`，A 机器人可以是 error、B 机器人是 warning，但阈值仍共用。

### 4.3 `baseline_whitelist` 也放顶层

白名单与 severity 同理，不能进 `defaults`：

- 它记录的是"这个具体数据集、这些具体 episode 的误报"，属于数据专属资产，不是通用参数；
- 放进 defaults 后会被 robot 段的 merge 覆盖，切换 `robot_type` 就丢配置；
- 它是**列表**，而 defaults 的合并语义是 dict 深合并，列表会被整体替换。

配置里还新增了 `frame_count.min_margin_ratio`（默认 0.05），配合 5.2 的"等长数据集"兜底使用。


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
sigma = 1.4826 * mad          # MAD=0 时另有兜底，见下文第 3 点
low  = max(floor, median - k_low * sigma)
high = min(ceiling, median + k_high * sigma)
```

细节：

1. **1.4826**：把 MAD 换算成正态分布下的 σ 估计，量纲才对得上。
2. **k_low=3 / k_high=8 不对称**：短 episode 基本是废数据（学不到时序）；长 episode 可能只是任务长或没切分，上界应宽松。
3. **MAD=0 要分两种情况**（当前实现）：

```python
if mad > 0:
    sigma = 1.4826 * mad
elif counts.size and np.ptp(counts) == 0:      # 所有 episode 完全等长
    sigma = 1.4826 * margin * abs(median)       # margin 默认 0.05
else:
    sigma = 0.0                                 # 有离散但仍 MAD=0
```

全部等长时 `sigma=0`，允许区间塌成单点，多一帧就判 error 并整条 drop——而"固定时长录制"是采集流水线的常态。此时补 5% 相对余量。

"存在离群值但 MAD=0"（100 集里 1 集超长、其余全等）必须保持塌缩：那唯一的长集就是真异常，补余量等于放过它。两种情形的判据不能合并。

最后再被 `floor/ceiling` 夹一次，防住极端值。

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

#### meta 重复行冲突

`_check_meta_duplicates()` 读取 loader 已经算好的 `view.meta_duplicate_conflicts`，非空就产出 `meta_episode_duplicate`（severity=error）。它放在数据集级而不是 episode 级，因为冲突证据来自 meta 文件本身，与某一集的帧数据无关。

#### 缺失列时的原因留痕

`_validate_non_video()` 在缺 `observation.state` 时不仅跳过 `nan_state`，还补 `joint_limit_violation` 的 skipped 原因；缺 `action` 时补 `action_jump` / `nan_action` / `duplicate_episode`。

不补的话报告里的"未执行检查"是一串裸 code，读者无法区分"没这列"和"检查被配置关掉"。


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

判定前还有两层降噪：

**1. 四元数整体取反不算跳变**

```python
quat_mask = _quaternion_dim_mask(dim_names, dim)   # 元数据可用时
mirror &= quat_mask[None, :]
flip_mask = mirror & (mirror.sum(axis=1, keepdims=True) >= flip_min_dims)
```

q 与 -q 表示同一姿态，但逐维差分会得到"所有维同时翻转"。只有**同一帧上多个维同时镜像翻转**才判为整体取反；单维翻转仍是真实跳变。

`dim_names` 来自 meta 的关节名：有元数据时只允许四元数维度参与，避免符号变化被宽泛地解释成"取反"而放过真实跳变；元数据缺失时才退回纯数值启发式。

**2. 邻域比较抑制连续快速运动**

```python
ref = median(邻域内其它帧的 |Δ|)
if ref <= 0 or deltas[f, d] >= spike_ratio * ref:
    保留为跳变
```

快速但连续的运动会同时抬高该维的 |Δ| 和它的邻域参考值，比值接近 1 而被过滤；真正的尖峰邻域几乎静止（`ref≈0`），一定被判为跳变。

`exceed_ratio` 还用于分级：阈值本身已是"每集峰值 3σ 上界"，1.0~1.5 倍命中大多属于正常快速动作（实测 pusht 17/206），按倍数分档可显著减少稳定误报。

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

**时间窗缺失时必须跳过，不能当黑屏**：

```python
if from_ts < 0:
    return   # v3 缺列时窗口是 (-1,-1)
```

`from_ts < 0` 表示 meta 给不出该集的视频时间窗。此时若从文件 0 点抽帧，实际判的是**同文件里别的 episode 的画面**，结论既不能证明这集黑屏、也不能证明它正常——属于"不可评估"，跳过是唯一正确选择。

#### `check_joint_limits`：越界维度的名字兜底

配置里的限位表可能比 `info.json` 声明的关节名更长（数据列比 info 宽是野外常见情况）。原实现用 `dataset.joint_names[d]` 直接索引，越界维会让**报错路径自己 IndexError**，整次质检失败。

现在：

```python
names[d] if d < len(names) else f"dim_{d}"
```

报错路径本身也必须不会崩——否则一个"发现问题"的规则反而成了故障源。

#### `_frame_labels`：证据里的帧号必须是有限值

```python
finite = np.asarray(vals, dtype=float)
finite = finite[np.isfinite(finite)]
return [int(v) for v in finite[:limit]]
```

`int(NaN)` 会抛 `ValueError`，被外层 `except Exception` 捕成 `episode_read_error`——于是"frame_index 有 NaN"这个数据问题被误报成"episode 读不出来"。先过滤非有限值，问题才能被正确归类。


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

### 5.6 parallel.py — 并发契约

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

注意 policy 看到的 `validation` 已经过基线白名单过滤（见 2.4.1），所以 `--accept-baseline` 确认过的误报不会在这里变成 review。

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

### 7.6 数据集指纹（`dataset_fingerprint`）

```
sha1( info.json 内容 | num_episodes|num_frames|codebase_version | 每个 data 分片的 rel_path:size:mtime_ns )
```

两处演进很重要：

- **覆盖全部分片，不再是前 20 个**。只取前 20 个时，改动发生在靠后的分片（v3 每 1000 集一个 chunk）指纹不变，报告会被认为"对应同一版数据"。
- **加 `mtime_ns`**。同尺寸原地覆盖（用脚本裁剪后重写）不改变 size，但内容已变；stat 级别的 mtime 是成本最低的变更信号。

`stat()` 失败（扫描与 stat 之间文件被删/无权限）时写入 `rel:stat-error:<异常类型>`，留痕而不是让整次质检崩掉——指纹只用于追溯，不值得为它中断。

指纹同时是基线白名单的锚点之一（`baseline.fingerprint = md5(数据集名|episode|规则)`），两者都刻意**不含阈值/帧数**，否则重跑后白名单会失效。


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
0. _surviving_episodes      （剔除「判定保留但一行都写不出」的 episode）
1. _collect_video_windows   （裁帧时按 fps 收窄 from/to）
2. _rewrite_frames          （重写 data parquet + 累计 stats）
3. _rebuild_videos          （按相机重建 mp4）
4. rewrite_episodes_meta    （新 episode_index / length / from_index / 视频窗口）
5. copy_tasks
6. rewrite_info             （total_episodes / total_frames / splits）
7. rewrite_stats_streaming  （用第 2 步累计的统计量）
```

顺序不能换：meta 的 `length` 和 `dataset_from_index` 依赖 `_rewrite_frames` 的返回值；视频窗口依赖 `kept_lengths`。

**第 0 步为什么必须存在**（[manual.py](file:///g:/Lerobot/pipeline/cleaner/manual.py#L300-L330)）：判定为 `keep`/`repair` 不代表能写出行。`total - head - tail <= 0` 时（例如 3 帧的 episode 被裁掉首尾各 2 帧），该集在产物里是零行，但 meta 里仍有它。结果是 parquet 的 `episode_index` 与 meta 不一致，后续按行查元信息 `KeyError`。

所以清洗前先算每个 episode 的**实际行数**（`_episode_row_counts`：优先复用已加载的 `frames`，否则只读 `episode_index` 一列），裁完仍为 0 的直接从 `kept` 移除并写 warning。warning 而不是异常——用户需要知道"这一集被剔了"，但不该因此中断。


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

**裁帧也必须按整集计数**。原实现每个文件从 0 重新数行，跨文件 episode 的 `total` 只等于单文件内行数，`head <= row_pos < total - tail` 会把第二个文件的行全部判成"超出尾部"而删掉。

修法分两半（[manual.py](file:///g:/Lerobot/pipeline/cleaner/manual.py#L383-L430)）：

- 预扫一遍所有源文件只读 `episode_index` 一列，得到 `episode_totals[new_ep] = 全数据集行数`；
- `row_pos_counter` 移出文件循环，跨文件累计行内序号。

预扫的代价是一次"只读一列"，换来的是裁帧边界正确。

**时间轴修复只作用于目标 episode**。原实现是：

```python
if any(e in rebuild_new for e in new_eps):
    sub = set_column(sub, "timestamp", frame_idx / fps)   # 整个文件一起重写
```

只要文件里有一集需要修，同文件所有 episode 的 `timestamp` 都会被改写成 `frame_idx/fps`。对于 timestamp 本来就是全局时间轴（不从 0 开始）的数据集，这等于把好数据改坏。

现在按掩码只改目标行：

```python
if "timestamp" in sub.column_names:
    ts = sub.column("timestamp").to_pylist()
else:
    ts = zeros(len(new_eps))
ts[rebuild_mask] = frame_idx[rebuild_mask] / fps
```

缺 `timestamp` 列时用 0 初始化再填目标行，而不是让 `set_column` 抛异常——清洗阶段要能处理"连列都不全"的脏数据集。


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

**文件级跳过要区分三种情况**（`_scan_columns`，[scanner.py](file:///g:/Lerobot/pipeline/pointcloud/scanner.py#L21-L34)）：

| 情况 | 处理 | 理由 |
|---|---|---|
| 没有点云特征列 | 跳过 | 这个分片本来就不参与点云质检 |
| 缺点云列但有数据 | 跳过 + warning | 同上，但要留痕 |
| 缺 `episode_index` / `frame_index` | 跳过 + warning | **不是损坏**，是索引不全；判损坏会把它算进 corrupted 并把整集标成读失败 |
| 读不出 row group | 记 `corrupted` | 这才是真损坏 |

把"缺索引列"和"文件损坏"分开，是因为前者在合并/裁剪过的数据集里很常见，而后者才是需要人工介入的故障。混在一起会让 corrupted 计数虚高。


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

**standalone 模式下，被跳过的分片也要原样复制**（[cleaner.py](file:///g:/Lerobot/pipeline/pointcloud/cleaner.py#L101-L116)）：

```python
if key not in names or "episode_index" not in names or "frame_index" not in names:
    if not same_tree:
        shutil.copy2(src_path, dst_path)
```

原地补丁模式（`same_tree=True`）不需要复制。但 standalone 输出是"完整数据集副本"，如果缺点云列的分片被跳过且不复制，产物里就静默少了文件——下游读到的 `data/files` 与 info 记录不再对应。

**产出校验只统计含点云列的分片**：`_verify()` 原来的 `rows` 是"所有分片行数之和"，被跳过的分片本来就不在 `validation.total_frames` 里，算进去会让校验无谓失败。现在遍历时先判 `feature_key in schema`，有则累加并置 `has_feature=True`。


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

**清洗缓存还绑定数据集归属**：

```python
store_cleaning(...):  payload["cleaning"]["dataset_root"] = payload["dataset_root"]
reusable_cleaning(): if cleaning["dataset_root"] != payload["dataset_root"]: return None
```

原因：`output_dir` 常被复用来跑不同数据集。处置结论可能有幸相同（都是"删第 3、7 集"），签名就一致，但产物目录里装的是**另一个数据集**的清洗结果。只比签名会把 A 数据的产物当成 B 的交付。

同时 `store_validation` 在 `dataset_root` 变化时主动 `pop("cleaning")`：换了数据集，旧清洗记录属于别的根，直接作废而不是留着被误判。


---

## 11. 旁路模块与辅助脚本

主链路之外还有四类模块，它们只读写已有产物，不参与数据正确性判定：

### 11.1 baseline.py — 基线白名单

```python
fingerprint = md5(f"{dataset_name}|{episode_index}|{rule_code}")[:16]
```

只锚定"数据集名 + episode 编号 + 规则"，**不含帧数/阈值**。原因：重跑会重新推导 median/MAD，若指纹含阈值，白名单每次重跑都失效——而白名单的价值恰恰是"这条我确认过是对的，以后别再问我"。

`whitelist_from_config()` 兼容两种写法（字符串、`{fingerprint: ...}`），并对非法指纹只 warning 不抛：配置里多一条写错的指纹，不应该让整次质检起不来。

`apply_whitelist()` 的三条约束：

- 从 `reasons` 摘除，同时移进 `ignored_baseline`；
- 不删 `details`（证据留着，便于日后反悔）；
- 不动阈值和 `severity`。

### 11.2 versions.py — 数据集版本管理

`DatasetVersionStore` 用对象库（按 sha1 存内容）+ 版本记录实现 `commit / log / diff / restore`，默认库目录 `<dataset>/.lrdq`。

两个安全点：

- `restore` 默认只打印 plan，必须 `--apply` 才写盘；
- 写盘前 `_safe_dest()` 校验 `dest.resolve()` 仍在 `root` 内，并把 `../../` 之类的相对路径直接拒绝。

第二条是必需的：版本记录里的 `path` 来自快照时的文件清单，属于**外部可写数据**。不加越界校验，一个被改过的 manifest 就能让 `restore` 往数据集目录外写文件。

对象缺失时 `restore` 直接抛 `RuntimeError` 列出前 5 个缺失文件，而不是部分写回——半套数据比不写更糟。

### 11.3 card.py — 数据集卡片

```bash
python -m pipeline.card --report outputs/demo/quality_report.json --out outputs/demo/DATASET_CARD.md
```

卡片**只读落盘产物**（`quality_report.json` / `meta/info.json` / `meta/stats.json`），不重算指标。这样卡片与报告永远一致，不会出现"卡片说 98 集、报告说 97 集"。

`skipped_checks` 渲染成"规则：原因"而不是裸 code 列表。裸 code 无法区分"没有这列"和"配置关掉了这条检查"，而这两者对应的修复动作完全不同。

`_mean_hint()` 对高维统计量只给形状摘要（`` `[29,3]` 张量``），否则卡片里会堆 87 个数。

### 11.4 export_clean_dataset.py / 其他脚本

- `export_clean_dataset.py`：把清洗产物导出成独立可分发数据集（重算 info/stats、必要时重建 splits）。
- `precheck.py`：不跑完整 pipeline 的轻量预检（meta/文件数/体积），用于先判断数据集值不值得跑。
- `tuning.py`：阈值调参辅助，基于历史报告分布给建议。
- `generate_report.py`：把 `quality_report.json` 转成人类可读摘要。
- `tools/dashboard_server.py`：本地 dashboard，读报告 JSON 做浏览。
- `visualizer.py`：按 `--plots-sample` 采样，输出 state/action 曲线与视频帧拼图到 `plots/`。

这些脚本失败都不影响主链路的退出码。


---

## 12. 中间产物契约

| 文件 | 产出阶段 | 关键字段 |
|---|---|---|
| `intermediate/01_dataset_summary.json` | loader | episodes/frames/schema/video_features |
| `intermediate/02_validation_result.json` | validator | results/reasons/details/resolved_thresholds/schema_contract |
| `intermediate/02b_baseline_ignored.json` | baseline | 被白名单摘除的告警（episode/rule/fingerprint） |
| `intermediate/03_decisions.json` | policy | 每条 episode 的 disposition/repair/trim/source |
| `quality_report.json` | reporter | summary/checks/episodes/attribution/cleaning/dataset_fingerprint |
| `review_queue.csv` | policy | 待人工复核项 |
| `pointcloud_scan_cache.json` | pointcloud | 帧级扫描结果 + envelope + 文件指纹 |
| `pointcloud_review_queue.csv` | pointcloud | 点云待复核项 |
| `DATASET_CARD.md` | card（可选） | 报告摘要 + 数据形状 + 跳过检查及原因 |
| `intermediate/failure_*.json` | 任一阶段 | stage/error/type/traceback/recoverable |
| `quality_manifest.json` | 收尾 | 配置摘要/episode 签名/清洗签名/dataset_root/verification |

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
2. `pipeline/schema.py::audit_schema` → 理解"能读"与"合规"的区别
3. `pipeline/validator/context.py::_thresholds_from_arrays` → 理解阈值怎么来
4. `pipeline/validator/__init__.py::_validate_non_video` → 看规则的调用顺序
5. `pipeline/validator/rules.py::check_finite` → 看"可修性"怎么被编码进证据
6. `pipeline/baseline.py::apply_whitelist` → 看"用户裁定"如何与规则输出解耦
7. `pipeline/policy.py::_decide_one` → 看证据怎么变成动作
8. `pipeline/cleaner/manual.py::_rewrite_frames` → 看动作怎么落到 parquet
9. `pipeline/pointcloud/rules.py::evaluate_frame` → 看另一套逐帧契约

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
| "能加载成功就说明 index 合规" | 全量读取不报错 ≠ 合规；退化成帧号的 index 只在过滤/`delta_timestamps` 时静默取错帧 |
| "meta 重复行直接去重就行" | 只做去重会掩盖 length/视频窗冲突，现报 `meta_episode_duplicate`(error) |
| "白名单会让报告少一条" | 告警移入 `ignored_baseline` 并落 `02b_baseline_ignored.json`，仍可追溯 |
| "同 output_dir 复跑能复用清洗产物" | 清洗缓存绑定 `dataset_root`，换数据集即失效 |
| "点云缺索引列的文件算损坏" | 分开处理：缺索引列只跳过并 warning，读不出 row group 才算 corrupted |
