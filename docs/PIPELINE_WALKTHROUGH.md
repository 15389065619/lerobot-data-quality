# LeRobot 质检流水线：执行主线 + 两处硬核实现

> 用途：面试前随时翻。上半部分回答「这条命令是怎么跑起来的」，下半部分回答两个会被追问到底的实现细节。
> 对应代码库：`G:\Lerobot`（LeRobot v3 数据质检与清洗流水线）

---

# 第一部分：一条命令背后的执行主线

## 0. 30 秒速览

整条链路只有 **一个进程、两个入口**，不是微服务：

```
python main.py
  └─ main.py                     ← CLI 入口：解析参数、准备输出目录、读增量状态
      ├─ pipeline/config.py      ← 合并配置（默认值 < 配置文件 < 机器人覆盖 < CLI）
      ├─ pipeline/loader.py      ← 把磁盘读成内存视图（Parquet + meta + 视频索引）
      ├─ pipeline/validator/     ← 逐 episode 跑规则，产出结论
      ├─ pipeline/baseline.py    ← 摘掉已确认的已知误报
      ├─ pipeline/policy.py      ← 结论翻译成动作：keep/repair/review/drop
      ├─ pipeline/reporter.py    ← 生成 quality_report.json + 终端报告
      ├─ pipeline/cleaner/       ← 按动作重写出一份新数据集
      ├─ pipeline/pointcloud/    ← 点云单独一条支线（有才跑）
      └─ pipeline/visualizer.py  ← 画图收尾
```

另一个入口是 `export_clean_dataset.py`：它不跑质检，只读已有报告做「按报告导出」（默认只剔除 drop）。

## 1. 第一步：`main.py` 到底先干了什么

1. **`build_parser()` + `parse_args()`**：把命令行参数解析成 `args`。
2. **`setup_logging()`**：把 stdout/stderr 改成 UTF-8（Windows 控制台默认 cp936，中文会炸）。
3. **分支判断**：带 `--accept-baseline` 时直接走白名单维护，**不加载数据集**、不跑质检。
4. **准备目录**：`output_dir.mkdir(parents=True, exist_ok=True)`。
5. **读增量状态**：`QualityManifest.load(<output_dir>/quality_manifest.json)`；
   `--incremental` 且无 `--force` 时先用 `load_dataset(..., load_frames=False)` 做轻量扫描，
   配置摘要 / episode 指纹 / 上次产物都没变就直接复用报告返回。
6. 打印 banner，进入第 1 步。

注意：**配置不是在入口第一步加载的**。`_build_config()` 要等 `dataset.info` 读出 `robot_type` 之后才调用，
因为阈值要按机器人型号覆盖。优先级见 `pipeline/config.py`：

```
代码内置 DEFAULT_CONFIG < --config 配置文件 < robots[robot_type] < CLI 快捷参数
```

## 2. 第二步：`loader.py` 怎么读数据

入口：`load_dataset(dataset_path, max_episodes=None, load_frames=True)`，只读、不判断质量：

| 顺序 | 做什么 | 关键函数 |
|---|---|---|
| 1 | 读 `meta/info.json`，缺了抛 `DatasetFormatError` | `_read_info` |
| 2 | 读 `meta/stats.json`（后面判 `stats_stale` 用） | `_read_json` |
| 3 | 读 `meta/tasks.parquet`，兼容 `tasks.jsonl` | `_read_tasks` |
| 4 | 按 `info.data_path` 模板枚举 `data/chunk-*/file-*.parquet` | `_discover_data_files` |
| 5 | 读 `meta/episodes/**/*.parquet` 或 `episodes.jsonl` | `_read_episodes_meta` |
| 6 | 扫描 `videos/**/*.mp4`，只建「相机 → (chunk,file) → 路径」索引 | `_discover_videos` |
| 7 | 识别点云等重型特征，默认不读进内存 | `heavy_feature_keys` |
| 8 | 对齐 data 与 meta 两侧 episode 编号（见第二部分） | `_reconcile_episode_ids` |
| 9 | Schema 契约审计（只读 schema 与两列极值） | `schema.audit_schema` |
| 10 | 真正读逐帧表，拼成一张长表 | `_read_frames` |

- Parquet 用 `pyarrow.parquet.read_table` **逐文件**读，带 `episode_index` 行过滤、跳过重型列，
  并加内部列 `__source_file` 记录来源；损坏文件记入 `corrupted_files` 不中断。
- 视频这一阶段**只建索引不解码**，真正的校验在 validator 第二阶段用 ffprobe/ffmpeg。
- 输出 `LeRobotDatasetView`：纯数据视图，不含任何结论。

## 3. 第三步：`validator/` 怎么分发规则

`validate_dataset(dataset, config, ...)`：

1. `DatasetContext.build()` 先算全局量：帧数区间（median ± k·MAD，或 manual）、action 逐维阈值
   （每集最大 |Δ| 的 median + k·MAD，再用按 fps 折算的 abs_floor 兜底）、数据集级问题
   （`stats_stale`、`meta_index_inconsistent`、`meta_episode_mismatch`）、`skipped_checks`。
2. 逐 episode 顺序调用 `_validate_non_video()`：

```
state 列存在?  → check_finite(nan_state) → check_static → check_joint_limits
action 列存在? → check_finite(nan_action) → check_action_jump → check_static → check_duplicate
不管有没有列  → check_frame_count → check_timestamp → check_alignment
             → check_meta_consistency → check_task
```

3. 视频探针阶段用 `ThreadPoolExecutor`（默认 2 路）跑 `frame_mismatch`（ffprobe 帧数 vs 实际行数）
   与 `blank_video`（ffmpeg 抽点算灰度标准差）。

三种模式：默认整表入内存；`--workers N` 走 `ProcessPoolExecutor`（纯函数 worker，结论与串行一致）；
`--streaming` 走 `validator/streaming.py`（见第二部分）。规则只记结论，不决定删不删。

## 4. 第四步：白名单与处置

- **`pipeline/baseline.py`**：`apply_whitelist()` 用指纹（数据集名 + episode + 规则码）摘掉确认过的误报，
  写进 `baseline_ignored`，不参与健康度。
- **`pipeline/policy.py`**：`decide()` 把结论翻译成 keep / repair / review / drop；
  两道保护：某规则命中 ≥ 95% episode 时触发**系统性熔断**降级为 review，禁止批量删除；
  `--overrides review_queue.csv` 可人工回灌并标记 `source=override`。
- 同时写 `review_queue.csv`（`reviewer_disposition` 留空待人填）。

## 5. 第五步：`reporter.py` 生成什么

`build_report()` 只汇总不重算：`dataset`（路径/版本/fps/指纹/编号对齐证据/Schema 违约）、
`summary`（正常/异常、严重度、处置计数、健康度）、`checks`、`invalid_episode_details`（含
`max_delta`、`worst_frame` 等规则细节）、`attribution`（按采集员/批次算异常率）、`recommendations`。
`write_report()` 落盘 `quality_report.json`，`print_summary()` 打终端版；清洗结果与图表随后回填同一份 JSON。

## 6. 第六步：`cleaner/` 怎么产出新数据集

`clean_dataset(dataset, decisions, output_dir, ...)`，源数据严格只读：

```
有 drop 或 repair？
├─ 否 → copy_untouched()：整份结构完整复制
└─ 是 → 无 repair 时优先调 lerobot-edit-dataset（官方 CLI，只能删）
         有 repair 时强制走内置 manual_clean()（CLI 不会修）
         内置实现用 pyarrow 的 table.take / set_column，避免破坏定长向量列
```

之后 `cleaner/meta.py` 重建 info/episodes/tasks/stats，`cleaner/util.py` 的 `verify()` 校验产出，
写 `cleaned_dataset/` + `cleaning_summary.md`。

`export_clean_dataset.py` 是另一个入口：读 `quality_report.json`，默认只物理剔除 `disposition == drop`
（review/repair 原样保留，需显式 `--drop-review` / `--drop-repair` 才删），并把 `episode_index` / `index`
从 0 连续重排、校验视频与 stats 完整性后才算导出成功。

## 7. 点云与可视化

- 点云（`pipeline/pointcloud/`）：仅在存在 `shape=[N,D]`、`D≥3` 浮点特征时运行，作用在**清洗后产物**上；
  修复只改点不改行；没有点云特征整段跳过。
- 可视化（`pipeline/visualizer.py`）：帧数分布、异常饼图、关节曲线、动作散点、清洗前后对比；失败只告警。

## 8. 一次运行留下什么

```
<output_dir>/
├─ quality_report.json / review_queue.csv / quality_manifest.json
├─ plots/                       # 5 张图
├─ intermediate/01..07_*.json   # 各阶段快照
├─ cleaned_dataset/
└─ pointcloud_cleaned_dataset/  # 有点云时才有
```

各阶段失败会写 `failure_<stage>.json`，主链路尽量不中断。

## 9. 面试速答（四问四答）

**Q1：入口启动后第一步？** `argparse` → 日志 → 输出目录 → 增量 manifest 快路径 → loader；
配置要等读到 `robot_type` 后再合并。

**Q2：loader 怎么读 Parquet 和视频？** 逐文件 `pyarrow.read_table`（列裁剪 + episode 过滤 + 来源标记），
视频只建索引不解码；读完后做编号对齐与 Schema 契约审计。

**Q3：数据怎么分发到规则？** 先建一次 `DatasetContext` 算阈值和数据集级问题，再逐 episode 顺序调规则，
第二阶段线程池跑视频探针；规则只产结论。

**Q4：报告和清洗谁接管？** `reporter.build_report()` 汇总 JSON；`cleaner.clean_dataset()` 按 policy 动作
重写新数据集（能修就修，修不了才删，源数据只读），结果回填报告；按报告导出由 `export_clean_dataset.py` 完成。

## 10. 容易答错的点

- 逐帧表只在 loader 读一次（流式模式例外），validator 用的是内存视图。
- 视频不做逐帧解码：绝大多数问题看容器元数据就够，只有黑屏才抽几帧。
- 规则不删数据；删除只发生在 cleaner，且有 95% 熔断与人工回灌。
- 清洗优先官方 CLI 是为了生态一致；CLI 不能修，所以有 repair 时必须走内置实现并记录原因。
- 大数据瓶颈在读表与视频探针，所以有流式、多进程、增量 manifest 三件套。

---

# 第二部分：巷战 —— 两个核心实现逐行拆

## 目标一：流式读取防 OOM

### 调用链

```
main.py --streaming --dataset_path ...
 └─ pipeline/validator/__init__.py: validate_dataset(..., streaming=True)
     └─ pipeline/validator/streaming.py: validate_dataset_streaming(dataset, config)
         ├─ Pass 1  _build_context()        ← 只扫出「阈值 / 计数 / 指纹」
         └─ Pass 2  逐 episode 校验          ← 只留下 EpisodeResult
              ↑ 两遍都只通过 pipeline/loader.py: iter_frame_chunks() 取数
```

核心在 `pipeline/loader.py` 的 `iter_frame_chunks()`：

```python
for path in dataset.data_files:              # ① 一次只打开一个 parquet
    file_wanted = wanted
    if wanted is not None:
        names = set(pq.ParquetFile(path).schema_arrow.names)  # ② 有的 v3 缺列，按文件真实 schema 过滤
        file_wanted = [c for c in wanted if c in names]

    if keep is not None:
        table = pq.read_table(                         # ③ 列裁剪 + 行组谓词下推
            path,
            columns=file_wanted,
            filters=[("episode_index", "in", sorted(keep))],   # 只取需要的 episode 行
        )
    else:
        table = pq.read_table(path, columns=file_wanted)

    df = table.to_pandas()                    # ④ 此刻内存里只有「这一个文件」的数据
    df[SOURCE_COL] = str(path)
    yield df                                  # ⑤ 交出去处理
    # ⑥ 下一轮 path 时 df / table 被重新绑定，引用计数归零，内存立刻释放
```

逐行说明：

- **①** `data_files` 是启动时枚举好的文件清单，循环本身不占内存。
- **②** 先读 parquet 的 schema（几十 KB），把调用方要的列与实际存在的列求交集，避免「请求不存在的列」直接报错。
- **③** 两个省内存的关键：`columns=` 只读需要的列（图像 / 点云列根本不读）；`filters=` 交给 pyarrow 做行组跳过，
  连解码都不做。老版本 pyarrow 不支持 filter 时会退化，但列裁剪仍然保留。
- **④** 这里才把 Arrow 表转成 pandas，且只转当前文件。
- **⑤** `yield` 是生成器：调用方处理完一块，控制权回到这里继续下一个文件，**不会预先攒一个全量列表**。
- **⑥** 这是「数据在哪被丢掉」的最终答案：没有任何显式 `del`，靠的是**变量重绑定 + CPython 引用计数**。
  `df` 被下一轮赋值覆盖 → 旧 DataFrame 引用计数归零 → 立即回收（大数组走 pymalloc 之外的路径，直接还给系统）。

### 为什么内存峰值 ≈ 单个文件

用 `--streaming` 跑 14 GB 数据时，常驻内存由三块组成：

| 组成 | 量级 | 说明 |
|---|---|---|
| 当前 parquet 文件 | 一个文件大小 | 处理完即释放，不累积 |
| 逐集聚合量 | episode 数 × 几十字节 | `counts`、`frames_by_episode`、`fingerprints` |
| 采样数组 | 最多 200 集 | `streaming.action_sample_episodes` 限制 |

Pass 1 的累加器就是这样「只记统计量、不留原始行」：

```python
counts: dict[int, int] = {}                    # 每集多少行——一个 int
fingerprints: dict[str, int] = {}              # 重复检测——hash -> 首个 episode
action_parts: dict[int, list[np.ndarray]] = {} # 只对采样的 200 集保留数组
```

Pass 2 同理，只存结论不存帧：

```python
for chunk in iter_frame_chunks(dataset, columns=pass2_columns, episodes=stream_eps):
    for raw_ep, ep_df in chunk.groupby("episode_index", sort=True):
        res = _validate_non_video(ep_df, ep_index, dataset, ctx, config, skipped)
        results[ep_index] = res        # 存 EpisodeResult（原因、阈值、极值），不存 ep_df
        pending_video.append((res, ep_index, want_blank))   # 视频探针也只排任务，不拿帧
```

`ep_df` 在内层循环下一轮就被替换，`chunk` 在外层下一轮被替换，两者都会立即释放。

### 「数据到底在哪被丢掉」清单

1. **列投影**：`columns=` 只读必需列，点云 / 图像列根本不进内存（最大的一刀）。
2. **谓词下推**：`filters=[("episode_index","in",...)]` 让 pyarrow 跳过无关 row group。
3. **逐文件分片**：`iter_frame_chunks` 一次只物化一个文件。
4. **生成器**：`yield` 而非 `return list`，不预聚合。
5. **变量重绑定**：上一块的 `df/table/chunk/ep_df` 引用计数归零，立即回收。
6. **采样上限**：action 阈值与 stats 时效只采样前 200 集，不保留全量数组。
7. **只留结论**：Pass 2 的 `results` 里是 `EpisodeResult`，不是原始帧。

实测同一份 14 GB 数据：pandas 一次性读峰值 **2,694 MB**、duckdb **1,509 MB**、本流水线流式读取 **732 MB**
（见 `docs/EVAL.md` §8）。差距不是「少读了几列」，而是**峰值不随数据集总量增长**。

### 诚实边界（面试官追问时主动说）

- v3 的 parquet 按写入 chunk 切分，通常单文件几十到几百 MB；如果上游把 14 GB 写进**一个** parquet，
  逐文件策略仍会吃满一个文件的内存。真正的兜底是 `ParquetFile.iter_batches(batch_size=...)`，
  点云路径就是这么做（`pipeline/pointcloud/scanner.py`）。
- 流式模式牺牲的是「随机回看」能力：跨 episode 的全局统计只能靠 Pass 1 的聚合量，不能再随手 `groupby` 全表。

---

## 目标二：ID 重映射防误杀

### 问题现场

behavior1k 这类数据集：`meta/episodes/*.parquet` 的 `episode_index` 是**全局 ID**（420010, 420020, …），
而 `data/*.parquet` 里是**局部 ID**（0..199）。不处理的话，`dataset.episodes` 按 420010 查不到任何行，
视频窗口、任务文本全部定位失败，200 个 episode 会被判成 200 个 error——**整批数据被误杀**。

### 入口与数据结构

`load_dataset()` 在读完 meta 和 data 文件清单后、读帧之前，先做对齐：

```python
data_summary = _data_episode_summary(frames if load_frames else None, data_files)
meta_id_map, reconciliation = _reconcile_episode_ids(episodes_meta, data_summary, fps)
```

- `data_summary`：每集**实际行数** `rows`（流式模式下只读 `episode_index` 一列扫一遍，128 万行约 10 MB）。
- `meta_id_map`：**一个普通 dict**，`{局部ID: 全局ID}`，例如 `{0: 420010, 1: 420020, ...}`。
- `reconciliation`：诊断字典（`mode`、两侧 ID 区间、`length_agreement`、`video_agreement`、样例），
  原样写进报告，面试时可以直接把「证据」摊开。

两者都挂在视图对象上：

```python
class LeRobotDatasetView:
    meta_id_map: dict[int, int]        # 数据侧 ID -> meta 侧 ID
    id_reconciliation: dict[str, Any]  # 对齐过程与证据

    def meta_episode_index(self, episode_index: int) -> int:
        # 没有映射时退化为恒等映射：空 dict.get 返回默认值，行为与原生数据集一致
        return int(self.meta_id_map.get(int(episode_index), int(episode_index)))
```

内存代价是 `O(episode 数)` 个整数对，2000 集也就几十 KB；**不做 DataFrame merge、不重写 parquet**。

### 判定流程（`_reconcile_episode_ids`）

```
两侧 ID 集合相同？
├─ 是 → mode=native，返回空映射（零成本）
└─ 否 → episode 数量不同？
         ├─ 是 → mode=ambiguous（数量都对不上，不可能一一映射，直接放弃）
         └─ 否 → 验证「顺序一一对应」的三重证据
                 ① 每集实际行数 rows == meta.length
                 ② 每集视频时间窗 (to_timestamp - from_timestamp) × fps ≈ rows
                 ③ agreement = min(①, ②) ≥ 0.99
                    ├─ 是 → mode=remapped，返回 {data_id: meta_id}
                    └─ 否 → mode=ambiguous（证据不足，禁止猜）
```

关键代码：

```python
rows_arr   = data_summary["rows"].to_numpy(dtype="int64")
length_arr = meta.sort_values("episode_index")["length"].to_numpy(dtype="int64")
length_ok  = int((rows_arr == length_arr).sum())

for 每个 videos/*/from_timestamp + to_timestamp 列对:
    window_frames = np.rint((to_ts - from_ts) * fps)
    video_ok += int((window_frames == rows_arr).sum())

evidence  = [length_agreement, video_agreement]   # 任一为 None 就不参与
agreement = min(evidence) if evidence else 0.0

if not evidence or agreement < 0.99:
    return {}, {"mode": "ambiguous", "reason": "证据不足，不做猜测映射"}

mapping = {d: m for d, m in zip(data_ids, meta_ids)}   # 顺序配对，O(n)
return mapping, {"mode": "remapped", ...}
```

为什么用 `min` 而不是 `max`：两个证据必须**同时**成立。只有行数对得上、视频窗口对不上，说明顺序假设在这批数据上
不成立（比如中间插了未登记的 episode），此时任何映射都可能是错的。用 `min` 等于「一票否决」。

为什么阈值是 0.99 而不是 1.0：`to_timestamp - from_timestamp` 是浮点秒数，乘 fps 再 `rint` 偶有 ±1 帧误差；
0.99 允许约 1% 的舍入抖动，但 200 集里错 2 集以上就会掉到 0.99 以下，仍能拦住系统性错配。

### 「对不上」时怎么处理（不重排、不猜）

1. **能证明 → 懒映射，不物理重排**
   帧数据里的 `episode_index` 保持原值（0..199），只在**访问 meta 时**翻译：

```python
def meta_row(self, episode_index: int) -> pd.Series | None:
    rows = self.episodes[self.episodes["episode_index"] == self.meta_episode_index(episode_index)]
    return rows.iloc[0] if len(rows) else None
```

视频定位、任务文本、`frames_by_episode` 的反向索引都走同一个入口。好处是：**不复制、不重排、不改源文件**，
避免「清洗前先改一遍 ID」这种不可逆操作。

2. **`--max_episodes` 时映射跟着裁剪**
   选前 N 集是用数据侧 ID 选的，裁剪 meta 时必须先把数据 ID 翻成 meta ID 再过滤，否则又会全落空：

```python
keep = set(all_eps[:max_episodes])                 # 数据侧 ID
wanted_meta = {meta_id_map.get(k, k) for k in keep}
episodes_meta = episodes_meta[episodes_meta["episode_index"].isin(wanted_meta)]
meta_id_map = {k: v for k, v in meta_id_map.items() if k in keep}   # 同步收缩映射
```

3. **证明不了 → 标记「不可评估」，而不是报错**
   `locator_available` 为 False 时，validator 不会判数据坏，而是把这些检查写进 `skipped_checks`：

```python
skipped["frame_mismatch"] = "meta 与 data 的 episode 编号空间不一致且无法可靠映射，属「不可评估」"
skipped["missing_task"]   = 同上
```

同时产生数据集级问题 `meta_episode_mismatch`，并标注：

```python
issue.detail["auto_reconciled"] = False
issue.detail["impact"] = "无法可靠映射，依赖 meta 的检查不可评估，禁止据此批量剔除"
```

即使误判成异常，policy 层的 95% 系统性熔断也会把「几乎全命中」的规则降级为人工复核，
不会出现「一条规则删掉整批数据」。

4. **映射成功时，报告要写「已解决」**
   `_annotate_reconciliation()` 给 `meta_episode_mismatch` 打上 `auto_reconciled=True` 和
   `length_agreement` / `video_agreement`，避免读者把已处理的问题当成未处理的阻断项。

### 一句话总结

> 映射就是内存里一个 `dict{局部ID: 全局ID}` 加一份证据字典；不改数据、不重排文件。
> 能证明就懒翻译，证明不了就标「不可评估」并禁止批量删除——**宁可少判，不可错杀**。

### 可能被追问

- **「为什么不用 meta 的 length 反推顺序？」** 行数相同只能证明长度一致，不能证明顺序；必须叠加视频时间窗这个独立证据。
- **「映射错了会怎样？」** 会静默错配任务和视频。所以阈值用 `min`、要求 ≥0.99，且样例写进报告，可回溯。
- **「为什么不在清洗时把 ID 改成一致？」** 清洗是下游动作，不该承担「修复元数据语义」的职责；
  而且改 ID 会破坏与上游标注系统的对应关系，属于不可逆变更。

---

## 附：两张速查表

| 你想讲 | 先打开的文件 | 一句话 |
|---|---|---|
| 入口与编排 | `main.py` | 参数 → 目录 → 增量快路径 → 七步流水线 |
| 读数据 | `pipeline/loader.py` | 逐文件读、列裁剪、只建视频索引 |
| 规则分发 | `pipeline/validator/__init__.py` | 先建上下文，再逐集顺序调规则 |
| 流式 | `pipeline/validator/streaming.py` | 两遍扫描，只留聚合量与结论 |
| 编号对齐 | `pipeline/loader.py: _reconcile_episode_ids` | 三重证据 + min ≥ 0.99，否则不猜 |
| 处置 | `pipeline/policy.py` | keep/repair/review/drop + 单规则 95% / 整体 80% 熔断 |
| 报告 | `pipeline/reporter.py` | 只汇总，不重算 |
| 清洗 | `pipeline/cleaner/__init__.py` | 源只读，能修就修，修不了才删 |
| 按报告导出 | `export_clean_dataset.py` | 默认只删 drop，ID 从 0 重排 |

| 关键数字 | 值 |
|---|---|
| 流式内存峰值（14 GB 数据） | 732 MB（pandas 2,694 MB / duckdb 1,509 MB） |
| 编号映射证据阈值 | `min(行数一致率, 视频窗一致率) ≥ 0.99` |
| 系统性熔断阈值 | 单规则命中 ≥ 95% episode → 降级 review |
| action 阈值采样上限 | 200 episode |
