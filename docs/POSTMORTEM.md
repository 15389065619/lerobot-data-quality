# 复盘：人类审核纠正的四次判定错误

本文记录这套质检系统在研发与真实数据验证阶段，四次「工具给出明确结论、但结论是错的」的事件。

四次事件的共同点是：**代码按设计正常工作，测试全绿，报告格式完整，数字看起来很有说服力**——错的是判定所依赖的前提。它们都不是崩溃类故障，而是「静默的错误结论」。这类问题无法靠 CI 或代码审查发现，只能靠人拿着业务常识去质疑一个已经自洽的结论。

记录的目的有两个：

1. 说明为什么这条流水线里必须保留人工复核环节，而不是把处置动作全自动化；
2. 把四次纠偏沉淀成可执行的验收标准——新增规则时，先问本文末尾「共性」一节的四个问题。

| 编号 | 事件 | 错误类型 | 影响面 | 修复提交 |
|---|---|---|---|---|
| 1 | 编号空间错位，200/200 全被判错 | 把「定位失败」表达成「数据不合格」 | 14 GB / 200 episode / 128 万帧，险些全量删除 | `d56e52f` |
| 2 | `stats_stale` 偏差公式分母错 | 比值型指标的分母不是尺度 | 全数据集级误报，偏移量虚高 913 倍 | `d56e52f` |
| 3 | v2.1 数据被按 v3.0 契约判定 | 判据隐含了「版本」这个隐变量 | 1418 个 episode 被置顶【阻断】 | `45e2b69` |
| 4 | `action_jump` 阈值退化 + 尖峰/平滑不可分 | 阈值落点未复测，判据维度不足 | 45 条命中（分布在 8 个数据集），单集最高命中率 8.25% | `d56e52f` / `45e2b69` |

---

## 案例 1：编号空间错位——AI 判 200/200 全错，人工抽查后保留 195 集

### 现象

在官方数据集 `behavior1k_task0042`（14 GB / 200 episode / 1,279,960 帧 / 9 路相机）上跑通全流程，耗时与吞吐均正常，但结论是：

| 指标 | 结果 |
|---|---|
| episode 判定 | 0 正常 / **200 error（100%）** |
| `frame_mismatch` | 200 条，「视频帧数与 state 行数不一致」 |
| `missing_task` | 200 条，「任务描述缺失或为空」 |
| 处置建议 | 剔除 200 个 episode / 约 128 万帧 |
| 健康度 | 严重（先修采集流程） |

两份彼此独立的规则在同一条数据上 100% 同时命中，且都是 error 级、都建议删除——按系统自身的策略，这份 14 GB 的官方数据集应当被整份丢弃。

### 怀疑与排查过程

「100% 全错」本身是可疑信号，而不是强证据。一个真实采集的数据集不可能每一条 episode 同时缺任务、同时视频帧数错位；更常见的解释是**我们没读到正确的位置**。

排查按三步收敛：

1. **人工抽查一条 episode。** 打开 ep0 的 mp4，用 `ffprobe` 取 `nb_frames`，与 parquet 中该 episode 的行数对比——两者相等。再看视频时间窗：`to_timestamp - from_timestamp` 乘以 fps，与行数相等。**数据本身是对的，是我们的判定路径没有指向它。**
2. **打印两侧编号集合。** `meta/episodes.parquet` 的 `episode_index` 是 `420010, 420020, …, 422980`（步长 10，中间 51 处缺口）；`data/*.parquet` 的 `episode_index` 是 `0..199`。**两个集合的交集为 0。**
3. **定位代码路径。** `video_location()` 与 `task_of()` 原先用 `episodes["episode_index"] == 数据侧编号` 查表。交集为空，查表必然返回空；返回空又被上游解释成「视频不存在 / 任务为空」，最终落到 `frame_mismatch` 与 `missing_task`。

即：**「我找不到」被表达成了「数据错了」。**

### 证据证伪

在数据侧不动、编号未对齐的前提下，有 3 条互相独立的证据可以证明「数据本体没坏」，且三者均 200/200 精确成立：

| 证据 | 判据 | 结果 |
|---|---|---|
| 行数与 meta 声明一致 | `meta/episodes.length` == 该 episode 在 parquet 中的实际行数 | 200/200 相等，合计 1,279,960 帧 |
| 视频窗与行数一致 | `(to_timestamp - from_timestamp) × fps` == 行数 | 200/200 相等（RGB 三路核对） |
| 全局 index 连续 | `dataset_from_index` == 前序 `length` 的累计和 | 200/200 相等 |

同时确认了「顺序映射」这个假设成立：`data 0..199` 与 `meta 420010..422980` 按升序一一对应时，行数一致率与视频窗一致率均为 **1.0**，不存在一对多或多对一。

### 代码修复方案

修复分四层，核心原则是**先证明映射成立，再使用映射；证明不了就拒绝猜测**。

1. **加载层对齐（`pipeline/loader.py`）**
   新增 `_reconcile_episode_ids()`：当两侧编号集合不同、且数量相等时，按升序建立映射，并要求两重证据同时达标（`length_agreement`、`video_agreement` ≥ 0.99）才接受；任一证据不足则标记 `mode="ambiguous"`，返回空映射并给出原因，**不做映射**。同时输出 `id_reconciliation` 诊断块（两侧编号范围、交集大小、步长、缺口数、一致率、前 3 条映射样例），写进报告供人工核查。

   ```python
   agreement = min(evidence)          # evidence = [length_agreement, video_agreement]
   if not evidence or agreement < min_agreement:   # min_agreement = 0.99
       diag.update(mode="ambiguous", reason="...证据不足，不做猜测映射")
       return {}, diag
   mapping = {d: m for d, m in zip(data_ids, meta_ids)}
   ```

2. **统一取数入口（`pipeline/loader.py`）**
   新增 `meta_row()`，`task_of()` / `video_location()` / `data_file_of()` / `frame_count_of()` 一律经它取 meta 行，映射只在入口处生效一次，避免某条路径漏改导致「同一编号在两处指不同 episode」。

3. **区分「不可评估」（`pipeline/validator/`）**
   对齐失败、`meta` 无定位信息时置 `locator_available=False`，此时 `frame_mismatch` / `missing_task` 不再产出 error，而是写入 `skipped_checks` 并说明原因。**判定结果从二态（通过/不通过）变为三态（通过/不通过/不可评估）**，这是本案例最关键的一处设计修改。

4. **系统级熔断（`pipeline/policy.py`）**
   `_apply_systemic_guard()`：单条规则在全体 episode 上命中率 ≥ 95% 时，把批量 `drop` 降级为 `review`。理由是这种情况更可能是工具缺陷而非数据缺陷，批量删除不可逆，必须先人工判断。

### 结果

| 指标 | 修复前 | 修复后 |
|---|---|---|
| episode 判定 | 0 正常 / 200 error | 195 正常 / 5 warning |
| `frame_mismatch` / `missing_task` | 200 / 200 | **0 / 0** |
| 处置建议 | 剔除 200 个 / 128 万帧 | 保留 195 / 复核 5 / 剔除 0 |
| `id_reconciliation` | 无此字段 | `mode=remapped`，一致率 1.0，映射样本可查 |

同时修复后暴露了这条数据**真实**存在的一个问题：`index` 列每个 episode 从 0 重启（`max=9431` 而 rows=133255），这属于案例 3 讨论的版本契约问题。

### 防回归测试用例

`tests/test_episode_id_reconciliation.py`：

| 用例 | 断言 |
|---|---|
| `test_global_meta_ids_are_mapped_by_order` | 全局 ID + 局部 ID 时按序映射成立，`mode=remapped` |
| `test_mapping_restores_task_lookup` | 映射后 `task_of()` 能取回正确任务文本（不再 `missing_task`） |
| `test_unmappable_id_space_is_not_evaluable` | 数量不等时 `mode=ambiguous`、映射为空、相关检查进 `skipped_checks`，**不得**判 error |
| `test_systemic_guard_blocks_batch_drop` | 单规则 ≥95% 命中时批量 drop 降级为 review |
| `test_minority_hits_still_drop` | 少数命中（<95%）仍按正常策略剔除，熔断不得过宽 |

---

## 案例 2：`stats_stale` 公式放大——微小漂移被报成 913 倍

### 现象

数据集级检查报出「`meta/stats.json` 与实测数据偏差过大（归一化统计已过期）」，证据字段 `max_relative_deviation` 为 **913** 量级，远超 5% 容忍，建议重算 stats。

问题在于：同一批数据的逐维绝对差看起来并不夸张，且不同数据集上这个数值分布极不稳定（0.15 / 1.57 / 3.63 / 913），无法据以判断严重程度。

### 怀疑与排查过程

`stats_stale` 判断的是「记录均值与实测均值的相对偏差」，本质是一个比值指标。比值指标出错，第一个要问的就是**分母是什么**。

1. 取该数据集 `stats.json` 的 `observation.state.mean` / `std`，与实测均值逐维相减，得到绝对差 `|Δ|`；
2. 观察最大 `|Δ|` 所在的维度：该维 `recorded_mean` 约 **1e-4**，`std` 约 **0.1**，`|Δ|` 约 **0.1**；
3. 代回旧式 `|Δ| / |recorded_mean|` ≈ 0.1 / 1e-4 ≈ **1000 量级**。而 `|Δ|` 与 `std` 同量级，说明**漂移是「相对噪声尺度」的正常量级，只是分母选错了**。

这类维度在机器人数据里很常见：关节零位附近的分量、对称往复信号、以及四元数中理论均值为 0 的分量。它们的 `|mean|` 天然接近 0，用它作分母必然爆表。

### 证据证伪

- 旧式：`|0.0991 - 1.1e-4| / |1.1e-4| ≈ 913`；
- 现式：`|0.0991 - 1.1e-4| / max(|1.1e-4|, 0.1) ≈ 0.99`；
- 同一份数据、同一组实测均值，仅换分母，量级从「荒谬」回到「可解释」。

因此结论明确：**913 不是数据问题，是度量缺陷**。修复后该数据集偏差 0.99 仍高于 5% 容忍，说明它的 stats 确实过期——只是这个结论现在可以说清「过期了多少」。

### 代码修复方案

`pipeline/validator/context.py::_stats_staleness_issue()`：

```python
# 旧：分母取 |mean|，近零维度爆表
deviations = np.abs(actual_mean - recorded_arr) / np.abs(recorded_arr)

# 新：分母取该维自身量级，带 1e-6 下限防止 0/0
scale = np.maximum.reduce([
    np.abs(recorded_arr), np.abs(recorded_std), np.full_like(recorded_arr, 1e-6)
])
deviations = np.abs(actual_mean - recorded_arr) / scale
```

配套改动：

- `detail` 中同时输出 `worst_dims`（偏差最大的 3 个维度下标）、`recorded_mean_first3`、`actual_mean_first3`，让人工复核能直接判断「是否真的过期」，而不必重跑；
- 子集扫描（`--max_episodes`）时该检查标记为**不可比**并写入 `skipped_checks`，因为子集均值与全量 stats 天然有差，报成「过期」是必然假阳性。

### 防回归测试用例

| 用例 | 断言 |
|---|---|
| `test_stale_stats_detected` | 注入过期 stats 后必须报 `stats_stale`，且 `max_relative_deviation > tolerance` |
| `test_fresh_stats_not_flagged` | 干净数据集不得报 |
| `test_streaming_stats_skipped_when_sampling_partial` | 部分采样时进 `skipped_checks`，不得判为过期 |
| `test_near_zero_mean_does_not_amplify_small_drift` | 本案例回归：`mean≈1e-4`、`std=1.0`、漂移 0.02 → 旧式报 199 倍，现式**必须不报** |
| `test_real_stats_drift_is_reported_at_true_magnitude` | 同一公式不得放过真实漂移：漂移 0.2 必须报，且值域落在 `[0.19, 0.21]`（旧式为 2000 量级） |

最后两条是本案例新增的：一条防假阳性，一条防「修过头」。

---

## 案例 3：版本契约冲突——同是 v2.1，一处报错一处不报

### 现象

对 17 个本地数据集做全量审计（v3.0 与 v2.1 混合，共 4108 个 episode）时，两个 **v2.1** 数据集的结论互相矛盾：

| 数据集 | 版本 | episode 数 | `index` 判定 |
|---|---|---|---|
| `kuavo4_hotel_services` | v2.1 | 484 | 不报 |
| `humanoid_everyday_h1` | v2.1 | 1418 | **error【阻断】**，置顶在所有剔除建议之前 |

两者都是「每个 episode 一个 parquet」。同一版本、同一物理布局，判据应当一致；出现不一致，说明判据里藏了一个我们没意识到的隐变量。

### 怀疑与排查过程

排查从「两者还有什么不同」入手，逐个排除：fps、相机路数、视频编码、字段名、`splits` 内容——都不是判定依据。最后落在加载路径上：

1. `kuavo4` 的 `meta/episodes` 行数与 `data` 中的 episode 数一致 → 加载层成功**重建**了全局连续的 `index`（0..N-1），送给契约审计的是一个合法 v3 形态，因此「不报」；
2. `humanoid` 的 `info.json` 声明 4885 个 episode，实际只有 1418 个 parquet → 重建被跳过（数量不匹配，无法安全重建），原始的「每集从 0 起」直接暴露给 v3 契约，因此「报 error」。

即：**判定结果取决于「是否重建」，而不是「数据是否合规」**。这是判据设计的错误——契约必须由数据自身的版本决定，不能由中间处理步骤的副作用决定。

### 证据证伪

直接读 `humanoid` 的 parquet：每个 episode 的 `index` 是 `0..length-1`，无缺失、无重复、无越界。这正是 **v2.1 的规范语义**（episode 内帧号），不是损坏。用 v3.0 的「全局唯一绝对帧号」去要求它，等于拿另一份格式规范去判它违规。

反过来逐条核验 `kuavo4` 的 `index`：同一 episode 内连续、不重复，但起点是全局偏移（ep1 为 `131..471`）——这同样合法，且说明 v2.1 的两个合法形态（每集归零 / 沿用全局偏移）都必须被接受。

### 代码修复方案

1. **契约按版本选择（`pipeline/schema.py::audit_schema`）**
   入参增加 `codebase_version`，`is_v2 = version.startswith("v2")`：
   - v3.0：`index` 必须是全局 `0..total_frames-1` 各出现一次（原判据保留）；
   - v2.x：只要求 **episode 内**连续且不重复，起点允许 0 或全局偏移；
   - 两种版本都追加「每集唯一值个数 == 行数」兜底，用于抓「重复与断号互相抵消」（如 `[0,1,1,3]`：4 行、跨度 4，仅判跨度会通过）。

2. **同一判据在三条路径上同步（`context.py` / `streaming.py`）**
   `_check_index_continuity()` 的 v2.x 分支改为「集内连续 + 唯一值个数」，`streaming.py` 的流式分支同步，避免「全量跑报、流式跑不报」。

3. **单 episode 数据集补集内判据**
   原先 `len(summary) > 1` 才判，单 episode 数据集完全放过。现补 `elif len(summary) == 1` 分支：只判「跨度 == 行数」「唯一值 == 行数」，**不判起点**——单集没有前序可比对象，判起点会把「保留全局偏移的合法子集」误杀。

4. **报错定位到真实编号**
   `_sample_episodes()` 取代 `summary.index[...]`：后者是 DataFrame 的位置下标，编号经过 remap 或非顺序时会指到别的 episode 上。

### 结果

| 数据集 | 修复前 | 修复后 |
|---|---|---|
| `humanoid_everyday_h1` | `index: error`【阻断】+ 无计数提示 | index 误报消失；改为报出真实缺陷 `total_frames` 差 **1,207,240**（info 声明 1,659,786，实际 452,546） |
| `kuavo4_hotel_services` | 不报（但同版本另一集报，判据不可信） | 不报，且结论可解释（v2.x 集内合规） |
| `behavior1k`（v3.0） | `index: error` | 保持 error（真实违约：每集归零），并给出总行数与取值范围 |

### 防回归测试用例

| 用例 | 断言 |
|---|---|
| `test_v2_index_restart_per_episode_is_compliant` | v2.1 每集归零 → **不得报** |
| `test_v2_global_offset_index_is_compliant` | v2.1 沿用全局偏移（如 `131..471`）→ 不得报 |
| `test_v2_index_gap_within_episode_is_error` | v2.1 集内断号 → 仍必须报 error（放宽不等于放过） |
| `test_v2_index_duplicate_cancelled_by_gap_is_error` | `[0,1,1,3]` 形态必须报，且证据里带唯一值个数 |
| `test_single_episode_index_damage_is_detected` | 单 episode 集内重复 → 必须报 |
| `test_index_restart_per_episode_is_error` | v3.0 每集归零 → 仍必须报 error（版本分支不得削弱 v3 判据） |

---

## 案例 4：`action_jump` 阈值退化——45 条命中全部贴着阈值

### 现象

对 17 个数据集做全量审计后，`action_jump` 共命中 45 条（分布在 8 个数据集），特征高度一致：

| 数据集 | 命中数 | 阈值全为 `abs_floor` | 超阈倍数 min / 中位 / max |
|---|---|---|---|
| `pusht` | 17（17/206 = **8.25%**） | 0 | 1.06 / 1.28 / 1.86 |
| `humanoid_everyday_h1` | 14 | **14（全部）** | 1.01 / 1.10 / 1.36 |
| `behavior1k_task0042` | 5 | 0 | 1.02 / 1.19 / 1.35 |
| `gym_xarm_pc` | 3 | 0 | 1.24 / 1.29 / 1.33 |
| `kuavo4_hotel_services` | 3 | 0 | 1.05 / 1.14 / 1.20 |
| `g1_place_grape_error` / `koch_pick_place_5_lego` / `svla_so101_pickplace` | 各 1 | 视机型而定 | 1.15 – 1.63 |

两个信号：

1. **没有任何一条是数量级级别的跳变**，全部落在阈值的 1.01–1.86 倍区间；
2. `humanoid` 的 14 条，所用阈值**全部恰好等于配置下限 `abs_floor=1.0`**，说明自适应阈值在该数据集上没有生效，退化为绝对值判据。

### 怀疑与排查过程

分两问排查，因为这两件事的根因完全不同。

**问一：为什么阈值会退化成 `abs_floor`？**
阈值定义是「逐维 = `max(abs_floor, median(每集最大|Δ|) + k · 1.4826 · MAD)`」，即对「每集峰值」这一分布取稳健上界。动捕类数据（120 Hz）每集峰值本身很小，`median + 3σ` 算出来低于 `abs_floor`，于是被夹到 1.0。此时阈值与数据分布无关，任何 1.0 以上的帧间差都会命中。同时该数据的单位是米/弧度混用，`abs_floor` 的标定还依赖 fps——120 Hz 与 30 Hz 的「每秒位移」并不等价。

**问二：为什么「平滑加速」和「单帧尖峰」会一起命中？**
原判据只看 `|Δ_t|` 是否超过阈值。一段快速但连续的加速运动，相邻帧的 `Δ` 同样很大，与「单帧突跳、前后帧都平缓」在 `max|Δ|` 这个统计量上不可区分。**这是判据维度不足，不是阈值不够大**——继续调大阈值会把真实尖峰一起放过。

### 证据证伪

对每条命中取该点邻域（±3 帧，去掉自身）的 `Δ` 中位数 `ref`，与命中值比较：

- 平滑快速运动：`ref` 与命中值同量级 → 比值接近 1，不是尖峰；
- 单帧尖峰：邻域近似静止，`ref ≈ 0` → 命中值是邻域的数十倍。

另一类被证伪的是动捕数据的四元数表示等价：整体取反（q 与 -q 表示同一姿态）会让 4 个分量在同一帧同时出现约 `2|x|` 的跳变。校验用 `|x_t + x_{t-1}| ≤ 0.25 · |x_t - x_{t-1}|` 且**同帧 ≥2 个维度同时满足**识别（单维翻转仍按真实跳变处理）。

### 代码修复方案

修复分两阶段，第一版只做「形态区分」，复测后才发现「落点分布」还需要第二版。

**阶段一：相对邻域突跳约束（`pipeline/validator/rules.py`）**

```python
# 邻域参考：同一维在 ±window 帧内（去掉自身）的 Δ 中位数
neighbours = np.delete(deltas[lo:hi, d], f - lo)
ref = float(np.median(neighbours))
# 邻域全静止（ref≈0）时任何超阈值的跳变都算尖峰；
# 否则要求命中值达到邻域的 spike_ratio 倍，才能与「快速但连续」区分开
if ref <= 0 or deltas[f, d] >= spike_ratio * ref:
    kept.append((f, d))
```

配置：`spike_ratio=3.0`、`spike_window=3`（`pipeline/config.py`）。同时加入四元数符号翻转的分组识别，以及 `abs_floor` 按数据集 fps 折算（以 `ref_fps` 标定）。

**阶段二：按超阈倍数分级（`rules.py` + `config/default.yaml`）**

复测发现阶段一之后仍有 45 条命中，且全部贴阈值——说明这些是「真实但幅度接近阈值」的边界样本，判 warning 只会制造稳定误报。因此按倍数分档：

| 超阈倍数 | 处置 | 理由 |
|---|---|---|
| < `downgrade_ratio`（1.5） | `info`（记录不判定） | 阈值本身取自峰值分布的 3σ 上界，1.0–1.5 倍属正常快速动作 |
| 1.5 – `escalate_ratio` | `warning`（转 review） | 需要人看 |
| ≥ `escalate_ratio`（默认 `null`，不启用） | `error` | 留出开关，默认不新增剔除 |

报告的 `details` 同步输出 `exceed_ratio`、`thresholds_used`、`spike_ratio`、`smooth_peaks_ignored`、`sign_flips_ignored`，让「为什么这条被放过」可追溯。

### 结果

- 平滑快动作与四元数取反不再命中（回归用例覆盖）；
- 45 条贴阈值命中按倍数分流：`pusht` 的处置由 `keep 189 / review 17` 变为 `keep 203 / review 3`（14 条降为 `info`、3 条仍达 warning 门槛），人工复核队列从 17 条压到 3 条；健康度仍为「中等」，因为那 3 条确实值得人看一眼——分级的目标是压缩队列，不是把报告刷成绿色；
- 真正的单帧尖峰仍按 `warning` 转人工——`spike_ratio` 只剔除形态，不放宽幅度。

### 防回归测试用例

| 用例 | 断言 |
|---|---|
| `test_action_jump_ignores_smooth_fast_motion` | 峰值超阈值但邻域同样大的连续运动不得命中；同数据集的单帧尖峰必须命中，且 `details["spike_ratio"] == 3.0` |
| `test_action_jump_uses_adaptive_threshold` | 正常抖动轨迹不得命中、单帧尖峰必须命中，且逐维阈值数量与 action 维度一致 |
| `test_action_jump_ignores_quaternion_sign_flip` | q / -q 整体取反不得命中 |
| `test_action_jump_thresholds_scale_with_fps` | 30 Hz 与 120 Hz 的同物理运动，阈值按 fps 折算 |
| `test_action_jump_near_threshold_downgrades_to_info` | 超阈 1.2 倍 → severity 覆盖为 `info` |
| `test_action_jump_far_over_threshold_keeps_default_severity` | 超阈 5 倍 → 保持配置默认级别，不得被降级 |

---

## 共性：四次纠偏提炼出的四条验收标准

四次事件的根因不同，但都可以归到四个「AI 写代码时不会主动问、而业务方必须回答」的问题上。

**1. 判据的隐含前提是否显式化了？**
案例 3 里，判据隐含了「数据已被重建」这一前提；案例 4 里，阈值隐含了「fps 与单位已标定」。前提不显式，判据就恒等于「看起来对」。落到工程上：任何规则必须写明它假设的格式版本、单位、采样率，并在不满足时降级为「不可评估」而不是「不通过」。

**2. 「不可评估」是否被当成了「不通过」？**
案例 1 的 200 条 error 全部来自这一处混淆。这条流水线现在的三态是硬约束：`skipped_checks` 里的项不得参与处置、不得计入健康度，也不得进剔除建议。

**3. 比值型指标的分母是什么量级？**
案例 2 的 913 倍是个纯算术事故。凡出现「相对偏差」「超阈倍数」「命中率」，都要能回答分母的物理含义与量级；分母接近 0 的维度必须有下限或改用尺度量（`max(|mean|, std)`）。

**4. 阈值是从分布推出来的，落点复测过吗？**
案例 4 的第一版修复只解决了形态，没解决落点；`humanoid` 14/14 命中全等于 `abs_floor` 这件事，只有把 `thresholds_used` 打出来逐条比对才会发现。规则上线前必须跑一遍全量数据，检查命中是否集中在阈值边界、单数据集命中率是否超过 5%。

### 人工审核在这套系统里的位置

这套工具的设计意图从来不是「全自动删数据」，而是「把 4000 个 episode 压缩成几十条需要人判断的条目」。四次纠偏中，人的作用都不是写代码，而是提出一个 AI 不会主动提出的反问：

- 「100% 全错，是数据太差，还是我们读错了位置？」
- 「913 倍这个数，物理上讲得通吗？」
- 「同一版本的两个数据集结论不同，哪一个是错的？」
- 「45 条全部贴着阈值，这叫命中还是叫误报？」

这四句话构成了本项目的验收标准：**任何一条结论，都必须能被业务常识反问一次而不倒。**
