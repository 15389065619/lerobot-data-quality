# 与主流方案的对比

选型时经常会被问到：已经有 LeRobot 官方工具、ROS 工具链、开源质检工具和商业数据平台，为什么还要这一层？

现有工具大致分两类：搬运类（load / convert / delete / repackage）和巡检类（check / score / gate）。本项目在巡检类的基础上多做一步：把判不准的样本变成可裁决的队列，把裁决结果变成可复用、可审计、可导出的处置记录（diagnose → review → override → export）。差异在处置闭环，不在检查项数量。

需要先说清楚：在检查项覆盖上，开源社区已有比本项目更全的实现（见下节 `lerobot-doctor` / DataDoctor）。本项目的规则不构成差异化优势，它的作用是把巡检结果接进一条可审计的处置链路。如果只需要一次性的体检报告，用 `lerobot-doctor` 更省事。

## 同生态质检工具

除上游工具链外，还有一类更接近的对手：专门做 LeRobot 数据集体检的巡检工具。截至 2026-11，公开可用的至少有四个：

| 工具 | 形态 | 覆盖范围 | 自动化接口 |
|---|---|---|---|
| `lerobot-doctor`（jashshah999） | CLI，源码安装 | 基础 check 集 | JSON / Markdown |
| DataDoctor（ForgeAI，基于上游扩展） | CLI，源码安装 | 12 类 check + 全帧视频解码审计 + URDF 物理限位 | `--ci` 退出码、`--fail-on=warn` |
| `lerobot-quality-gates`（PyPI） | CLI，轻量 | 元信息 / 时间戳 / 形状 / 视频引用 / 标注缺失 | `--fail-on`、HF dataset card 输出 |
| Physical Intelligence `validate.py`（`pi-data-sharing`） | 内部脚本 | 元信息契约 + 标注 JSON 跨字段一致性 | 上传前校验 |

DataDoctor 还提供 `fix`（自动修复，带备份）、`trim`（裁首尾空闲帧）、`score`（按平滑度、覆盖度、一致性、长度给 episode 排序并推荐删除）、`gate`（按 ACT / Diffusion / SmolVLA / Pi0 检查兼容性）、`merge-check`（合并前后校验）。

这些功能本项目没有，也不打算补。`gate` 属于训练侧契约，`merge-check` 属于数据集编排，都超出单数据集质量治理的边界。

商业平台不在逐项对照之列。Foxglove Data Search & Curation（2026-04 上线）、Encord、Roboto 解决的是采集、索引、检索、标注的规模化问题，能力更强，但需要接入采集链路、按席位付费，目标不是拿一个离线数据集做一次性治理。把它们放进表格对比检查项，等于拿两种不同的东西比大小。

### 逐项对照

| 维度 | DataDoctor / lerobot-doctor | 本项目 |
|---|---|---|
| 检查项 | DataDoctor 12 类（含 URDF 物理限位、全帧解码审计）；上游 `lerobot-doctor 0.2.0` 实为 11 类，无这两项 | 17 条数值/文件规则 + 8 条点云规则，无 URDF |
| 严重度模型 | PASS / WARN / FAIL 三档 | keep / repair / review / drop 四档 |
| 判不准的怎么办 | 归入 WARN 或 per-episode drilldown，由使用者判断 | 进 `review_queue.csv`，带视频路径、时间窗、证据片段 |
| 人工裁决回灌 | 无。没有 episode 级处置状态，`score` 只给一次性建议 | `--overrides` 把裁决写回，重跑时复用，结论稳定 |
| 对源数据 | `fix` / `trim` 原地改写，备份仅含 `meta/`，被重写的 parquet 没有备份 | 源目录只读，清洗产物独立落盘 |
| 子集模式 | 跳过项记为 PASS，默认视图不显示该消息 | 检测到覆盖不足时拒绝导出，不给部分结论 |
| 增量重检 | 无（incremental 仅指 HF 下载分批取数） | `--incremental` + manifest 复用 |
| 下游收益量化 | 无（`score` 只排序，不量化清洗前后分布） | `evaluation/analyze_action_distribution.py` 三段测量（选择效应与修复效应分离） |

### 能力清单

只列上面已经核实过的事实，不新增判断。

双方都有：

- 数值和元信息类检查：NaN、时间戳、帧数、动作形态
- v2 / v3 布局识别
- 逐 episode 下钻的 JSON 报告
- `--max-episodes` 子集模式

对方有、本项目没有：

- URDF 物理限位：需要外部真值，本项目手里多数是来源不明的公开数据
- 全帧视频解码审计：本项目只做采样探针，避免全量解码的 IO 成本
- `--ci` / `--fail-on` 训练门禁：属于训练侧契约，超出单数据集质量治理的边界
- `fix` / `trim` / `score` / `merge-check`：原地编辑和数据集编排，交给官方 CLI 或下游
- 注：`kinematics` 与全帧审计属于 DataDoctor fork，PyPI `0.2.0` 包内不存在，本章不对其源码下结论

本项目有、对方没有：

- episode 级 disposition + `--overrides` 回灌，重跑结论可复现
- 源目录只读，清洗产物独立落盘
- 覆盖不足时拒绝导出，不给部分结论
- `--incremental` + manifest 增量重检
- 三段测量法（选择效应与修复效应分离）

### 三条实质差异

1. 自动判定还是分派给人。DataDoctor 的目标是给出 PASS/FAIL 并接进 CI，它必须自己下结论。本项目承认部分样本（`action_jump` 超阈 1.0–1.5 倍这类）在规则层面判不了，于是把它们放进 review 档并生成可裁决的队列。这不是更聪明，是承认边界：规则能判的判，判不了的写成问题交给人。实测在 demo 集上，13 处注入若一刀切要删 12 处，分级处置只删 7 处。

2. 就地修复还是源不可变。`fix` / `trim` 原地改写数据集本身（备份仅含 `meta/`），适合「这份数据就是我的，改完继续用」。本项目按源只读、产物独立处理，适合「数据是别人给的，我要留证据说明改了什么」。两者都对，取决于你是否需要事后回答「原始长什么样」。

3. 单次巡检还是结论可复现。CI gate 每次全量重跑没有负担，但治理场景下第二次运行的结论必须能和第一次对上，否则增量质检会漂移。这也是本项目要持久化 `severity_overrides`、要求 `from_dict` roundtrip 的原因。

### 源码核实

核实范围：PyPI `lerobot-doctor 0.2.0`（wheel 51 KB，Python 4,117 行，`runner.py` 实际注册 11 个 check）。README 宣称的 12 类中，`kinematics`（URDF 物理限位）与全帧视频审计在 0.2.0 包内不存在，它们是 DataDoctor fork 的扩展，本表未核实那部分源码。

六条结论都能落到源码：

| # | 问题 | 结论 | 证据 |
|---|---|---|---|
| 1 | episode 级 disposition 持久化 | 无。没有「这集已处置」的状态概念，只有检查结果 | 全仓检索 disposition / override / decision / review，仅命中 `gate.py:73` 的一句 CLI 帮助文本 |
| 2 | `score` 推荐删除能否回灌 | 不能。`score` 只支持 `--json` 打到 stdout，没有输出文件参数；`recommended_drops` 仅作为返回值，`fix` / `trim` 都不读它 | `cli.py:50-54`、`score.py:116` |
| 3 | `fix` / `trim` 写入路径 | 原地改写，且 `backup` 只复制 `meta/`，被重写的 `data/*.parquet` 不在备份内 | `fix.py:46-49`（copytree 仅 meta）、`fix.py:170/236/284`（`pq.write_table` 直写原文件）、`trim.py:136` |
| 4 | 增量重检 | 无。incremental 只出现在 HF 下载阶段的分批取数，没有质检缓存或 manifest | `dataset_loader.py:293-400` |
| 5 | 部分模式下被跳过的检查如何呈现 | 记为 PASS，且默认（非 verbose）视图不打印该消息，用户只看到 `[PASS] Metadata` | `metadata.py:81-83`、`report.py:116`（PASS 消息在非 verbose 时被跳过） |
| 6 | 独立导出干净数据集 | 无。全仓唯一一处 `copytree` 就是第 3 条那个只含 meta 的备份 | `grep copytree` → `fix.py:49` 单处 |

据此修正上表两处措辞：人工裁决回灌和增量重检都不是「公开 README 未提及」，而是源码里没有这个设计。

有两点需要说明，否则会变成踩竞品。

第 5 条不是缺陷，是取舍。`--max-episodes` 是用户主动抽样，把抽样下无法判断的项记 PASS 对 CI gate 是合理的，否则每次抽样都 FAIL。真正的差异在用途：它的报告是 gate 的输入，抽样 PASS 可以接受；本项目的报告要当结论用，所以覆盖不足时必须拒绝导出。这一条保留自己的做法，但不认为对方错了。

第 3 条值得提醒。备份只含 `meta/`，而 `fix` 的 `reindex` / `timestamps` / `nan` 三类修复都直接重写 `data/*.parquet`。也就是说，如果误用 `fix` 改坏了数值，`*_backup` 目录里没有原始数据可回滚。这不是设计缺陷（有 `--dry-run` 可以先预览），但 README 只写 "creates backup"，容易被读成全量备份。

复现方式不需要访问 GitHub，PyPI 即可：

```bash
pip download lerobot-doctor --no-deps -d /tmp/ld
cd /tmp/ld && unzip -o *.whl -d src
sed -n '46,50p'  src/lerobot_doctor/fix.py                 # 备份范围：只 copytree meta
sed -n '81,84p'  src/lerobot_doctor/checks/metadata.py     # partial 模式记 PASS
sed -n '114,118p' src/lerobot_doctor/report.py             # 非 verbose 下 PASS 消息被跳过
```

行号对应 `0.2.0`，后续版本以源码为准。

## Schema 契约严谨度对照

前文比较的是检查项和处置闭环。这一节回答另一个问题：本项目的 schema 契约，严谨度对不对得上官方和同类工具。

### 一个前提

`lerobot-doctor`（jashshah999）不是官方工具，也不依赖 `lerobot` 包。官方的质检分散在三处：写侧（`record` 的 `validate_frame`、`finalize()` 强制收敛）、读侧加载器（刻意宽松）、官方 CLI（`lerobot-edit-dataset` 只删不修）。v3 文档第一段写明设计目标是 "decoupling storage from the user API"，读侧信任元数据、软失败，严谨性被前置到生产者。

所以比较的正确姿势是：官方读侧不是标准答案，本项目补的是它不查的洞。

### 官方读侧查什么

据官方 main 分支源码（`datasets/utils.py`、`lerobot_dataset.py`，核实于 2026-11）：

| 行为 | 证据位置 |
|---|---|
| `DatasetInfo` 只强制 `fps > 0`、`chunks_size > 0`、两个文件大小字段 > 0 | `DatasetInfo.__post_init__` |
| `info.json` 未知字段忽略（warning，不报错） | `DatasetInfo.from_dict` |
| `splits` 仅声明为 `dict[str, str]`，不校验区间长度、连续性、与 `total_episodes` 的一致性 | 字段定义 |
| 越界 episode index：丢弃并 warning，不报错 | `resolve_episode_indices`、`LeRobotDataset.__init__` |
| 不核对 meta/episodes 行数与 parquet 实际行数，也不核对 `total_episodes` / `total_frames` | 全链路未见断言 |
| 时间戳同步检查存在，但由 `tolerance_s` 参数化 | reader 模块与 `lerobot_dataset.py` 文档字符串 |

读侧信任包装标签：标签写 50 箱就按 50 箱上架，缺箱记一笔警告照收。软失败的后果正是本项目 `schema.py` docstring 开头那句：「数据集能跑起来」不等于「Schema 合规」。

### 三方对照

| 契约点 | 官方读侧 | `lerobot-doctor` | 本项目 |
|---|---|---|---|
| `info.json` 必填字段 + `fps > 0` | ✓（仅此） | ✓ | 读，缺字段走兜底（见 G3） |
| `index` 全局唯一且起点等于前序累计行数（v3）；集内连续（v2，按版本分支） | ✗（重复键在 `_build_index_mapping` 处塌缩，是官方自身隐患） | 部分（temporal 查 broken index） | ✓ 独有，`schema.py:166` |
| `total_episodes` / `total_frames` 对账 | ✗ | ✓ | ✓ `schema.py:245`，超 1% 升 error |
| `splits` 区间长度 == `total_episodes` | ✗ | ✗ | ✓ 独有，`schema.py:224` |
| 时间戳单调 / 与 fps 一致 | ✓（`tolerance_s` 参数化） | ✓ | ✓（`step_tolerance_ratio` 0.25） |
| `features` 声明 dtype / shape 与实际是否一致 | 仅保证结构存在，不验值 | ✓（跨数据集 `consistency`） | ✗ 缺口 G1 |
| `total_tasks` / `total_videos` / `total_chunks` 对账 | ✗ | 部分（tasks.parquet 相关） | ✗ 缺口 G2 |
| chunks 布局不变量（chunk / file 编号） | 写侧有 rollover 规则，读侧不查 | 部分（portability） | ✗ 缺口 G3 |
| 去重 / 跨 split 泄漏 | ✗ | ✓ `identity`（SHA-256 全列签名 + 泄漏判 FAIL） | 近似：`duplicate_episode`，按 episode 判 |
| URDF 物理限位 | ✗ | ✓ `kinematics` | ✗（刻意，无外部真值） |
| 黑屏 / 点云 8 条 / 自适应阈值 | ✗ | 视频部分，无黑屏、无点云 | ✓ 独有 |

### 结论

契约面上，本项目不是「达不到官方严谨度」。上表前三条硬契约（index 语义、splits 长度、两级对账）官方读侧根本不查，本项目是在补洞，不是追标准。官方的严谨在写侧和 v2 时代，v3 读侧是刻意宽松。

但确实有三个缺口，按优先级排列：

| 编号 | 缺口 | 风险 | 修法 | 状态 |
|---|---|---|---|---|
| G3 | `chunks_size` 缺声明时按 `1000` 兜底，两处各自兜底：`loader.py:404`（拼路径）、`loader.py:760`（扫分片） | 归因错误。把「元数据没声明」偷换成「文件丢失」，用户会去查存储而不是查 `info.json` | 统一为单一解析函数，兜底值保留但标记来源为 assumed 并产出 warning | 未修 |
| G2 | 无 `total_tasks` / `total_videos` / `total_chunks` 对账，未查 `task_index` 越界 | 与已修的 episodes / frames 两条同构，属同类漏检 | 复用现有对账结构，约 10 行，warning 起步 | 未修 |
| G1 | 只把实际 dtype 打印出来（`loader.py:260-277`），从不拿 `info.json` 的声明对账 | 声明 `action` 7 维、实际 8 维时，`delta_timestamps` 取帧静默错位；dtype 不一致则依赖下游 cast | 两条检查：声明 dtype vs parquet physical dtype；声明 shape vs 实际列长。`image` / `video` 列必须白名单跳过 | 未修 |

G1 需要先在本地数据集上跑一遍统计误报率，再决定严重级（口径统一走 `tools/dataset_inventory.py` 实时统计，2026-11-17 实测 30 个根目录 / 6648 episode）。`image` / `video` 列的 parquet dtype 与声明天然不同，直接全量比较会制造噪声。

### 规则层缺口（2026-11-17 复测新增，当日修复）

来源：从 hf-mirror.com 新拉取 `lerobot/tokyo_u_lsmo`（24 MB / v3.0 / 50 集 / 11925 帧 / fps=5 / `robot_type` 未声明）全量复测，契约检查 0 违反，判定结果 24 review（health=severe）——问题不在契约层而在规则/阈值层；0 剔除、0 修复，全部转人工（数据见 [EVAL.md §2.6](EVAL.md)）。

> 归因修正：初版（8451913）把 G5 归因为「规则不识别 ±π 卷绕」。评审用原始 parquet
> 逐帧核对后修正：state 侧卷绕属实（端点精确贴 ±π），但 action 侧的 ±2π 脉冲
> 是成对（连续两帧）超阈 Δ 互为对方唯一邻域、互相抬高中位数导致的**漏报**——
> 机制与卷绕正交，任何两帧成对异常都会中招。G5 据此拆为 a/b/c 三条。

| 编号 | 缺口 | 风险 | 修法 | 状态 |
|---|---|---|---|---|
| G4 | `frame_count_abnormal` 在 MAD=0 且 ptp>0 时区间塌缩成单点（旧实现刻意塌缩以抓唯一离群集）：tokyo 帧数 237/238/239 三种（1/23/26 集），±1~2 帧抖动把 24/50 集全体判越界送复核，health=severe、valid_ratio=0.52——**已观测的主损失** | 次风险：若中位数落在另一侧，过长集走 error→drop 不可逆路径（M3 的 ep306 是先例）；塌缩区间使录制抖动即可触发批量越界 | 最小宽度 `median ± max(1, ceil(ratio×median))`，`frame_count.mad0_min_width_ratio` 可配（默认 0.01）；评审否决观测 [min,max] 方案（用被检样本定义边界 = 零检测，截断集永久合规）；长侧边际降级（M3）保留为第二道防线，未修 | **已修**（b8e00d7） |
| G5a | spike_window 低帧率塌缩（fps=5 → 1 帧），**成对超阈 Δ 互为对方唯一邻域、互相抬高中位数**，spike_ratio 判不过 → 整对漏检；与卷绕正交，任何两帧成对异常都中招（tokyo action dim5 的 4 个 ±2π 脉冲全部漏报） | 训练目标里的 ±2π 指令脉冲静默漏检——state 侧只是表示等价，action 是训练输入 | 邻域 ref 剔除其他超阈 Δ；剩余邻居 <2 时，仅**异号往返**（\|Δ1+Δ2\| ≤ 0.2·max\|Δ\|）的 2-diff 脉冲回退全局逐维 median(正 Δ)（`diff_ref` 记录口径；同号 2-diff 是快速斜坡，不兜底）；超阈 Δ 按 run 分组，≥3 为运动包络整体跳过（反向护栏）；脉冲不做幅度降级（下限 warning 进复核） | **已修**（194d52e + 形状门槛补丁） |
| G5b | state 侧 ±π 卷绕被当 `state_jump` 送复核——表示层等价而非数据缺陷 | 合法表示被当突跳，复核队列混入不需要人处理的条目 | 端点判据（两侧距 ±π 与 \|Δ\| 距 2π 均在 `wrap_endpoint_tol`=0.05 rad 内）转独立 info 规则 `state_wrap` 留痕（不修复，下游按连续量使用需自行 unwrap）；**不用整维值域判据**（tokyo dim3 合法落在 [−π,π] 内会被误豁免） | **已修**（31bb938） |
| G5c | action 侧 ±2π 脉冲：dim5 与 `next_state − state` 逐帧一致（corr 0.983、卷绕帧逐位相等）→ 单独修 action 会破坏 state-action 自洽，真实物理量不可知；且 4 个脉冲把该维 std 从 0.003 抬到 0.115，**归一化统计已被污染** | 按记录值归一化会在脉冲帧上放大离群；静默改 stats 等于编造数据 | 处置 review 不修复；新增 `stats_pollution` 证据（rules 出单集 observed/trimmed std，reporter 加权池化并对照 stats.json 记录值，仅当记录值 ≥1.5× 剔尖峰值才发数据集级提示；stats.json 不动，子集运行标 partial） | **已修**（31bb938） |

### 不补的项

- SHA-256 全列签名去重：`duplicate_episode` 是近似判据（长度 + 采样帧），对同长度不同内容和同内容不同长度都有边界，文档标明即可，不引入全列哈希的 IO 成本；
- 跨 split 泄漏：v3 的 splits 是单一 `train` 区间，暴露面远小于 NLP 数据集的多 split 场景；
- URDF 物理限位：无外部真值，前文已说明；
- 字段命名规范前缀：属于训练侧契约，下游自会失败。

### 定位

官方把契约压在生产侧，读侧是软失败；`lerobot-doctor` 用巡检补读侧；本项目在读侧补了 3 条官方根本不查的硬契约（index 语义、splits 长度、两级对账），缺 features 和 chunks 两张表，但多一条分级处置闭环。官方写侧仍有本项目不具备的强约束，所以不能说成「比官方严得多」。

### 与前文的关系

前文源码核实的是 PyPI `0.2.0`（11 个 check）。截至 2026-11，其 README 已声明 13 类，新增 `identity` 与 `kinematics`；新增项的源码未复核，因此本节 `lerobot-doctor` 一列只按 README 的能力描述对照，不与 0.2.0 的源码结论混用。

## 上下游工具链对比

| 维度 | 本项目<br/>lerobot-data-quality | LeRobot 官方<br/>`lerobot-edit-dataset` 等 | ROS 2 工具链<br/>rosbag2 / rosbag_editor | OpenVLA 数据管线<br/>RLDS/OXE 转换脚本 |
|---|---|---|---|---|
| 定位 | 数据质量治理层 | 数据编辑层 | 录制/转换层 | 数据打包层 |
| 输入 | LeRobot v3 / v2.1（parquet + mp4） | LeRobot 数据集（本地或 Hub） | rosbag2（`.mcap` / `.db3`） | Open X-Embodiment RLDS/TFDS 分片 |
| 逐帧质量诊断 | 17 条数值/文件规则 + 8 条点云规则，自适应阈值 | 无（不做质量判定） | 基本无（偏传输完整性、时间戳检查） | 无（以轨迹级筛选为主） |
| 自动修复 | 有：裁边界坏帧、重建时间轴、重算 meta 索引与 stats、点云定长补点 | 有编辑能力（如按 episode 删除），不做内容修复 | 有帧裁剪/重采样，但不是为训练数据质量设计 | 无 |
| 分级处置 | Keep / Repair / Review / Drop 四档 | 以删除为主 | 不涉及 | 保留/丢弃轨迹 |
| 审计与可追溯 | 报告含数据指纹、阈值快照、逐帧证据、修复动作清单，可 diff | 操作日志有限 | 依赖用户自建 | 依赖用户自建 |
| 人工复核回路 | 有：导出 `review_queue.csv`（带视频路径、时间窗、证据），填写后 `--overrides` 回灌 | 无 | 无 | 无 |
| 点云质量 | 有：NaN、稀疏、退化、重复、离群、越界、颜色、时序跳变 8 条规则 | 无 | 无（ROS 侧通常另行处理 PointCloud2） | 视下游而定，通常不做质量判定 |
| 并发/续传 | `--workers` 多进程；`--incremental` 两级断点续传 | 视实现而定 | 无 | 无 |
| 与本项目的关系 | — | 纯删除场景可委托官方 CLI（`--no-cli` 可禁用） | 需先转换到 LeRobot 格式，再接本项目 | 本项目在其上游做质检，避免脏数据进入打包 |

## 适用场景

适合本项目的场景：

- 采集数据存在静默污染（视频与 state 行数错位、meta length 漂移、stats.json 过期），训练不报错但学不动；
- 数据稀缺，不能一刀切删除，需要「能修则修、不能修才删、拿不准转人工」的分级处置；
- 需要向采集或标注团队归因（哪一批、哪个人、哪台机器异常率高）；
- 点云特征需要在定长契约内修复（训练张量形状不能变）；
- 需要一份能被复核、能被 diff 的质检报告。

不适合的场景：

- 只需要一次性体检报告、不需要处置记录，用 `lerobot-doctor`，一条命令出 JSON/Markdown，且不依赖 `lerobot` 包；
- 需要按 URDF 校验动作是否物理可达，用 DataDoctor 的 `--checks kinematics`，本项目没有外部真值，只能做数据集自证；
- 需要训练前按具体策略（ACT / Diffusion / SmolVLA / Pi0）做兼容性 gate，用 `lerobot-doctor gate`；
- 只需要按 episode 删除或拆分数据集，用官方 `lerobot-edit-dataset` 更轻；
- 需要从 rosbag 录制或转换，用 rosbag2 工具链，本项目不读 rosbag；
- 需要语义级检查（物体缺失、遮挡、标注错误），本项目只做数值、几何、时序，语义层留给模型或标注复核。

推荐的组合：

```
rosbag2 录制 / any4lerobot 转换（AgiBot / OpenX / LIBERO / RoboMIND → LeRobot）
  → [本项目：schema 契约审计 → 质检分级 + 修复 + 报告]
  → 干净数据集 → （可选）lerobot-doctor gate → RLDS/训练管线
  ↘ 纯删除场景可交给 lerobot-edit-dataset
```

### 接入实测（2026-11-17）

两个组合件在真实数据上做过接入对照实验（同一数据、同一流水线代码、隔离环境；原始产物与完整复现命令在 `G:\pipeline-lab\EXPERIMENT.md`，实验外置于仓库）：

| 工具 | 实测内容 | 结果 | 结论 |
|---|---|---|---|
| any4lerobot `ds_version_convert/v21_to_v30` | `g1_place_grape_error`（43 集 / v2.1）副本转换后过全量流水线，与原 v2.1 直跑对照 | 最难的 index 语义重建做对（0..41277 全局唯一）、帧数声明=实测、视频对账 0 违约、处置判定与直跑逐位一致；源数据自带的 splits 缺陷（train 0:42 vs 43 集）被忠实搬运，由本项目 schema 审计抓出 | **建议接入**：链路最上游的格式归一；转换产物必须过本项目契约审计再进主流程 |
| lerobot-doctor 0.2.0 | `tokyo_u_lsmo` 上 A/B：入线 check + 出线 score/gate（fix/trim 禁用） | 数据安全零影响（源 19 文件未动、清洗产物逐字节一致、判定可复现）；其 jump 检查（z=|Δ|/按维池化 σ、跨维 mean>8）在本数据上 13 集命中均为「每集末帧微小收尾步」（相对阈值在近常数维上放大噪声）+ 漏报 ±π 卷绕（跨维 mean 稀释，z 5.72<8）；独有价值 = 零 std 维归一化告警 + 4 策略 gate 全 PASS | **不进主链路**：仅在「清洗产物 → 训练」之间可选调 `gate`；jump/fix/trim 不用（修复权归本项目 cleaner） |

接入实测的两条补充结论（2026-11-17 评审后补记）：

1. **score 出线建议与本项目红线直接冲突**：`doctor_score.json` 的 `recommended_drops = [10, 48]`，恰是两个 ±π 卷绕集——卷绕是表示等价、数据自洽，按 doctor 建议执行就是删数据。这比「13 集命中多为噪声」更能说明为什么修复权不能外放。
2. **「13 集为噪声」依赖故障定义，须与「doctor 报了 13 集」分开表述**。严格的对照实验需先给 50 集打标签（ok / truncated / wrap / action_spike / zero_var），按检查项算 precision/recall；当前材料支撑「能力对照 + 源码级机制复算」，升级为正式 P/R 表是待办。

实测踩到的接入成本：

- any4lerobot 转换器：要求 **Python ≥ 3.12** + 官方 lerobot 包源码安装（torch 系）+ `lerobot[dataset]` extra（README 未写，首跑即 ImportError）；**就地转换**（`rmtree/move` 原目录，必须副本操作）；`--push-to-hub` 默认 `true`，本地转换也必须显式关掉。
- lerobot-doctor：Windows 下路径必须用正斜杠，反斜杠会被误判成 HF repo_id。
- doctor 的跨维 mean 稀释与本项目的「证据合并取 min」互为反面教材：同一处 2π 卷绕，本项目 `state_jump` 直接命中，doctor 的 jump 检查以 5.72 < 8 放过。

## 一处取舍

官方 `lerobot-edit-dataset` 只能删、不能修。所以本项目在 cleaner 阶段做了分支：只有当本次清洗计划只包含删除时，才优先调用官方 CLI；一旦存在任何修复项（裁帧、时间轴重建、stats 重算、点云补点），强制走内置实现，并在报告里标注 `method: manual`。这样既不重复造轮子，也不因为 CLI 的能力边界丢功能。

本节对比基于官方工具的公开行为，本项目侧能力都有单测和实测记录（见 [EVAL.md](EVAL.md)）。官方工具若新增能力，以官方文档为准。
