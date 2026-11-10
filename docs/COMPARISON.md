# 与主流方案的对比

本文回答一个面试和选型时几乎必被问到的问题：**已经有 LeRobot 官方工具、ROS 工具链、开源质检工具、商业数据平台了，为什么还需要这一层？**

一句话结论：现有工具分两类——**搬运类**（load / convert / delete / repackage）和**巡检类**（check / score / gate）。本项目在巡检类的基础上多走了一步：把「判不准的」变成可裁决的队列，把「裁决结果」变成可复用、可审计、可导出的处置记录（diagnose → review → override → export）。差异在**处置闭环**，不在检查项数量。

> **先说清楚一件事**：在「检查项覆盖」这个维度上，开源社区已有比我更全的实现（见下节 `lerobot-doctor` / DataDoctor）。本项目的检查规则不构成差异化优势，它存在的意义是把巡检结果接进一条可审计的处置链路。如果只需要一次性的体检报告，用 `lerobot-doctor` 更省事。

## 同生态质检工具：最直接的对照

上游工具链之外，还有一类更接近的对手：**专门做 LeRobot 数据集体检的巡检工具**。截至 2026-11，公开可用的至少有四个：

| 工具 | 形态 | 覆盖范围 | 自动化接口 |
|---|---|---|---|
| `lerobot-doctor`（jashshah999） | CLI，源码安装 | 基础 check 集 | JSON / Markdown |
| **DataDoctor**（ForgeAI，基于上游扩展） | CLI，源码安装 | 12 类 check + 全帧视频解码审计 + URDF 物理限位 | `--ci` 退出码、`--fail-on=warn` |
| `lerobot-quality-gates`（PyPI） | CLI，轻量 | 元信息 / 时间戳 / 形状 / 视频引用 / 标注缺失 | `--fail-on`、HF dataset card 输出 |
| Physical Intelligence `validate.py`（`pi-data-sharing`） | 内部脚本 | 元信息契约 + 标注 JSON 跨字段一致性 | 上传前校验 |

DataDoctor 还提供 `fix`（自动修复，带备份）、`trim`（裁首尾空闲帧）、`score`（按平滑度/覆盖度/一致性/长度给 episode 排序并推荐删除）、`gate`（按 ACT / Diffusion / SmolVLA / Pi0 检查兼容性）、`merge-check`（合并前后校验）。

这几项**本项目没有，也不打算补**：`gate` 属于训练侧契约，`merge-check` 属于数据集编排，都超出「单数据集质量治理」的边界。

商业平台不在逐项对照之列：Foxglove Data Search & Curation（2026-04 上线）、Encord、Roboto 解决的是**采集—索引—检索—标注**的规模化问题，能力显著更强，但需要接入采集链路、按席位付费，目标不是「拿一个离线数据集做一次性治理」。把它们放进表格对比检查项，等于拿两种不同的东西比大小。

### 三向对照

| 维度 | DataDoctor / lerobot-doctor | 本项目 |
|---|---|---|
| 检查项 | 12 类，含 URDF 物理限位、全帧解码审计 | 17 条数值/文件规则 + 8 条点云规则，无 URDF |
| 严重度模型 | PASS / WARN / FAIL 三档 | keep / repair / review / **drop** 四档 |
| 判不准的怎么办 | 归入 WARN 或 per-episode drilldown，由使用者自行判断 | 进 `review_queue.csv`，带视频路径、时间窗、证据片段 |
| 人工裁决回灌 | 公开 README 未提及（待核实） | `--overrides` 把裁决写回，重跑时复用，结论稳定 |
| 对源数据 | `fix` / `trim` 原地修改（先备份） | 源目录只读，清洗产物独立落盘 |
| 子集模式 | `--max-episodes` 时自动跳过依赖全量的检查 | 检测到覆盖不足时**拒绝导出**，不给部分结论 |
| 增量重检 | 公开 README 未提及（待核实） | `--incremental` + manifest 复用 |
| 下游收益量化 | 无（`score` 只排序，不量化清洗前后分布） | `tools/analyze_action_distribution.py` 三段测量（选择效应 / 修复效应分离） |

### 三条实质差异，而不是功能多少

**1. 自动判定，还是分派给人。** DataDoctor 的目标是给出 PASS/FAIL 并接进 CI——它必须自己下结论。本项目承认部分样本（`action_jump` 超阈 1.0–1.5 倍这类）在规则层面判不了，于是把它们挪进 `review` 档并生成可裁决的队列。这不是更聪明，是承认边界：**规则能判的判，判不了的写成问题交给人**。实测在 demo 集上，人工裁决把 8 条待删数据里的 2 条救了回来。

**2. 就地修复，还是源不可变。** `fix` / `trim` 改写数据集本身（先备份），适合「这份数据就是我的，改完继续用」。本项目按「源只读 + 产物独立」处理，适合「数据是别人给的，我要留证据说明我改了什么」。两者都对，取决于你是否需要事后回答「原始长什么样」。

**3. 单次巡检，还是结论可复现。** CI gate 每次全量重跑没有负担；但治理场景下，第二次运行的结论必须能和第一次对上，否则增量质检会漂移。这就是本项目为什么要持久化 `severity_overrides`、要求 `from_dict` roundtrip 的那条测试。

### 待核实清单

上表中标注「未提及」的条目，来自公开 README，**不等于对方没有**。下列六条需读源码确认后回填，核实方法附后：

| # | 待核实问题 | 为什么重要 | 核实方法 |
|---|---|---|---|
| 1 | 是否有 episode 级 disposition 持久化（「这集已判 drop 并被记录」，而非仅 WARN） | 决定能否做增量与结论复用 | 检索 `disposition` / `override` / `decision` |
| 2 | `score` 的推荐删除能否回灌为下一轮输入 | 决定是「建议」还是「闭环」 | 检索 `score` 的输出是否被 `fix` / `trim` 读取 |
| 3 | `fix` / `trim` 是「备份后改写源目录」还是「写新目录」 | 决定源数据是否可变 | 读 `fix` 实现的写入路径 |
| 4 | 是否存在增量模式（只重检变更的 episode） | 决定大集上的可重复成本 | 检索 `incremental` / `manifest` / `cache` |
| 5 | `--max-episodes` 部分模式下，被跳过的检查在报告里如何呈现 | 决定部分结论会不会被误读为全量结论 | 读报告渲染逻辑，看是否标注「未评估」 |
| 6 | 是否有「导出干净数据集」的独立产物（而非原地改） | 决定能否并存原始与清洗两份 | 检索 `export` / `copy` / `--output` |

核实命令（本机连不上 GitHub 时可用 PyPI 包替代）：

```bash
pip download lerobot-doctor --no-deps -d /tmp/ld && cd /tmp/ld && unzip -o *.whl -d src
grep -rn "disposition\|override\|incremental\|manifest" src/ | head -30
```

> 说明：以上对比基于各工具的公开 README 与设计目标（2026-11），本项目侧能力均有单测与实测记录（见 [EVAL.md](EVAL.md)）。竞品若已补齐上述条目，请以源码为准并在本表回填——**这张表的价值在于它可被推翻，而不在于它证明我更好**。

## 上下游工具链对比

| 维度 | 本项目<br/>lerobot-data-quality | LeRobot 官方<br/>`lerobot-edit-dataset` 等 | ROS 2 工具链<br/>rosbag2 / rosbag_editor | OpenVLA 数据管线<br/>RLDS/OXE 转换脚本 |
|---|---|---|---|---|
| 定位 | 数据**质量治理**层 | 数据**编辑**层 | 数据**录制/转换**层 | 数据**打包**层 |
| 输入 | LeRobot v3 / v2.1（parquet + mp4） | LeRobot 数据集（本地或 Hub） | rosbag2（`.mcap` / `.db3`） | Open X-Embodiment RLDS/TFDS 分片 |
| 逐帧质量诊断 | **17 条数值/文件规则 + 8 条点云规则**，自适应阈值 | 无（不做质量判定） | 基本无（偏传输完整性、时间戳检查） | 无（以轨迹级筛选/过滤为主） |
| 自动修复 | **有**：裁边界坏帧、重建时间轴、重算 meta 索引与 stats、点云定长补点 | 有编辑能力（如按 episode 删除），但不做内容修复 | 有帧裁剪/重采样，但不是为训练数据质量设计 | 无 |
| 分级处置 | Keep / Repair / Review / Drop 四档 | 以删除为主 | 不涉及 | 保留/丢弃轨迹 |
| 审计与可追溯 | 报告含数据指纹、阈值快照、逐帧证据、修复动作清单，可 diff | 操作日志有限 | 依赖用户自建 | 依赖用户自建 |
| 人工复核回路 | **有**：导出 `review_queue.csv`（带视频路径/时间窗/证据），填写后 `--overrides` 回灌 | 无 | 无 | 无 |
| 点云质量 | **有**：NaN/稀疏/退化/重复/离群/越界/颜色/时序跳变 8 条规则 | 无 | 无（ROS 侧通常另行处理 PointCloud2） | 视下游而定，通常不做质量判定 |
| 并发/续传 | `--workers` 多进程；`--incremental` 两级断点续传 | 视实现而定 | 无 | 无 |
| 与本项目的关系 | — | 纯删除场景可**委托**官方 CLI（`--no-cli` 可禁用） | 需先转换到 LeRobot 格式，再接本项目 | 本项目在其上游做质检，避免脏数据进入打包 |

## 各自更合适的场景

**该用本项目的场景**

- 采集侧数据存在**静默污染**（视频与 state 行数错位、meta length 漂移、stats.json 过期），训练不报错但学不动
- 数据稀缺，**不能一刀切删除**，需要"能修则修、不能修才删、拿不准转人工"的分级处置
- 需要向采集/标注团队**归因**（哪一批、哪个人、哪台机器异常率高）
- 点云特征需要在**定长契约**内修复（训练张量形状不能变）
- 需要一份能被复核、能被 diff 的**质检报告**

**不该用本项目的场景**

- 只需要一次性体检报告、不需要处置记录 → 用 `lerobot-doctor`，一条命令出 JSON/Markdown，且不依赖 `lerobot` 包
- 需要按 URDF 校验动作是否物理可达 → 用 DataDoctor 的 `--checks kinematics`；本项目没有外部真值可对照，只能做数据集自证
- 需要训练前按具体策略（ACT / Diffusion / SmolVLA / Pi0）做兼容性 gate → 用 `lerobot-doctor gate`
- 只需要按 episode 删除或拆分数据集 → 用官方 `lerobot-edit-dataset` 更轻
- 需要从 rosbag 录制/转换 → 用 rosbag2 工具链，本项目不读 rosbag
- 需要语义级检查（物体缺失、遮挡、标注错误）→ 本项目只做数值/几何/时序，语义层留给模型或标注复核

**推荐的组合用法**

```
rosbag2 录制 → 转换为 LeRobot v3 → [本项目：质检分级 + 修复 + 报告]
                                   → 干净数据集 → RLDS/训练管线
                                   ↘ 纯删除场景可交给 lerobot-edit-dataset
```

## 一处刻意的取舍

官方 `lerobot-edit-dataset` 只能删、不能修。所以本项目在 `cleaner` 阶段做了分支：**当且仅当本次清洗计划只包含删除**时才优先调用官方 CLI；一旦存在任何修复项（裁帧、时间轴重建、stats 重算、点云补点），强制走内置实现并在报告里标注 `method: manual`。这样既不重复造轮子，也不因为 CLI 的能力边界而丢功能。

> 本节对比基于官方工具的公开行为；本项目侧能力均有单测与实测记录（见 [EVAL.md](EVAL.md)）。官方工具若新增能力，以官方文档为准。
