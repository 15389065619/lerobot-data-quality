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
- 处置（默认策略，复核未裁决）：保留 24 / 修复 2（边界裁帧、时间轴重建）/ 复核 3 / 剔除 7；把 ep7 判 drop 后才是 保留 24 / 修复 2 / 复核 2 / 剔除 8
- 修复验证：裁帧后 NaN 归零，重建的时间轴与 `frame_index/fps` 完全一致
- 清洗产物：默认策略 29 episode / 2393 帧，裁决 ep7 后 28 episode / 2335 帧（ep7 有 58 帧，2393 − 58 = 2335，与 §10 的口径同一份产物）；两种产物的 `length==rows`、视频窗口==行数、index 连续、stats 对齐都通过
  - 这一格原来写的是 2388，那来自仓库里一份**旧口径的 `outputs/cleaned_dataset`**（那一轮 ep11 的 5 帧被 auto-drop、ep7 留在复核，`removed = [3,9,11,13,17,19,27,31]`、`drop 8 / review 2`）。2026-11-13 用当前代码重跑 `python tools/demo.py --force` 复核：默认是 `keep 24 / repair 2 / review 3 / drop 7`、产物 29 ep / 2393 帧、`verification.ok=true`，与 README 的控制台快照逐位对上；2388 已经不是当前代码能复现出来的数。
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

> 口径：`data/` 下原有 18 个数据集根目录，17 = 这 18 个减去 §2.4 单独交叉验证的 `chingmu_classical_dance`（18 ep）；合成 `demo_dataset`（36 ep）归 §1，不计入。4108 是这 17 个的**实测 episode 之和**，不是 info 声明求和——humanoid 声明 4885 实测 1418、gym_xarm_pc 声明 50 实测 5。用 `python tools/dataset_inventory.py` 可复算。加上 §2.5 两份外部集 = **19 个数据集 / 5855 episode**（4108 + 1693 + 54）。

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

### 2.4 第二份光捕数据集（120 Hz 古典舞）：换一份分布做交叉验证

`chingmu_subset` 覆盖的几乎全是跳跃类高动态动作，为确认结论不是「对单一样本集过拟合」，从同一原始库另切一份**零重叠**子集：`1.14.6.Classical_Dance` 下全部 18 条片段（18 ep / 16,975 帧 / 36 维 / 120 Hz / 无视频），沿用同一份 `config/chingmu_manual.yaml` 跑完整流水线。

| 指标 | 结果 |
|---|---|
| episode 判定 | 16 正常 / 2 warning / 0 error |
| 命中规则 | `frame_count_abnormal` 1、`action_jump` 1 |
| 处置 | keep 16 / review 2 / drop 0；清洗 0 删 0 修，18 集、16,975 帧原样保留 |
| 跳过项 | `frame_mismatch`、`blank_video`（无视频特征）；点云（无二维点云特征） |

两条 warning 逐条核对后**都不是规则误报**：

- ep6（`DP_Five_Position_Brush`）599 帧，比手填下限 600 少 1 帧（4.99 s）。这是业务硬约束的正常命中，不是 bug；若 5 s 只是名义下限，把 `frame_count.min` 降到 500 即可滤掉这类边界噪声。
- ep11（`DP_Ground_Kick`）frame 1009 的 dim 8/9 同时跳 0.4685 / 0.3193（阈值 0.25，1.87×），但前序 Δ 是 0.12→0.19→0.24→0.28 的连续加速、之后停在高位平台，属于真实快速踢腿而非单帧尖峰，转人工复核（不剔除）是正确处置。放宽邻域判定会一并放过「加速后停住的持续型跳变」，故不改规则。

反向漏报检查：ep1 有 16 处超阈值跳变，全部是 dim 3–6 的四元数 `q ↔ -q` 整体取反，已被 sign-flip 规则正确忽略，未发现真正的漏报。

### 2.5 外部公开数据集交叉验证（libero 1693 ep / dual-lidar-umi 54 ep）

§2.3 那 17 个数据集是我自己挑的（其中不少也来自公开渠道，但选哪些由我定），为了排除「规则只对我挑的样本成立」，从 Hugging Face 直连下载两份**我从没碰过的第三方全量集**（不 `--allow-patterns` 裁剪、不抽分片），跑同一套流水线。

| 数据集 | 版本 / fps / 机型 | 规模 | info 声明 vs 实测 | 判定 |
|---|---|---|---|---|
| `lerobot/libero` | v3.0 / 10 Hz / panda | 1693 ep / 273465 帧 / 1.94 GB / 457 文件 / 2 相机 / state 8 维 / action 7 维 / 40 任务 | **逐位一致**，编号对账 `mode=native`，schema 契约 0 违反 | 全量模式 keep 1692 / drop 1 / review 0（health minor），端到端 **1617 s**（8 workers，含清洗与复检） |
| `brandonyang/dual-lidar-umi-filtered` | v3.0 / 30 Hz / robot_type=null | 54 ep / 52067 帧 / 716 MB / 222 文件 / 2 相机 / state 12 维 / **无 action 列** / 1 任务 | **逐位一致**，`mode=native`，schema 契约 0 违反 | keep 54 / review 0 / drop 0（health clean，79 s） |

两份首次运行就精确对上 `total_episodes` / `total_frames`。§2.3 为 humanoid、gym_xarm_pc 的计数虚报而加的那两条对账断言，第一次在「声明本来就正确」的数据集上被验证为**不响**——只报警不误报才算对上。dual-lidar 是全仓**唯一没有 `action` 特征**的数据集，它和 `robot_type=null` 一起把两条退化路径跑通了。

#### 外部数据推翻的 4 条结论

这四条在我自己挑的数据上都测不出来，逐条给机制与复算证据。

| # | 现象 | 机制 | 证据 |
|---|---|---|---|
| M1 | `missing_task` 在 libero 上命中 **1693/1693**。`--streaming` 把 health 判成 `severe`、valid_ratio 0.0，建议「先修采集流程再清洗」；默认全量模式对同一份数据判 keep 1692 / review 0 | [loader.py:342](../pipeline/loader.py) `task_of()` 只有两条解析路径：`meta/episodes` 的 `tasks` 列，或逐帧 `task_index`。libero 的 `meta/episodes` 14 列里没有任何 task 列，只能走逐帧；而 [precheck.py:96](../pipeline/precheck.py) 用 `load_frames=False` 加载、流式路径同样不把整表读进内存 → 两条路同时断掉，返回空串 | 全量模式 `reason_frequency.missing_task = 0`，`--streaming` = 1693，precheck = 1692 条 review；连**清洗产物的自动复检**都被它判成 `review 1692`（见下方「清洗产物复检」）。dual-lidar 有 `tasks` 列（`['bimanual umi demo']`），三种模式全是 0 |
| M2 | 全量模式报 `stats_stale`：`max_relative_deviation = 0.4809`（容差 0.05）、`worst_dims = [4,5,2]`，但 `stats.json` 其实是**对的** | [context.py:372](../pipeline/validator/context.py) 取 `stale_stats.sample_episodes`（默认 200）个 episode 算均值，取的是**前缀**而非随机样本；libero 按任务排序，前 200 集只覆盖部分任务 → 前缀有偏。同函数 :363-371 已经承认「子集均值与全量 stats 不可比」，但只在用户传 `--max_episodes` 时跳过，没管自己的 200 集上限 | 我按前缀长度独立重算：200→**0.4809**（与报告逐位吻合）、400→0.3486、800→0.2364、全量 1693→**0.0002**。偏差随样本增大单调收敛 = 抽样偏差，不是 stats 过期。产物未受损：cleaner 从清洗后数据整体重算 stats（[meta.py:209](../pipeline/cleaner/meta.py)），假阳性只停在报告文本里。**同一条规则已经出过两次假阳性**：[POSTMORTEM 案例 2](POSTMORTEM.md) 修的是分母（`\|Δ\|/\|mean\|` → `\|Δ\|/max(\|mean\|,std)`，913 → 0.99），这次是分子——分母修好后残余的 0.15 / 0.48 这类数，来源就是前缀样本本身有偏 |
| M3 | 全场唯一的 drop：ep306 有 505 帧，超自适应上限 496 达 **1.8%**，直接判 error 剔除 | `frame_count` 的 MAD 上限算在**跨任务池化**分布上（median=140 mad=30）。全库最长的 6 集——505/466/458/457/455/450——**全部**属于 task 6「put both moka pots on the stove」；该任务 29 集、长度中位 401，是全局中位的 2.9 倍 → 最难的任务类别被当成了分布尾部 | 方向不对称本身是刻意设计（[rules.py:360-362](../pipeline/validator/rules.py)：过短降 warning 交人审、过长保持 error 剔除，注释里还专门点了 LIBERO）；但过长侧没有 `action_jump` 那种 <1.5× 边际降级。报告证据含 median/mad/bounds，**不含 task_index**，复核者看得见 1.8% 的边距、看不见它为什么长 |
| M4 | dual-lidar 根本没有 `action` 列，但 20 条检查里只有 1 条被标成 skipped | [validator/\_\_init\_\_.py:308](../pipeline/validator/__init__.py) 的无 action 分支只塞了 `action_jump` 一个代理键；[reporter.py:152](../pipeline/reporter.py) 的 `status` 就是对 `skipped_checks` 做成员判断 → `nan_action`、`duplicate_episode` 显示 `enabled / 0 命中`，而这两个函数只在 action 分支内调用（:300、:306） | `checks` 数组 19 条 enabled + 1 条 skipped，逐条回看调用点才确认虚报的是这 2 条。`static_episode` 标 enabled 是**对的**，state 分支同样调用它（:288） |

M1 和 M4 错在**危险的那一面**：不是把好的说成坏的，是把没跑过的检查报成「跑过且通过」。

#### 没有被推翻的 3 条

- **熔断成立。** missing_task 命中率 99.9%，但 `protected_by_systemic_guard = 0` 是正确的——该规则 severity 为 warning，处置本来就是 review，没有批量 drop 需要拦；报告另有「异常率 100% 偏高，属系统性问题」的告警。「不会一次删光你的数据」在 libero 上成立。
- **源数据集只读成立。** 全流程跑完后源目录最新 mtime 仍等于下载完成时刻，早于流水线启动时间，0 个源文件被改写。
- **无脏可洗时不硬造。** dual-lidar 走 `method=noop-copy`，仍产出结构完整的 54 集副本，`meta/stats.json` 重算 10 个特征。

#### 清洗产物复检（第四行口径）

清洗不是终点，报告里必须回答「洗出来的东西是不是真的更干净」。两份外部集的复检字段：

| 数据集 | `cleaning.verification` | 复检结论 |
|---|---|---|
| `lerobot/libero` | `ok=true`，`episodes=1692`、`frames=272960`、`episode_indices_contiguous=true` | **结构层全过**：ep306 剔除后编号连续、帧数与 `info.json` 逐位一致。**规则层被 M1 污染**：`rule_recheck = {keep 0, review 1692, drop 0}`，附带文案「1692 个 episode 有告警，建议跑完整流水线修复」——一份刚洗完、全量模式判 `review 0` 的数据集，被自己的复检判成「还需要再修」。复检走流式路径，所以 task 解析失败在这里第三次复现，而且这次打在**流水线自己的产物**上 |
| `brandonyang/dual-lidar-umi-filtered` | `{}`（`method=noop-copy`，无脏可洗，不触发复检） | 54 集副本结构完整，`meta/stats.json` 重算 10 个特征 |

这条是 M1 最有力的证据：它不再只是「换个模式结论不同」，而是**流水线的自检环节会对自己刚产出的干净数据集给出与全量模式相反的结论**。清洗本身没问题——问题在复检路径复用了那条坏掉的 task 解析。

#### 自适应阈值的一处边界（是代价，不是 bug）

dual-lidar 的 12 个 state 维，最终阈值 **12/12 全部等于 `abs_floor`**：自适应项 `med + 3·1.4826·MAD` 逐维只有 0.0115–0.0467，而 floor 按 30 Hz 标定后是 1.0/帧（[context.py:271](../pipeline/validator/context.py) `floor = abs_floor · ref_fps / eff_fps`）。全库 52067 帧实测最大逐帧 Δ = 0.352 → 规则只在 >1.0 时报，对 0.35–1.0 这一段（典型尾部 p99.9 = 0.040 的 9–25 倍）不敏感。`config/default.yaml:95` 已注明 floor 是「弧度/米混用数据集的兜底」，所以这是「换机器人不改代码」换来的代价：低速高精度机型上 `state_jump` 退化为只抓灾难性瞬移。同一现象在 120 Hz 的 chingmu_subset 上是 36 维里 32 维落在 floor（0.250/帧）。

#### 这 4 条现在的状态：已定位，**未修**

我只做了定位和复算，没有动流水线代码——四条的修法代价不同，不宜在同一批里一起改：

| # | 建议修法 | 为什么没顺手改 |
|---|---|---|
| M1 | 流式分片时聚合 `episode_index → task_index`（每集一个整数，不需要留整表），让 `task_of()` 在不加载帧时也能解析 | 会改 `--streaming`、precheck **以及清洗产物复检**的既有结论，17 个数据集那一轮的 disposition 需要重跑一遍才能继续引用 |
| M2 | 把 200 集**前缀**换成蓄水池采样（统计侧已有 `Welford + reservoir`，复用即可），或在 detail 里写死「基于前 N 集」 | 换采样会改变所有历史 `stats_stale` 数值，含 [POSTMORTEM 案例 2](POSTMORTEM.md) 引用的 0.15 / 1.57 / 3.63 三个数，得同步改写 |
| M3 | 二选一：`frame_count` 上限按 `task_index` 分组算；或给过长侧补上和 `action_jump` 一样的边际降级（<1.5× 降 review） | 分组算会遇到小样本问题——task 6 只有 29 集，MAD 本身不稳；这条要先定「按任务分组」是否是想要的语义，不是纯工程 |
| M4 | 无 action 分支把 `nan_action` / `duplicate_episode` 一并写进 `skipped_checks`，而不是只留 `action_jump` 一个代理键 | 代价最小、方向最该修（把没跑的检查报成通过），改完要加一条负向用例 |

M4 是四条里唯一「不改结论、只改报告诚实度」的，真要排优先级我把它放第一个；M1 让三条路径给出互相矛盾的结论——同一份数据全量判 `review 0`、流式判 1693 条 review，而清洗后自检又对刚产出的干净副本给出 1692 条 review，这对用户最致命，因为第三种连我自己都会被误导。

#### 复现

```bash
# 直连 HF 全量下载（不走镜像，不裁剪文件）
python -c "from huggingface_hub import snapshot_download as s; \
           s('lerobot/libero', repo_type='dataset', local_dir='data/libero')"
python main.py --dataset_path data/libero --output_dir outputs_libero --workers 8               # keep 1692 / drop 1
python main.py --dataset_path data/libero --output_dir outputs_libero_stream --streaming --skip-clean  # review 1692
python -m pipeline.precheck --dataset_path data/libero                                          # 同样 1692 条复核
python main.py --dataset_path data/dual_lidar_umi_filtered --output_dir outputs_lidar            # 79 s，clean
```

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

- **300 个单测**（本地实测 `300 passed`，`tests/` 下 30 个测试文件，本机有收集的 29 个）：清洗矩阵 36 + 校验器 32 + 审计加固 27 + 清洗策略 24 + 增量 16 + 精确率/基准 16 + 配置 14 + 动作分布 14 + loader 边界 13 + ROS Bag 9 + 点云 8 + 容错续传 8 + 基线白名单 8 = **225**；其余 16 个文件 **75**（schema 契约 6 / 120Hz 6 / Dashboard 6 / 对齐 6 / 调参 5 / Mocap CSV 5 / FZMotion CSV 5 / 导出 5 / 编号核对 5 / 清洗 stats 5 / 版本 4 / 非标准字段 4 / 流式 3 / 并发 3 / 数据卡片 3）
  - 另有 `tests/test_rosbag_loader_e2e.py` 的 2 项：本机未装可选依赖 `rosbags`，模块顶层 `importorskip` 使**整文件在收集阶段就被跳过**，所以既不计进 300、也不显示成 `skipped` 计数；CI 装了 `rosbags`，那一轮是 302。
  - 这一格之前写 209，是 `test_cleaner_matrix.py`（+36）等文件落地前的旧口径一直没跟着改。复算只要 `python -m pytest tests -q --collect-only`，它逐文件报数。文档里的计数不留「大概」，只留能一键重算的。
- 覆盖率 **73.73%**（branch=true；2026-11-13 那次 CI 运行 `Required test coverage of 64.0% reached. Total coverage: 73.73%`，门槛由 pyproject 的 `fail_under=64` 强制，只升不降）
  - 这一格原来写 65.35%，是套件还小得多时候的基线，一直没跟着跑。**这个数在本机当前 venv 里一键算不出来**：`pytest-cov` 写在 `requirements-dev.txt` 但本机未装，直接 `--cov` 会 `unrecognized arguments` 退出码 4；要复算要么 `pip install -r requirements-dev.txt` 再 `python -m pytest tests -q --cov --cov-branch`，要么直接看 Actions 里最新一次 CI 的「运行测试」日志（公开可读，比我的文档更难狡辩）。CI 三个 Python 版本都装了 `rosbags`，所以那一轮收集的是 302 项，本机未装依赖是 300 项。
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
- **不加载逐帧表的两条路径（`--streaming`、precheck）解析不了 task**：`meta/episodes` 没有 task 列的数据集（如 `lerobot/libero`）会被全量误报 `missing_task`，清洗后自检也走这条路。修好前这两条的结论不能当全量用——见 §2.5 M1
- **`checks` 里标 `enabled` 不等于真的跑过**：无 `action` 列的数据集只把 `action_jump` 记进 `skipped_checks`，`nan_action` / `duplicate_episode` 会以「已跑且 0 命中」出现（§2.5 M4）
- 两处度量样本口径仍有已知偏差：`stats_stale` 取前缀 200 集而非随机样本（任务排序的集上会假阳性，§2.5 M2），`frame_count` 上下限算在跨任务池化分布上（最长任务类别会被判成分布尾部，§2.5 M3）
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

<!-- ACTION_DIST_BEHAVIOR1K:BEGIN -->
## 11. 真实规模复核：behavior1k 的 5 条跳变 review 判为 drop

> 原始：`behavior1k_task0042`（200 ep / 1,279,960 帧） → 清洗：`仅保留集（代理）`（195 ep / 1,251,667 帧） **（代理口径，见下）**
> 中间基线：同一份原始数据，只取清洗后保留的 episode —— 用来把「剔除」和「修复」两种效应拆开。
> 跳变阈值：`|Δ|∞ > 1.000`（`action_jump.abs_floor` 按 fps 折算）；口径：逐维总体方差（ddof=0）跨维平均。
> 跳变统计排除 3 个离散维（取值种数 ≤ 3）：dim6、dim14、dim22 —— 与规则 `action_jump` 同口径；不排除的话夹爪开合（一次 ±2）会被当成跳变，每条轨迹都「有跳变」。
> 复现：`python tools/analyze_action_distribution.py --raw behavior1k_task0042 --proxy-from-kept --report <quality_report.json> --update-eval docs/EVAL.md --section-id ACTION_DIST_BEHAVIOR1K --section-title "## 11. 真实规模复核：behavior1k 的 5 条跳变 review 判为 drop"`

### 11.1 总量指标

> **本节的「清洗后」是代理口径。** 报告记录 `trim_frames=0`、`repair_frames=0`，本次清洗只做整集剔除、不改任何一帧的内容，因此「原始仅保留集」与真实清洗产物在动作数值上逐帧等价；本节据此省略了清洗产物（生成它需要重编码全部视频，耗时数小时，而本节只比较动作数值）。代价是**修复效应无法测量**，表中记为 n/a 而不是 0。
| 指标 | 原始（全部） | 原始（仅保留集） | 清洗后 | 总变化 |
|---|---|---|---|---|
| episode 数 | 200 | 195 | 195 | -5 |
| 帧数 | 1,279,960 | 1,251,667 | 1,251,667 | -2.2% |
| 动作维度 | 23 | 23 | 23 | — |
| **逐维方差均值** | 0.1691 | 0.1671 | 0.1671 | **-1.2%** |
| 动作标准差 | 0.4112 | 0.4088 | 0.4088 | -0.6% |
| 离差平方和（SSE） | 4,978,784 | 4,810,366 | 4,810,366 | -3.4% |
| 最大跳变 max\|Δ\|∞ | 1.349 (ep40) | 0.9972 | 0.9972 | -26.1% |
| 每集最大跳变中位数 | 0.2691 | 0.2633 | 0.2633 | -2.1% |
| 每集最大跳变 P90 | 0.5418 | 0.5063 | 0.5063 | -6.5% |
| 超阈帧数 | 5 | 0 | 0 | -5 |
| 均值（跨维平均） | -0.0398 | -0.0391 | -0.0391 | 6.36e-04 |
| 归一化跳变 max\|Δ\|/std | 3.281 | 2.440 | 2.440 | -25.7% |

### 11.2 变化归因：剔除 vs 修复

删样本一定会改变方差，所以「方差下降 X%」必须先拆开，否则等于把删数据说成提纯。SSE 可跨 episode 相加，是这里唯一能做加法分解的量：

| 效应 | 定义 | 方差变化 | 说明 |
|---|---|---|---|
| 选择（剔除） | 原始全部 → 原始仅保留集 | -1.2% | 剔除 5 个 episode、28,293 帧（占帧数 2.210%），带走 3.370% 的离差平方和 |
| 修复 | 原始仅保留集 → 清洗后 | n/a | 未生成清洗产物，修复效应未测量（不是 0） |
| **合计** | 原始全部 → 清洗后 | **-1.2%** | 只由剔除构成（无逐帧修复，见 §11.5） |

均值漂移：逐维均值差的最大值 0.0111（按原始 std 归一化 0.0270）；因未生成清洗产物，无法再拆出「裁剪/重建」的贡献。

### 11.3 被剔除 episode 的尾部贡献

| episode | 帧数 | 帧数占比 | 方差 | max\|Δ\|∞ | SSE 占比 | 剔除原因 |
|---|---|---|---|---|---|---|
| 34 | 5,500 | 0.4297% | 0.1455 | 1.188 | 0.9101% | action_jump |
| 11 | 7,699 | 0.6015% | 0.1822 | 1.284 | 0.8576% | action_jump |
| 40 | 5,035 | 0.3934% | 0.1376 | 1.349 | 0.7485% | action_jump |
| 179 | 6,328 | 0.4944% | 0.1432 | 1.017 | 0.4590% | action_jump |
| 106 | 3,731 | 0.2915% | 0.1993 | 1.160 | 0.3953% | action_jump |
| **合计** | 28,293 | 2.210% | — | — | **3.370%** | — |

### 11.4 结论

- 剔除 5 个 episode（28,293 帧）后，整体动作方差由 0.1691 降至 0.1671，**-1.2%**；其中选择效应 -1.2%（本次只做剔除，修复效应未测量）。
- 归一化跳变（max|Δ|/std）由 3.281 收窄到 2.440（-25.7%）；单帧最大位移从 1.349 降到 0.9972，已回到阈值以内。
- 超过 `|Δ|∞ > 1.000` 的帧：5 → 0（涉及 5 → 0 个 episode）。
- 均值漂移 0.0111（0.0270 个 std，可忽略，动作中心未被移动）。

### 11.5 边界（这段不能删）

- **没有训练模型。** 本节给的是代理指标：异常跳变帧减少、归一化跳变幅度收窄、对应梯度被放大的通道变少。这不等于「精度提升」，也不等于「loss 曲线更平滑」；要那条结论必须做清洗前后的训练对照（同 seed、同 step、同评测集）。
- **方差下降不等于数据变好。** 剔除高方差 episode 必然降低整体方差——若这些 episode 是真实的高动态任务（而非碰撞/断连），降方差反而是信息损失。判断依据只能是 §11.3 的剔除原因，而不是方差这个数字本身。
- 逐维方差对量纲敏感；跨机器人/跨任务混采的数据集不能直接和本表比较。
- 跳变阈值取配置 `abs_floor`（按 fps 折算），不做量纲自适应：像素动作（如 pusht 的 0–512）上该阈值没有物理意义，重跑时用 `--jump-threshold` 显式指定。
- 修复阶段只裁剪边界帧与重建时间戳，不动动作幅值，因此「修复效应」理论上应接近 0；若它明显非 0，说明清洗改写了动作内容，需要单独解释。
- 跳变指标排除了 3 个离散维（dim6、dim14、dim22，取值种数 ≤ 3）。这一条不是可选项：不排除时夹爪一次开合就是 |Δ|=2，「超阈帧数」会退化成「夹爪动作次数」，和规则侧的命中完全对不上。
- **「清洗后」是代理值，不是产物。** 本节未生成清洗后的数据集：报告证明本次清洗只做整集剔除（`trim_frames` / `repair_frames` 均为 0），故以「原始仅保留集」等价代替。若报告里的这两个计数非 0，工具会直接拒绝该模式（见 `_require_drop_only`）——等价关系不成立时不允许出数。

<!-- ACTION_DIST_BEHAVIOR1K:END -->

### 11.6 口径校验：同一份数据，两套答案

这一节第一次跑出来的数字不是上面这些，而是：`max|Δ|∞ = 2.000`（ep0）、**每集**最大跳变的中位数也是 2.000、超过阈值的帧 **1,763**、覆盖**全部 200 个 episode**。

照这个读数，结论应该是「这台机器人 200 条轨迹全都有跳变，平均每条 8.8 次」，而质检规则对同一份数据、同一个阈值只报了 5 个 episode。两个数差 300 倍。

差在动作向量的构成：这个数据集的 23 维动作里，**dim14 / dim22 是二值夹爪（取值 {-1, +1}），dim6 恒为 0**。夹爪一次开合就是 |Δ|=2，是正常控制指令，不是异常；把它的 L∞ 直接和 1.0 比阈值，等于在数「夹爪开关了多少次」。

规则侧早就处理了这件事——`check_action_jump` 里 `discrete_max_unique: 3`：先按每集取值种数判定离散维、把它置零，再比阈值和邻域尖峰比。对比脚本第一版没有这一步，于是把「夹爪动作次数」写成了「跳变次数」。

修法是把同一套判据写进工具默认路径（抽样 20 万行判取值种数），结果与规则逐条对上：

| 口径 | max\|Δ\|∞ | 超阈帧数 | 命中 episode | 与规则对照 |
|---|---|---|---|---|
| 含离散维（`--discrete-max-unique 0`） | 2.000 | 1,763 | 200 / 200 | 与规则的 5 个差 300 倍，且「每条轨迹都有」 |
| 排除离散维（默认） | 1.349 | 5 | 5 / 200（11、34、40、106、179） | 与规则命中逐条一致 |

规则按 `(帧, 维)` 计数得 6，脚本按「帧」计数得 5——ep40 的两处命中共用同一帧。这个差异是两种计数单位，不是判据不一致。

记这一条的理由不是「改了个 bug」，而是：**代理指标必须和规则同口径**。两边各用各的口径时，同一份数据能给出相差 300 倍的结论，而错的那一版在文档里长得完全正常——表格整齐、数字自洽，只有结论是错的。

### 11.7 这组数字真正说明了什么

把 §10（demo，36 集，注入了一处 ±18 的碰撞尖峰）和本节（真实数据集，1.28 M 帧，5 条边缘命中）放在一起看，会得到一个不太舒服但重要的结论：

| | demo_dataset（§10） | behavior1k（本节） |
|---|---|---|
| 剔除 episode | 8 / 36（22.3% 帧） | 5 / 200（2.2% 帧） |
| 被剔除集的跳变幅度 | 34.0（阈值的 34 倍） | 1.02–1.35（阈值的 1.0–1.35 倍） |
| 逐维方差变化 | **−10.0%** | **−1.2%**（全部来自删样本，选择效应） |
| max\|Δ\|∞ | 34.018 → 0.2121（−99.4%） | 1.349 → 0.9972（−26.1%） |
| 超阈帧数 | 6 → 0 | 5 → 0 |

两边的**跳变指标都是干净归零**的：清洗之后，超过阈值的单帧位移一处不剩。但**方差几乎没有动**（−1.2%），而且这 1.2% 完全来自「删掉了 2.2% 的帧」这一件事，不是清洗让数据变纯。

所以「剔除 N 个跳变 episode，整体方差降低 X%，消除模型震荡风险」这句话不能通用：它在 demo 上成立（−10.0%）是因为 demo 里注入的尖峰是阈值的 34 倍、且集中在 8 个集里；在真实规模上，方差这个统计量对 5 个边缘命中的集几乎不敏感，能被稳定观测到的是**跳变指标**（超阈帧数、归一化跳变峰值），而不是方差。

要验证「清洗是否值得」，得看的是那 5 条 review 在规则里的原始定位：它们的 `exceed_ratio` 只有 1.02–1.35，本来就落在「快速但连续」与「真实尖峰」之间——这正是需要人裁决的区间，也是复查回灌这一步的价值所在：**工具负责把它们挑出来，人负责决定删不删，删完的收益要用对指标去量。**
