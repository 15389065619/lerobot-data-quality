# LeRobot 数据质检流水线 · 三层讲解教程

> **读者画像**：学过 Python 基础语法（变量 / 函数 / import / if-for），不要求熟悉数据工程与并发。
> **贯穿类比**：本项目是一家「洗衣厂」——数据集是脏衣服，质检清洗流水线是分拣与洗涤车间。
> **来源**：2026-11 对本仓库全量代码走读后整理的教学文档；所有代码引用均可在仓库内定位到行号。

---

## 目录

- [0. 怎么用这份教程](#0-怎么用这份教程)
- [第 1 层 · 全局地图](#第-1-层--全局地图)
- [第 2 层 · 模块拆解](#第-2-层--模块拆解)
- [第 3 层 · 代码精讲（四轮）](#第-3-层--代码精讲四轮)
  - [3.1 `_as_int` 与 `_reconcile_episode_ids`（编号对齐）](#31-_as_int-与-_reconcile_episode_ids编号对齐)
  - [3.2 `check_action_jump`（跳变检测）](#32-check_action_jump跳变检测)
  - [3.3 `_decide_one` 与 `_rewrite_frames`（分诊与手术）](#33-_decide_one-与-_rewrite_frames分诊与手术)
  - [3.4 点云 `_decide_one` 与 `repair_frame`（特护线）](#34-点云-_decide_one-与-repair_frame特护线)
- [附录：贯穿全程的六条设计思想](#附录贯穿全程的六条设计思想)

---

## 0. 怎么用这份教程

三层递进，每层回答一个深度的问题：

| 层 | 回答的问题 | 类比 |
|---|---|---|
| 第 1 层 全局地图 | 这个项目**是什么**，数据从哪来到哪去 | 洗衣厂的鸟瞰图 |
| 第 2 层 模块拆解 | 每台机器**负责什么**，为什么这么分工 | 走进每个车间 |
| 第 3 层 代码精讲 | 关键函数**怎么写**，为什么这么写而不是别的方式 | 拆开机器看齿轮 |

每节末尾有检验题，本教程为自学模式——**参考答案直接附在题后**。建议先自己作答再核对。

---

## 第 1 层 · 全局地图

### 1.1 这个项目解决什么问题

机器人学做任务（比如「把红薯放进篮子」）要靠**模仿学习**：给模型看几千条「人类遥控机器人做这件事」的记录，它照着学。这些记录统称**训练数据**。

问题：采集数据的机器人不在真空里工作。控制程序丢包会写出无效值（**NaN**：Not a Number，「这个格子里的数丢了/坏了」的特殊浮点值）；相机被挡会录出黑屏；数据合并时同一条记录可能存两次。**模型吃到脏数据不会报错，它会安静地学错**——等几周后发现问题，几百小时 GPU 算力已经烧完。

本项目是挡在「原始数据」和「训练」之间的**质检 + 清洗流水线**：

> **输入**：一个按 LeRobot v3 格式组织的本地数据集目录（LeRobot 是机器人社区通用的数据格式规范；一份数据集 = 说明书 `meta/info.json` + 数据表 `data/*.parquet` + 相机视频 `videos/*.mp4`。**parquet** 是列式存储文件——压缩率更高、读取更快的「Excel」；**episode** 是机器人一次完整任务，是数据的最小逻辑单位）。
>
> **输出**：一个 `outputs/` 目录——体检报告（JSON）、待人工复核清单（CSV）、洗完可直接训练的 `cleaned_dataset/`、图表（PNG）。

### 1.2 洗衣厂类比（贯穿全程）

| 洗衣厂环节 | 项目里的模块 | 干什么 |
|---|---|---|
| 收衣登记台 | `loader.py` | 把衣服登记造册，翻出吊牌（元数据）。**只登记，不判断脏不脏** |
| 分拣检查台 | `validator/` | 一件件检查：破洞？染色？（13+ 条规则） |
| 分诊决定台 | `policy.py` | 决定每件怎么办：**直接洗**（repair）/ **打电话问顾客**（review）/ **报废**（drop）/ **没问题**（keep） |
| 开小票 | `reporter.py` | 出具质检报告 + 待确认清单 |
| 洗衣车间 | `cleaner/` | 真正动手删/修，产出干净的一批 |
| 皮衣特护线 | `pointcloud/` | 3D 点云这种「贵重面料」走单独工序 |
| 展示橱窗 | `visualizer.py` | 画图表给人看 |
| 会员档案 | `manifest.py` | 记住上次洗过什么，没脏的不用再洗（增量/断点续跑） |
| 「污渍是设计」登记本 | `baseline.py` | 确认过的误报登记在案，下次不再拦 |
| 洗衣参数面板 | `config.py` | 阈值参数，可按衣料（机器人型号）定制 |

### 1.3 仓库目录树

```text
G:\Lerobot\
├── main.py                    # 总调度员：按顺序串起 8 个阶段，管退出码和失败降级
├── pipeline/                  # ★ 核心流水线包
│   ├── loader.py              # 收衣登记台：把数据集读成内存结构（只读，不做判断）
│   ├── schema.py              # 格式合规员：数据集「摆放方式」符不符合规范
│   ├── validator/             # 分拣检查台：context 推阈值、rules 逐条检查、streaming/parallel 加速跑法
│   ├── policy.py              # 分诊决定台：体检结果 → 保留/修复/复核/剔除
│   ├── reporter.py            # 报告科：quality_report.json + 复核队列 CSV
│   ├── cleaner/               # 洗衣车间：删、裁帧、重建时间轴、重算统计量
│   ├── pointcloud/            # 皮衣特护线：3D 点云的独立质检+清洗小流水线
│   ├── visualizer.py          # 展示橱窗：matplotlib 画统计图
│   ├── config.py              # 参数面板：默认阈值 + 四层合并
│   ├── manifest.py            # 会员档案：增量/断点续跑状态
│   ├── baseline.py            # 白名单本：已确认误报的指纹登记
│   ├── precheck.py            # 训练门口的保安：脏数据直接拦下
│   ├── versions.py            # 时光机：数据集快照/对比/回滚
│   ├── tuning.py              # 学习笔记：从人工复核提炼阈值建议
│   ├── card.py                # 简历生成器：报告浓缩成数据集卡片
│   └── parallel.py            # 并发调度器：多进程执行与结果归并
├── src/lerobot_data_quality/  # 可选上游适配器：ROS Bag / 动捕 CSV → 标准格式
├── tools/                     # 周边工具：演示脏数据、网页 Dashboard、效果分析
├── config/default.yaml        # 默认配置文件（与 config.py 内置默认一一对应）
└── tests/                     # 300 个自动化测试
```

### 1.4 产出目录树（运行后生成）

```text
outputs/
├── run_summary.json           # 本次运行「成绩单」，CI/监控只看这一个
├── quality_report.json        # ★ 总体检报告：每集结论+证据，可 diff 历史
├── review_queue.csv           # ★ 待人工复核清单：Excel 填一列就能回灌
├── quality_manifest.json      # 增量模式的会员档案
├── pointcloud_report.json     # 点云专报
├── pointcloud_review_queue.csv# 点云待复核清单
├── cleaned_dataset/           # ★ 洗完的数据集，可直接给训练管线
├── pointcloud_cleaned/        # （仅当主清洗失败时）点云独立清洗产物
├── plots/                     # 图表 PNG
└── intermediate/              # 8 个阶段中间产物，任一步中断可复用
```

### 1.5 一份脏数据集的完整旅程

敲下 `python main.py --dataset_path 我的机器人数据 --output_dir outputs` 后：

1. **`main.py`** 接管，解析参数、建输出目录。车间主任，自己不干活，只按编号喊人。
2. **[1/8] `loader.py`** 收衣登记：读说明书、读吊牌、把 parquet 读成大表。顺带两件事：`schema.py` 查摆放合规；两套集号对不上时尝试自动对齐（证据不足则标记「无法对齐」，拒绝瞎猜）。**损坏文件不中断，记账跳过**。
3. **[2/8] `validator/`** 体检：`context.py` 用数据自身分布**推导**阈值（自适应），`rules.py` 逐集跑 13+ 条规则，每条结论带证据（第几帧、超了多少）。视频检查交给 `probes.py` 调 ffmpeg。三种跑法：串行 / 流式 / 多进程——结论必须一致。
4. **`baseline.py`** 划掉白名单里的已确认误报。
5. **[3/8] `policy.py`** 分诊：四档处置。人工复核 CSV（`--overrides`）在这里覆盖机器判定。保险丝：单规则命中 ≥95% → 熔断降级复核。
6. **[4/8] `reporter.py`** 开小票：`quality_report.json` + `review_queue.csv` + 终端摘要。
7. **[5/8] `cleaner/`** 动手洗：删报废、裁首尾坏帧、重建时间轴、重算统计量。**先写暂存目录，校验通过才改名发布**。原数据集全程只读。
8. **[6/8] `pointcloud/`** 特护线：逐帧流式检查 8 条几何规则。
9. **[7/8] 点云清洗**：只修「点」不动「行」，坏帧用同 episode 最近好帧占位。
10. **[8/8] `visualizer.py`** 出橱窗。
11. **收尾**：写 `run_summary.json`，按情况返回退出码（0 成功 / 2 输入错 / 3~5 阶段失败 / 6 部分降级但报告保留）。

### 1.6 模块间调用与依赖

关键设计一句话：**各阶段互相不直接调用，全部由 `main.py` 居中传递**。

```text
            ┌─────────────── main.py（唯一的编排者）───────────────┐
            │  loader → validator → policy → reporter → cleaner    │
            │                → pointcloud → visualizer             │
            └──────────────────────────────────────────────────────┘
                  ↑ 所有阶段都依赖这两个底座 ↑
        config.py（读参数）    loader.py（拿数据视图）

  横切工具（不参与主链，被 main.py 按需调用）：
    manifest.py（增量）  baseline.py（白名单）  schema.py（被 loader 用）
    precheck.py / versions.py / tuning.py / card.py（独立小工具）
```

为什么？如果 reporter 直接调 cleaner，链条会变成蜘蛛网。现在每个阶段只认识「输入格式」和「输出格式」，中间结果统一落盘 `intermediate/`——任何一步挂了，前面产物照样能用。这就是流水线和「一个几千行 main 函数」的区别。

### 1.7 第 1 层检验题与答案

**Q1** 为什么「检查」（validator）和「决定怎么处理」（policy）要拆开？合成「查完直接删」违背哪个核心设计？

> **答**：判定是事实，处置是策略。机器只能客观说「首尾有 NaN」，但「首尾 NaN 可裁掉救回来」是业务判断。合成一个模块就没有人工推翻机器的空间——而「质检结论必须能被人推翻」（`--overrides` 回灌）正是本项目核心设计。

**Q2** 「这批数据整体能不能用」和「第 7 集删了什么修了什么」分别看哪个文件？

> **答**：整体看 `quality_report.json`（或终端摘要）；单集删改明细看 `cleaned_dataset/meta/quality_cleaning_notes.json`（留痕文件，随清洗产物走）。

**Q3** `main.py` 挂在半路（如清洗失败）时，前几步的成果还在吗？

> **答**：在。每阶段中间产物落盘 `intermediate/`；清洗失败走「降级不中断」：报告照常产出，退出码 6 告诉 CI「这不是能训练的数据」。

---

## 第 2 层 · 模块拆解

排序按数据流经顺序：进来（loader）→ 判断（validator）→ 决定（policy）→ 动手（cleaner）→ 报告（reporter）→ 三块支撑地板（config / manifest / baseline）→ 特护线（pointcloud）→ 保安与周边工具。

> 「接口」= 一个模块允许别人调用的函数，相当于柜台窗口。

### 2.1 `loader.py` —— 收衣登记台（全项目的地基）

- **负责**：把硬盘上的数据集变成内存里的「数据集视图」对象。只读不判断——出问题能立刻分清「读错了」还是「判错了」。
- **接口**：主函数 `load_dataset(路径)` 返回 `LeRobotDatasetView`；视图方法：`episode_frame()` 取某集帧、`meta_row()` 查吊牌、`video_location()` 定位视频与时间窗、`episode_indices` 列集号。
- **被谁调**：main.py、precheck、cleaner 复检、dashboard——所有想碰数据的地方。
- **依赖**：`schema.py`。
- **为什么返回视图对象而不是大表+字典**：真实数据集的吊牌集号与数据集号可能是两套系统（全局编号 420010 vs 0,1,2…）。对齐逻辑收拢在视图里（对不上且证据不足就标「无法对齐」），下游永远问「第 7 集的吊牌」，视图保证拿到正确那张。**翻译工作收拢在一个人身上，下游全员说同一种话。**
- **另一个细节**：单个 parquet 损坏不抛异常，记 `corrupted_files` 账本继续跑——毒衣服不能让流水线停摆，validator 事后把这些集标成「读取失败」走正常处置。

### 2.2 `validator/` —— 分拣检查台（项目的「大脑」）

| 文件 | 职责 |
|---|---|
| `types.py` | 规则清单与结论的数据结构（全项目通用语言） |
| `context.py` | 看全局分布推导阈值 |
| `rules.py` | 13+ 条具体检查 |
| `probes.py` | 调 ffmpeg 探测视频帧数/亮度 |
| `streaming.py` / `parallel.py` | 同一套规则的两个加速跑法 |

- **接口**：`validate_dataset(视图, 配置)` → `ValidationResult`。
- **设计一：注册表模式**。`RULES` 表登记每条规则的名字/说明/可否修复/软硬等级。新增规则 = 登记表加一行 + 写检查函数 + 配置加默认值，不动调度代码。不用 if-elif 大分支——分支漏写一个 elif 就是静默漏检。
- **设计二：自适应阈值**。不同机器人量纲差 50 倍，死阈值必然误报。阈值从数据自身分布推：`中位数 ± k × MAD`（MAD 是抗极端值干扰的波动度量）。配置数字只是保底下限。
- **设计三：结论带证据不返回布尔值**。规则说「第 12/13/17 帧 NaN，坏帧只在首尾、可裁剪」。复核靠证据判断、报告靠证据 diff、修复器靠「首尾几帧」干活——布尔值啥都干不了。
- **设计四：三种跑法一个结论**。串行/流式/并行跑同一套规则函数，测试强制「并行与串行逐位一致」——否则你不敢信任何一个报告。**这条有一个已知反例**：流式不把整张逐帧表读进内存，所以 `meta/episodes` 缺 `tasks` 列的数据集（`lerobot/libero`，1693 ep）在流式里解析不出任务文本，会被整批误判 `missing_task`，而同一份数据全量模式判 `review 0`。逐帧类规则仍然一致，涉及元数据解析的要看 `docs/EVAL.md` §2.5 M1。

### 2.3 `policy.py` —— 分诊决定台

- **接口**：`decide()`、`read_overrides()`、`write_review_queue()`、`summarize()`。
- **设计一：可修复性写在 policy 不写在 validator**。「这个事实能不能修」是业务决定，会随认知变化；规则本身不动。
- **设计二：三层防误删保险丝**：① 软规则（state_jump）配成 error 也不许自动删；② 单规则命中 ≥95% → 全部降级复核（八成是元数据错了）；③ 整体剔除超 80% → 熔断。类比：95% 的衣服有污渍，正确反应是怀疑检测探头坏了，不是全扔。
- **设计三：人工复核回灌**。导出 CSV → 人填一列 → `--overrides` 读回覆盖并留痕。质检从「一次性脚本」变成「流程的一环」。

### 2.4 `cleaner/` —— 洗衣车间（唯一有写权限的地方）

- **接口**：`clean_dataset(数据集, 处置决定, 输出目录)`；内部三条路：`manual_clean`（全功能）、`copy_untouched`（无事可做产出副本）、`reindex_untouched`（只重建索引）。
- **设计一：先暂存、后原子发布**。写入先落 `cleaned_dataset.tmp-xxxx`，校验通过后一步 `rename` 到位。直接往目标写，写一半崩溃就得到半成品还覆盖了上一版好的。
- **设计二：洗完自己再考一遍**。发布前 `verify()` 重新完整加载、重跑规则引擎，确认修复没引入新问题。
- **设计三：能用官方工具就用官方工具**。纯删除优先调官方 `lerobot-edit-dataset`；有修复项才走内置实现（官方只能删不能修）。
- **设计四：用 pyarrow 改写不用 pandas**。直接在 Arrow 表上替换列——pandas 往返会破坏固定形状嵌套数组的类型信息，下游训练库读不了。

### 2.5 `reporter.py` —— 报告科

- **接口**：`build_report()` / `write_report()` / `print_summary()` / `attach_cleaning_result()` / `attach_artifacts()`。报告在清洗**之前**生成，清洗结果事后回填——「报告不因清洗失败而丢失」的关键。
- **设计**：报告塞了配置快照、阈值推导依据、数据集指纹。结论的价值取决于**可追溯**——三个月后有人问「当时为什么删了这条」，报告必须自己能回答。

### 2.6 `config.py` —— 参数面板（支撑地板之一）

- **接口**：`load_config()`（四层合并：代码默认 ← 配置文件 ← 机器人型号 ← 命令行）、`cfg_get(config, "a.b.c", 默认值)`、`severity_of()`。
- **设计**：`cfg_get` 兜底让配置文件只需写「与默认不同的几行」。但拼错的键会被兜底**静默**当「没配置」——所以还有 `validate_config` 严格模式在加载时拦拼写错误。兜底是双刃剑，严格校验负责补刀。

### 2.7 `manifest.py` —— 会员档案（支撑地板之二）

- **接口**：`QualityManifest` 类（load/save/store_results/unchanged_episodes/reusable_cleaning）。
- **设计一：廉价指纹**。指纹 = 文件大小 + 修改时间（纳秒级）+ 小文件加内容哈希。「判断有没有变」的成本不能接近「重跑一遍」。代价：内容篡改但大小/时间恰好不变时失灵——小文件内容哈希补最常见的洞。
- **设计二：配置也参与指纹**。阈值变了旧结论全作废——缓存复用的前提是「条件完全一样」。

### 2.8 `baseline.py` —— 白名单本（支撑地板之三）

- **接口**：`apply_whitelist()`、`append_to_config()`。
- **设计**：摘除位置在 policy **之前**（白名单才能影响处置）；摘掉的告警转入 `ignored_baseline` 留痕。「这是误报」是人的裁决，不混进规则引擎代码。

### 2.9 `pointcloud/` —— 皮衣特护线（自成体系的子系统）

- **接口**：与主线刻意同构——`validate_pointcloud_dataset()` / `decide()` / `clean_pointclouds()` / `write_review_queue()`。学一套会用两套。
- **为什么独立子系统**：点云是「每帧 512 点 × 6 数」的二维数组，塞进 pandas 表撑爆内存——用逐批流式读取；修复方式也不同（补点 vs 裁行）。硬塞进主线，validator 会被特例撑爆——**分开才能各自简单**。
- **核心契约：「只改点，不改行」**。一帧修不了不能删行——删行让 state/动作/视频全错位。坏帧用同 episode 最近好帧**占位**（hold-last-good），行数永远不变。

### 2.10 门口保安与周边工具

- **`precheck.py`**：训练脚本开头调一行 `precheck(路径, strict=True)`，有 drop 级直接抛异常。validator 的「轻装版」——复用规则但只读、几秒出结论。
- **`versions.py`**：数据集快照/回滚。内容寻址存储：按文件哈希存对象，同一文件多版本只存一份。
- **`tuning.py`**：读历史复核 CSV 统计「人最终判删比例」，产出阈值建议 YAML。**只建议不自动生效**——判定规则必须可审计。
- **`card.py` / `visualizer.py` / `tools/`**：纯展示层。
- **`parallel.py`**：多进程调度通用工具。核心契约「worker 是纯函数」（只读输入、只返回结果、不写外部）——纯函数不会打架，多进程无需加锁。

### 2.11 第 2 层检验题与答案

**Q1** 去掉视图对象、让 validator 直接拿原始表查吊牌，会在什么场景出什么事故？

> **答**：meta 用全局编号（420010…）、data 用 0..N-1 的数据集：validator 拿集号 5 查吊牌表「第 5 行」，查到的是全局编号 5 的**别人家**吊牌——任务描述、视频时间窗全张冠李戴且不报错。视图对象把这个翻译收拢在一处。

**Q2** 新增一条「视频亮度闪烁」检查要动哪几处？if-elif 版本最容易出什么静默错误？

> **答**：三处：`types.py` RULES 表登记 → `rules.py` 写检查函数 → `config.py` 加默认严重度。if-elif 版：忘写 elif 或顺序被前面分支截胡——不报错，只是这条检查永远不跑。

**Q3** 「暂存 → 校验 → 原子发布」组合拳分别防哪两类事故？

> **答**：暂存防「洗到一半崩溃，半成品覆盖上一版好的」（保过程）；`verify()` 重跑规则引擎防「洗完了但修复引入新问题」（保结果）。

---

## 第 3 层 · 代码精讲（四轮）

每轮按固定模板：一句话作用 → 逐段拆解（讲为什么）→ 语法/框架 API/设计模式 → 踩坑 → 更简洁写法讨论。

### 3.1 `_as_int` 与 `_reconcile_episode_ids`（编号对齐）

位置：`pipeline/loader.py`（658-668 行 / 1054-1164 行）。

#### 热身：`_as_int`（11 行）

**一句话作用**：把「可能是任何鬼东西」的值安全转成整数；转不动返回 None，**绝不抛异常**。

```python
def _as_int(value: Any) -> int | None:
    if value is None:                      # ① 先挡 None
        return None
    try:
        num = float(value)                 # ② 宽容入口：字符串/np 类型都能吃
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num) or not num.is_integer():   # ③ 严格出口
        return None
    return int(num)
```

- **为什么先 float 再 int**：`int("3.7")` 抛异常而 `float("3.7")` 成功——宽容入口 + 严格出口。
- **③ 是精髓**：`math.isfinite` 拦 NaN/Inf（NaN 参与任何比较都返回 False，必须用专门函数判断）；`is_integer()` 拦 3.7——集号必须整数。
- **模式**：*宽容解析、严格校验、失败返回 None*。脏数据是常态不是异常——失败给信号不给崩溃。
- **踩坑**：`int(3.7)` 会静默截断成 3（丢数据不报警，比崩溃更难查）。
- 同文件 `_as_float` 是双胞胎，少了 `is_integer()` 那关。

#### 主菜：`_reconcile_episode_ids`

**一句话作用**：吊牌本（meta）和货物（data）用两套集号时，尝试建立映射——**只有证据足够才认，证据不足宁可宣布「对不了账」**。

**类比**：仓库对账——货箱贴流水号 0,1,2…，登记本写会员号 420010…。要确认对应关系，必须找**独立证据**：数每箱件数对不对得上档案（证据一），核对录像时长（证据二）。两项都 ≥99% 才认。

**段 1：签名——类型系统预先声明三种结局**

```python
def _reconcile_episode_ids(
    episodes_meta: pd.DataFrame | None,
    data_summary: pd.DataFrame | None,
    fps: float,
    *,
    min_agreement: float = 0.99,
) -> tuple[dict[int, int], dict[str, Any]]:
```

返回 `(映射表, 诊断信息)`。诊断里 `mode` 取四值：`native`（本来一样）/ `remapped`（成功）/ `ambiguous`（证据不足拒绝猜）/ `unavailable`（没料）。**为什么多带诊断**：调用方需要知道「为什么没对上」才能区分「不可评估」（不怪数据）和「没得查」。把「结论」和「为什么」绑在一起返回，是贯穿全项目的模式。

**段 2：卫语句（guard clauses）**

```python
    if episodes_meta is None or episodes_meta.empty or "episode_index" not in episodes_meta.columns:
        return {}, {"mode": "unavailable", ...}
```

异常路径全部前置、提前 return，主逻辑不缩进。对比嵌套金字塔（三层 if 套主逻辑）——卫语句让眼睛不做括号配对体操。

**段 3：捷径先行**

集合相等 → native 零成本退出；数量不等 → 一一映射数学上不可能，直接认输。**模糊地带的正确姿势是声明模糊，不是硬猜。**

**段 4：证据一——行数对账**

```python
    rows_arr = data_summary["rows"].to_numpy(dtype="int64")
    meta_sorted = (...).sort_values("episode_index")          # 吊牌按集号升序
    length_ok = int((rows_arr == length_arr).sum())
```

行数是数据的客观属性，与集号体系无关——映射正确则每集行数必须等于吊牌 length。**配对双方必须先统一排序基准**，这是所有「按位置配对」代码的命门。

- **numpy 向量化**：`(rows_arr == length_arr)` 一次性比较全部位置得布尔数组，`sum()` 时 True 当 1。新手 for 循环慢一个数量级。**NumPy 心智模型：把「对每个元素做某事」改写成「对整个数组做一次某事」**（广播）。
- **⚠️ 真 bug（审查轮 1）**：`meta_sorted["length"]` 先访问、列存在性后判断——守卫写在访问之后形同虚设，meta 缺 length 列时 KeyError 崩掉整次加载。教训：**守卫必须出现在它守卫的访问之前**（「先写主逻辑后补防御」的书写顺序最容易放错位置）。

**段 5：证据二——视频窗口对账**

```python
        window = to_vals - from_vals                          # 视频窗时长（秒）
        frames_in_window = np.rint(window * max(fps, 1e-6))   # 折算帧数
        frames_in_window = np.where(np.isfinite(frames_in_window), frames_in_window, -1).astype("int64")
```

时长 × fps = 应有帧数，与行数比对。**为什么需要第二证据**：所有集恰好 200 帧（仿真常见）时行数证据对**任何**配对都 100% 吻合——证据必须来自相互独立的维度。

- `max(fps, 1e-6)`：防御层内部继续防御。
- `np.rint`：四舍五入（199.9999 直接转 int 会变 199 误判）。
- `np.where(..., -1)`：无效值换哨兵值 -1——一个技巧同时解决「无效值」和「比较会崩」。

**段 6：裁决**

```python
    evidence = [r for r in (length_agreement, video_agreement) if r is not None]
    agreement = min(evidence) if evidence else 0.0
    if not evidence or agreement < min_agreement:
        return {}, {"mode": "ambiguous", ...}
    mapping = {d: m for d, m in zip(data_ids, meta_ids)}
```

- **取 `min` 不取平均**：证据 1.0 和 0.5，平均 0.75 会稀释坏消息。链断在最弱环节。
- **没有证据 ≠ 通过**：`not evidence` 也进 ambiguous——证据为零敢配对等于闭眼签字。
- **0.99 不是 1.0**：给现实噪声留缝，一条脏吊牌不该否掉整个正确映射。
- **⚠️ 真 bug（审查轮 1）**：`diag["data_order_sorted"]` 算了但从未参与裁决——`zip` 位置配对隐含假设「数据按集号升序出现」，假设不成立时配对整体错位且证据校验给假象。**「诊断信息算了但没用上」是经典代码坏味道——要么用上要么删掉。**

**本节四件事**：三态返回 + 诊断信息；证据思维（独立证据、取最弱）；向量化；卫语句前置 + 哨兵值。

**检验题与答案**

> **Q1** 为什么「先转 float 再转 int」？`int(3.7)` 在原函数里会怎样？
> **答**：`int(value)` 是严格入口（`int("3.0")` 直接抛异常）；先过 `float()` 宽容之门能吃下各种变体。`int(3.7)` 在原函数里到不了最后一步——`is_integer()` 拦下返回 None。集号必须整数，宁可「没法用」不可静默截断。
>
> **Q2** 为什么不加第三条证据（如任务文本），而视频窗这种「独立物理维度」才行？
> **答**：任务文本、长度都写在同一本吊牌册上——吊牌册错了三条证据一起错（相关性高的证据没有独立证明力）。视频窗是物理时间维度，才能互相制衡。加证据关键在**独立性**不在数量。
>
> **Q3** `min(evidence)` 换 `mean(evidence)` 会放过什么？
> **答**：证据 1.0 和 0.98：min=0.98<0.99 拒绝；mean=0.99 放行——放进去的是 2% 错位映射（200 集里 4 集全错）。平均数稀释坏消息。

### 3.2 `check_action_jump`（跳变检测）

位置：`pipeline/validator/rules.py`（93-196 行）。

**一句话作用**：检测 action（发给机器人的控制指令序列）相邻帧跳变异常——**只抓真跳变，放过合法快动作**。

**为什么它伤疤最厚**：`docs/POSTMORTEM.md` 记录了三次误判进化史：度制机器人（±100°）几乎每集误报；步进保持型指令误报；干净数据集 17/206 集边缘误报。**好规则不是设计出来的，是被真实数据教育出来的。**

**类比**：地震台报异常震动。不拿全国统一值当标准，而是看本地历史正常最大震动范围（自适应阈值）；超过范围还分两种：连着几天越来越猛——正常施工；一夜孤立爆一下——才是地震。

**段 1：Δ 的准备**

```python
    deltas = np.abs(np.diff(arr, axis=0))       # 相邻帧变化量，T 帧变 T-1 步
    deltas = np.nan_to_num(deltas, nan=0.0, posinf=np.inf)
    dim = min(deltas.shape[1], len(thresholds))  # schema 漂移时能查多少查多少
```

**一行两个深思熟虑**：NaN 的 Δ 置 0（缺失由 `nan_state` 专职处理，职责分离）；Inf 保留 Inf（超过任何阈值，自然成为候选）。**缺失可以豁免，故障必须留声。**

**段 2：离散维豁免（夹爪条款）**

```python
    nunique = np.array([len(np.unique(arr[np.isfinite(arr[:, d]), d])) ... for d in range(dim)])
    discrete_dims = np.where(nunique <= discrete_max)[0]
    if discrete_dims.size:
        deltas[:, discrete_dims] = 0.0
```

夹爪 action 只有两个值（开/合），每次开合就是 ±2 跳变——不豁免则夹爪每抓一次报一次异常。**用「取值种类数 ≤3」反推离散维**：连续物理量不可能只取 3 个值。

**踩坑**：`deltas[:, discrete_dims] = 0.0` 原地修改——此处安全（deltas 是本函数刚造的），重构复用外部数组时会静默污染。**改数组前永远问：这个内存块归谁？**

**段 3：海选——便宜的全局过滤**

```python
    over = deltas[:, :dim] > thr        # 必要条件，不是结论
    frames, dims = np.where(over)
```

**两段式过滤（海选 → 面试）**：向量化一行扫全部一万步，候选压到个位数；昂贵的逐个核验只花在候选人身上。**向量化大海，循环打捞针。**

**段 4：四元数条款（全函数最妙）**

背景：**四元数**用 4 个数 (x,y,z,w) 表示三维旋转，特性是 **q 和 −q 表示同一个姿势**（「向左转 90°」=「向右转 270°」）。动捕设备常在相邻帧切换写法，数值上像跳了 2|x|，物理上什么都没发生。

```python
    head, tail = arr[:-1, :dim], arr[1:, :dim]
    with np.errstate(invalid="ignore"):
        step = np.abs(tail - head)
        residual = np.abs(head + tail)
        mirror = (residual <= flip_tol * step) & (step > 0)   # 变号：和≈0，差巨大
    mirror &= finite
    quat_mask = _quaternion_dim_mask(dim_names, dim)
    if quat_mask is not None:
        mirror &= quat_mask[None, :]          # 有元数据时只在四元数维里认翻转
    flip_mask = mirror & (mirror.sum(axis=1, keepdims=True) >= flip_min_dims)  # ≥2 维同时翻
```

**一行漂亮数学**：设 a→−a（变号）：差 `2|a|` 巨大、和 `0`。真跳变：差大和也大。判据 `残差 ≤ tol × 步长` 用**和与差的结构关系**识别符号翻转，对幅值、单位、机器人型号完全免疫。

- **元数据优先、启发式兜底**：有维度名只在四元数维认翻转（非四元数维的变号是真跳变）；没名字才退回数值启发式。
- **`flip_min_dims ≥ 2`**：真翻转是 4 分量同时变号；单维变号更可能是真跳变。
- `np.errstate` 静音 NaN 比较警告，`mirror &= finite` 随后统一清理——**先静音，后打扫，必须有人接手**。

**段 5：面试——孤立尖峰核验**

```python
    for f, d in zip(frames.tolist(), dims.tolist()):
        if flip_mask[f, d]:
            sign_flips += 1
            continue
        lo, hi = max(0, f - window), min(deltas.shape[0], f + window + 1)
        neighbours = np.delete(deltas[lo:hi, d], f - lo)      # 去掉自己
        ref = float(np.median(neighbours)) if neighbours.size else 0.0
        if ref <= 0 or deltas[f, d] >= spike_ratio * ref:     # 孤立尖峰才算
            kept.append((int(f), int(d)))
```

问每个候选：**「你这步相对邻居是不是孤立的？」**取前后 window 帧（按 fps 折算）同维 Δ 的中位数当参照。

- **`ref <= 0` 反而保留**：邻居全零（静止后突然动）时任何超阈 Δ 都算尖峰——周围越安静越可疑。
- **用中位数不用均值**：邻域可能有别的坏点，均值会被拖走。稳健统计专门伺候脏数据。
- **踩坑**：`f` 是 Δ 下标（对应「f→f+1 这一步」），不是帧号本身；`hi` 的 +1 因为切片右边界不含。

**段 6：量刑**

```python
    exceed_ratio = float(kept_deltas[worst_pos] / max(float(thr[worst_dim]), 1e-9))
    if escalate_ratio is not None and exceed_ratio >= float(escalate_ratio):
        severity = "error"
    elif exceed_ratio < downgrade_ratio:      # 默认 1.5
        severity = "info"
```

阈值本来就压在分布尾部，1.0~1.5 倍命中几乎全是边缘擦线 → 降 info 只记录；真跳变通常超 5~100 倍。**分级量刑的本质：规则的置信度不均匀，报告的呈现必须反映这一点。**

**段 7：出证——连豁免都记账**

证据里不止「抓到了什么」，还有「放过了什么」（`smooth_peaks_ignored` / `sign_flips_ignored`）。**审计的完整性 = 定罪记录 + 无罪记录。**

**检验题与答案**

> **Q1** 为什么 NaN 置零、Inf 保留？反过来会怎样？
> **答**：NaN 是「缺数据算不出来」，不是跳变——缺失可豁免（nan_state 专职）；Inf 是「数值真的炸了」，置零等于把故障洗白成正常。反过来：NaN 保留会让比较返回 False 静默溜走（还好），Inf 置零把真故障藏起来（致命）。**缺失可豁免，故障必须留声。**
>
> **Q2** 用数值验证四元数判据。
> **答**：① a=0.7→−0.7：step=1.4，residual=0 → 0 ≤ 0.35 → True。② a=0.7→0.9：step=0.2，residual=1.6 → 1.6 ≤ 0.05 不成立 → False。变号时 residual≈0、step 巨大；真跳变两者同量级。
>
> **Q3** 「邻居全零 + 一步超阈」该放行还是保留？
> **答**：保留更安全。误报代价是人工看一眼；漏报代价是碰撞跳变污染归一化、训练静默学错——且漏报永远没人发现。**代价不对称原则：误报可弥补，漏报不可见。**

### 3.3 `_decide_one` 与 `_rewrite_frames`（分诊与手术）

位置：`pipeline/policy.py`（126-164 行）/ `pipeline/cleaner/manual.py`（348-389 行）。

#### 主角一：`_decide_one` —— 五级决策瀑布

**一句话作用**：把一个 episode 的体检结论走一遍优先级瀑布，落进四档之一。

**模式**：决策瀑布（cascade）vs 打分制。瀑布的优势是**可解释**——「为什么 repair」永远能一行行念出条件；医疗分诊也是瀑布：先看生命危险，再看保守治疗。**优先级编码的是价值观。**

```python
    # 第 0 级：没病 → keep
    if not res.reasons:
        return Decision(res.episode_index, "keep", severity, [], "保留")

    # 第 1 级：能救先救（边界 NaN 裁剪）
    trim = res.trim_plan()
    if trim and set(res.reasons) <= {"nan_state", "nan_action"}:   # 子集判断！
        ...

    # 第 2 级：能修的修（病种全部在 REPAIR_ACTIONS 清单里）
    if all(code in REPAIR_ACTIONS for code in res.reasons):
        repair = REPAIR_ACTIONS[res.reasons[0]]

    # 第 3 级：error → drop，软规则逃生门
    if severity == "error":
        soft = [c for c in res.reasons if (RULE_MAP.get(c) and RULE_MAP[c].tier == "soft")]
        if soft:
            return Decision(..., "review", ...)     # 软规则不自动剔除
        return Decision(..., "drop", ...)

    # 第 4 级：info → keep 仅记录
    # 第 5 级：warning → review（或 --on-review drop 显式剔除）
```

- **第 1 级双条件**：`trim` 存在（坏帧全在首尾、占比不超限；state/action 取 max 合并——裁到两者都干净）+ **子集判断**——病种必须只有 NaN。如果同时还报 `frame_mismatch`，裁帧会让错位更严重。**治你的病之前，先确认你没有别的病。**
- **第 3 级**：`RULE_MAP.get(c)` 防御性查表——动态加的结论（video_probe_error）不在表里，下标会 KeyError。**表查询永远 .get + 兜底**。软规则即便配成 error 也只 review——删数据不可逆，error 级别不等于有把握。
- **`on_review` 开关**：批量删除必须是用户显式命令行决定，不能是默认行为。
- **个体决策后、集体守卫前**：两个熔断（95% 单规则 / 80% 总量）在所有个体决策之后统一跑——**局部决策、全局否决**。
- `Decision.source` 字段留痕：rule / override / systemic-guard——每条决定记住「我是谁拍板的」。

#### 主角二：`_rewrite_frames` 核心段 —— 数据外科手术

**一句话作用**：按处置清单逐源文件动刀——删行、裁边界帧、重排三套编号、重建时间轴——同步更新账本。

**背景**：v3 一个 parquet 文件装**多个** episode。「文件粒度」≠「集粒度」——记住这句话，两个真 bug 都从它来。

```python
        mask = np.isin(ep_values, np.asarray(kept, dtype=np.int64))
        sub = table.take(pa.array(np.where(mask)[0], type=pa.int64()))   # 矢量化筛选
        new_eps = ...ep_index_map...                                     # 旧集号→新集号重映射
```

**切口一：裁帧（348-369 行）**

```python
            row_pos[i] = c                    # 本集第几行（⚠️ 文件内计数）
            keep_row[i] = head <= row_pos[i] < total - tail
            sub = sub.take(pa.array(np.where(keep_row)[0], type=pa.int64()))
```

裁帧用纯算术：先用计数器铺「行位置」轨道，再一个不等式切出保留区——把逻辑问题翻译成区间问题。

**⚠️ 真 bug（审查轮 4）**：`counter`/`totals` 每个源文件重置。episode 跨文件时，file-001 的首行被当成集内第 0 行——头裁两遍、尾各裁一遍，**中段好帧被误删**且视频窗按整集收窄，数据错位无告警。修复：计数器和总数必须集粒度、活在文件循环**外**（同函数的 `running_frames` 就是正确写法，两把计数器一把踩坑一把正确）。教训：**状态的生命周期必须匹配它的语义粒度**。

**切口二：三套编号同步重排（371-382 行）**

| 编号 | 语义 |
|---|---|
| `episode_index` | 集号 |
| `frame_index` | 集内帧号（每集从 0） |
| `index` | 全局绝对帧号（0..总帧−1，全库唯一） |

`running_frames` 定义在文件循环之外、跨文件连续累计。`index` 用 `np.arange` 铺连续区间——**cleaner 是 Schema 审计那条「index 全局唯一」契约的执法者**：修复产物必须让审计复核通过。这就是契约思维：修复不是把数据改对，是让产物满足下游一切隐含假设。

**切口三：时间轴重建（384-389 行）**

```python
        if any(int(e) in rebuild_new for e in new_eps):
            ts = (frame_idx / fps).astype(np.float64)
            sub = set_column(sub, "timestamp", ts)     # ⚠️ 整列替换！
```

**⚠️ 真 bug（审查轮 4）**：意图粒度是「集」，操作粒度是「列」——`set_column` 替换整个文件的 timestamp 列，同文件**没病**的集也被重写：相对时间轴数据丢掉时钟抖动（轻度）；全局时间戳数据集被归零、与视频窗彻底错位（重度），且复检只拦 drop 级静默通过。修复：行级掩码，只重写命中集的行。

教训与切口一同源反向：那里「状态粒度太窄」，这里「操作粒度太宽」。**每次写操作前问：我的意图是行级、集级还是文件级？三层粒度各用各的工具。**

**收尾：账本同步**——把每集真实行数、index 起止、所在文件记进 `data_meta`，meta 重建全靠它。**数据和账本必须同一次手术内同步更新。**

**检验题与答案**

> **Q1** 构造一个违反子集约束就出事故的例子。
> **答**：`reasons = [nan_state（可裁）, frame_mismatch]`：裁帧后行数变了、视频没重切（frame_mismatch 不在自动修复清单）——错位从「差几帧」变「差更多帧」。子集判断的本质：裁帧只治 NaN，其它病它治不了也不该装治。
>
> **Q2** 有人把 state_jump 配成 error 想自动删，实际会发生什么？原则是什么？出口怎么加？
> **答**：走软规则逃生门进 review。原则：删数据不可逆，error 级别≠有把握。出口学第 5 级 `on_review`：加显式 CLI/config 开关（如 `--allow-soft-drop state_jump`），用户主动声明才生效且留痕——出口可以有，但必须明确、逐规则、可审计。
>
> **Q3** `counter` 与 `running_frames` 结果不同的最小触发条件？
> **答**：一个 episode 的行分布在两个 parquet 文件里且它有裁帧需求。`head_frames` 天然集粒度：validator 在内存中跨文件分组后的**完整 episode 序列**上数「开头连续坏帧数」——它没有「文件」概念。

### 3.4 点云 `_decide_one` 与 `repair_frame`（特护线）

位置：`pipeline/pointcloud/policy.py`（55-77 行）/ `pipeline/pointcloud/cleaner.py`（179-221 行）。

背景概念：**点云帧** = 一帧里 512 个空间点、每点 6 个数（x,y,z + R,G,B）的二维数组；**固定点数契约** = info.json 声明每帧必须恰好 512 点，下游 DataLoader 按这个形状开内存。

**类比**：清洗一帧点云 = 整理一个**规定 512 人的合唱方阵**：请走伤员（NaN）、出界的、捣乱的（离群）、站重位的（重复）；人不够就**克隆**补位（克隆体挪一点免得重叠）；人多了砍掉。无论怎么折腾，「方阵必须 512 人」不可动摇。

#### 主角一：点云版 `_decide_one` —— 同构不同形

与主线骨架相同（瀑布 / severity / review 逃生门），**但瀑布顺序几乎完全颠倒**：

| 优先级 | 主线（episode 级） | 点云线（帧级） |
|---|---|---|
| 第 1 级 | **能救先救**（trim） | **判死刑**（有效点过少/几何退化 → drop） |
| 第 2 级 | 能修的修 | 不可修复项 → 人工 |
| 第 3 级 | error → drop（软规则逃生） | **坏点比例超预算 → drop** |
| 第 4 级 | info → keep | 修 |
| 第 5 级 | warning → 人工 | （无） |

**为什么顺序相反**：单位成本不同。episode 是几分钟人力遥操作，删一条心疼——主线先抢救；一帧只是 1/18000 秒采样，删一帧无妨——但「能修」门槛更硬：**无法凭空造出 512 个点的空间结构**。主线「宁枉勿纵地救人」，点云线「修不了干脆别修，别用假几何污染训练」。**同一个瀑布模式，装什么内容由业务的成本结构决定——复骨架，不复制价值观。**

```python
    if "pc_empty_or_sparse" in res.reasons or "pc_degenerate" in res.reasons:
        return PointCloudDecision(key, "drop", ...)      # 信息已不存在，任何修复都是编造
```

判定用 `in res.reasons` 不查 severity——这两个病**无论配什么级别**都 drop：凑不齐方阵是物理事实，不是阈值观点。

```python
    non_repairable = [c for c in res.reasons if c not in REPAIRABLE]
```

`REPAIRABLE` 是集合推导式 `{r.code for r in RULES if r.repair}`——**从规则登记表自动生成**，加规则零维护。「能不能修」的唯一事实来源是规则登记表，别处只许引用。

```python
    ratio = max(float(res.metrics.get("non_finite_ratio", 0.0)),
                float(res.metrics.get("outlier_ratio", 0.0))
                + float(res.metrics.get("duplicate_ratio", 0.0)))
    if ratio > max_repair:      # 默认 0.35
        return PointCloudDecision(key, "drop", ...)
```

**第 3 级：修复预算**。修复链会**删点**，删太多再克隆补位，「补」出来的是假几何——修之前先估账，超 35% 报废。类比：毛衣破洞超过三成，织补就不是修复而是重织。

**⚠️ 真 bug（审查轮 5）**：预算漏了 `outside_ratio`（越界占比）——而越界裁剪**恰恰是删点动作**。一帧 100% 越界：预算 0% → 放行手术 → bounds 清空全部点 → 空数组 → Arrow 写入对固定 512 点类型爆炸 → **整个文件修复回滚**。对照：`color_bad_ratio` 不用进预算——颜色修复 `np.clip` 裁数值**不删人**。**预算口径必须精确等于「修复链将删除的点的全集」**——多算浪费，少算送命。

另一点不对称：点云版 `decide()` 没有主线那套熔断守卫。**复用模式时连防护也要一起复用，不然只搬了半套。**

#### 主角二：`repair_frame` —— 512 人方阵的修复链

**一句话作用**：对判 repair 的帧执行固定顺序修复流水线，**无论死伤多少，必须交出恰好 512 个点**。

**段 1：动作清单两个来源取并集**

```python
    actions = set(decision.repair_actions) | {RULE_MAP[r].repair for r in decision.reasons if RULE_MAP[r].repair}
```

决策上记录的 + 按结论实时反查的取并集——防「决策和结论不同步」。**关键参数单一事实来源（规则表），其它引用都是投影。**

**段 2：五道工序，顺序即语义**

```python
    # ① NaN 先清场（离群检测要算距离，NaN 会污染整个距离矩阵）
    pts = pts[np.isfinite(pts).all(axis=1)]
    # ② 越界裁剪（廉价判断），在离群之前
    inside = np.all((xyz(pts) >= ctx.workspace_lower) & (xyz(pts) <= ctx.workspace_upper), axis=1)
    # ③ 离群检测（昂贵计算 O(N²)）——别把算力浪费在注定要删的点上
    # ④ 去重（删离群会改变重复比例，先去重白算）
    _, idx = np.unique(np.round(xyz(pts), ctx.duplicate_decimals), axis=0, return_index=True)
    pts = pts[np.sort(idx)]
    # ⑤ 颜色裁剪（裁数值不动几何）
    pts[:, 3:6] = np.clip(pts[:, 3:6], 0.0, ctx.color_scale)
    # ⑥ 补点必须最后——不删完不知道要补几个
```

**顺序一个都不能换**：NaN 先清场 → 廉价筛除在昂贵计算之前 → 去重在删点之后 → 不动数量的放最后 → 补点收尾。「海选 → 面试」原则这次用在**工序排序**上。

浮点细节：先 `np.round` 4 位再比对——**浮点相等判定必须量化**（0.1 的二进制误差让直接比较几乎永不相等）；`np.sort(idx)` 恢复原顺序，删完人队伍不乱。

**段 3：补点——确定性随机 + 契约落地**

```python
        if kept < target:
            rng = np.random.default_rng(abs(decision.key[0]) * 100000 + decision.key[1])  # 确定性种子！
            idx = np.arange(target) % kept        # 循环索引：人人至少出场一次
            order = rng.permutation(target)       # 打乱，克隆体不扎堆
            scale = spacing if spacing > 0 else float(ctx.resolved.get("scene_extent") or 1.0)
            jitter = rng.normal(0.0, ctx.jitter_ratio * max(scale, 1e-6), size=(target, 3))
            picked = pts[idx[order]].copy()
            picked[:, :3] += jitter               # 克隆体挪一点位置（只抖 xyz）
            pts = picked
        else:
            pts = pts[:target]                    # 人多了砍到 512
```

- **种子 = (集号, 帧号)**：同一帧不管跑多少遍、在哪个进程跑，克隆结果**逐位相同**。项目承诺「并发与串行逐位一致」——用真随机这测试永远过不了。**随机性可以有，但必须可复现。**
- **`idx = np.arange(target) % kept`**：取模实现「用完了从头再排」。
- **抖动幅度 = 0.005 × 相邻点平均间距**：参照数据自身密度尺度——比间距小则克隆体不与本体重叠（白去重），太大则扭曲几何。**噪声尺度跟随数据自身尺度**——和自适应阈值一个哲学。
- **只抖 xyz**：克隆的是「这个位置附近的点」，颜色照抄本体。
- **语法点**：`pts[idx[order]]` 是花式索引——**numpy 花式索引返回拷贝，切片返回视图**。这条分界线是 numpy 最大的坑源。

**⚠️ 真 bug 一（审查轮 5）**：jitter 加在全部 512 行上——原班好点也被挪了亚毫米级，违背「只改坏点」。修法：前 kept 行保持原坐标，只对克隆行加噪声。

**⚠️ 真 bug 二（同一审查轮）**：`kept == 0` 返回空数组——方阵一个人不剩。上游预算门漏算 `outside_ratio` 没拦住 → 空单元格 → Arrow 写入爆炸 → 文件级 except 接住 → **整个文件修复作废**。一个漏算的预算项顺着「决策 → 修复 → 写盘」三层传递，炸成整文件回滚——**边界 bug 的破坏力总是大于它的体积**。

**检验题与答案**

> **Q1** 主线第 1 级「能救先救」、点云第 1 级「直接判死」，除单位成本还有什么信息论理由？
> **答**：sparse/degenerate 是**信息不存在**（没有几何可救），而「NaN 在首尾」是信息存在但边界损坏（裁掉即可）。可救性取决于信息是否还在，不取决于愿意花多少成本。
>
> **Q2** 预算为什么必须覆盖所有删点动作，color 却不用？
> **答**：预算估算的是「修复链将删掉的点的全集」。bounds/outliers/dedupe/filter 都改**数量**，必须计入；color 只改**坐标数值**（clip），人数不变，不计入。
>
> **Q3** 种子公式的碰撞边界与后果？
> **答**：帧号 ≥100000 时 `abs(ep)*100000+fi` 会与下一集的种子相同（如 ep0 frame 100001 = ep1 frame 1）。后果是「修复结果仍然合法」——种子只决定克隆哪些点、抖动方向，碰撞只意味着这两帧的补点结果恰好相同（可复现性没破，只是少了一种随机序列），不是错误。

---

## 附录：贯穿全程的六条设计思想

| # | 思想 | 在代码里的样子 |
|---|---|---|
| 1 | **证据思维**：结论必须带证据，连「无罪释放」都要记账 | 规则证据、诊断 mode、`smooth_peaks_ignored`、`Decision.source` |
| 2 | **粒度意识**：状态与操作的生命周期必须匹配语义粒度 | 跨文件计数器 vs 文件内计数器；行级掩码 vs 整列替换 |
| 3 | **确定性**：同数据 + 同配置 = 逐位一致 | 并发归并排序重建、补点种子化 |
| 4 | **契约思维**：修复不是改对数据，是让产物满足下游一切隐含假设 | 三套编号同步重排、固定点数补位、`verify()` 复检 |
| 5 | **代价不对称**：误报可弥补（进复核），漏报不可见（静默学错） | 软规则逃生门、熔断守卫、`ref<=0` 保留尖峰 |
| 6 | **职责分离 + 单位成本定价值观**：复骨架，重写价值观 | 判定/处置分离、只读/写入分离、episode vs 帧的瀑布顺序颠倒 |
