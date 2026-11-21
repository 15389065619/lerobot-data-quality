# 代码审查修复清单（2026-11 六轮全量审查 · 46 条）

> **来源**：2026-11-12/13 对本仓库的六轮静态审查，每条带行号定位与修法，完整分析在审查对话中。
> **状态基线**：`f5f7b6d`（2026-11-14/15 夜，17 个修复 commit）。**46 条中已修复 16 条**（含 M4 共 17 个 commit），R2-1 为部分修复；全量测试 327 passed。
> **与外部交叉验证的关系**：`docs/EVAL.md §2.5` 的 M1-M4 对照见文末。
> 本文件是内部跟踪清单，是否入库/公开由你决定。

## 统计（基线 f5f7b6d 后）

| 类别 | 已修 | 未修 |
|---|---|---|
| 严重（11） | 8：R1-1, R1-2, R2-1*, R2-2, R4-1, R4-2, R4-3, R5-1 | 3：**R5-2**、R3-1、R3-2 |
| 一般（22） | 7：R1-4, R2-3, R2-4, R3-7, R5-4, R6-2, R6-3 | 15：R1-3, R1-5, R1-6, R2-5, R2-6, R3-3, R3-4, R3-5, R3-6, R4-5, R4-6, R4-7, R5-3, **R6-1** |
| 优化（13） | 1：R6-5 | 12：R1-7, R1-8, R2-7, R4-8, R4-9, R4-10, R5-5, R5-6, R5-7, R5-8, R5-9, R6-4 |

R2-1 标 * ：**部分修复**——`da3c5ce` 只救了 `ptp==0`（全等长）场景；`mad==0 且 ptp>0`（如 99 集 100 帧 + 1 集 101 帧）仍塌缩，±1 帧偏离判 error→drop。残余场景建议与 M3 合并解决：给 `too_many` 加边际降级（超阈 <1.5× → review，与 action_jump 的 `downgrade_ratio` 同款）。

另有审查后新增项 **M4 + state 孪生**（无 state/action 侧 skipped 虚报）已修复（83a01c9）。

---

## 修复优先级（合并 M1-M4 后的队列）

### P0 · 数据正确性 / 崩溃

- [x] **R1-1** `loader.py:1113-1114` meta 缺 length 列守卫反置 → KeyError 无报告 → 守卫前置（`f5523c4`）
- [x] **R2-2** `rules.py:758-760` `_frame_labels` 对 NaN frame_index 崩溃，episode 被重建为读取失败、证据丢失 → 取证跳过非有限值（`aaae706`）
- [x] **R1-2** `loader.py:1091,1107-1154` 编号映射忽略 `data_order_sorted` 证据，等长数据集错位配对可通过校验 → 乱序判 ambiguous；配套修复 `_data_episode_summary` 首现顺序（`20844c8`）
- [x] **R2-1*** `validator/context.py:145-151` MAD=0 帧数区间塌缩 → 只救 ptp==0 场景（`da3c5ce`）；**mad==0 且 ptp>0 的残余场景未修**，建议与 M3 合并用「too_many 边际降级 <1.5× → review」解决
- [x] **R4-1** `cleaner/manual.py:349-369` 裁帧按单文件计数，跨文件 episode 裁掉中段好帧 → 两遍式：预扫整集总数 + 跨文件持久游标（`55b419d`；硬化备注：预扫 `int(old)` 遇脏单元格会 TypeError，下一批可并入 dirty 防御）
- [x] **R4-2** `cleaner/manual.py:385-389` 时间轴重建按整列生效，殃及同文件无关 episode → 行掩码部分重写（`089c757`）
- [x] **R4-3** `cleaner/meta.py:93` + `manual.py:469` 人工 override 保留无数据行 episode → KeyError 崩溃 → `_surviving_episodes` 提前剔除并留痕（`1f0a0cb`）
- [x] **R5-1** `pointcloud/cleaner.py:104-106` standalone 模式丢缺点云列的 parquet，verify 无法发现 → 跳过分支补复制 + `_verify` 口径同步（`f5f7b6d`）
- [ ] **R5-2** `pointcloud/policy.py:69-71` + `cleaner.py:143-145,204-206` 修复预算漏 outside_ratio → 100% 越界帧进修复 → 空单元格 → ArrowInvalid → **整文件修复静默回滚** → ratio 纳入 outside_ratio + 空帧降级 hold-last-good（**P0 剩余，11 条严重中最后一条未修的写盘路径问题**）
- [ ] **R3-1** `loader.py:1250-1259` + `streaming.py:119,220` 流式零容错：单损坏 parquet → exit 3 无报告 → 记账跳过（依赖 R3-4 兜底）
- [ ] **R3-2** `streaming.py:123-147` 跨文件 episode 逐片校验、后片覆盖前片 → 跨片缓冲 carry

### P0 · 口径一致性

- [x] **M4 + state 孪生**（审查后新增）无 state/action 侧 skipped 虚报 → 两侧补齐 + 负向用例（`83a01c9`）
- [ ] **R6-1** `cleaner/util.py:97-100` + `precheck.py:98` 清洗复检不透传用户 config（复检结论已进 EVAL 被引用）→ 三层透传；修复前先在 EVAL 补「复检使用内置默认阈值」口径声明
- [ ] **R2-6（需确认）** `rules.py:369-371` frame_count_abnormal 过短降级 warning 与 README/config 三方不一致 → 定口径后同步
  - 与 R2-1 残余 + M3 的统一方案相关：若引入「too_many 边际降级」，R2-6 的口径问题可一并定稿

### P1 · 结论丢失 / 路径发散

- [ ] **R3-5（= M1）** `streaming.py:84` + `loader.py:359-366` 流式 task_of 无法走 task_index 兜底（libero 三路矛盾）→ pass 2 记 `first_task_index[ep]` + 视图适配器；修完重跑 17 集口径与外部两份集
- [ ] **R3-3** `streaming.py:296-307` truncated_to 守卫放错分支，假 stats_stale → 截断判断提前（与 M2 同族不同病）
- [ ] **R3-4** `streaming.py:111-153` 流式缺 missing_eps 兜底，无数据行 episode 静默消失 → 补兜底循环
- [ ] **R3-6** `validator/parallel.py:93-100` meta 无 length 列时并行丢视频窗列 → 无 length 保留窗列
- [ ] **R1-3** `loader.py:562-569,626` `--max_episodes` ≥ 实际集数被当截断，跳过计数对账 → 仅真截断置值
- [ ] **R1-5（需确认）** `schema.py:127-143` v3 index 起点校验用扫描出现顺序，跨文件交错误报 → 按 idx_min 排序（R1-2 修复后乱序数据已判 ambiguous，本条影响面缩小）
- [ ] **R1-6** `schema.py:156-176` v3 分支缺「集内跨度==行数」判据，末集尾部断号漏报 → 补 span_bad
- [ ] **R2-5** `rules.py:645-661` + `loader.py:428-431` 单文件兜底伪造窗口，多集误报 frame_mismatch → 兜底窗口 + 多集记 unlocatable
- [ ] **R4-5** `cleaner/meta.py:76-99` kept 中源 meta 缺失的 episode 输出 meta 静默缺行 → 合成行兜底
- [ ] **R4-6** `cleaner/cli.py:95-102` `_find_cli_output` 子串匹配 + 无条件兜底 → 精确匹配、拒绝兜底
- [ ] **R4-7（需确认）** `baseline.py:34-37` 指纹只锚定目录名，同名数据集白名单互染 → 锚定 resolve 路径
- [ ] **R5-3（需确认）** `pointcloud/context.py:72-74,120-147` 包围盒只采样数据集头部帧，多场景误报且无熔断 → 均匀抽帧或补熔断
- [ ] **M2** stats_stale 前缀采样有偏（有意缓修：蓄水池采样 + POSTMORTEM 历史数字联动）
- [ ] **M3** frame_count 跨任务池化（有意缓修：先证据富化 how_derived 补 task_index，再议行为；可与 R2-1 残余合并成「too_many 边际降级」）

### P2 · 优化

- [x] **R6-5** `card.py:122-127` skipped_checks 只遍历键丢原因 → 遍历 items（`915d138`）
- [ ] **R4-4** `util.py:80-117`（有意保留告警性质；仅建议「加载失败」与「复检未过」字段拆分）
- [ ] **R1-7** `loader.py:846-859` jsonl episodes 未去重
- [ ] **R1-8** `loader.py:541,954-957` `_data_episode_summary` 零 IO 分支不可达，注释误导
- [ ] **R2-7** `validator/__init__.py:159` `probe.max_cameras` 截断相机不留痕
- [ ] **R4-8** `policy.py:114-118` override repair 兜底 index 语义错
- [ ] **R4-9** `stats_streaming.py:61,116-117` 重型特征 to_pylist + 蓄水池 ≈1.2GB/特征
- [ ] **R4-10** `meta.py:25-42` `_load_source_episodes` 不去重
- [ ] **R5-5** `pointcloud/cleaner.py:208-217` 抖动加在全部行含原好点
- [ ] **R5-6** `pointcloud/cleaner.py:118-141` last_good 按文件行序非 frame_index 序
- [ ] **R5-7** `pointcloud/cleaner.py:361-372` `--pointcloud-max-frames` 下 _verify 恒 false
- [ ] **R5-8** `pointcloud/context.py:236-237` radius 回退值用倍率 2.5
- [ ] **R5-9** `pointcloud/rules.py:91-92` 颜色切片按声明维度非实际列宽
- [ ] **R6-4** `tools/dashboard_server.py:237-239` 曲线 API 对异常行整段 500

---

## 与 EVAL §2.5 外部交叉验证（M1-M4）的对照

| 外部发现 | 清单关系 | 状态 |
|---|---|---|
| M1 流式/precheck/复检 task_of 断裂 | ≈ R3-5 | 未修（P1 首位） |
| M2 stats_stale 前缀采样有偏 | 与 R3-3 同族不同病 | 有意缓修（未修） |
| M3 frame_count 跨任务池化 | 与 R2-1 互补；建议与 R2-1 残余合并为「too_many 边际降级」 | 有意缓修（未修） |
| M4 无 action 分支 skipped 虚报 | 审查盲区新增项 + state 孪生 | **已修**（83a01c9） |

另：f6c3f34（2026-11-13）修复的「输出包含源」反向护栏为审查双方清单外的新发现，已修复并有 3 条钉住测试。

---

## 修复纪律（沿用既有约定）

1. 一个修复 = 一个 commit = 一个先红后绿的钉住测试；
2. 凡导致既有文档数字变化的，单独出「口径重跑」commit，同步 README / EVAL / INTERVIEW；
3. 每批修完跑：全量 pytest → demo 集 → `tools/dataset_inventory.py` 17 集口径 → 外部两份集 precheck，对照三路结论收敛。

## 下一步（按序）

1. **补齐 P0 残余**：R5-2（最后一条严重级写盘问题）→ R6-1 → R2-6 定口径（可顺带定「too_many 边际降级」，一并了结 R2-1 残余与 M3）；
2. **口径重跑 commit**：R1-2 / R2-1 / R4-1~3 / R5-1 都会改变结论或指纹口径，按纪律跑四层回归并同步 README / EVAL / INTERVIEW 数字；
3. **流式批次**：R3-1 ~ R3-6（R3-5/M1 优先）——注意 M1 修复后 libero 的 precheck/复检结论将大幅变化，是口径重跑的大头；
4. **P2 批次**：12 条优化项按模块归组一次收尾。
