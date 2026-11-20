# Kaggle A/B 清洗收益实验——第 1 步本地构建

**假设 H1**：相同初始化/seed/步数/归一化下，完整清洗组（B）的 hold-out flow-matching loss
低于最小修复组（A）。

## 数据真值（以 parquet 为准，不以 info.json 为准）

| 数据集 | 集 | 帧 | 说明 |
|---|---:|---:|---|
| `data/external/so101_bomb_pick_place_v1` | 150 | 88782 | 源数据。**其 info.json 写的 103 集/57211 帧是陈旧值**，meta/episodes 有 172 行（22 个坏集各两行）|
| `outputs_ext_bomb/cleaned_dataset` | 121 | 63185 | 完整清洗产物（29 drop 留痕在 `meta/quality_cleaning_notes.json`）|

三份产物（`exports/kaggle_ab/`）：

| 产物 | 集 | 角色 |
|---|---:|---|
| `so101_minrepair` | 157 块 | **A 组**：bomb 原始数据剔 15 eval 块；结构修复（重排 index、按物理块重建 info/episodes 元数据），内容（timestamp/action/视频）不动，含全部 29 个坏集。157 = 172 物理块 − 15 eval：坏集 81–102 在数据里物理存在两份分歧拷贝（meta 172 行的真相），连续 from/to 无法表达同 id 两块分离数据，只能按块拆开重编号——行数不变（81229 = 88782 − eval 7553），只多 22 个 episode 边界 |
| `so101_clean` | 106 | **B 组**：清洗产物剔 15 eval 集 |
| `so101_eval` | 15 | 两臂共用的固定评测集（重排为 0..14）|

**v2 重跑产物（2026-11-19，cleaner 帧数/pts 修复 `84daa17`+`65fa94a` 后）**：
`so101_clean_v2` / `so101_eval_v2`（同 106/15 集，`build_datasets.py --source
outputs_ext_bomb_v2/cleaned_dataset --suffix _v2` 重建），`verify_report_v2.json`
全绿——帧数差 0/34 文件（v1 为 9）、pts 34/34 首帧归零（v1 有 4 文件偏 33ms，
即 B@4000 崩溃根因；自 v2 起 ①' 需用 v2 源重跑并恢复零容差验收，脚本
`verify_v2.py`）。同 epoch 对照重训用 v2 数据 + eval′；A 续训沿用其原 stats
（上传包 = 现盘 md5 `4da81c51`，无归一化漂移）。

## 本地流程（顺序不能换）

```bash
cd kaggle_ab
python -I make_eval_split.py            # ① 冻结 eval 划分 + 320 评测窗口（已跑，可复现已验证）
python -I build_datasets.py             # ② B/eval：clean_dataset drop-only 重排（已跑，校验通过）
python -I rebuild_minrepair_blocks.py --fresh   # ②' A 组专用：物理块级最小修复
                                        #    （clean_dataset 处理不了「同 id 两份分歧拷贝」，
                                        #     按 172 块重编号 + 视频字节级复制 + 元数据按源行匹配）
python -I transcode_h264.py             # ③ 视频统一 h264（源是 AV1，Kaggle 4 vCPU 软解会卡；
                                        #    也是为消除「分段重编码文件 h264 / 整文件复制文件 av1」的混杂）
python -I apply_stats_fixed.py          # ④ B 组算一份 stats（数值=parquet 精确；图像=cv2 解码抽帧），
                                        #    字节级覆盖 A/B/eval——loss 可比的前提
python -I verify_datasets.py            # ⑤ 总验收：结构层三份必过；内容层 B/eval 零帧数差；
                                        #    validator 三连跑（A 的 flagged 经映射回原编号 ⊆ 29 坏集
                                        #    且原因内容级；B/eval 全 keep）
```

注：A 组视频在块级重建时与源字节一致（后续步骤 ③ 统一转码为逐帧透传的 h264，
帧序列不变、字节不同），其中含 eval 集画面帧残留在共享文件里，
但 meta 窗口不引用、LeRobotDataset 读不到，无泄漏（留痕在 `rebuild_minrepair_manifest.json`）。

索引文件（两臂共用，seed=0 全程可复现）：

- `exports/kaggle_ab/eval_episodes.json`：15 集双编号（cleaned 编号给 B/eval 用，bomb 原编号给 A 用）
- `exports/kaggle_ab/eval_windows.json`：320 个 `(episode_index, start_frame)`，
  语义 = obs@start_frame，action chunk = `start_frame..start_frame+49`；
  范围避开每集前 8 帧（WARMUP）且保证 chunk 不越集（HORIZON=50，与 SmolVLA 配置一致，
  Kaggle Session 1 需核对 policy 的 chunk_size；若实际为 16，改用备份
  `eval_windows_h16.json`）

留痕 manifest：`build_manifest.json` / `rebuild_minrepair_manifest.json` /
`transcode_manifest.json` / `verify_report.json`；执行日志 `rerun.log` / `verify2.log`
（06:44–07:13 的 pipeline*.log 是早期失败迭代的留痕）。

## 已知口径（写进 results.md）

- **A 组 157 块（不是方案初稿的 135 集）**：坏集 81–102 物理存在两份分歧拷贝
  （如 old-81 = 564 帧版 + 380 帧版），validator 的 29 drop 是把每个重复 id 整体丢。
  A 臂要保留全部坏数据，同一 id 的两块分离数据无法用连续 from/to 表达，唯一结构级
  修复 = 按块拆开重编号：172 块 − 15 eval = 157 块。行数口径不变（81229 = 88782 −
  eval 7553，一行不丢），只多 22 个 episode 边界。
- A 157 块（81,229 帧）vs B 106 集（55,632 帧）：B 少 31.5% 帧数是清洗的固有代价。
  B 赢 = 强结论；A 赢/平手无法区分「坏数据有害」与「数据量有害」。
- 图像统计用 stride=5 抽帧（数值特征全量精确）；三份共用同一 stats.json
  （三份字节级一致，当前 md5 4da81c51）。
- A 组内容级缺陷（video_span_mismatch / frame_mismatch / timestamp 乱序）**有意保留**，
  这是「最小修复」的定义；帧数差逐文件留痕在 `rebuild_minrepair_manifest.json`
  （20/24 文件）。validator 只标 9 块（映射回原编号 19/23/47/68/69/84/98/120/126，
  全部属于 29 坏集）——拆块后 frame_index_gap / meta_index_inconsistent 等聚合级
  症状消失，内容级症状由帧数差留痕补足。
- B/eval 允许每文件 |帧数差| ≤ max(3, 0.5%)：clean_dataset 的分段重编码
  （-ss 关键帧 seek）有 ±帧级边界效应，实测 B 最大 13 帧（0.17%）、eval 最大 2 帧。

## 验收结论（2026-11-18，verify_report.json 六门全 PASS）

| 门 | 结果 |
|---|---|
| 结构层（episode 连续 / index 对齐 / info 计数 / meta 无重复行） | A 157 块 ✓ / B 106 ✓ / eval 15 ✓ |
| stats 三份 md5 一致 + 特征齐全 + 图像统计在 0–255 语义内 | 41cb7c6c ✓ |
| eval 320 窗口全部合法 | 0 非法 ✓ |
| validator A | 9 块 flagged，映射回 ⊆ 29 坏集，原因均为内容级 ✓ |
| validator B | 106 keep / 0 flag ✓ |
| validator eval | 15 keep / 0 flag ✓ |

## Kaggle Session 对齐（对应总体方案，含过审修正）

1. **Session 1（环境 + pilot）**：
   - 锁版本装 lerobot 后先 `lerobot-train --help` 核对 flags（`--steps` 还是
     `--training.steps`、`--mixed_precision` 是否存在），并加 `--wandb.enable=false`；
   - `LeRobotDataset` 读 A 迭代 20 batch 之后，**必须对 A 全量随机访问扫描**
     （每集抽首/中/尾帧）——坏集的视频 seek 失败只有扫过才知道；
     加载失败的集从 A 剔除并记录（预先定的口径）；
   - 磁盘预算：SmolVLA checkpoint ≈ 权重 1.8GB + optimizer.pt ≈ 3.6GB ≈ 5.4GB/个；
     每落盘一个新 ckpt 就删旧 ckpt 的 optimizer.pt（评测只需 safetensors）。
2. **Session 2/3（训练）**：两臂只改 `--dataset.root` 与 `--output_dir`，seed/步数/归一化全同。
3. **Session 4（评测 + 出图）**：320 窗口两臂共用；flow-matching 噪声固定 = 每个
   batch 前 `torch.manual_seed(batch_idx)` + `policy.eval()` + `no_grad`（SmolVLA
   在 forward 内部采 time/noise，不要假设能传参注入）；**统计用配对差值 SE / 按窗口
   bootstrap**（两臂同窗口同噪声，是配对设计，不要用「batch 间标准误」）。

## 上传

三份各打一个 zip（数据集已解压可用），private Kaggle Dataset，先传 A 验速。
