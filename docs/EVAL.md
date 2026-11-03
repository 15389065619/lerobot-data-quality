# 实测记录

所有数字均来自本机真实运行（Windows 11 / Python 3.11 / 同一台机器）。

## 1. demo_dataset（36 episode / 3008 帧 / 2 相机）

- 13 处注入异常全部命中，**0 误报**（自适应阈值下正常轨迹不触发跳变告警）
- 处置：保留 24 / 修复 2（边界裁帧、时间轴重建）/ 复核 2 / 剔除 8
- 修复验证：裁帧后 NaN 归零，重建的时间轴与 `frame_index/fps` 完全一致
- 清洗后 28 episode / 2388 帧，`length==rows`、视频窗口==行数、index 连续、stats 对齐
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

- **89 个单测**：数值/清洗/增量/流式 35 个 + 点云 8 个 + 并发一致性 2 个 + 容错续传 6 个 + 配置 8 个 + FZMotion 适配 5 个 + 多模态对齐 6 个 + 预检/版本/策略/Dashboard 19 个
- GitHub Actions 在 **Python 3.10 / 3.11 / 3.12** 三个版本 × 干净 Linux 环境全部通过

## 6. 已知边界（如实说明）

- 单机运行，适配 ≤ 数十 GB / 数千 episode；更大规模可按 parquet 文件分片并行（规则函数已无状态，扩展点在 main 的批处理循环）
- 视频重建用 libx264 重编码（crf20），画质有轻微损失；官方 CLI 场景由其自行处理编码
- 视频内容检测（黑屏）默认只抽首相机、限 400 次探针，可通过 config 调整
- `--overrides` 只改变处置动作，不改数据内容（如「补标任务」需在上游完成后重跑质检）
- `lerobot-edit-dataset` 分支按官方 CLI 契约实现但未在本机实测（未安装 lerobot），不可用时自动回退并在报告标注 `method: manual`
- 点云只做几何/颜色/时序质量，不做语义级检查（物体缺失、遮挡、标注错误）
- 固定点数契约下，`drop` 帧用 hold-last-good 占位：能保证训练张量对齐，但占位帧不是真实观测，全部记录在 `pointcloud_cleaning_notes.json`
- 离群半径/工作空间是**整数据集自适应**；多机器人、多场景混采的数据集建议在 config 里用 `workspace.mode=fixed` 明确包围盒
- 时序规则只看质心平移，不检测纯旋转（需要位姿/坐标系参考才能判断）
