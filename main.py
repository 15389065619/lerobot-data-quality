#!/usr/bin/env python3
"""LeRobot v3 数据质量校验与清洗 Pipeline — CLI 入口。

流程：loader -> validator -> policy -> reporter -> cleaner -> pointcloud -> visualizer

每一步都把中间结果落盘到 <output_dir>/intermediate/，任何一步异常中断，
已完成的阶段产物仍然可用（例如质检过了但清洗挂了，报告照样能交出去）。

用法：
    python main.py --dataset_path /path/to/dataset --output_dir outputs
    python main.py --dataset_path /path/to/dataset --output_dir outputs --max_episodes 20
    python main.py --dataset_path /path/to/dataset --output_dir outputs --on-review drop
    python main.py --dataset_path /path/to/dataset --output_dir outputs --overrides outputs/review_queue.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from pipeline import cleaner as cleaner_mod
from pipeline import baseline as baseline_mod
from pipeline import pointcloud as pointcloud_mod
from pipeline import policy as policy_mod
from pipeline import reporter as reporter_mod
from pipeline import visualizer as visualizer_mod
from pipeline.config import cfg_get, load_config
from pipeline.loader import DatasetFormatError, load_dataset
from pipeline.manifest import (
    MANIFEST_FILENAME,
    QualityManifest,
    cleaning_signature,
    config_digest,
)
from pipeline.validator import validate_dataset

LOG = logging.getLogger("pipeline")

BANNER = "=" * 76


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lerobot-data-quality",
        description="LeRobot v3 数据集质检与清洗：加载 -> 质检 -> 处置 -> 报告 -> 清洗 -> 可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset_path", default=None, help="本地 LeRobot v3 数据集根目录（含 meta/info.json）")
    p.add_argument("--output_dir", default="outputs", help="输出根目录（报告 / 清洗结果 / 图表）")
    p.add_argument("--max_episodes", type=int, default=None, help="只处理前 N 个 episode（调试大数据集时用）")

    p.add_argument("--config", default=None,
                   help="质检策略配置文件（YAML/JSON，按扩展名识别；阈值、严重度、机器人覆盖，缺失键自动用代码默认值）")
    p.add_argument("--accept-baseline", nargs="+", metavar="FINGERPRINT", default=None,
                   help="把报告里的误报指纹写入 --config 的 baseline_whitelist；"
                        "可单独使用（不需要 --dataset_path），下次运行自动忽略这些告警")
    p.add_argument("--overrides", default=None, help="人工复核结果 CSV（来自 review_queue.csv）")
    p.add_argument("--on-review", choices=["keep", "drop"], default="keep",
                   help="未经人工复核的 warning 级 episode 如何处理")

    # 阈值快捷覆盖：会切换到固定阈值模式，正式使用建议写在 --config 里
    p.add_argument("--min-frames", type=int, default=None, help="固定模式下的最小帧数（覆盖自适应阈值）")
    p.add_argument("--max-frames", type=int, default=None, help="固定模式下的最大帧数")
    p.add_argument("--action-jump-threshold", type=float, default=None,
                   help="action 跳变的绝对下限（覆盖自适应阈值）")

    p.add_argument("--skip-clean", action="store_true", help="只做质检与报告，不生成清洗后数据集")
    p.add_argument("--skip-pointcloud", action="store_true", help="跳过 3D 点云质检与清洗")
    p.add_argument("--pointcloud-feature", default=None,
                   help="点云特征列名（默认自动识别，如 observation.environment_state）")
    p.add_argument("--pointcloud-max-frames", type=int, default=None,
                   help="只校验前 N 帧点云（调试大数据集时用）")
    p.add_argument("--pointcloud-overrides", default=None,
                   help="点云人工复核结果 CSV（来自 pointcloud_review_queue.csv）")
    p.add_argument("--skip-plots", action="store_true", help="跳过可视化")
    p.add_argument("--no-cli", action="store_true", help="不调用 lerobot-edit-dataset，强制使用内置清洗实现")
    p.add_argument("--plots-sample", type=int, default=20, help="可视化采样 episode 数量")
    p.add_argument("--incremental", action="store_true",
                   help="增量/断点续传：episode 级复用 manifest 中未变更的结论，"
                        "点云帧级复用 pointcloud_scan_cache.json；损坏/新增/改动自动重检")
    p.add_argument("--manifest", default=None,
                   help=f"增量状态文件路径（默认 <output_dir>/{MANIFEST_FILENAME}）")
    p.add_argument("--force", action="store_true", help="忽略增量缓存，强制全量重跑")
    p.add_argument("--streaming", action="store_true",
                   help="流式模式：按 parquet 文件分片扫描，不把整张逐帧表读进内存（大数据集用；暂不支持清洗）")
    p.add_argument("--workers", type=int, default=1,
                   help="多进程并发数：>1 时 episode 质检与点云扫描用 ProcessPoolExecutor 并行"
                        "（worker 为纯函数不写盘，结果由主进程统一归并落盘；1=串行）")
    p.add_argument("--verbose", "-v", action="store_true", help="输出 DEBUG 日志")
    return p


def setup_logging(verbose: bool) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # Windows 控制台默认 cp936，中文/符号会炸
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)-7s %(name)-14s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def step(title: str, index: int, total: int) -> None:
    print()
    print(BANNER)
    print(f"[{index}/{total}] {title}")
    print(BANNER)


def save_intermediate(output_dir: Path, name: str, payload: Any) -> Path:
    path = output_dir / "intermediate" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    LOG.debug("中间结果已保存: %s", path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    # 白名单维护是独立的维护动作，不需要（也不应该）重新跑一遍质检
    if args.accept_baseline:
        return _accept_baseline(args)
    if not args.dataset_path:
        parser.error("--dataset_path 是必填参数（使用 --accept-baseline 维护白名单时除外）")

    dataset_path = Path(args.dataset_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if args.streaming and not args.skip_clean:
        LOG.warning("--streaming 模式暂不支持清洗（需要整表重写），本次自动跳过 cleaner；"
                    "如需清洗请去掉 --streaming")
        args.skip_clean = True
    if args.streaming and args.workers > 1:
        LOG.info("--streaming 模式下 episode 质检走流式单进程，--workers 仅作用于点云阶段")

    print(BANNER)
    print("LeRobot v3 数据质量校验与清洗 Pipeline")
    print(f"数据集: {dataset_path}")
    print(f"输出  : {output_dir.resolve()}")
    print(BANNER)

    total_steps = 8

    # ----------------------------------------------------- 0. 增量状态
    manifest_path = Path(args.manifest).expanduser() if args.manifest else output_dir / MANIFEST_FILENAME
    manifest = QualityManifest.load(manifest_path)
    incremental = args.incremental and not args.force
    if args.incremental and args.max_episodes is not None:
        LOG.warning("--max_episodes 与增量模式不兼容，本次按全量处理")
        incremental = False

    if incremental and manifest.has_results:
        try:
            scan = load_dataset(dataset_path, load_frames=False)
            scan_config = _build_config(args, scan.info.get("robot_type"))
            if _manifest_fresh(manifest, scan, scan_config, output_dir,
                               require_cleaning=not args.skip_clean):
                print(BANNER)
                print("增量：未检测到数据或配置变化，复用上次结果（--force 可强制重跑）")
                print(f"  质检报告    : {output_dir / 'quality_report.json'}")
                print(f"  复核队列    : {output_dir / policy_mod.REVIEW_QUEUE_FILENAME}")
                print(f"  增量状态    : {manifest_path}")
                print(BANNER)
                return 0
        except Exception as exc:  # noqa: BLE001 - 增量判断失败不应阻断全量流程
            LOG.warning("增量预检失败，按全量处理: %s", exc)

    # ---------------------------------------------------------- 1. loader
    step("加载数据集（loader）", 1, total_steps)
    try:
        dataset = load_dataset(dataset_path, max_episodes=args.max_episodes,
                               load_frames=not args.streaming)
    except DatasetFormatError as exc:
        LOG.error("数据集格式错误: %s", exc)
        _write_failure(output_dir, "loader", exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        LOG.error("加载失败: %s\n%s", exc, traceback.format_exc())
        _write_failure(output_dir, "loader", exc)
        return 2

    summary = dataset.summary()
    save_intermediate(output_dir, "01_dataset_summary.json", summary)
    print(f"  episode: {summary['num_episodes']}   帧数: {summary['num_frames']}   "
          f"tasks: {summary['total_tasks']}   相机: {len(summary['video_features'])}")
    print(f"  state 维度: {summary['num_state_dims']}   action 维度: {summary['num_action_dims']}")

    # ------------------------------------------------------- 2. validator
    step("逐 episode 质检（validator）", 2, total_steps)
    try:
        config = _build_config(args, dataset.info.get("robot_type"))
        prior = None
        only: set[int] | None = None
        if incremental and manifest.has_results and manifest.matches_root(dataset):
            unchanged = manifest.unchanged_episodes(dataset, config)
            all_eps = set(dataset.episode_indices)
            changed = all_eps - unchanged
            if 0 < len(changed) < len(all_eps):
                prior = manifest.cached_results()
                only = changed
                LOG.info("增量模式：%d 个 episode 变更，%d 个复用历史结论",
                         len(changed), len(all_eps) - len(changed))

        validation = validate_dataset(dataset, config, prior=prior, only=only,
                                      streaming=args.streaming, workers=args.workers)
        if prior is not None and not _thresholds_equal(
            validation.resolved_thresholds, manifest.resolved_thresholds()
        ):
            LOG.info("阈值随数据集变化而改变，增量缓存失效，回退全量重检")
            validation = validate_dataset(dataset, config, streaming=args.streaming,
                                          workers=args.workers)
    except Exception as exc:  # noqa: BLE001
        LOG.error("质检失败: %s\n%s", exc, traceback.format_exc())
        _write_failure(output_dir, "validator", exc)
        return 3

    # 基线白名单：已确认的已知误报在这里摘除，不参与处置决策与健康度计算
    baseline_mod.apply_whitelist(validation, dataset_path.name, config)
    if validation.baseline_ignored:
        save_intermediate(output_dir, "02b_baseline_ignored.json", validation.baseline_ignored)

    validation_dict = validation.to_dict()
    save_intermediate(output_dir, "02_validation_result.json", validation_dict)
    _print_thresholds(validation)
    sev = validation.severity_summary()
    print(f"  正常 {validation.valid_episodes} / 异常 {len(validation.invalid_episodes)}"
          f"（error {sev.get('error', 0)} / warning {sev.get('warning', 0)} / info {sev.get('info', 0)}）")
    if validation.baseline_ignored:
        ignored_eps = {item["episode_index"] for item in validation.baseline_ignored}
        print(f"  基线白名单: 已忽略 {len(validation.baseline_ignored)} 条已知告警"
              f"（{len(ignored_eps)} 个 episode，不计入健康度）")
    for code, count in validation.reason_frequency().items():
        if count:
            print(f"    - {code:<26} {count} 个 episode")
    for issue in validation.dataset_issues:
        LOG.warning("数据集级问题 [%s] %s", issue.code, issue.label)

    # ---------------------------------------------------------- 3. policy
    step("处置决策（policy：保留/修复/复核/剔除）", 3, total_steps)
    try:
        overrides = policy_mod.read_overrides(args.overrides) if args.overrides else {}
        decisions = policy_mod.decide(validation, on_review=args.on_review, overrides=overrides)
    except Exception as exc:  # noqa: BLE001
        LOG.error("处置决策失败: %s\n%s", exc, traceback.format_exc())
        _write_failure(output_dir, "policy", exc)
        return 4

    save_intermediate(output_dir, "03_dispositions.json",
                      {str(ep): d.to_dict() for ep, d in sorted(decisions.items())})
    # --incremental --force：强制全量重跑后仍要写回 manifest，否则旧格式（缺
    # severity_overrides）的缓存永远无法刷新；--max_episodes 只是子集，禁止落盘。
    store_manifest = args.incremental and args.max_episodes is None
    signatures = manifest.episode_signatures(dataset) if store_manifest else {}
    if store_manifest:
        manifest.store_results(dataset, config, validation, decisions, signatures)
        manifest.save()
    counts = policy_mod.summarize(decisions)["counts"]
    print(f"  保留 {counts['keep']}   修复 {counts['repair']}   "
          f"复核 {counts['review']}   剔除 {counts['drop']}")
    for ep, d in sorted(decisions.items()):
        if d.disposition != "keep":
            print(f"    ep {ep:>4}  {d.severity:<8} {d.disposition:<7} {d.recommended_action}")

    # -------------------------------------------------------- 4. reporter
    step("生成质检报告（reporter）", 4, total_steps)
    report_path: Path | None = None
    review_path: Path | None = None
    try:
        report = reporter_mod.build_report(dataset, validation, decisions, dataset_path=dataset_path)
        report_path = reporter_mod.write_report(report, output_dir)
        reporter_mod.print_summary(report)
        review_path = policy_mod.write_review_queue(
            output_dir / policy_mod.REVIEW_QUEUE_FILENAME, validation, decisions, dataset
        )
        print(f"人工复核队列: {review_path}  （填 reviewer_disposition 后可用 --overrides 回灌）")
    except Exception as exc:  # noqa: BLE001
        LOG.error("报告生成失败: %s\n%s", exc, traceback.format_exc())
        _write_failure(output_dir, "reporter", exc)
        return 5

    # --------------------------------------------------------- 5. cleaner
    cleaning_dict: dict[str, Any] | None = None
    cleaning_sig = cleaning_signature(decisions, signatures) if (store_manifest and not args.skip_clean) else None
    reused_cleaning = manifest.reusable_cleaning(cleaning_sig) if cleaning_sig else None
    if args.skip_clean:
        step("清洗（cleaner）— 已跳过", 5, total_steps)
    elif reused_cleaning is not None:
        step("清洗（cleaner）— 复用上次产出", 5, total_steps)
        cleaning_dict = reused_cleaning
        save_intermediate(output_dir, "04_cleaning_result.json", cleaning_dict)
        print(f"  清洗计划未变化，复用: {cleaning_dict.get('output_path')}")
        if report_path is not None:
            reporter_mod.attach_cleaning_result(report_path, cleaning_dict)
    else:
        step("执行清洗（cleaner：删除 + 修复）", 5, total_steps)
        try:
            cleaning = cleaner_mod.clean_dataset(
                dataset, decisions, output_dir, prefer_cli=not args.no_cli,
                dataset_issues=validation.dataset_issues,
            )
            cleaning_dict = cleaning.to_dict()
            save_intermediate(output_dir, "04_cleaning_result.json", cleaning_dict)
            print(f"  方式: {cleaning.method}    删除 {len(cleaning.removed_episodes)} 个 / "
                  f"修复 {len(cleaning.repaired_episodes)} 个    保留 {cleaning.kept_episodes} 个")
            if cleaning.output_root:
                print(f"  输出: {cleaning.output_root}")
            for warning in cleaning.warnings:
                LOG.warning("清洗警告: %s", warning)
            if report_path is not None:
                reporter_mod.attach_cleaning_result(report_path, cleaning_dict)
            if cleaning_sig:
                manifest.store_cleaning(cleaning_sig, cleaning_dict)
                manifest.save()
        except Exception as exc:  # noqa: BLE001 - 清洗失败不应丢掉报告
            LOG.error("清洗失败: %s\n%s", exc, traceback.format_exc())
            _write_failure(output_dir, "cleaner", exc, keep_going=True)
            print("  ! 清洗阶段失败，但 quality_report.json 仍然保留，可单独重跑 cleaner。")

    # ------------------------------------------- 6. pointcloud validation
    pc_validation = None
    pc_decisions: dict[Any, Any] = {}
    pc_clean_dict: dict[str, Any] | None = None
    pc_report_path: Path | None = None
    pc_review_path: Path | None = None
    pc_feature = args.pointcloud_feature or cfg_get(config, "pointcloud.feature_key")
    # 点云阶段作用在「行已稳定」的清洗产物上；没有清洗产物时退回原数据集（只读，独立输出）
    pc_source_root = (Path(cleaning_dict["output_path"])
                      if cleaning_dict and cleaning_dict.get("output_path") else dataset_path)
    pc_patch_in_place = bool(cleaning_dict and cleaning_dict.get("output_path"))

    if args.skip_pointcloud:
        step("3D 点云质检（pointcloud）— 已跳过", 6, total_steps)
    else:
        step("3D 点云质检（pointcloud：几何 / 颜色 / 时序 8 条规则）", 6, total_steps)
        try:
            pc_validation = pointcloud_mod.validate_pointcloud_dataset(
                pc_source_root, config, feature_key=pc_feature,
                max_frames=args.pointcloud_max_frames, workers=args.workers,
                # 断点续传：增量模式下复用文件指纹未变的逐帧扫描结论
                cache_path=(output_dir / "pointcloud_scan_cache.json") if args.incremental else None,
            )
            pc_validation_dict = pc_validation.to_dict()
            save_intermediate(output_dir, "05_pointcloud_validation.json", pc_validation_dict)
            _print_pointcloud_thresholds(pc_validation)

            pc_overrides = (pointcloud_mod.read_overrides(args.pointcloud_overrides)
                            if args.pointcloud_overrides else {})
            pc_decisions = pointcloud_mod.decide(
                pc_validation, on_review=args.on_review, overrides=pc_overrides
            )
            save_intermediate(output_dir, "06_pointcloud_dispositions.json",
                              {f"{e}:{f}": d.to_dict() for (e, f), d in sorted(pc_decisions.items())})

            pc_report = pointcloud_mod.build_report(pc_validation, pc_decisions,
                                                    dataset_path=pc_source_root)
            pc_report_path = pointcloud_mod.write_report(pc_report, output_dir)
            pointcloud_mod.print_summary(pc_report)
            pc_review_path = pointcloud_mod.write_review_queue(
                output_dir / pointcloud_mod.REVIEW_FILENAME, pc_validation, pc_decisions
            )
            print(f"点云复核队列: {pc_review_path}  （填 reviewer_disposition 后可用 --pointcloud-overrides 回灌）")
        except pointcloud_mod.PointCloudFormatError as exc:
            LOG.info("未检测到可用的点云特征，跳过点云阶段：%s", exc)
            print("  未检测到二维点云特征（shape=[N,D], D≥3），跳过点云质检与清洗。")
            pc_validation = None
        except Exception as exc:  # noqa: BLE001 - 点云失败不应中断数值质检链路
            LOG.error("点云质检失败: %s\n%s", exc, traceback.format_exc())
            _write_failure(output_dir, "pointcloud_validator", exc, keep_going=True)
            print("  ! 点云质检失败，主链路继续。")
            pc_validation = None

    # ----------------------------------------------- 7. pointcloud cleaning
    if pc_validation is None:
        step("3D 点云清洗（pointcloud）— 已跳过", 7, total_steps)
    else:
        step("3D 点云清洗（pointcloud：只改点、不改行）", 7, total_steps)
        try:
            pc_cleaning = pointcloud_mod.clean_pointclouds(
                pc_source_root, pc_validation, pc_decisions, output_dir,
                feature_key=pc_validation.feature_key,
                target_root=str(pc_source_root) if pc_patch_in_place else None,
            )
            pc_clean_dict = pc_cleaning.to_dict()
            save_intermediate(output_dir, "07_pointcloud_cleaning.json", pc_clean_dict)
            print(f"  方式: {pc_cleaning.method}    修复 {len(pc_cleaning.repaired_frames)} 帧 / "
                  f"占位 {len(pc_cleaning.held_frames)} 帧 / 保留 {pc_cleaning.kept_frames} 帧")
            print(f"  点数: 移除 {pc_cleaning.removed_points} / 补回 {pc_cleaning.added_points}"
                  f"    stats 更新: {pc_cleaning.stats_updated}")
            print(f"  输出: {pc_cleaning.output_root}")
            for warning in pc_cleaning.warnings[:5]:
                LOG.warning("点云清洗警告: %s", warning)
            if not pc_cleaning.verification.get("ok", False):
                LOG.warning("点云清洗校验未通过: %s", pc_cleaning.verification)

            # 用清洗后的真实结果重建点云报告
            pc_report = pointcloud_mod.build_report(pc_validation, pc_decisions, pc_cleaning,
                                                    dataset_path=pc_source_root)
            pc_report_path = pointcloud_mod.write_report(pc_report, output_dir)
        except Exception as exc:  # noqa: BLE001 - 点云清洗失败保留报告
            LOG.error("点云清洗失败: %s\n%s", exc, traceback.format_exc())
            _write_failure(output_dir, "pointcloud_cleaner", exc, keep_going=True)
            print("  ! 点云清洗失败，但 pointcloud_report.json 仍然保留。")

    # ------------------------------------------------------ 8. visualizer
    if args.skip_plots:
        step("可视化（visualizer）— 已跳过", 8, total_steps)
        plots: dict[str, Path] = {}
    else:
        step("生成可视化（visualizer）", 8, total_steps)
        try:
            plots = visualizer_mod.generate_plots(
                dataset, validation, output_dir, cleaning=cleaning_dict, sample_size=args.plots_sample
            )
            for name, path in plots.items():
                print(f"  {name:<24} {path}")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("可视化失败（不影响主流程）: %s", exc)
            if args.verbose:
                LOG.debug(traceback.format_exc())
            plots = {}

    if report_path is not None and (plots or review_path):
        artifacts = {name: str(p) for name, p in plots.items()}
        if review_path is not None:
            artifacts["review_queue"] = str(review_path)
        if cleaning_dict and cleaning_dict.get("output_path"):
            artifacts["cleaned_dataset"] = cleaning_dict["output_path"]
        if pc_report_path is not None:
            artifacts["pointcloud_report"] = str(pc_report_path)
        if pc_review_path is not None:
            artifacts["pointcloud_review_queue"] = str(pc_review_path)
        if pc_clean_dict and pc_clean_dict.get("output_path"):
            artifacts["pointcloud_cleaned_dataset"] = str(pc_clean_dict["output_path"])
        reporter_mod.attach_artifacts(report_path, artifacts)

    # ------------------------------------------------------------ 收尾
    elapsed = time.time() - started
    print()
    print(BANNER)
    print(f"完成，用时 {elapsed:.1f}s")
    print(f"  质检报告    : {report_path}")
    print(f"  复核队列    : {review_path}")
    if cleaning_dict:
        print(f"  清洗后数据  : {cleaning_dict.get('output_path')}")
    if pc_report_path is not None:
        print(f"  点云报告    : {pc_report_path}")
    if pc_clean_dict:
        print(f"  点云清洗后  : {pc_clean_dict.get('output_path')}")
    if plots:
        print(f"  可视化      : {output_dir / 'plots'}")
    print(f"  中间结果    : {output_dir / 'intermediate'}")
    if validation.invalid_episodes:
        print(f"  结论        : 剔除 {counts['drop']} 个 / 修复 {counts['repair']} 个 / "
              f"待复核 {counts['review']} 个，预计丢弃 {report['summary']['drop_frames']} 帧")
    else:
        print("  结论        : 未发现结构性脏数据，可直接进入训练")
    print(BANNER)
    return 0


def _manifest_fresh(
    manifest: QualityManifest,
    scan,
    config: dict[str, Any],
    output_dir: Path,
    *,
    require_cleaning: bool,
) -> bool:
    """增量快路径：数据和配置都没变，且上次产物仍在，才允许直接返回。"""
    if manifest.payload.get("config_digest") != config_digest(config):
        return False
    if not (output_dir / "quality_report.json").is_file():
        return False
    unchanged = manifest.unchanged_episodes(scan, config)
    if not unchanged or unchanged != set(scan.episode_indices):
        return False
    if require_cleaning:
        cleaning = (manifest.payload.get("cleaning") or {}).get("result") or {}
        output = cleaning.get("output_path")
        if not output or not Path(str(output)).is_dir():
            return False
        if (cleaning.get("verification") or {}).get("ok") is False:
            return False
    return True


def _thresholds_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """阈值解析结果比较；JSON 归一化后比较，避免 list/dict 顺序与类型差异。"""
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def _accept_baseline(args: argparse.Namespace) -> int:
    """把误报指纹写回配置文件（`--accept-baseline` 的独立维护模式）。"""
    if not args.config:
        print("--accept-baseline 需要配合 --config 指定要写入的配置文件（YAML/JSON）")
        return 2
    try:
        added, existed = baseline_mod.append_to_config(args.config, args.accept_baseline)
    except (FileNotFoundError, ValueError) as exc:
        print(f"写入基线白名单失败：{exc}")
        return 2
    path = Path(args.config).expanduser()
    if added:
        print(f"已写入 {len(added)} 个基线指纹 -> {path}")
        for fp in added:
            print(f"  + {fp}")
    else:
        print(f"没有新增指纹（{len(existed)} 个已在白名单中）-> {path}")
    print("重新运行质检命令即可生效：命中的告警会标记 ignored_by_baseline，且不再影响健康度。")
    return 0


def _build_config(args: argparse.Namespace, robot_type: str | None) -> dict[str, Any]:
    """配置优先级：默认值 < 配置文件 < robot_type 覆盖 < CLI 快捷覆盖。"""
    overrides: dict[str, Any] = {}
    if args.min_frames is not None or args.max_frames is not None:
        overrides["frame_count"] = {
            "mode": "fixed",
            "floor": args.min_frames if args.min_frames is not None else 8,
            "ceiling": args.max_frames if args.max_frames is not None else 20000,
        }
    if args.action_jump_threshold is not None:
        overrides["action_jump"] = {"abs_floor": args.action_jump_threshold}
    return load_config(args.config, robot_type=robot_type, overrides=overrides)


def _print_thresholds(validation) -> None:
    t = validation.resolved_thresholds or {}
    frame = t.get("frame_count") or {}
    mode_label = {"fixed": "固定", "manual": "人工指定", "auto": "自适应"}.get(
        frame.get("mode"), "自适应"
    )
    print(f"  阈值({mode_label}) : 帧数区间 [{frame.get('min')}, {frame.get('max')}] "
          f"(median={frame.get('median')}, mad={frame.get('mad')}, mode={frame.get('mode')})")
    jump = t.get("action_jump") or {}
    per_dim = jump.get("per_dim") or []
    if per_dim:
        print(f"  action 跳变 : 逐维阈值 {[round(float(x), 2) for x in per_dim]}")
    if t.get("video_frame_tolerance") is not None:
        print(f"  视频帧容差 : ±{t['video_frame_tolerance']} 帧")


def _print_pointcloud_thresholds(validation) -> None:
    r = validation.resolved or {}
    key = validation.feature_key
    n = validation.expected_points or "变长"
    print(f"  点云特征    : {key}  ({n} 点 × {validation.dim} 维)")
    print(f"  自适应阈值  : 场景尺度 {r.get('scene_extent')} m   离群半径 {r.get('outlier_radius')} m   "
          f"最少有效点 {r.get('min_points')}  颜色量程 {r.get('color_scale'):g}")
    if r.get("workspace_lower"):
        print(f"  工作空间    : {r.get('workspace_mode')} 包围盒 "
              f"{[round(float(x), 2) for x in r['workspace_lower']]} ~ "
              f"{[round(float(x), 2) for x in r['workspace_upper']]}")


def _write_failure(output_dir: Path, stage: str, exc: BaseException, keep_going: bool = False) -> None:
    payload = {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "recoverable": keep_going,
    }
    try:
        save_intermediate(output_dir, f"failure_{stage}.json", payload)
    except OSError:
        LOG.debug("写入失败信息失败")


if __name__ == "__main__":
    sys.exit(main())
