#!/usr/bin/env python3
"""generate_report.py — 把质检流水线的产物渲染成单文件 HTML 可视化大屏。

读取的输出文件（全部来自某次 main.py 运行的 --output_dir）：

1. quality_manifest.json          —— 增量状态文件，含每个 episode 的 decision
                                      （keep/repair/review/drop + reasons + severity）。
2. pointcloud_review_queue.csv    —— 点云复核队列，含坏帧的 episode/frame/规则/metrics。
3. quality_report.json            ——（可选）主质检报告，用于顶部大盘的真实总帧数等。
4. pointcloud_report.json         ——（可选）点云质检报告，用于点云帧健康环图。

用法：
    python generate_report.py --output_dir outputs_fault
    python generate_report.py --output_dir outputs_fault --report my_report.html --top 30

设计说明：
- 纯标准库（json/csv/argparse），不依赖后端与 Jinja2；HTML 用模板字符串注入数据。
- 图表用 ECharts（jsdelivr CDN），无网络时图表区域会显示占位文字但表格仍可读。
- 深色科技风：深蓝黑底 + 霓虹青/品红点缀 + 毛玻璃卡片 + 等宽数字。
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

LOG = logging.getLogger("generate_report")

# ---------------------------------------------------------------------------
# 处置档位 -> 展示颜色（与流水线 policy/pointcloud 的四档一一对应）
# ---------------------------------------------------------------------------
DISPOSITION_COLORS = {
    "keep": "#22d3a7",    # 绿：保留
    "repair": "#38bdf8",  # 蓝：已自动修复
    "review": "#fbbf24",  # 黄：待人工复核
    "drop": "#f43f5e",    # 红：剔除
}
DISPOSITION_LABELS = {
    "keep": "保留 keep",
    "repair": "修复 repair",
    "review": "复核 review",
    "drop": "剔除 drop",
}
SEVERITY_COLORS = {"error": "#f43f5e", "warning": "#fbbf24", "info": "#38bdf8", "ok": "#22d3a7"}


# ---------------------------------------------------------------------------
# 数据加载：每一步都容错（缺文件/字段缺失时不让整个报告崩掉）
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict[str, Any]:
    """读 JSON；文件不存在或损坏时返回空 dict 并告警。"""
    if not path.is_file():
        LOG.warning("缺少文件 %s，对应区块将显示为「无数据」", path.name)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("读取 %s 失败: %s", path.name, exc)
        return {}


def load_manifest(output_dir: Path) -> dict[str, Any]:
    """quality_manifest.json：增量模式才会生成；非增量运行时可能没有。"""
    return _load_json(output_dir / "quality_manifest.json")


def load_review_queue(output_dir: Path) -> list[dict[str, str]]:
    """pointcloud_review_queue.csv -> 行字典列表。

    CSV 列：episode_index, frame_index, severity, reasons,
             suggested_disposition, suggested_action, metrics(JSON 字符串), ...
    """
    path = output_dir / "pointcloud_review_queue.csv"
    if not path.is_file():
        LOG.warning("缺少文件 pointcloud_review_queue.csv，坏帧明细表为空")
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def load_quality_report(output_dir: Path) -> dict[str, Any]:
    return _load_json(output_dir / "quality_report.json")


def load_pointcloud_report(output_dir: Path) -> dict[str, Any]:
    return _load_json(output_dir / "pointcloud_report.json")


# ---------------------------------------------------------------------------
# 数据聚合：把原始 JSON/CSV 变成「图表直接可用」的结构
# ---------------------------------------------------------------------------
def aggregate(manifest: dict[str, Any], quality: dict[str, Any],
              pointcloud: dict[str, Any], queue_rows: list[dict[str, str]]) -> dict[str, Any]:
    """统一汇总出渲染层需要的所有数据。"""

    # ---- episode 级处置分布：优先 manifest（增量），退回 quality_report 的 summary ----
    ep_dispositions: Counter[str] = Counter()
    ep_reasons: Counter[str] = Counter()
    episodes = manifest.get("episodes") or {}
    for entry in episodes.values():
        decision = (entry or {}).get("decision") or {}
        disp = str(decision.get("disposition") or "keep")
        ep_dispositions[disp] += 1
        for r in decision.get("reasons") or []:
            ep_reasons[str(r)] += 1

    summary = quality.get("summary") or {}
    if not ep_dispositions and summary:
        # 没有 manifest 时，用主报告里的处置汇总兜底
        for disp, n in (summary.get("disposition") or {}).items():
            ep_dispositions[str(disp)] += int(n or 0)
        for code, n in (quality.get("reason_frequency") or {}).items():
            ep_reasons[str(code)] += int(n or 0)

    # ---- 点云帧级分布与规则频次 ----
    pc_summary = pointcloud.get("summary") or {}
    pc_dispositions: Counter[str] = Counter()
    pc_reasons: Counter[str] = Counter()
    for disp, n in (pc_summary.get("disposition") or {}).items():
        pc_dispositions[str(disp)] += int(n or 0)
    for code, n in (pc_summary.get("reason_frequency") or pointcloud.get("reason_frequency") or {}).items():
        pc_reasons[str(code)] += int(n or 0)

    # ---- 规则排行榜：episode 规则 + 点云规则合并 ----
    rule_board = sorted(
        {**{f"[episode] {k}": v for k, v in ep_reasons.items()},
         **{f"[点云] {k}": v for k, v in pc_reasons.items()}}.items(),
        key=lambda kv: kv[1], reverse=True,
    )

    # ---- 顶部大盘 KPI ----
    dataset_info = quality.get("dataset") or pointcloud.get("dataset") or {}
    kpis = {
        "dataset": dataset_info.get("root") or manifest.get("dataset_root") or "-",
        "robot_type": dataset_info.get("robot_type") or "-",
        "fps": dataset_info.get("fps") or "-",
        "total_episodes": summary.get("total_episodes") or len(episodes) or "-",
        "total_frames": summary.get("total_frames")
            or pc_summary.get("total_frames") or "-",
        "invalid_episodes": summary.get("invalid_episodes", "-"),
        "pc_bad_frames": sum(v for k, v in pc_dispositions.items() if k != "keep") or "-",
        "health": (summary.get("health_level")
                   or pc_summary.get("health_level") or "unknown"),
    }

    return {
        "kpis": kpis,
        "ep_dispositions": dict(ep_dispositions),
        "pc_dispositions": dict(pc_dispositions),
        "rule_board": rule_board,
        "generated_at": quality.get("generated_at") or pointcloud.get("generated_at") or "",
    }


# ---------------------------------------------------------------------------
# 坏帧明细表：CSV 前 N 行 + metrics JSON 的关键指标展开
# ---------------------------------------------------------------------------
def build_frame_rows(queue_rows: list[dict[str, str]], top: int) -> list[dict[str, str]]:
    """把 CSV 行转成模板友好的结构，并解析 metrics 里的关键数值。"""
    rows: list[dict[str, str]] = []
    for raw in queue_rows[:top]:
        # metrics 列是 JSON 字符串，解析出最有说服力的几个指标
        metrics: dict[str, Any] = {}
        try:
            metrics = json.loads(raw.get("metrics") or "{}")
        except json.JSONDecodeError:
            pass
        metric_bits = []
        for key, label in (("point_count", "点数"), ("non_finite_ratio", "NaN率"),
                           ("duplicate_ratio", "重复率"), ("outlier_ratio", "离群率"),
                           ("outside_ratio", "越界率"), ("color_bad_ratio", "坏色率")):
            if key in metrics:
                val = metrics[key]
                if isinstance(val, float):
                    metric_bits.append(f"{label} {val:.3f}")
                else:
                    metric_bits.append(f"{label} {val}")
        rows.append({
            "episode": raw.get("episode_index", "-"),
            "frame": raw.get("frame_index", "-"),
            "severity": raw.get("severity", "-"),
            "reasons": raw.get("reasons", "-"),
            "action": raw.get("suggested_action", "-"),
            "metrics": "，".join(metric_bits) or "-",
        })
    return rows


# ---------------------------------------------------------------------------
# HTML 模板：深色监控大屏。数据通过 __DATA_JSON__ 占位符注入。
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeRobot 数据集质检大屏</title>
<!-- ECharts CDN：无网络时图表显示占位文字，表格不受影响 -->
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
  :root {
    --bg: #070b16;            /* 页面底色：接近黑的深蓝 */
    --card: rgba(17, 25, 46, 0.72);   /* 卡片毛玻璃底 */
    --border: rgba(56, 189, 248, 0.18);/* 卡片描边：淡青色 */
    --text: #dbe7ff;          /* 主文字 */
    --muted: #7c8db5;         /* 次要文字 */
    --cyan: #38bdf8;          /* 科技青 */
    --green: #22d3a7; --blue: #38bdf8; --yellow: #fbbf24; --red: #f43f5e;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: radial-gradient(1200px 600px at 80% -10%, rgba(56,189,248,.12), transparent),
                radial-gradient(900px 500px at -10% 110%, rgba(244,63,94,.08), transparent),
                var(--bg);
    color: var(--text);
    font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
    padding: 28px 36px 48px;
  }
  /* 顶部标题栏：左侧标题 + 右侧生成时间 */
  header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 22px; }
  header h1 { font-size: 26px; letter-spacing: 2px; }
  header h1 .accent { color: var(--cyan); }
  header .meta { color: var(--muted); font-size: 12px; text-align: right; line-height: 1.7; }

  /* 通用卡片 */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    backdrop-filter: blur(6px);
    box-shadow: 0 8px 28px rgba(0,0,0,.35);
  }
  .card h2 { font-size: 14px; color: var(--muted); font-weight: 600; letter-spacing: 1px; margin-bottom: 12px; }

  /* KPI 大盘：一行六张卡 */
  .kpis { display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin-bottom: 18px; }
  .kpi { text-align: left; }
  .kpi .label { font-size: 12px; color: var(--muted); margin-bottom: 8px; }
  .kpi .value { font-size: 30px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .kpi .sub { font-size: 11px; color: var(--muted); margin-top: 6px; }

  /* 图表区：左环图 + 中点云环图 + 右排行榜 */
  .charts { display: grid; grid-template-columns: 1fr 1fr 1.4fr; gap: 14px; margin-bottom: 18px; }
  .chart { height: 300px; }
  .chart-fallback { color: var(--muted); font-size: 13px; padding: 40px 0; text-align: center; }

  /* 坏帧明细表 */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: var(--muted); font-weight: 600; padding: 10px 12px;
       border-bottom: 1px solid var(--border); letter-spacing: 1px; }
  td { padding: 10px 12px; border-bottom: 1px solid rgba(56,189,248,.07); }
  tr:hover td { background: rgba(56,189,248,.06); }
  .tag { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px;
         border: 1px solid currentColor; }
  .mono { font-family: Consolas, "Courier New", monospace; }
  .legend { display:flex; gap:18px; flex-wrap:wrap; margin-top:10px; font-size:12px; color:var(--muted); }
  .legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }
  footer { margin-top: 26px; color: var(--muted); font-size: 12px; text-align: center; }
</style>
</head>
<body>

<header>
  <h1>LeRobot 数据集<span class="accent">质检大屏</span></h1>
  <div class="meta">
    数据集：__DATASET__<br>
    机器人类型：__ROBOT__ ｜ FPS：__FPS__<br>
    生成时间：__GENERATED_AT__
  </div>
</header>

<!-- ============================ 顶部 KPI 大盘 ============================ -->
<section class="kpis">
  <div class="card kpi"><div class="label">Episode 总数</div>
    <div class="value">__KPI_EPISODES__</div><div class="sub">来自 manifest / quality_report</div></div>
  <div class="card kpi"><div class="label">总帧数</div>
    <div class="value">__KPI_FRAMES__</div><div class="sub">逐帧质检覆盖量</div></div>
  <div class="card kpi"><div class="label">异常 Episode</div>
    <div class="value" style="color:var(--red)">__KPI_BAD_EPS__</div><div class="sub">帧数 / 时间轴 / 视频 等问题</div></div>
  <div class="card kpi"><div class="label">异常点云帧</div>
    <div class="value" style="color:var(--yellow)">__KPI_BAD_PC__</div><div class="sub">NaN / 离群 / 重复 / 越界 等</div></div>
  <div class="card kpi"><div class="label">待复核帧</div>
    <div class="value" style="color:var(--yellow)">__KPI_REVIEW__</div><div class="sub">pointcloud_review_queue.csv 行数</div></div>
  <div class="card kpi"><div class="label">健康度</div>
    <div class="value" style="font-size:22px;color:var(--cyan)">__KPI_HEALTH__</div><div class="sub">流水线综合评级</div></div>
</section>

<!-- ============================ 图表区 ============================ -->
<section class="charts">
  <div class="card"><h2>EPISODE 健康占比（keep / repair / review / drop）</h2>
    <div id="chartEp" class="chart"></div></div>
  <div class="card"><h2>点云帧健康占比</h2>
    <div id="chartPc" class="chart"></div></div>
  <div class="card"><h2>异常规则排行榜（episode 级 + 点云帧级）</h2>
    <div id="chartRules" class="chart"></div></div>
</section>

<!-- ============================ 坏帧明细表 ============================ -->
<section class="card">
  <h2>坏帧明细（pointcloud_review_queue.csv 前 __TOP__ 条，需人工复核）</h2>
  <table>
    <thead><tr>
      <th>Episode</th><th>帧号</th><th>严重度</th><th>触发规则</th><th>建议动作</th><th>关键指标</th>
    </tr></thead>
    <tbody>__TABLE_ROWS__</tbody>
  </table>
  <div class="legend">__LEGEND__</div>
</section>

<footer>由 generate_report.py 渲染 ｜ 数据源：quality_manifest.json + pointcloud_review_queue.csv（+ 可选 quality/pointcloud 报告）</footer>

<script>
// 由 Python 注入的聚合数据
const DATA = __DATA_JSON__;

// 处置档位的统一配色（与 Python 端 DISPOSITION_COLORS 保持一致）
const DISP_COLORS = {"keep":"#22d3a7","repair":"#38bdf8","review":"#fbbf24","drop":"#f43f5e"};
const DISP_LABELS = {"keep":"保留","repair":"修复","review":"复核","drop":"剔除"};

// CDN 加载失败时的优雅降级：显示占位文字而不是整页报错
function renderOrFallback(id, option) {
  const el = document.getElementById(id);
  if (!window.echarts) { el.innerHTML = '<div class="chart-fallback">ECharts CDN 未加载（离线环境），图表不可用</div>'; return; }
  const chart = echarts.init(el, null, {renderer: "canvas"});
  chart.setOption(option);
  window.addEventListener("resize", () => chart.resize());
}

// ---------- 通用深色饼/环图配置 ----------
function doughnutOption(title, dataMap) {
  const entries = Object.entries(dataMap).filter(([,v]) => v > 0);
  return {
    backgroundColor: "transparent",
    tooltip: {trigger: "item", backgroundColor: "#0f1830", borderColor: "#38bdf8",
              textStyle: {color: "#dbe7ff"}, formatter: "{b}: {c} ({d}%)"},
    legend: {bottom: 0, textStyle: {color: "#7c8db5", fontSize: 11}},
    series: [{
      type: "pie", radius: ["48%", "72%"], center: ["50%", "44%"],
      itemStyle: {borderColor: "#070b16", borderWidth: 3},
      label: {color: "#dbe7ff", fontSize: 12, formatter: "{b}\\n{c} 条"},
      data: entries.map(([k, v]) => ({name: DISP_LABELS[k] || k, value: v,
                                      itemStyle: {color: DISP_COLORS[k] || "#38bdf8"}})),
    }],
    title: entries.length ? undefined :
      {text: "无异常，全部 keep", left: "center", top: "middle",
       textStyle: {color: "#22d3a7", fontSize: 14}},
  };
}

// ---------- 规则排行榜：横向条形图 ----------
function rulesOption(board) {
  const items = board.slice(0, 12);           // 最多展示 12 条，防止图表过挤
  const names = items.map(kv => kv[0]);
  const values = items.map(kv => kv[1]);
  return {
    backgroundColor: "transparent",
    grid: {left: 10, right: 30, top: 10, bottom: 10, containLabel: true},
    tooltip: {trigger: "axis", backgroundColor: "#0f1830", borderColor: "#38bdf8",
              textStyle: {color: "#dbe7ff"}},
    xAxis: {type: "value", axisLabel: {color: "#7c8db5"},
            splitLine: {lineStyle: {color: "rgba(56,189,248,.08)"}}},
    yAxis: {type: "category", inverse: true, data: names,
            axisLabel: {color: "#dbe7ff", fontSize: 11, fontFamily: "Consolas, monospace"}},
    series: [{
      type: "bar", data: values, barWidth: 14,
      itemStyle: {borderRadius: [0, 7, 7, 0],
        color: {type: "linear", x: 0, y: 0, x2: 1, y2: 0,
                colorStops: [{offset: 0, color: "#38bdf8"}, {offset: 1, color: "#f43f5e"}]}},
      label: {show: true, position: "right", color: "#dbe7ff", fontSize: 11},
    }],
  };
}

renderOrFallback("chartEp", doughnutOption("Episode", DATA.ep_dispositions));
renderOrFallback("chartPc", doughnutOption("点云帧", DATA.pc_dispositions));
renderOrFallback("chartRules", rulesOption(DATA.rule_board));
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 渲染：模板占位符替换 + JSON 注入
# ---------------------------------------------------------------------------
def render(agg: dict[str, Any], frame_rows: list[dict[str, str]], top: int) -> str:
    kpis = agg["kpis"]

    # 表格行：严重度用彩色 tag，其余转义防 XSS
    def esc(s: Any) -> str:
        return html.escape(str(s))

    if frame_rows:
        body = "\n".join(
            "<tr>"
            f"<td class='mono'>{esc(r['episode'])}</td>"
            f"<td class='mono'>{esc(r['frame'])}</td>"
            f"<td><span class='tag' style='color:{SEVERITY_COLORS.get(r['severity'], '#7c8db5')}'>"
            f"{esc(r['severity'])}</span></td>"
            f"<td class='mono'>{esc(r['reasons'])}</td>"
            f"<td>{esc(r['action'])}</td>"
            f"<td>{esc(r['metrics'])}</td>"
            "</tr>"
            for r in frame_rows
        )
    else:
        body = "<tr><td colspan='6' style='text-align:center;color:var(--muted)'>无待复核帧，点云全部通过</td></tr>"

    # 图例：处置四档
    legend = "".join(
        f"<span><i style='background:{color}'></i>{DISPOSITION_LABELS[key]}</span>"
        for key, color in DISPOSITION_COLORS.items()
    )

    page = HTML_TEMPLATE
    replacements = {
        "__DATASET__": esc(kpis["dataset"]),
        "__ROBOT__": esc(kpis["robot_type"]),
        "__FPS__": esc(kpis["fps"]),
        "__GENERATED_AT__": esc(agg["generated_at"] or "-"),
        "__KPI_EPISODES__": esc(kpis["total_episodes"]),
        "__KPI_FRAMES__": esc(kpis["total_frames"]),
        "__KPI_BAD_EPS__": esc(kpis["invalid_episodes"]),
        "__KPI_BAD_PC__": esc(kpis["pc_bad_frames"]),
        "__KPI_REVIEW__": esc(len(frame_rows)),
        "__KPI_HEALTH__": esc(kpis["health"]),
        "__TOP__": str(top),
        "__TABLE_ROWS__": body,
        "__LEGEND__": legend,
        # 图表数据以 JSON 形式注入 <script>，html 层面不需要转义引号
        "__DATA_JSON__": json.dumps({
            "ep_dispositions": agg["ep_dispositions"],
            "pc_dispositions": agg["pc_dispositions"],
            "rule_board": agg["rule_board"],
        }, ensure_ascii=False),
    }
    for placeholder, value in replacements.items():
        page = page.replace(placeholder, value)
    return page


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_report",
        description="读取流水线输出目录，生成单文件 HTML 质检大屏",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output_dir", required=True,
                   help="main.py 的输出目录（含 quality_manifest.json / pointcloud_review_queue.csv 等）")
    p.add_argument("--report", default=None,
                   help="输出 HTML 路径（默认 <output_dir>/quality_dashboard.html）")
    p.add_argument("--top", type=int, default=20, help="坏帧明细表展示的行数上限")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_dir():
        LOG.error("输出目录不存在: %s", output_dir)
        return 2

    # 1) 加载四份产物（都容错，缺哪个都不致命）
    manifest = load_manifest(output_dir)
    queue_rows = load_review_queue(output_dir)
    quality = load_quality_report(output_dir)
    pointcloud = load_pointcloud_report(output_dir)

    # 2) 聚合 + 渲染
    agg = aggregate(manifest, quality, pointcloud, queue_rows)
    frame_rows = build_frame_rows(queue_rows, args.top)
    page = render(agg, frame_rows, args.top)

    # 3) 写出单文件 HTML
    report_path = Path(args.report) if args.report else output_dir / "quality_dashboard.html"
    report_path.write_text(page, encoding="utf-8")
    print(f"HTML 质检大屏已生成: {report_path.resolve()}")
    print(f"  KPI: episodes={agg['kpis']['total_episodes']} frames={agg['kpis']['total_frames']} "
          f"异常episode={agg['kpis']['invalid_episodes']} 异常点云帧={agg['kpis']['pc_bad_frames']}")
    print(f"  规则榜 {len(agg['rule_board'])} 条，坏帧明细 {len(frame_rows)} 行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
