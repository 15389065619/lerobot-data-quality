#!/usr/bin/env python3
"""交互式质检 Dashboard：在线浏览每个 episode 的 action/state 曲线与视频片段。

与 ``generate_report.py`` 的区别：后者是「一次性渲染的静态 HTML 快照」，本文件是
「常驻服务 + 按需取数」——动作曲线直接从数据集 parquet 里按 episode 流式读取，
视频通过 HTTP Range 转发给浏览器做片段预览，数据量大时不会把整份数据集塞进内存。

用法::

    python tools/dashboard_server.py --dataset_path demo_dataset --output_dir outputs
    # 浏览器打开 http://127.0.0.1:8765

只依赖标准库 + 项目已有的 pyarrow/pandas；不引入 Flask/FastAPI，部署时可直接
``python -m tools.dashboard_server`` 起服务。

路由：
  GET /                        单页应用
  GET /api/summary             大盘 KPI + episode 列表
  GET /api/episode/<n>         单集曲线、问题、视频区间
  GET /video/<n>/<camera>      mp4（支持 Range: bytes=，供 <video> 拖动）
  GET /healthz                 存活探针
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import mimetypes
import re
import sys
import threading
import webbrowser
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.loader import LeRobotDatasetView, load_dataset  # noqa: E402
from pipeline.validator.types import RULE_MAP  # noqa: E402

LOG = logging.getLogger("dashboard")

DISPOSITION_COLORS = {"keep": "#22d3a7", "repair": "#38bdf8", "review": "#fbbf24", "drop": "#f43f5e"}
SEVERITY_COLORS = {"error": "#f43f5e", "warning": "#fbbf24", "info": "#38bdf8", "ok": "#22d3a7"}
MAX_POINTS_DEFAULT = 600
FRAME_CACHE_SIZE = 16


# ---------------------------------------------------------------------------
# 状态：启动时只读元信息，帧数据按 episode 点播（带小容量 LRU）
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("读取 %s 失败: %s", path.name, exc)
        return {}


def _feature_names(features: dict[str, Any], key: str, dim: int) -> list[str]:
    spec = features.get(key) or {}
    names = spec.get("names")
    if isinstance(names, dict):
        vals = list(names.values())
        if all(isinstance(v, (list, tuple)) for v in vals):
            names = [str(x) for v in vals for x in v]
        else:
            try:
                names = [names[k] for k in sorted(names, key=lambda s: int(s))]
            except (TypeError, ValueError):
                names = [str(v) for v in vals]
    if isinstance(names, (list, tuple)) and len(names) == dim:
        return [str(n) for n in names]
    return [f"dim_{i}" for i in range(dim)]


class DashboardState:
    """把「数据集 + 流水线产物」组织成 API 可直接消费的结构。"""

    def __init__(self, dataset_path: str | Path, output_dir: str | Path | None = None,
                 max_points: int = MAX_POINTS_DEFAULT) -> None:
        self.dataset: LeRobotDatasetView = load_dataset(dataset_path, load_frames=False)
        self.max_points = max(10, int(max_points))
        out = Path(output_dir).expanduser().resolve() if output_dir else None
        self.report = _load_json(out / "quality_report.json") if out else {}
        self.manifest = _load_json(out / "quality_manifest.json") if out else {}
        self.pointcloud = _load_json(out / "pointcloud_report.json") if out else {}
        self.pc_queue = self._read_pc_queue(out)
        self._frame_cache: dict[int, tuple[pd.DataFrame, list[int]]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ 元信息
    @staticmethod
    def _read_pc_queue(out: Path | None) -> list[dict[str, str]]:
        if out is None:
            return []
        path = out / "pointcloud_review_queue.csv"
        if not path.is_file():
            return []
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))

    def _decision_of(self, episode_index: int) -> dict[str, Any]:
        entry = (self.manifest.get("episodes") or {}).get(str(episode_index)) or {}
        decision = entry.get("decision")
        return decision if isinstance(decision, dict) else {}

    def _report_detail_of(self, episode_index: int) -> dict[str, Any]:
        for ep in self.report.get("invalid_episode_details") or []:
            if int(ep.get("episode_index", -1)) == episode_index:
                return ep
        return {}

    def episode_meta(self, episode_index: int) -> dict[str, Any]:
        decision = self._decision_of(episode_index)
        detail = self._report_detail_of(episode_index)
        disposition = str(decision.get("disposition") or detail.get("disposition") or "keep")
        severity = str(decision.get("severity") or detail.get("severity")
                       or ("ok" if disposition == "keep" else "warning"))
        reasons = list(decision.get("reasons") or detail.get("reasons") or [])
        details = detail.get("details") or {}
        videos = {}
        for cam in self.dataset.video_index:
            loc = self.dataset.video_location(cam, episode_index)
            if loc:
                videos[cam] = {"from": round(loc[1], 3) if loc[1] >= 0 else None,
                               "to": round(loc[2], 3) if loc[2] >= 0 else None}
        bad = [{"frame_index": int(r.get("frame_index", -1)),
                "reasons": r.get("reasons", ""),
                "severity": r.get("severity", ""),
                "action": r.get("suggested_action", "")}
               for r in self.pc_queue
               if str(r.get("episode_index", "")).strip() == str(episode_index)]
        return {
            "episode_index": episode_index,
            "frame_count": self.dataset.frame_count_of(episode_index),
            "task": self.dataset.task_of(episode_index),
            "disposition": disposition,
            "severity": severity,
            "reasons": reasons,
            "reason_labels": [RULE_MAP[c].label for c in reasons if c in RULE_MAP],
            "recommended_action": decision.get("recommended_action") or detail.get("recommended_action"),
            "details": details,
            "videos": videos,
            "pointcloud_bad_frames": bad,
        }

    def episodes(self) -> list[dict[str, Any]]:
        out = []
        for ep in self.dataset.episode_indices:
            meta = self.episode_meta(ep)
            out.append({
                "episode_index": ep,
                "frame_count": meta["frame_count"],
                "task": meta["task"],
                "disposition": meta["disposition"],
                "severity": meta["severity"],
                "reasons": meta["reasons"],
                "has_video": bool(meta["videos"]),
                "pointcloud_bad": len(meta["pointcloud_bad_frames"]),
            })
        return out

    def summary(self) -> dict[str, Any]:
        eps = self.episodes()
        disps = Counter(e["disposition"] for e in eps)
        severities = Counter(e["severity"] for e in eps)
        pc_summary = (self.pointcloud.get("summary") or {})
        rule_board = sorted((self.report.get("reason_frequency") or {}).items(),
                            key=lambda kv: kv[1], reverse=True)
        total_frames = sum(e["frame_count"] for e in eps)
        return {
            "dataset": {
                "root": str(self.dataset.root),
                "robot_type": self.dataset.info.get("robot_type"),
                "fps": self.dataset.fps,
                "cameras": sorted(self.dataset.video_index),
                "generated_at": self.report.get("generated_at") or "",
            },
            "kpis": {
                "total_episodes": len(eps),
                "total_frames": total_frames,
                "invalid_episodes": sum(1 for e in eps if e["severity"] != "ok"),
                "pc_bad_frames": sum(v for k, v in (pc_summary.get("disposition") or {}).items()
                                    if k != "keep"),
                "review_queue": len(self.pc_queue),
                "health": (self.report.get("summary") or {}).get("health_level")
                          or pc_summary.get("health_level") or "unknown",
            },
            "dispositions": dict(disps),
            "severities": dict(severities),
            "rule_board": rule_board,
            "episodes": eps,
            "colors": {"disposition": DISPOSITION_COLORS, "severity": SEVERITY_COLORS},
        }

    # ------------------------------------------------------------ 曲线数据
    def _frames(self, episode_index: int) -> tuple[pd.DataFrame, list[int]]:
        with self._lock:
            hit = self._frame_cache.get(episode_index)
            if hit is not None:
                return hit
        df = self.dataset.episode_frame(episode_index)
        n = len(df)
        stride = max(1, -(-n // self.max_points))
        idx = list(range(0, n, stride))
        if n and idx[-1] != n - 1:
            idx.append(n - 1)
        with self._lock:
            if len(self._frame_cache) >= FRAME_CACHE_SIZE:
                self._frame_cache.pop(next(iter(self._frame_cache)))
            self._frame_cache[episode_index] = (df, idx)
        return df, idx

    def episode_series(self, episode_index: int) -> dict[str, Any]:
        meta = self.episode_meta(episode_index)
        df, idx = self._frames(episode_index)
        if df.empty:
            meta.update({"timestamp": [], "action": [], "action_names": [],
                         "state": [], "state_names": []})
            return meta

        def to_vectors(key: str) -> tuple[list[list[float]], list[str]]:
            if key not in df.columns:
                return [], []
            raw = df.iloc[idx][key].tolist()
            arr = [list(map(float, row)) for row in raw]
            return arr, _feature_names(self.dataset.features, key, len(arr[0]) if arr else 0)

        action, action_names = to_vectors("action")
        state, state_names = to_vectors("observation.state")
        ts = df.iloc[idx]["timestamp"].tolist() if "timestamp" in df.columns else []
        meta.update({
            "timestamp": [round(float(x), 4) for x in ts],
            "action": action,
            "action_names": action_names,
            "state": state,
            "state_names": state_names,
        })
        return meta

    def video_path(self, episode_index: int, camera: str) -> Path | None:
        loc = self.dataset.video_location(camera, episode_index)
        return loc[0] if loc else None


# ---------------------------------------------------------------------------
# HTTP 层
# ---------------------------------------------------------------------------
def _finite(obj: Any) -> Any:
    """NaN/Inf 不是合法 JSON，转换成 null（曲线在坏数据处断开，正好暴露问题）。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "LeRobotQualityDashboard/1.0"
    state: DashboardState  # 由 make_server 注入到子类

    # ------------------------------------------------------------ 输出助手
    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(_finite(payload), ensure_ascii=False, allow_nan=False,
                          default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, page: str) -> None:
        body = page.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def log_message(self, fmt: str, *args: Any) -> None:  # 不污染 stdout
        LOG.debug("%s - %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------ 路由
    def do_GET(self) -> None:  # noqa: N802
        try:
            self._route(head=False)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_HEAD(self) -> None:  # noqa: N802
        try:
            self._route(head=True)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _route(self, head: bool) -> None:
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            self._html(PAGE if not head else "")
            return
        if path == "/healthz":
            self._json({"ok": True})
            return
        if path == "/api/summary":
            self._json(self.state.summary())
            return
        m = re.fullmatch(r"/api/episode/(\d+)", path)
        if m:
            ep = int(m.group(1))
            if ep not in self.state.dataset.episode_indices:
                self._error(HTTPStatus.NOT_FOUND, f"episode {ep} 不存在")
                return
            self._json(self.state.episode_series(ep))
            return
        m = re.fullmatch(r"/video/(\d+)/([^/]+)", path)
        if m:
            self._video(int(m.group(1)), m.group(2), head=head)
            return
        self._error(HTTPStatus.NOT_FOUND, f"未知路径: {path}")

    # ------------------------------------------------------------ 视频
    def _video(self, episode_index: int, camera: str, head: bool) -> None:
        if camera not in self.state.dataset.video_index:
            self._error(HTTPStatus.NOT_FOUND, f"相机 {camera} 不存在")
            return
        path = self.state.video_path(episode_index, camera)
        if path is None or not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, f"episode {episode_index} / {camera} 无视频")
            return
        # 只允许数据集目录内的文件，防止路径穿越
        try:
            path.resolve().relative_to(self.state.dataset.root)
        except ValueError:
            self._error(HTTPStatus.FORBIDDEN, "非法路径")
            return

        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not m or (not m.group(1) and not m.group(2)):
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start = int(m.group(1) or 0)
            end = int(m.group(2)) if m.group(2) else size - 1
            end = min(end, size - 1)
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head:
            return
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def make_server(state: DashboardState, host: str = "127.0.0.1",
                port: int = 8765) -> ThreadingHTTPServer:
    handler = type("_BoundDashboardHandler", (DashboardHandler,), {"state": state})
    server = ThreadingHTTPServer((host, int(port)), handler)
    server.daemon_threads = True
    return server


# ---------------------------------------------------------------------------
# 单页前端（ECharts 走 CDN；离线时只退化为文字，不影响数据浏览）
# ---------------------------------------------------------------------------
PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeRobot 质检 Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
  :root { --bg:#070b16; --card:rgba(17,25,46,.72); --border:rgba(56,189,248,.18);
          --text:#dbe7ff; --muted:#7c8db5; --cyan:#38bdf8;
          --green:#22d3a7; --blue:#38bdf8; --yellow:#fbbf24; --red:#f43f5e; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:radial-gradient(1200px 600px at 80% -10%,rgba(56,189,248,.12),transparent),
         radial-gradient(900px 500px at -10% 110%,rgba(244,63,94,.08),transparent),var(--bg);
         color:var(--text); font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif; }
  header { display:flex; justify-content:space-between; align-items:flex-end;
           padding:20px 28px 14px; }
  header h1 { font-size:22px; letter-spacing:2px; }
  header h1 .accent { color:var(--cyan); }
  header .meta { color:var(--muted); font-size:12px; text-align:right; line-height:1.7; }
  .layout { display:grid; grid-template-columns:320px minmax(0,1fr); gap:14px; padding:0 28px 28px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:14px;
          padding:14px 16px; box-shadow:0 8px 28px rgba(0,0,0,.35); }
  .card h2 { font-size:12px; color:var(--muted); letter-spacing:1px; font-weight:600; margin-bottom:10px; }
  .kpis { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-bottom:14px; }
  .kpi .label { font-size:11px; color:var(--muted); }
  .kpi .value { font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; margin-top:4px; }
  .chips { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }
  .chip { font-size:12px; padding:3px 10px; border-radius:999px; cursor:pointer;
          border:1px solid var(--border); color:var(--muted); background:transparent; }
  .chip.active { color:var(--cyan); border-color:var(--cyan); }
  #search { width:100%; padding:7px 10px; margin-bottom:10px; border-radius:8px;
            border:1px solid var(--border); background:rgba(7,11,22,.6); color:var(--text); }
  #list { max-height:calc(100vh - 320px); overflow:auto; }
  .ep { display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:8px;
        cursor:pointer; border:1px solid transparent; }
  .ep:hover { background:rgba(56,189,248,.06); }
  .ep.active { border-color:rgba(56,189,248,.5); background:rgba(56,189,248,.10); }
  .dot { width:8px; height:8px; border-radius:50%; flex:none; }
  .ep .idx { font-family:Consolas,monospace; font-size:13px; width:44px; flex:none; }
  .ep .why { font-size:11px; color:var(--muted); overflow:hidden; text-overflow:ellipsis;
             white-space:nowrap; }
  .ep .frames { margin-left:auto; font-size:11px; color:var(--muted); flex:none; }
  .detail-head { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:6px; }
  .detail-head .title { font-size:18px; font-family:Consolas,monospace; }
  .tag { display:inline-block; padding:2px 10px; border-radius:999px; font-size:11px;
         border:1px solid currentColor; }
  .task { color:var(--muted); font-size:12px; margin:6px 0 10px; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  .chart { height:260px; }
  video { width:100%; max-height:340px; background:#000; border-radius:10px; }
  .hint { color:var(--muted); font-size:11px; margin-top:6px; }
  .bad { color:var(--yellow); font-size:12px; line-height:1.8; word-break:break-all; }
  footer { color:var(--muted); font-size:11px; text-align:center; padding:0 0 24px; }
</style>
</head>
<body>
<header>
  <h1>LeRobot 数据集<span class="accent">质检 Dashboard</span></h1>
  <div class="meta" id="meta">加载中…</div>
</header>

<div class="layout">
  <aside class="card">
    <h2>EPISODE 列表</h2>
    <div class="chips" id="chips">
      <button class="chip active" data-filter="all">全部</button>
      <button class="chip" data-filter="keep">保留</button>
      <button class="chip" data-filter="repair">修复</button>
      <button class="chip" data-filter="review">复核</button>
      <button class="chip" data-filter="drop">剔除</button>
    </div>
    <input id="search" placeholder="按 episode 号 / 规则码过滤">
    <div id="list"></div>
  </aside>

  <main>
    <section class="kpis" id="kpis"></section>
    <section class="card" id="detail">
      <h2>EPISODE 详情</h2>
      <div id="empty" class="hint">左侧选择一个 episode</div>
      <div id="detail-body" hidden>
        <div class="detail-head" id="ep-head"></div>
        <div class="task" id="ep-task"></div>
        <div class="bad" id="pc-bad"></div>
        <div class="grid2" style="margin-top:12px">
          <div><h2>ACTION 曲线</h2><div id="chartAction" class="chart"></div></div>
          <div><h2>STATE 曲线</h2><div id="chartState" class="chart"></div></div>
        </div>
      </div>
    </section>
    <section class="card" style="margin-top:14px">
      <h2>视频预览（按 meta 的 from/to 自动截取片段）</h2>
      <div id="video-wrap"><div class="hint">暂无视频</div></div>
    </section>
  </main>
</div>
<footer>数据按需从 parquet 流式读取 ｜ 视频走 HTTP Range ｜ 判定结果来自 quality_manifest / quality_report</footer>

<script>
const COLORS = {keep:"#22d3a7", repair:"#38bdf8", review:"#fbbf24", drop:"#f43f5e"};
const PALETTE = ["#38bdf8","#22d3a7","#fbbf24","#f472b6","#a78bfa","#fb7185","#34d399","#60a5fa"];
let SUMMARY = null, CURRENT = null, FILTER = "all", SEARCH = "";
const charts = {};

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g,
    c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function renderKpis() {
  const k = SUMMARY.kpis;
  document.getElementById("kpis").innerHTML = [
    ["Episode 总数", k.total_episodes],
    ["总帧数", k.total_frames],
    ["异常 Episode", `<span style="color:var(--red)">${k.invalid_episodes}</span>`],
    ["异常点云帧", `<span style="color:var(--yellow)">${k.pc_bad_frames}</span>`],
    ["健康度", `<span style="color:var(--cyan);font-size:16px">${esc(k.health)}</span>`],
  ].map(([label, value]) =>
    `<div class="card kpi"><div class="label">${label}</div><div class="value">${value}</div></div>`
  ).join("");
  const d = SUMMARY.dataset;
  document.getElementById("meta").innerHTML =
    `数据集：${esc(d.root)}<br>机器人：${esc(d.robot_type || "-")} ｜ FPS：${esc(d.fps)}<br>` +
    `相机：${esc((d.cameras || []).join(", ") || "无")}<br>生成时间：${esc(d.generated_at || "-")}`;
}

function renderList() {
  const list = SUMMARY.episodes.filter(e =>
    (FILTER === "all" || e.disposition === FILTER) &&
    (!SEARCH || String(e.episode_index).includes(SEARCH) ||
     e.reasons.join(",").includes(SEARCH)));
  document.getElementById("list").innerHTML = list.map(e => `
    <div class="ep ${CURRENT === e.episode_index ? "active" : ""}" data-ep="${e.episode_index}">
      <span class="dot" style="background:${COLORS[e.disposition] || "#7c8db5"}"></span>
      <span class="idx">#${e.episode_index}</span>
      <span class="why">${esc(e.reasons.join(" · ") || "—")}</span>
      <span class="frames">${e.frame_count}f${e.has_video ? " ▶" : ""}</span>
    </div>`).join("") || `<div class="hint">没有匹配的 episode</div>`;
  document.querySelectorAll(".ep").forEach(el =>
    el.onclick = () => selectEpisode(Number(el.dataset.ep)));
}

function lineOption(names, rows, ts) {
  const series = (names.length ? names : ["value"]).map((name, dim) => ({
    name, type: "line", showSymbol: false, smooth: 0.15, lineStyle: {width: 1.4},
    color: PALETTE[dim % PALETTE.length],
    data: rows.map((row, i) => [ts[i], row[dim]]),
  }));
  return {
    backgroundColor: "transparent", animation: false,
    tooltip: {trigger: "axis", backgroundColor: "#0f1830", borderColor: "#38bdf8",
              textStyle: {color: "#dbe7ff"}},
    legend: {type: "scroll", bottom: 0, textStyle: {color: "#7c8db5", fontSize: 10}},
    grid: {left: 44, right: 18, top: 14, bottom: 40},
    xAxis: {type: "value", name: "t(s)", nameTextStyle: {color: "#7c8db5"},
            axisLabel: {color: "#7c8db5", fontSize: 10},
            splitLine: {lineStyle: {color: "rgba(56,189,248,.08)"}}},
    yAxis: {type: "value", axisLabel: {color: "#7c8db5", fontSize: 10},
            splitLine: {lineStyle: {color: "rgba(56,189,248,.08)"}}},
    series,
  };
}

// 离线兜底：没有 ECharts 时用原生 canvas 画折线，保证内网/无外网环境也能看曲线
function drawFallback(el, names, rows, ts) {
  el.innerHTML = "";
  const w = Math.max(el.clientWidth, 320), h = 260, dpr = window.devicePixelRatio || 1;
  const cv = document.createElement("canvas");
  cv.width = w * dpr; cv.height = h * dpr;
  cv.style.width = w + "px"; cv.style.height = h + "px";
  el.appendChild(cv);
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  const pad = {l: 44, r: 12, t: 12, b: 34};
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
  const values = rows.flat().filter(v => typeof v === "number" && isFinite(v));
  if (!values.length || !ts.length) {
    ctx.fillStyle = "#7c8db5"; ctx.font = "12px sans-serif";
    ctx.fillText("暂无可绘制的数值", pad.l, h / 2);
    return;
  }
  const vmin = Math.min(...values), vmax = Math.max(...values);
  const span = (vmax - vmin) || 1, t0 = ts[0], t1 = ts[ts.length - 1] || 1;
  const x = t => pad.l + ((t - t0) / ((t1 - t0) || 1)) * plotW;
  const y = v => pad.t + (1 - (v - vmin) / span) * plotH;

  ctx.strokeStyle = "rgba(56,189,248,.12)"; ctx.fillStyle = "#7c8db5"; ctx.font = "10px sans-serif";
  for (let g = 0; g <= 4; g++) {
    const yy = pad.t + (g / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
    const label = (vmax - (g / 4) * span).toFixed(2);
    ctx.fillText(label, 4, yy + 3);
  }
  ctx.fillText(t0.toFixed(2) + "s", pad.l, h - 18);
  ctx.fillText((t1 || 0).toFixed(2) + "s", w - pad.r - 36, h - 18);

  names.forEach((name, dim) => {
    ctx.strokeStyle = PALETTE[dim % PALETTE.length]; ctx.lineWidth = 1.4;
    ctx.beginPath(); let started = false;
    rows.forEach((row, i) => {
      const v = row[dim];
      if (typeof v !== "number" || !isFinite(v)) { started = false; return; }
      const px = x(ts[i]), py = y(v);
      started ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      started = true;
    });
    ctx.stroke();
  });
  let lx = pad.l, ly = h - 8;
  names.forEach((name, dim) => {
    const text = name.length > 12 ? name.slice(0, 11) + "…" : name;
    ctx.fillStyle = PALETTE[dim % PALETTE.length];
    ctx.fillRect(lx, ly - 7, 6, 6);
    ctx.fillStyle = "#7c8db5"; ctx.fillText(text, lx + 9, ly);
    lx += 14 + text.length * 6;
  });
}

const fallback = {};

function drawChart(id, names, rows, ts) {
  const el = document.getElementById(id);
  fallback[id] = {names, rows, ts};
  if (!window.echarts) { drawFallback(el, names, rows, ts); return; }
  const chart = charts[id] || (charts[id] = echarts.init(el, null, {renderer: "canvas"}));
  chart.setOption(lineOption(names, rows, ts), true);
  chart.resize();
}

function renderVideo(data) {
  const wrap = document.getElementById("video-wrap");
  const cams = Object.keys(data.videos || {});
  if (!cams.length) {
    wrap.innerHTML = '<div class="hint">该 episode 没有可定位的视频文件（或 meta 未记录 from/to）</div>';
    return;
  }
  const cam = cams[0];
  const v = data.videos[cam];
  const base = `/video/${data.episode_index}/${encodeURIComponent(cam)}`;
  const frag = (v.from != null && v.to != null && v.to > v.from) ? `#t=${v.from},${v.to}` : "";
  wrap.innerHTML = `
    <select id="cam" class="chip" style="margin-bottom:8px">
      ${cams.map(c => `<option value="${esc(c)}" ${c === cam ? "selected" : ""}>${esc(c)}</option>`).join("")}
    </select>
    <video id="vid" controls preload="metadata"></video>
    <div class="hint">片段 ${v.from != null ? v.from + "s – " + v.to + "s" : "全片"}（浏览器若忽略 #t 片段，可手动拖动进度条）</div>`;
  const vid = document.getElementById("vid");
  const apply = (c) => {
    const vv = data.videos[c] || {};
    const f = (vv.from != null && vv.to != null && vv.to > vv.from) ? `#t=${vv.from},${vv.to}` : "";
    vid.src = `/video/${data.episode_index}/${encodeURIComponent(c)}${f}`;
  };
  apply(cam);
  document.getElementById("cam").onchange = (ev) => apply(ev.target.value);
}

async function selectEpisode(ep) {
  CURRENT = ep;
  renderList();
  const data = await api(`/api/episode/${ep}`);
  document.getElementById("empty").hidden = true;
  document.getElementById("detail-body").hidden = false;
  const head = document.getElementById("ep-head");
  document.getElementById("ep-task").textContent = data.task || "（无任务描述）";
  head.innerHTML = `
    <span class="title">Episode #${ep}</span>
    <span class="tag" style="color:${COLORS[data.disposition] || "#7c8db5"}">${esc(data.disposition)}</span>
    <span class="tag" style="color:${data.severity === "error" ? "#f43f5e" : "#fbbf24"}">${esc(data.severity)}</span>
    <span class="hint">${data.frame_count} 帧 ｜ ${esc(data.recommended_action || "—")}</span>
    <span class="hint">${(data.reason_labels || []).map(esc).join(" / ")}</span>`;
  document.getElementById("pc-bad").innerHTML = data.pointcloud_bad_frames.length
    ? "点云坏帧：" + data.pointcloud_bad_frames.slice(0, 30)
        .map(b => `#${b.frame_index}(${esc(b.reasons)})`).join("，")
    : "";
  drawChart("chartAction", data.action_names || [], data.action || [], data.timestamp || []);
  drawChart("chartState", data.state_names || [], data.state || [], data.timestamp || []);
  renderVideo(data);
}

async function boot() {
  SUMMARY = await api("/api/summary");
  renderKpis();
  renderList();
  if (SUMMARY.episodes.length) {
    const first = SUMMARY.episodes.find(e => e.disposition !== "keep") || SUMMARY.episodes[0];
    selectEpisode(first.episode_index);
  }
}

document.getElementById("chips").onclick = (ev) => {
  const btn = ev.target.closest(".chip");
  if (!btn) return;
  FILTER = btn.dataset.filter;
  document.querySelectorAll("#chips .chip").forEach(b => b.classList.toggle("active", b === btn));
  renderList();
};
document.getElementById("search").oninput = (ev) => { SEARCH = ev.target.value.trim(); renderList(); };
window.addEventListener("resize", () => {
  Object.values(charts).forEach(c => c.resize());
  Object.entries(fallback).forEach(([id, d]) =>
    drawFallback(document.getElementById(id), d.names, d.rows, d.ts));
});
boot().catch(err => {
  document.getElementById("meta").textContent = "加载失败：" + err.message;
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dashboard_server",
        description="启动质检 Dashboard：在线查看每集 action 曲线与视频片段",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset_path", required=True, help="LeRobot 数据集根目录")
    p.add_argument("--output_dir", default=None,
                   help="main.py 输出目录（读取 quality_report/manifest/点云复核队列）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max_points", type=int, default=MAX_POINTS_DEFAULT,
                   help="单条曲线的最大采样点数（防止长 episode 卡顿）")
    p.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        state = DashboardState(args.dataset_path, args.output_dir, args.max_points)
    except Exception as exc:  # noqa: BLE001 - 启动失败要给出可读原因
        LOG.error("加载数据集失败: %s", exc)
        return 2
    server = make_server(state, args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"
    LOG.info("Dashboard 已启动: %s（Ctrl+C 退出）", url)
    if args.open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("正在关闭…")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
