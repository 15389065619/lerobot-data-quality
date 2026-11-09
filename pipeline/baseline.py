"""基线白名单（Baseline Whitelist）：把「已确认的已知误报」沉淀成可复用的白名单。

纯统计学阈值在不同数据集上总会有少量稳定误报（例如片段库的长度分布、光捕的
四元数重采样）。每次重跑都人工复核同一条告警是浪费，因此给每条告警一个稳定指纹：

    fingerprint = md5(dataset_name|episode_index|rule_code)[:16]

用户确认是误报后，把指纹写进配置文件顶层的 ``baseline_whitelist``。命中的告警会从
episode 的 ``reasons`` 中摘除（不再参与处置决策与健康度计算），同时保留在
``ignored_baseline`` 里，报告仍能追溯「哪条告警被白名单忽略了」。

指纹只锚定「数据集名 + episode 编号 + 规则」，不包含帧数/阈值等易变信息，
因此重跑、重新推导阈值都不会让白名单失效。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from .config import read_config_file, severity_of
from .validator.types import ValidationResult

LOG = logging.getLogger(__name__)

FINGERPRINT_LEN = 16
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{16}$")


def fingerprint(dataset_name: str, episode_index: int, rule_code: str) -> str:
    """告警指纹：同一数据集/同一 episode/同一规则的指纹跨运行稳定。"""
    raw = f"{dataset_name}|{int(episode_index)}|{rule_code}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:FINGERPRINT_LEN]


def whitelist_from_config(config: dict[str, Any] | None) -> set[str]:
    """读取配置里的白名单，兼容字符串与 ``{fingerprint: ...}`` 两种写法。"""
    entries = (config or {}).get("baseline_whitelist") or []
    out: set[str] = set()
    for entry in entries:
        value = entry.get("fingerprint") if isinstance(entry, dict) else entry
        text = str(value or "").strip().lower()
        if not text:
            continue
        if not _FINGERPRINT_RE.match(text):
            LOG.warning("baseline_whitelist 中的指纹 %r 格式不合法（应为 16 位十六进制），已忽略", value)
            continue
        out.add(text)
    return out


def apply_whitelist(
    validation: ValidationResult,
    dataset_name: str,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """按白名单摘除告警，返回全量被忽略记录（写进报告）。

    只摘除 ``reasons``，``details`` 转存到忽略记录里，因此：
    - ``valid_episodes`` / 健康度 / 处置决策都按「忽略后」计算；
    - 报告仍能列出被忽略的原始证据与指纹。
    """
    cfg = config if config is not None else validation.config
    allowed = whitelist_from_config(cfg)

    if allowed:
        for res in validation.results.values():
            if not res.reasons:
                continue
            kept: list[str] = []
            for code in res.reasons:
                fp = fingerprint(dataset_name, res.episode_index, code)
                if fp not in allowed:
                    kept.append(code)
                    continue
                severity = res.severity_overrides.pop(code, None) or severity_of(cfg, code)
                detail = res.details.pop(code, None) or {}
                res.ignored_baseline.append(
                    {"code": code, "fingerprint": fp,
                     "severity": severity, "detail": detail}
                )
                LOG.info("基线白名单：忽略 ep%d 的 %s（%s）", res.episode_index, code, fp)
            res.reasons = kept

    # 增量模式下复用的历史结论也带着 ignored_baseline，统一汇总到结果对象
    validation.baseline_ignored = [
        {"episode_index": res.episode_index, **item}
        for res in sorted(validation.results.values(), key=lambda r: r.episode_index)
        for item in res.ignored_baseline
    ]
    return validation.baseline_ignored


def append_to_config(path: str | Path, fingerprints: Iterable[str]) -> tuple[list[str], list[str]]:
    """把指纹写回配置文件，返回 (新增, 已存在)。

    YAML 文件里没有 ``baseline_whitelist`` 键时用文本追加，尽量保留人工写的注释；
    键已存在（或 JSON）时才整份重写，此时注释会丢失并给出告警。
    """
    p = Path(path).expanduser()
    raw = read_config_file(p)

    existing: list[str] = []
    for entry in raw.get("baseline_whitelist") or []:
        value = entry.get("fingerprint") if isinstance(entry, dict) else entry
        text = str(value or "").strip().lower()
        if text:
            existing.append(text)

    incoming: list[str] = []
    invalid: list[str] = []
    for item in fingerprints:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if not _FINGERPRINT_RE.match(text):
            invalid.append(str(item))
        elif text not in incoming:
            incoming.append(text)
    if invalid:
        raise ValueError(
            "指纹格式不合法（应为 16 位十六进制，如 0f3a...）：" + ", ".join(invalid)
        )

    added = [fp for fp in incoming if fp not in set(existing)]
    if not added:
        return [], existing

    merged = existing + added
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml") and "baseline_whitelist" not in raw:
        text = p.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            text += "\n"
        block = (
            "\n# 基线白名单：已确认的已知误报，指纹 = md5(dataset_name|episode_index|rule_code)[:16]\n"
            "baseline_whitelist:\n" + "".join(f"  - {fp}\n" for fp in merged)
        )
        p.write_text(text + block, encoding="utf-8")
    else:
        raw["baseline_whitelist"] = merged
        if suffix in (".yaml", ".yml"):
            import yaml

            LOG.warning("配置文件已有 baseline_whitelist，整份重写会丢失注释: %s", p)
            p.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
        else:
            with p.open("w", encoding="utf-8") as fh:
                json.dump(raw, fh, ensure_ascii=False, indent=2)
    return added, existing
