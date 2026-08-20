#!/usr/bin/env python3
"""Index Genesis shirt-fold run JSON files into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    return parser.parse_args()


def face_count(path: str | None, cache: dict[str, int | None]) -> int | None:
    if not path:
        return None
    if path in cache:
        return cache[path]
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        cache[path] = None
        return None
    count = 0
    with candidate.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("f "):
                count += 1
    cache[path] = count
    return count


def contact_counts(metrics: dict[str, Any]) -> tuple[int, int, int]:
    passed = failed = unknown = 0
    summary = metrics.get("contact_grasp_summary") or {}
    for hand in ("left", "right"):
        for event in ((summary.get(hand) or {}).get("events") or []):
            verdict = str(event.get("verdict", "")).upper()
            if verdict == "PASS":
                passed += 1
            elif verdict == "FAIL":
                failed += 1
            else:
                unknown += 1
    return passed, failed, unknown


def metric_verdict(metrics: dict[str, Any], field: str) -> str:
    value = metrics.get(field)
    return str(value.get("verdict", "")) if isinstance(value, dict) else ""


def health(metrics: dict[str, Any]) -> str:
    _, failed, unknown = contact_counts(metrics)
    second_motion = metric_verdict(metrics, "second_fold_motion_summary")
    layering = metric_verdict(metrics, "fold_layering_summary")
    third_motion = metric_verdict(metrics, "third_fold_motion_summary")
    # Motion/contact failures are hard failures.  The legacy layering metric
    # uses a world-X partition and is only a coarse guardrail, so it must not
    # overrule visual review or the future garment-semantic metric by itself.
    if failed or "FAIL" in (second_motion, third_motion):
        return "FAIL"
    frames = int(metrics.get("frames") or 0)
    if unknown or (frames >= 619 and not second_motion) or (frames >= 979 and not third_motion):
        return "INCOMPLETE"
    if layering == "FAIL":
        return "REVIEW"
    return "REVIEW"


def load_runs(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    runs = []
    for path in sorted(root.rglob("*.json")):
        if any(part.endswith(("_state", "_keyframes", "_evaluation")) for part in path.parts):
            continue
        if path.name.endswith(("_summary.json", ".provenance.json")):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("backend") == "Genesis IPC / FEM.Cloth":
            runs.append((path, value))
    return runs


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    prefix = (
        args.output_prefix.expanduser().resolve()
        if args.output_prefix
        else root / "experiment_index"
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    faces: dict[str, int | None] = {}
    rows = []
    for path, metrics in load_runs(root):
        passed, failed, unknown = contact_counts(metrics)
        rows.append(
            {
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "run": path.stem,
                "json": str(path),
                "health": health(metrics),
                "frames": metrics.get("frames", ""),
                "executed_frames": metrics.get("executed_frames", ""),
                "mesh_faces": face_count(metrics.get("shirt_obj"), faces) or "",
                "mean_wall_fps": round(float(metrics.get("mean_wall_fps", 0.0)), 3),
                "contact_pass": passed,
                "contact_fail": failed,
                "contact_unknown": unknown,
                "second_motion": metric_verdict(metrics, "second_fold_motion_summary"),
                "layering": metric_verdict(metrics, "fold_layering_summary"),
                "third_motion": metric_verdict(metrics, "third_fold_motion_summary"),
                "checkpoint_loaded": metrics.get("persistent_third_fold_checkpoint_loaded") or "",
                "checkpoint_saved": metrics.get("persistent_third_fold_checkpoint_saved") or "",
            }
        )
    rows.sort(key=lambda row: row["modified"], reverse=True)
    fieldnames = list(rows[0]) if rows else ["run", "json", "health"]
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = prefix.with_suffix(".md")
    lines = [
        "# Genesis IPC 折衣实验索引",
        "",
        f"共识别 {len(rows)} 个运行结果。`health` 只是自动初筛，REVIEW 不等于成功。",
        "",
        "| 时间 | 运行 | 初筛 | 帧 | 面数 | FPS | 抓取 P/F/? | 二折移动 | 叠层 | 三折 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        relative = Path(row["json"]).relative_to(root)
        display = dict(row)
        display["json"] = relative.as_posix()
        lines.append(
            "| {modified} | [{run}]({json}) | {health} | {frames} | {mesh_faces} | "
            "{mean_wall_fps} | {contact_pass}/{contact_fail}/{contact_unknown} | "
            "{second_motion} | {layering} | {third_motion} |".format(**display)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"runs={len(rows)}")
    print(f"csv={csv_path}")
    print(f"markdown={md_path}")


if __name__ == "__main__":
    main()
