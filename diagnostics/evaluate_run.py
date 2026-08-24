#!/usr/bin/env python3
"""Offline stage-gate evaluation for a Genesis IPC shirt-fold run.

This deliberately refuses to turn logs alone into a success claim.  Automatic
checks can produce FAIL or INCOMPLETE; a clean run remains REVIEW until a human
has reviewed the mandatory multi-view keyframes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SPEC = SCRIPT_DIR / "fold_evaluation_spec.json"
CURRENT_SIM1_ROOT = (
    Path(os.environ["SIM1_ROOT"]) if os.environ.get("SIM1_ROOT") else None
)
STATUS_PRIORITY = {
    "FAIL": 4,
    "INCOMPLETE": 3,
    "REVIEW": 2,
    "PASS": 1,
    "NOT_RUN": 0,
}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    source: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_json", type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--manual-review", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless every executed stage is PASS.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def combine(checks: list[Check]) -> str:
    if not checks:
        return "INCOMPLETE"
    return max((check.status for check in checks), key=STATUS_PRIORITY.__getitem__)


def resolve_recorded_file(path: Path | None) -> tuple[Path | None, Path | None]:
    """Return (resolved, recorded) while preserving explicit relocation evidence."""
    if path is None:
        return None, None
    recorded = path.expanduser()
    parts = recorded.parts
    if (
        "SIM1" in parts
        and CURRENT_SIM1_ROOT is not None
        and CURRENT_SIM1_ROOT.exists()
    ):
        marker = max(index for index, part in enumerate(parts) if part == "SIM1")
        candidate = CURRENT_SIM1_ROOT.joinpath(*parts[marker + 1 :])
        if candidate.is_file():
            return candidate, recorded
    if recorded.is_file():
        return recorded, recorded
    return recorded, recorded


def find_contact_event(
    metrics: dict[str, Any], hand: str, close_frame: int, tolerance: int
) -> dict[str, Any] | None:
    summary = metrics.get("contact_grasp_summary") or {}
    events = ((summary.get(hand) or {}).get("events") or [])
    candidates = [
        event
        for event in events
        if isinstance(event, dict)
        and abs(int(event.get("close_frame", -100000)) - close_frame) <= tolerance
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda event: abs(int(event["close_frame"]) - close_frame))


def locate_visual_index(metrics: dict[str, Any]) -> Path | None:
    raw = metrics.get("keyframe_diagnostics_dir")
    if not raw:
        return None
    candidate = Path(raw).expanduser() / "visual_index.json"
    return candidate.resolve()


def image_guardrail(path: Path, rule: dict[str, Any]) -> dict[str, Any] | None:
    """Return simple image-space cloth-mask measurements when PIL/numpy exist."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    blue = rgb[..., 2]
    mask = (
        (blue >= int(rule["blue_min"]))
        & (blue >= rgb[..., 0] + int(rule["blue_over_red"]))
        & (blue >= rgb[..., 1] + int(rule["blue_over_green"]))
    )
    # The diagnostic label is painted in the upper-left corner.  Excluding the
    # first 36 rows prevents antialiased text from entering the cloth mask.
    mask[:36, :] = False
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"cloth_pixels": 0, "centroid_xy_px": None, "bbox_xyxy_px": None}
    return {
        "cloth_pixels": int(len(xs)),
        "centroid_xy_px": [float(xs.mean()), float(ys.mean())],
        "bbox_xyxy_px": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    }


def visual_checks(
    stage: dict[str, Any],
    metrics: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    measurements: dict[str, Any] = {}
    index_path = locate_visual_index(metrics)
    if index_path is None or not index_path.is_file():
        checks.append(
            Check(
                "mandatory_visual_evidence",
                "INCOMPLETE",
                "visual_index.json is missing; logs cannot substitute for keyframes",
                str(index_path) if index_path else None,
            )
        )
        return checks, measurements

    visual_index = load_json(index_path)
    available_frames = set(visual_index.get(f"{stage['id']}_frames", []))
    required_frames = set(stage.get("visual_frames", []))
    # Older visual indexes predate first_fold_frames. Infer availability from
    # actual files so legacy runs are reported accurately rather than rejected
    # solely because their index schema is old.
    if not available_frames:
        available_frames = {
            int(path.name.split("_")[1])
            for path in index_path.parent.glob("frame_*_overhead.png")
            if len(path.name.split("_")) >= 3
        }
    missing_frames = sorted(required_frames - available_frames)
    required_views = spec["visual_views"]
    missing_files: list[str] = []
    for frame in sorted(required_frames & available_frames):
        frame_metrics: dict[str, Any] = {}
        for view in required_views:
            image_path = index_path.parent / f"frame_{frame:04d}_{view}.png"
            if not image_path.is_file():
                missing_files.append(str(image_path))
                continue
            guardrail = image_guardrail(image_path, spec["image_guardrail"])
            if guardrail is not None:
                frame_metrics[view] = guardrail
        if frame_metrics:
            measurements[str(frame)] = frame_metrics

    if missing_frames or missing_files:
        detail = []
        if missing_frames:
            detail.append(f"missing frames={missing_frames}")
        if missing_files:
            detail.append(f"missing images={len(missing_files)}")
        checks.append(
            Check(
                "mandatory_visual_evidence",
                "INCOMPLETE",
                "; ".join(detail),
                str(index_path),
            )
        )
    else:
        checks.append(
            Check(
                "mandatory_visual_evidence",
                "PASS",
                f"all {len(required_frames)} frames x {len(required_views)} views exist",
                str(index_path),
            )
        )
    return checks, measurements


def provenance_checks(metrics: dict[str, Any], run_json: Path) -> list[Check]:
    checks: list[Check] = []
    for field in ("trajectory", "shirt_obj", "urdf"):
        raw = metrics.get(field)
        path, recorded = resolve_recorded_file(Path(raw) if raw else None)
        relocated = path is not None and recorded is not None and path != recorded
        detail = str(path) if path is not None else "path not recorded"
        if relocated:
            detail = f"recorded={recorded}; relocated={path}"
        checks.append(
            Check(
                f"input:{field}",
                "PASS" if path is not None and path.is_file() else "INCOMPLETE",
                detail,
                str(path) if path is not None else None,
            )
        )
    provenance_path = run_json.with_suffix(".provenance.json")
    checks.append(
        Check(
            "reproducibility_manifest",
            "PASS" if provenance_path.is_file() else "INCOMPLETE",
            "hashes and repository states captured"
            if provenance_path.is_file()
            else "run diagnostics/capture_provenance.py for this run",
            str(provenance_path),
        )
    )
    semantic_path = run_json.with_suffix(".semantic_state_analysis.json")
    checks.append(
        Check(
            "garment_semantic_analysis",
            "PASS" if semantic_path.is_file() else "INCOMPLETE",
            "shirt-local region/layer trajectories measured"
            if semantic_path.is_file()
            else "run diagnostics/analyze_semantic_states.py; world-axis masks are insufficient",
            str(semantic_path),
        )
    )
    return checks


def make_review_template(
    run_json: Path, spec: dict[str, Any], target_frames: int
) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage in spec["stages"]:
        if target_frames < int(stage["completion_frame"]):
            continue
        stages[stage["id"]] = {
            "visual_verdict": "PENDING",
            "checks": {text: None for text in stage.get("manual_checks", [])},
            "notes": "",
        }
    return {
        "schema_version": 1,
        "run_json": str(run_json),
        "reviewer": "",
        "reviewed_at": "",
        "stages": stages,
    }


def evaluate_stage(
    stage: dict[str, Any],
    metrics: dict[str, Any],
    spec: dict[str, Any],
    manual: dict[str, Any] | None,
) -> dict[str, Any]:
    source_frames = int(metrics.get("frames") or 0)
    if source_frames < int(stage["completion_frame"]):
        return {
            "id": stage["id"],
            "name_cn": stage["name_cn"],
            "status": "NOT_RUN",
            "checks": [],
            "visual_measurements": {},
        }

    checks: list[Check] = []
    for expected in stage.get("contact_events", []):
        event = find_contact_event(
            metrics,
            expected["hand"],
            int(expected["close_frame"]),
            int(expected.get("tolerance", 0)),
        )
        label = f"contact:{expected['hand']}@{expected['close_frame']}"
        if event is None:
            checks.append(Check(label, "INCOMPLETE", "matching close event not recorded"))
        else:
            verdict = str(event.get("verdict", "INCOMPLETE")).upper()
            status = verdict if verdict in ("PASS", "FAIL") else "INCOMPLETE"
            checks.append(
                Check(
                    label,
                    status,
                    str(event.get("reason", "no reason recorded")),
                    "contact_grasp_summary",
                )
            )

    for metric_spec in stage.get("numeric_metrics", []):
        field = metric_spec["field"]
        value = metrics.get(field)
        if not isinstance(value, dict):
            checks.append(Check(f"metric:{field}", "INCOMPLETE", "summary is missing"))
            continue
        verdict = str(value.get("verdict", "INCOMPLETE")).upper()
        if verdict == "PASS":
            status = "PASS"
        elif verdict == "FAIL" and metric_spec.get("fail_on_metric_fail", True):
            status = "FAIL"
        elif verdict == "FAIL":
            status = "REVIEW"
        else:
            status = "INCOMPLETE"
        checks.append(
            Check(
                f"metric:{field}",
                status,
                f"reported verdict={verdict}; definition={value.get('definition', 'not recorded')}",
                field,
            )
        )

    image_checks, measurements = visual_checks(stage, metrics, spec)
    checks.extend(image_checks)

    manual_stage = ((manual or {}).get("stages") or {}).get(stage["id"])
    if not isinstance(manual_stage, dict):
        checks.append(
            Check(
                "manual_visual_review",
                "REVIEW",
                "mandatory review is pending; automatic checks cannot prove layer correctness",
            )
        )
    else:
        verdict = str(manual_stage.get("visual_verdict", "PENDING")).upper()
        if verdict in ("PASS", "FAIL"):
            checks.append(
                Check(
                    "manual_visual_review",
                    verdict,
                    str(manual_stage.get("notes", "")),
                )
            )
        else:
            checks.append(Check("manual_visual_review", "REVIEW", f"verdict={verdict}"))

    return {
        "id": stage["id"],
        "name_cn": stage["name_cn"],
        "status": combine(checks),
        "checks": [asdict(check) for check in checks],
        "visual_measurements": measurements,
    }


def match_failure_route(name: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    for route in spec.get("failure_routing", []):
        pattern = str(route.get("pattern", ""))
        if pattern.endswith("*") and name.startswith(pattern[:-1]):
            return route
        if name == pattern:
            return route
    return None


def select_next_action(stages: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    """Route the earliest unapproved Gate to one bounded next experiment."""
    target_index = next(
        (index for index, stage in enumerate(stages) if stage["status"] != "PASS"),
        None,
    )
    if target_index is None:
        return {
            "target_gate": "final_validation",
            "status": "READY",
            "objective": "在约 14000 面网格上运行至少 3 次完整三折并做最终人工验收",
            "primary_check": None,
            "allowed_changes": ["网格分辨率与重复运行种子；不得再改变已冻结轨迹"],
            "recommended_actions": ["保留全部 provenance、关键帧和最终静置状态"],
            "blocked_downstream_gates": [],
        }

    target = stages[target_index]
    checks = target.get("checks", [])
    primary = (
        max(checks, key=lambda check: STATUS_PRIORITY.get(check["status"], -1))
        if checks
        else None
    )
    route = match_failure_route(primary["name"], spec) if primary else None
    stage_spec = next(
        (item for item in spec.get("stages", []) if item["id"] == target["id"]),
        {},
    )
    if target["status"] == "NOT_RUN":
        objective = f"从上一批准 checkpoint 运行并验收{target['name_cn']}"
        recommended = ["只运行当前 Gate 的最短后缀", "自动证据与人工视觉均通过后才创建下一 checkpoint"]
    else:
        objective = (
            route.get("objective")
            if route
            else f"解决{target['name_cn']}的首个未通过检查"
        )
        recommended = list(route.get("recommended_actions", [])) if route else []
    if target["id"] == "provenance":
        allowed = ["路径、缺失输入、版本记录和诊断采集；禁止调轨迹、网格或物理"]
    elif route and route.get("allowed_changes"):
        allowed = list(route["allowed_changes"])
    else:
        allowed = list(stage_spec.get("allowed_changes", []))
    return {
        "target_gate": target["id"],
        "target_gate_name_cn": target["name_cn"],
        "status": target["status"],
        "objective": objective,
        "primary_check": primary,
        "allowed_changes": allowed,
        "recommended_actions": recommended,
        "blocked_downstream_gates": [
            stage["id"] for stage in stages[target_index + 1 :] if stage["id"] != "provenance"
        ],
        "rule": "前一 Gate 未经数值与人工视觉共同批准，不得调试或批准下游 Gate。",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# 折衣实验验收：{Path(report['run_json']).stem}",
        "",
        f"- 总状态：**{report['overall_status']}**",
        f"- 运行结果：`{report['run_json']}`",
        f"- 规则版本：`{report['spec_schema_version']}`",
        "",
        "> PASS 必须包含人工关键帧复核；自动日志或几何指标不能单独证明折叠成功。",
        "",
        "## 下一轮唯一目标",
        "",
        f"- 目标 Gate：`{report['next_action']['target_gate']}`（{report['next_action']['status']}）",
        f"- 目标：{report['next_action']['objective']}",
    ]
    primary = report["next_action"].get("primary_check")
    if primary:
        lines.append(
            f"- 首要证据：`{primary['status']}` {primary['name']} — {primary['detail']}"
        )
    for item in report["next_action"].get("allowed_changes", []):
        lines.append(f"- 允许改动：{item}")
    for item in report["next_action"].get("recommended_actions", []):
        lines.append(f"- 建议动作：{item}")
    blocked = report["next_action"].get("blocked_downstream_gates", [])
    if blocked:
        lines.append(f"- 暂停下游：{', '.join(blocked)}")
    lines.append("")
    for stage in report["stages"]:
        lines.extend([f"## {stage['name_cn']} — {stage['status']}", ""])
        for check in stage["checks"]:
            source = f"（{check['source']}）" if check.get("source") else ""
            lines.append(
                f"- `{check['status']}` {check['name']}：{check['detail']}{source}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_json = args.run_json.expanduser().resolve()
    spec_path = args.spec.expanduser().resolve()
    metrics = load_json(run_json)
    spec = load_json(spec_path)
    manual = load_json(args.manual_review.expanduser().resolve()) if args.manual_review else None
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_json.parent / f"{run_json.stem}_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = provenance_checks(metrics, run_json)
    stages = [
        {
            "id": "provenance",
            "name_cn": "可复现性",
            "status": combine(provenance),
            "checks": [asdict(check) for check in provenance],
            "visual_measurements": {},
        }
    ]
    stages.extend(evaluate_stage(stage, metrics, spec, manual) for stage in spec["stages"])
    executed_statuses = [stage["status"] for stage in stages if stage["status"] != "NOT_RUN"]
    overall = max(executed_statuses, key=STATUS_PRIORITY.__getitem__)
    report = {
        "schema_version": 2,
        "spec_schema_version": spec["schema_version"],
        "run_json": str(run_json),
        "overall_status": overall,
        "stages": stages,
    }
    report["next_action"] = select_next_action(stages, spec)
    report_path = output_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = output_dir / "evaluation.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    review_path = output_dir / "manual_review.json"
    if not review_path.exists():
        review_path.write_text(
            json.dumps(
                make_review_template(run_json, spec, int(metrics.get("frames") or 0)),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"overall={overall}")
    print(f"report={report_path}")
    print(f"review_template={review_path}")
    if args.strict and overall != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
