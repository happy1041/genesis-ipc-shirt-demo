#!/home/happy1041/work/genesis-ipc-env/bin/python
"""Score second-fold placement from saved cloth states.

The simulator's old motion verdict only inspected the trajectory up to frame
583.  That is useful for detecting gross body drag, but it cannot distinguish
an under-placed flap from a flap that slips after release.  This tool keeps the
mesh-independent initial-X semantic split and evaluates placement, release,
and settling at frames 583, 600, and 619.  An optional known-good run supplies
the target geometry when comparing different mesh resolutions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FRAMES = (0, 583, 600, 619)


def load_positions(directory: Path) -> dict[int, np.ndarray]:
    positions: dict[int, np.ndarray] = {}
    for frame in FRAMES:
        path = directory / f"frame_{frame:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as state:
            positions[frame] = np.asarray(state["cloth_pos"], dtype=np.float64)
    return positions


def occupied(points: np.ndarray, origin: np.ndarray, cell_m: float) -> set[tuple[int, int]]:
    cells = np.floor((points - origin) / cell_m).astype(np.int64)
    return {tuple(cell) for cell in cells}


def overlap(a: np.ndarray, b: np.ndarray, cell_m: float = 0.01) -> dict[str, float | int]:
    origin = np.minimum(np.min(a, axis=0), np.min(b, axis=0))
    a_cells = occupied(a, origin, cell_m)
    b_cells = occupied(b, origin, cell_m)
    intersection = len(a_cells & b_cells)
    union = len(a_cells | b_cells)
    return {
        "cell_m": cell_m,
        "cells_a": len(a_cells),
        "cells_b": len(b_cells),
        "intersection": intersection,
        "coverage": intersection / max(1, min(len(a_cells), len(b_cells))),
        "iou": intersection / max(1, union),
    }


def vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in value]


def analyze(directory: Path) -> dict:
    positions = load_positions(directory)
    initial_x = positions[0][:, 0]
    low, high = np.quantile(initial_x, (0.35, 0.65))
    masks = {
        "first": initial_x <= low,
        "base": (initial_x > low) & (initial_x < high),
        "second": initial_x >= high,
    }
    frames: dict[str, dict] = {}
    for frame in FRAMES[1:]:
        current = positions[frame]
        centroids = {
            name: np.mean(current[mask], axis=0) for name, mask in masks.items()
        }
        frames[str(frame)] = {
            "centroids_xyz_m": {name: vector(value) for name, value in centroids.items()},
            "second_minus_first_centroid_xyz_m": vector(centroids["second"] - centroids["first"]),
            "second_minus_base_centroid_xyz_m": vector(centroids["second"] - centroids["base"]),
            "overlap": {
                "first_second": overlap(current[masks["first"], :2], current[masks["second"], :2]),
                "first_base": overlap(current[masks["first"], :2], current[masks["base"], :2]),
                "second_base": overlap(current[masks["second"], :2], current[masks["base"], :2]),
            },
            "z_q10_q50_q90_m": {
                name: vector(np.quantile(current[mask, 2], (0.10, 0.50, 0.90)))
                for name, mask in masks.items()
            },
            "cloth_xy_q05_q95_m": {
                "q05": vector(np.quantile(current[:, :2], 0.05, axis=0)),
                "q95": vector(np.quantile(current[:, :2], 0.95, axis=0)),
            },
        }

    start = positions[583]
    end = positions[619]
    release_drift = {}
    for name, mask in masks.items():
        delta = np.mean(end[mask], axis=0) - np.mean(start[mask], axis=0)
        release_drift[name] = {
            "centroid_xyz_m": vector(delta),
            "centroid_xy_norm_m": float(np.linalg.norm(delta[:2])),
        }
    whole_delta = np.mean(end, axis=0) - np.mean(start, axis=0)
    release_drift["whole"] = {
        "centroid_xyz_m": vector(whole_delta),
        "centroid_xy_norm_m": float(np.linalg.norm(whole_delta[:2])),
    }
    summary = {
        "directory": str(directory.resolve()),
        "vertices": int(len(initial_x)),
        "region_definition": "initial world-X low 35%=first, middle 30%=base, high 35%=second",
        "initial_x_bounds_m": {"p35": float(low), "p65": float(high)},
        "frames": frames,
        "release_drift_583_to_619": release_drift,
    }
    motion_path = directory / "second_fold_motion_summary.json"
    if motion_path.is_file():
        motion = json.loads(motion_path.read_text(encoding="utf-8"))
        full = motion.get("intervals", {}).get("393_to_583", {})
        if full:
            whole = np.asarray(full.get("all_vertices_centroid_xy_m", (0.0, 0.0)))
            summary["motion_393_to_583"] = {
                "stationary_body_median_xy_m": float(
                    full.get("stationary_body_median_xy_m", 0.0)
                ),
                "stationary_body_p90_xy_m": float(
                    full.get("stationary_body_p90_xy_m", 0.0)
                ),
                "whole_centroid_xy_m": vector(whole),
                "whole_centroid_xy_norm_m": float(np.linalg.norm(whole)),
            }
    return summary


def compare(candidate: dict, reference: dict) -> dict:
    candidate_619 = candidate["frames"]["619"]
    reference_619 = reference["frames"]["619"]
    candidate_gap = np.asarray(candidate_619["second_minus_first_centroid_xyz_m"])
    reference_gap = np.asarray(reference_619["second_minus_first_centroid_xyz_m"])
    gap_error = candidate_gap - reference_gap
    candidate_drift = candidate["release_drift_583_to_619"]["second"]["centroid_xy_norm_m"]
    reference_drift = reference["release_drift_583_to_619"]["second"]["centroid_xy_norm_m"]
    candidate_coverage = candidate_619["overlap"]["first_second"]["coverage"]
    reference_coverage = reference_619["overlap"]["first_second"]["coverage"]

    # X is the controlled stacking direction.  Y depends more strongly on the
    # tessellation's sleeve/collar sampling, so it is reported but weighted
    # less.  Release drift prevents a deceptively good placement that cannot
    # remain stable after the grippers open.
    score = (
        abs(float(gap_error[0])) * 1000.0
        + 0.25 * abs(float(gap_error[1])) * 1000.0
        + max(0.0, float(candidate_drift - reference_drift)) * 500.0
        + max(0.0, float(reference_coverage - candidate_coverage)) * 20.0
    )
    motion_comparison = None
    if "motion_393_to_583" in candidate and "motion_393_to_583" in reference:
        candidate_motion = candidate["motion_393_to_583"]
        reference_motion = reference["motion_393_to_583"]
        median_excess = max(
            0.0,
            candidate_motion["stationary_body_median_xy_m"]
            - reference_motion["stationary_body_median_xy_m"],
        )
        centroid_excess = max(
            0.0,
            candidate_motion["whole_centroid_xy_norm_m"]
            - reference_motion["whole_centroid_xy_norm_m"],
        )
        # Both terms are in metres.  A 2x500 multiplier means that each
        # excess millimetre contributes one score point in total when the
        # stationary-body and whole-centroid signals agree.  This rejects a
        # deceptively aligned flap obtained by dragging the entire shirt.
        motion_penalty = (median_excess + centroid_excess) * 500.0
        score += motion_penalty
        motion_comparison = {
            "candidate_stationary_body_median_xy_m": float(
                candidate_motion["stationary_body_median_xy_m"]
            ),
            "reference_stationary_body_median_xy_m": float(
                reference_motion["stationary_body_median_xy_m"]
            ),
            "candidate_whole_centroid_xy_norm_m": float(
                candidate_motion["whole_centroid_xy_norm_m"]
            ),
            "reference_whole_centroid_xy_norm_m": float(
                reference_motion["whole_centroid_xy_norm_m"]
            ),
            "excess_motion_penalty": float(motion_penalty),
        }
    verdict = "PASS" if abs(float(gap_error[0])) <= 0.008 and candidate_drift <= 0.025 else "TUNE"
    result = {
        "reference_directory": reference["directory"],
        "frame_619_gap_error_xyz_m": vector(gap_error),
        "frame_619_first_second_coverage": {
            "candidate": float(candidate_coverage),
            "reference": float(reference_coverage),
        },
        "second_region_release_drift_xy_m": {
            "candidate": float(candidate_drift),
            "reference": float(reference_drift),
        },
        "score_lower_is_better": float(score),
        "verdict": verdict,
        "thresholds": {"stack_x_error_m": 0.008, "release_drift_xy_m": 0.025},
    }
    if motion_comparison is not None:
        result["motion_393_to_583"] = motion_comparison
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("debug_dir", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = analyze(args.debug_dir)
    if args.reference is not None:
        summary["comparison"] = compare(summary, analyze(args.reference))
    output = args.output or args.debug_dir / "second_fold_quality_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if "comparison" in summary:
        result = summary["comparison"]
        print(
            f"second_fold_quality verdict={result['verdict']} "
            f"score={result['score_lower_is_better']:.3f} "
            f"x_error_mm={result['frame_619_gap_error_xyz_m'][0] * 1000.0:.2f} "
            f"release_drift_mm={result['second_region_release_drift_xy_m']['candidate'] * 1000.0:.2f}"
        )
    print(output)


if __name__ == "__main__":
    main()
