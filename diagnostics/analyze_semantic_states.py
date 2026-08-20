#!/usr/bin/env python3
"""Measure fold-state geometry using the shirt-local garment atlas.

These measurements are conservative diagnostics, not success labels.  They
replace fragile world-axis vertex partitions with immutable OBJ vertex labels
and expose layer/region trajectories for later threshold calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_json", type=Path)
    parser.add_argument("--atlas", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve_atlas(metrics: dict[str, Any], explicit: Path | None) -> tuple[Path, Path]:
    if explicit is not None:
        json_path = explicit.expanduser().resolve()
        if json_path.suffix == ".npz":
            npz_path = json_path
            json_path = Path(str(json_path)[:-4] + ".json")
        else:
            npz_path = Path(str(json_path).removesuffix(".json") + ".npz")
        return json_path, npz_path
    mesh_name = Path(str(metrics.get("shirt_obj", ""))).stem
    candidates = [
        Path(str(metrics.get("shirt_obj", ""))).with_suffix("").with_name(
            f"{mesh_name}.garment_atlas.json"
        ),
        PROJECT_ROOT
        / "outputs"
        / "episode_000000"
        / "assets"
        / f"{mesh_name}.garment_atlas.json",
    ]
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            return candidate, Path(str(candidate)[:-5] + ".npz")
    raise FileNotFoundError(f"No garment atlas found; tried: {candidates}")


def load_states(raw: str | None) -> dict[int, np.ndarray]:
    if not raw:
        return {}
    directory = Path(raw).expanduser().resolve()
    states: dict[int, np.ndarray] = {}
    for path in sorted(directory.glob("frame_*.npz")):
        try:
            frame = int(path.stem.split("_")[-1])
            with np.load(path) as payload:
                states[frame] = np.asarray(payload["cloth_pos"], dtype=np.float64).copy()
        except (ValueError, KeyError, OSError):
            continue
    return states


def region_summary(points: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    region = points[mask]
    return {
        "vertices": int(len(region)),
        "centroid_xyz_m": np.mean(region, axis=0).tolist(),
        "bbox_min_xyz_m": np.min(region, axis=0).tolist(),
        "bbox_max_xyz_m": np.max(region, axis=0).tolist(),
        "height_q10_q50_q90_m": np.quantile(region[:, 2], (0.10, 0.50, 0.90)).tolist(),
    }


def dilated_occupancy(
    points: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], bins: int = 96
) -> np.ndarray:
    low, high = bounds
    scale = np.maximum(high - low, 1.0e-9)
    indices = np.floor((points[:, :2] - low) / scale * (bins - 1)).astype(int)
    indices = np.clip(indices, 0, bins - 1)
    occupancy = np.zeros((bins, bins), dtype=bool)
    occupancy[indices[:, 1], indices[:, 0]] = True
    # One-pixel dilation makes the score less sensitive to mesh decimation.
    padded = np.pad(occupancy, 1)
    return np.logical_or.reduce(
        [padded[dy : dy + bins, dx : dx + bins] for dy in range(3) for dx in range(3)]
    )


def overlap_scores(points: np.ndarray, width_band: np.ndarray) -> dict[str, float]:
    low = np.min(points[:, :2], axis=0) - 0.01
    high = np.max(points[:, :2], axis=0) + 0.01
    occupancy = {
        name: dilated_occupancy(points[width_band == value], (low, high))
        for name, value in (("negative", 0), ("center", 1), ("positive", 2))
    }

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        union = np.count_nonzero(a | b)
        return float(np.count_nonzero(a & b) / union) if union else 0.0

    triple = occupancy["negative"] & occupancy["center"] & occupancy["positive"]
    smallest = min(np.count_nonzero(value) for value in occupancy.values())
    return {
        "negative_center_iou": iou(occupancy["negative"], occupancy["center"]),
        "positive_center_iou": iou(occupancy["positive"], occupancy["center"]),
        "negative_positive_iou": iou(occupancy["negative"], occupancy["positive"]),
        "triple_overlap_over_smallest_region": (
            float(np.count_nonzero(triple) / smallest) if smallest else 0.0
        ),
    }


def sequence_analysis(
    states: dict[int, np.ndarray],
    width_band: np.ndarray,
    length_zone: np.ndarray,
    surface_layer: np.ndarray,
    keypoints: dict[str, Any],
) -> dict[str, Any] | None:
    if not states:
        return None
    expected_vertices = len(width_band)
    mismatched = {frame: len(points) for frame, points in states.items() if len(points) != expected_vertices}
    if mismatched:
        return {
            "status": "INCOMPLETE",
            "reason": f"atlas vertices={expected_vertices}; mismatched state frames={mismatched}",
        }
    reference_frame = 0 if 0 in states else min(states)
    reference = states[reference_frame]
    center_mask = width_band == 1
    frame_summaries: dict[str, Any] = {}
    for frame, points in states.items():
        horizontal = np.linalg.norm(points[:, :2] - reference[:, :2], axis=1)
        frame_summaries[str(frame)] = {
            "whole_centroid_xyz_m": np.mean(points, axis=0).tolist(),
            "center_body_median_xy_from_reference_m": float(np.median(horizontal[center_mask])),
            "regions": {
                "negative_outer": region_summary(points, width_band == 0),
                "center": region_summary(points, center_mask),
                "positive_outer": region_summary(points, width_band == 2),
                "hem": region_summary(points, length_zone == 0),
                "table_up": region_summary(points, surface_layer == 0),
                "table_facing": region_summary(points, surface_layer == 1),
            },
            "width_band_xy_overlap": overlap_scores(points, width_band),
        }

    pair_tracks: dict[str, Any] = {}
    for name, definition in keypoints.items():
        upper = int(definition["table_up_vertex"])
        lower = int(definition["table_facing_vertex"])
        pair_tracks[name] = {
            str(frame): {
                "table_up_xyz_m": points[upper].tolist(),
                "table_facing_xyz_m": points[lower].tolist(),
                "pair_distance_m": float(np.linalg.norm(points[upper] - points[lower])),
                "pair_xy_distance_m": float(np.linalg.norm(points[upper, :2] - points[lower, :2])),
            }
            for frame, points in states.items()
        }

    intervals: dict[str, Any] = {}
    ordered = sorted(states)
    for start, end in zip(ordered[:-1], ordered[1:]):
        displacement = states[end] - states[start]
        intervals[f"{start}_to_{end}"] = {
            "whole_median_xy_m": float(np.median(np.linalg.norm(displacement[:, :2], axis=1))),
            "center_median_xy_m": float(
                np.median(np.linalg.norm(displacement[center_mask, :2], axis=1))
            ),
            "negative_outer_median_xy_m": float(
                np.median(np.linalg.norm(displacement[width_band == 0, :2], axis=1))
            ),
            "positive_outer_median_xy_m": float(
                np.median(np.linalg.norm(displacement[width_band == 2, :2], axis=1))
            ),
        }
    return {
        "status": "MEASURED_REVIEW_REQUIRED",
        "reference_frame": reference_frame,
        "frames": ordered,
        "frame_summaries": frame_summaries,
        "interval_motion": intervals,
        "surface_pair_tracks": pair_tracks,
        "warning": (
            "No hard success thresholds are applied until one human-approved target run "
            "calibrates expected overlap and layer-pair behavior."
        ),
    }


def main() -> None:
    args = parse_args()
    run_json = args.run_json.expanduser().resolve()
    metrics = json.loads(run_json.read_text(encoding="utf-8"))
    atlas_json_path, atlas_npz_path = resolve_atlas(metrics, args.atlas)
    atlas_manifest = json.loads(atlas_json_path.read_text(encoding="utf-8"))
    with np.load(atlas_npz_path) as atlas:
        width_band = np.asarray(atlas["width_band"], dtype=np.int8)
        length_zone = np.asarray(atlas["length_zone"], dtype=np.int8)
        surface_layer = np.asarray(atlas["surface_layer"], dtype=np.int8)
    result = {
        "schema_version": 1,
        "run_json": str(run_json),
        "atlas_json": str(atlas_json_path),
        "atlas_npz": str(atlas_npz_path),
        "atlas_mesh_sha256": atlas_manifest["mesh_sha256"],
        "second_fold": sequence_analysis(
            load_states(metrics.get("debug_second_fold_dir")),
            width_band,
            length_zone,
            surface_layer,
            atlas_manifest["keypoints"],
        ),
        "third_fold": sequence_analysis(
            load_states(metrics.get("debug_third_fold_dir")),
            width_band,
            length_zone,
            surface_layer,
            atlas_manifest["keypoints"],
        ),
    }
    output = (
        args.output.expanduser().resolve()
        if args.output
        else run_json.with_suffix(".semantic_state_analysis.json")
    )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"semantic_analysis={output}")


if __name__ == "__main__":
    main()
