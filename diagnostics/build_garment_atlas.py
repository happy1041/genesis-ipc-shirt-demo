#!/usr/bin/env python3
"""Build stable shirt-local semantic labels for an OBJ cloth mesh.

The SIM1 shirt is authored with raw X across the sleeves, raw Z from hem to
collar, and raw Y across the front/back thickness.  Genesis applies
Euler(-90, 0, 0), so lower raw-Y vertices become the table-up surface.  The
atlas keeps these asset-local labels independent of camera and robot frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument(
        "--positive-x-side",
        choices=("unknown", "wearer_left", "wearer_right"),
        default="unknown",
        help="One-time semantic calibration; do not infer this from a camera view.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_obj(path: Path) -> tuple[np.ndarray, int]:
    vertices: list[list[float]] = []
    faces = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
            elif line.startswith("f "):
                faces += 1
    if not vertices or not faces:
        raise ValueError(f"OBJ has no usable vertices/faces: {path}")
    return np.asarray(vertices, dtype=np.float64), faces


def nearest_surface_pair(
    vertices: np.ndarray,
    table_up: np.ndarray,
    target_xz: tuple[float, float],
    scale_xz: np.ndarray,
) -> dict[str, object]:
    target = np.asarray(target_xz, dtype=np.float64)
    distances = np.linalg.norm((vertices[:, (0, 2)] - target) / scale_xz, axis=1)
    result: dict[str, object] = {}
    for name, mask in (("table_up", table_up), ("table_facing", ~table_up)):
        candidates = np.flatnonzero(mask)
        vertex_id = int(candidates[np.argmin(distances[candidates])])
        result[f"{name}_vertex"] = vertex_id
        result[f"{name}_raw_xyz"] = vertices[vertex_id].tolist()
    return result


def main() -> None:
    args = parse_args()
    mesh = args.mesh.expanduser().resolve()
    if not mesh.is_file():
        raise FileNotFoundError(mesh)
    prefix = (
        args.output_prefix.expanduser().resolve()
        if args.output_prefix
        else mesh.with_suffix("").with_name(f"{mesh.stem}.garment_atlas")
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    vertices, face_count = read_obj(mesh)

    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0e-12)
    canonical_u = (vertices[:, 0] - minimum[0]) / span[0]
    canonical_v = (vertices[:, 2] - minimum[2]) / span[2]

    width_low, width_high = np.quantile(vertices[:, 0], (0.35, 0.65))
    width_band = np.full(len(vertices), 1, dtype=np.int8)
    width_band[vertices[:, 0] <= width_low] = 0
    width_band[vertices[:, 0] >= width_high] = 2

    # Hem is raw-Z minimum; collar is raw-Z maximum for this official asset.
    length_zone = np.full(len(vertices), 1, dtype=np.int8)
    length_zone[canonical_v <= 0.25] = 0
    length_zone[canonical_v >= 0.72] = 2

    surface_mid = float(np.median(vertices[:, 1]))
    # Rx(-90 deg) maps world Z = -raw Y, hence smaller raw Y faces upward.
    table_up = vertices[:, 1] <= surface_mid
    surface_layer = np.where(table_up, 0, 1).astype(np.int8)

    hem_region = canonical_v <= 0.25
    hem_x_low, hem_x_high = np.quantile(vertices[hem_region, 0], (0.05, 0.95))
    negative_tip = vertices[:, 0] <= np.quantile(vertices[:, 0], 0.02)
    positive_tip = vertices[:, 0] >= np.quantile(vertices[:, 0], 0.98)
    target_xz = {
        "hem_negative": (float(hem_x_low), float(minimum[2])),
        "hem_center": (float((hem_x_low + hem_x_high) / 2.0), float(minimum[2])),
        "hem_positive": (float(hem_x_high), float(minimum[2])),
        "collar_center": (0.0, float(maximum[2])),
        "negative_x_sleeve_tip": (
            float(np.median(vertices[negative_tip, 0])),
            float(np.median(vertices[negative_tip, 2])),
        ),
        "positive_x_sleeve_tip": (
            float(np.median(vertices[positive_tip, 0])),
            float(np.median(vertices[positive_tip, 2])),
        ),
    }
    scale_xz = span[[0, 2]]
    keypoints = {
        name: {
            "target_raw_xz": list(target),
            **nearest_surface_pair(vertices, table_up, target, scale_xz),
        }
        for name, target in target_xz.items()
    }

    npz_path = Path(f"{prefix}.npz")
    np.savez_compressed(
        npz_path,
        rest_vertex_id=np.arange(len(vertices), dtype=np.int64),
        rest_raw_xyz=vertices,
        canonical_uv=np.column_stack((canonical_u, canonical_v)),
        width_band=width_band,
        length_zone=length_zone,
        surface_layer=surface_layer,
    )
    json_path = Path(f"{prefix}.json")
    positive_mapping = {
        "unknown": {
            "positive_x": "UNRESOLVED",
            "negative_x": "UNRESOLVED",
        },
        "wearer_left": {
            "positive_x": "wearer_left",
            "negative_x": "wearer_right",
        },
        "wearer_right": {
            "positive_x": "wearer_right",
            "negative_x": "wearer_left",
        },
    }[args.positive_x_side]
    manifest = {
        "schema_version": 1,
        "mesh": str(mesh),
        "mesh_sha256": sha256(mesh),
        "vertices": int(len(vertices)),
        "faces": int(face_count),
        "raw_bounds_xyz": {"min": minimum.tolist(), "max": maximum.tolist()},
        "genesis_transform": {
            "euler_degrees": [-90.0, 0.0, 0.0],
            "axis_mapping": {
                "raw_x": "world_x / shirt width",
                "raw_z": "world_y / hem-to-collar",
                "raw_y": "-world_z / front-back thickness",
            },
        },
        "semantics": {
            "raw_z_min": "hem/bottom",
            "raw_z_max": "collar/top",
            "raw_y_below_median": "table_up_surface",
            "raw_y_above_median": "table_facing_surface",
            "raw_x_mapping": positive_mapping,
            "raw_x_warning": (
                "Calibrate wearer-left/right exactly once from the asset, never from "
                "camera-left/right or robot-left/right."
            ),
        },
        "integer_labels": {
            "width_band": {"0": "negative_x_outer", "1": "center", "2": "positive_x_outer"},
            "length_zone": {"0": "hem", "1": "torso", "2": "shoulder_collar"},
            "surface_layer": {"0": "table_up", "1": "table_facing"},
        },
        "thresholds": {
            "width_raw_x_35_65_percentile": [float(width_low), float(width_high)],
            "surface_raw_y_median": surface_mid,
            "length_normalized_v": [0.25, 0.72],
        },
        "keypoints": keypoints,
        "arrays": str(npz_path),
        "warning": (
            "Surface labels are topology guardrails for double-layer grasp analysis; "
            "open sleeves/seams still require visual confirmation."
        ),
    }
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"atlas_json={json_path}")
    print(f"atlas_npz={npz_path}")
    print(f"vertices={len(vertices)} faces={face_count}")


if __name__ == "__main__":
    main()
