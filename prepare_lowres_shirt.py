#!/usr/bin/env python3
"""Build a low-resolution shirt mesh and a source-vertex nearest-neighbour map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--faces", type=int, default=2500)
    args = parser.parse_args()

    source = trimesh.load(args.source, process=False)
    if not isinstance(source, trimesh.Trimesh):
        raise TypeError(f"Expected one Trimesh, got {type(source)}")
    simplified = source.simplify_quadric_decimation(face_count=args.faces, aggression=5)
    if simplified.body_count != 1:
        raise RuntimeError(f"Simplification split the shirt into {simplified.body_count} bodies")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    simplified.export(args.output)
    distances, mapping = cKDTree(simplified.vertices).query(source.vertices, k=1)
    mapping_path = args.output.with_suffix(".vertex_map.npz")
    np.savez_compressed(
        mapping_path,
        source_to_lowres=mapping.astype(np.int64),
        source_to_lowres_distance=distances.astype(np.float64),
    )
    metadata = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "source_vertices": int(len(source.vertices)),
        "source_faces": int(len(source.faces)),
        "vertices": int(len(simplified.vertices)),
        "faces": int(len(simplified.faces)),
        "body_count": int(simplified.body_count),
        "watertight": bool(simplified.is_watertight),
        "max_source_vertex_mapping_distance_obj_units": float(distances.max()),
        "mean_source_vertex_mapping_distance_obj_units": float(distances.mean()),
        "vertex_map": str(mapping_path.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
