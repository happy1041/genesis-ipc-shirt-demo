#!/usr/bin/env python3
"""Export the SIM1 shirt USD mesh to a triangle OBJ without changing coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prim", default="/root/Mesh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = Usd.Stage.Open(str(args.source))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {args.source}")
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(args.prim))
    if not mesh:
        raise KeyError(f"Mesh prim not found: {args.prim}")

    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if not np.all(counts == 3):
        raise ValueError("The SIM1 cloth exporter currently requires triangle faces")
    faces = indices.reshape(-1, 3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii") as stream:
        stream.write("# SIM1 short-shirt mesh; source coordinates are millimetres\n")
        for x, y, z in points:
            stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in faces + 1:
            stream.write(f"f {a} {b} {c}\n")

    extent_m = np.ptp(points, axis=0) * 0.001
    print(f"Wrote {args.output.resolve()}")
    print(f"vertices={len(points)}, triangles={len(faces)}, raw_extent_m={extent_m}")


if __name__ == "__main__":
    main()
