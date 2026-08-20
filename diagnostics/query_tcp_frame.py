#!/usr/bin/env python3
"""Query executed Acone TCP poses by source trajectory frame."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HAND_TO_SNAPSHOT_KEY = {
    "left": "left_link16_tcp",
    "right": "right_link26_tcp",
}


def print_pose(frame: int, hand: str, tcp_m: np.ndarray, extra: str = "") -> None:
    x_mm, y_mm, z_mm = tcp_m * 1000.0
    suffix = f" | {extra}" if extra else ""
    print(
        f"frame={frame:4d} hand={hand:5s} "
        f"TCP_world_mm=(x={x_mm:9.3f}, y={y_mm:9.3f}, z={z_mm:9.3f}){suffix}"
    )


def query_csv(path: Path, target: int, window: int) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    frames = sorted({int(row["source_frame"]) for row in rows})
    selected = [frame for frame in frames if abs(frame - target) <= window]
    if not selected:
        nearest = min(frames, key=lambda frame: abs(frame - target))
        selected = [nearest]
        print(f"没有 frame={target}；显示最近的 frame={nearest}")
    for frame in selected:
        for row in rows:
            if int(row["source_frame"]) != frame:
                continue
            tcp = np.array(
                [float(row["tcp_x_m"]), float(row["tcp_y_m"]), float(row["tcp_z_m"])]
            )
            extra = (
                f"step_dxyz_mm=({float(row['step_dx_mm']):.3f},"
                f" {float(row['step_dy_mm']):.3f}, {float(row['step_dz_mm']):.3f}) "
                f"openness={float(row['openness']):.3f}"
            )
            print_pose(frame, row["hand"], tcp, extra)


def query_snapshots(debug_dir: Path, target: int, window: int) -> None:
    snapshots: dict[int, Path] = {}
    for path in debug_dir.glob("frame_*.npz"):
        try:
            frame = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        with np.load(path) as data:
            if any(key in data.files for key in HAND_TO_SNAPSHOT_KEY.values()):
                snapshots[frame] = path
    if not snapshots:
        raise FileNotFoundError(f"没有包含 TCP 的关键帧快照：{debug_dir}")
    selected = [frame for frame in sorted(snapshots) if abs(frame - target) <= window]
    if not selected:
        nearest = min(snapshots, key=lambda frame: abs(frame - target))
        selected = [nearest]
        print(f"旧运行没有 frame={target}；显示最近的关键帧 frame={nearest}")
    for frame in selected:
        with np.load(snapshots[frame]) as data:
            for hand, key in HAND_TO_SNAPSHOT_KEY.items():
                if key in data.files:
                    print_pose(frame, hand, np.asarray(data[key], dtype=np.float64))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按源轨迹帧查询实际执行后的左右机械臂 TCP，坐标以 mm 显示。"
    )
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument(
        "--window", type=int, default=0, help="同时显示目标帧前后多少帧"
    )
    args = parser.parse_args()

    run_json = args.run_json.expanduser().resolve()
    metrics = json.loads(run_json.read_text(encoding="utf-8"))
    telemetry_value = metrics.get("tcp_trajectory_csv")
    if telemetry_value:
        telemetry = Path(telemetry_value).expanduser()
        if telemetry.is_file():
            print(f"数据源：逐帧实际 TCP {telemetry}")
            query_csv(telemetry, args.frame, args.window)
            return
    debug_value = metrics.get("debug_second_fold_dir")
    if debug_value:
        debug_dir = Path(debug_value).expanduser()
        if debug_dir.is_dir():
            print(f"数据源：旧运行关键帧快照 {debug_dir}")
            query_snapshots(debug_dir, args.frame, args.window)
            return
    raise FileNotFoundError(
        "运行结果既没有逐帧 TCP CSV，也没有可用的第二折关键帧快照。"
    )


if __name__ == "__main__":
    main()
