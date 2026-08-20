#!/usr/bin/env python3
"""Print SIM1 Acone TCP motion phases without running cloth IPC."""

from __future__ import annotations

import argparse
from pathlib import Path

import genesis as gs
import numpy as np

from run_genesis_ipc import SOURCE_JOINT_NAMES, TCP_LOCAL, as_numpy, quat_wxyz_to_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim1-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--start", type=int, default=330)
    parser.add_argument("--end", type=int, default=610)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("/tmp/sim1_tcp.csv"))
    args = parser.parse_args()

    with np.load(args.trajectory) as data:
        source_q = np.asarray(data["joint_q"], dtype=np.float64)
        openness = np.asarray(data["openness"], dtype=np.float64)

    gs.init(backend=gs.gpu, logging_level="warning", seed=0)
    scene = gs.Scene(show_viewer=False)
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(args.sim1_root / "assets/acone/acone.urdf"),
            pos=(0.0, 0.0, 0.17),
            fixed=True,
        )
    )
    genesis_joint_names = tuple(joint.name for joint in robot.joints if joint.n_qs)
    source_index = {name: i for i, name in enumerate(SOURCE_JOINT_NAMES)}
    genesis_from_source = np.array([source_index[name] for name in genesis_joint_names])
    q = source_q[:, genesis_from_source]
    for joint in robot.joints:
        if joint.n_qs:
            joint._init_qpos = np.zeros(joint.n_qs, dtype=np.float64)
    scene.build()

    links = (
        robot.get_link(name="left_link16"),
        robot.get_link(name="right_link26"),
    )
    previous = None
    rows = ["frame,hand,openness,tcp_x,tcp_y,tcp_z,dxy_from_previous,dz_from_previous"]
    frames = list(range(args.start, args.end + 1, args.step))
    for required in (332, 393, 439, 455, 500, 540, 560, 583, 600):
        if args.start <= required <= args.end:
            frames.append(required)
    for frame in sorted(set(frames)):
        robot.set_qpos(q[frame], zero_velocity=True)
        current = []
        for hand_index, (name, link) in enumerate(zip(("left", "right"), links)):
            pos = as_numpy(link.get_pos()).reshape(3)
            rot = quat_wxyz_to_matrix(link.get_quat())
            tcp = pos + rot @ TCP_LOCAL
            current.append(tcp)
            if previous is None:
                dxy = dz = float("nan")
            else:
                dxy = float(np.linalg.norm(tcp[:2] - previous[hand_index][:2]))
                dz = float(tcp[2] - previous[hand_index][2])
            rows.append(
                f"{frame},{name},{openness[frame, hand_index]:.6f},"
                f"{tcp[0]:.8f},{tcp[1]:.8f},{tcp[2]:.8f},{dxy:.8f},{dz:.8f}"
            )
        previous = current
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
