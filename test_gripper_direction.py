#!/usr/bin/env python3
"""Minimal, cloth-free Acone gripper direction test for Genesis."""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import genesis as gs
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
FINGER_JOINT_NAMES = (
    "left_joint17",
    "left_joint18",
    "right_joint27",
    "right_joint28",
)
SOURCE_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "left_joint11",
    "left_joint12",
    "left_joint13",
    "left_joint14",
    "left_joint15",
    "left_joint16",
    "left_joint17",
    "left_joint18",
    "right_joint21",
    "right_joint22",
    "right_joint23",
    "right_joint24",
    "right_joint25",
    "right_joint26",
    "right_joint27",
    "right_joint28",
)
FINGER_LINK_PAIRS = {
    "left": ("left_link17", "left_link18"),
    "right": ("right_link27", "right_link28"),
}
SIM1_CLOSED_Q = 0.001
SIM1_OPEN_Q = 0.044


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-json",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/episode_000000/genesis_ipc_three_folds_arc_lift80_top40_8000f_save.json",
        help="Existing demo metrics JSON; supplies the exact URDF and trajectory paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/gripper_direction_test",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--seconds-per-command", type=float, default=1.0)
    parser.add_argument("--log-level", default="warning")
    return parser.parse_args()


def urdf_finger_facts(urdf_path: Path) -> dict[str, dict]:
    root = ET.parse(urdf_path).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    facts = {}
    for name in FINGER_JOINT_NAMES:
        joint = joints[name]
        limit = joint.find("limit")
        axis = joint.find("axis")
        mimic = joint.find("mimic")
        facts[name] = {
            "type": joint.attrib["type"],
            "axis": [float(value) for value in axis.attrib["xyz"].split()],
            "lower": float(limit.attrib["lower"]),
            "upper": float(limit.attrib["upper"]),
            "mimic": None if mimic is None else dict(mimic.attrib),
        }
    return facts


def link_distance(robot, link_names: tuple[str, str]) -> float:
    first = as_numpy(robot.get_link(name=link_names[0]).get_pos(relative=False)).reshape(3)
    second = as_numpy(robot.get_link(name=link_names[1]).get_pos(relative=False)).reshape(3)
    return float(np.linalg.norm(first - second))


def annotate(frame: np.ndarray, phase: str, command: float, target: float, actual: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    lines = (
        f"{phase}",
        f"normalized command={command:.1f}",
        f"Genesis target q={target:.4f} m",
        "actual q=" + ", ".join(f"{value:.4f}" for value in actual),
    )
    cv2.rectangle(image, (18, 18), (760, 158), (0, 0, 0), thickness=-1)
    for index, line in enumerate(lines):
        color = (80, 255, 80) if index == 0 else (255, 255, 255)
        cv2.putText(
            image,
            line,
            (36, 52 + index * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
    return image


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.seconds_per_command <= 0:
        raise ValueError("fps and seconds-per-command must be positive")

    demo = json.loads(args.demo_json.read_text(encoding="utf-8"))
    urdf_path = Path(demo["urdf"]).resolve()
    trajectory_path = Path(demo["trajectory"]).resolve()
    urdf_facts = urdf_finger_facts(urdf_path)
    with np.load(trajectory_path) as trajectory:
        source_q = np.asarray(trajectory["joint_q"][0], dtype=np.float64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "gripper_direction_A_open_B_closed.mp4"
    csv_path = args.output_dir / "gripper_direction_frames.csv"
    facts_path = args.output_dir / "gripper_direction_facts.json"
    open_screenshot = args.output_dir / "command_A_OPEN.png"
    closed_screenshot = args.output_dir / "command_B_CLOSED.png"

    gs.init(backend=gs.gpu, logging_level=args.log_level)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / args.fps, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, enable_joint_limit=True),
        vis_options=gs.options.VisOptions(ambient_light=(0.55, 0.55, 0.55)),
        show_viewer=False,
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(urdf_path),
            pos=(0.0, 0.0, 0.17),
            fixed=True,
            convexify=False,
            decimate=False,
        ),
        surface=gs.surfaces.Plastic(color=(0.58, 0.64, 0.72, 1.0)),
    )
    camera = scene.add_camera(
        res=(1280, 720),
        pos=(0.34, 0.25, 1.72),
        lookat=(0.34, 0.25, 1.05),
        up=(1.0, 0.0, 0.0),
        fov=28,
        GUI=False,
    )

    actuated_joints = tuple(joint for joint in robot.joints if joint.n_qs)
    source_joint_names = tuple(demo.get("source_joint_order", SOURCE_JOINT_NAMES))
    source_index = {name: index for index, name in enumerate(source_joint_names)}
    genesis_names = tuple(joint.name for joint in actuated_joints)
    q = np.array([source_q[source_index[name]] for name in genesis_names], dtype=np.float64)
    local_index = {name: index for index, name in enumerate(genesis_names)}
    finger_indices = np.array([local_index[name] for name in FINGER_JOINT_NAMES], dtype=np.int64)
    q[finger_indices] = SIM1_OPEN_Q

    q_cursor = 0
    for joint in robot.joints:
        if joint.n_qs:
            # Genesis uses init_qpos as an imported joint-frame offset. All
            # joints must keep the URDF zero; the first pose is applied after
            # build with set_qpos().
            joint._init_qpos = np.zeros(joint.n_qs, dtype=np.float64)
            q_cursor += joint.n_qs
    scene.build()

    robot.set_qpos(q, zero_velocity=True)
    robot.set_dofs_kp(np.full(len(q), 500.0))
    robot.set_dofs_kv(np.full(len(q), 50.0))
    robot.set_dofs_kp(np.full(len(finger_indices), 1000.0), dofs_idx_local=finger_indices)
    robot.set_dofs_kv(np.full(len(finger_indices), 50.0), dofs_idx_local=finger_indices)

    runtime_facts = {}
    for name in FINGER_JOINT_NAMES:
        joint = robot.get_joint(name=name)
        runtime_facts[name] = {
            "dof_index_local": int(local_index[name]),
            "limit": np.asarray(joint.dofs_limit).tolist(),
            "translation_axis": np.asarray(joint.dofs_motion_vel).tolist(),
            "rotation_axis": np.asarray(joint.dofs_motion_ang).tolist(),
        }

    phases = (
        ("COMMAND A - OPEN", 1.0, SIM1_OPEN_Q, open_screenshot),
        ("COMMAND B - CLOSED", 0.0, SIM1_CLOSED_Q, closed_screenshot),
    )
    frames_per_command = round(args.seconds_per_command * args.fps)
    rows: list[dict] = []
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (1280, 720),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    global_frame = 0
    for phase, command, target_q, screenshot_path in phases:
        commanded_q = q.copy()
        commanded_q[finger_indices] = target_q
        for phase_frame in range(frames_per_command):
            # Match run_genesis_ipc.py's corrected direct drive path exactly.
            # Direct qpos writes and PD actuation must not be enabled together.
            robot.set_qpos(commanded_q, zero_velocity=True)
            scene.step()
            actual = as_numpy(robot.get_dofs_position(dofs_idx_local=finger_indices)).reshape(-1)
            left_distance = link_distance(robot, FINGER_LINK_PAIRS["left"])
            right_distance = link_distance(robot, FINGER_LINK_PAIRS["right"])
            row = {
                "frame": global_frame,
                "phase_frame": phase_frame,
                "phase": phase,
                "normalized_command": command,
                **{f"target_{name}": target_q for name in FINGER_JOINT_NAMES},
                **{f"actual_{name}": float(actual[i]) for i, name in enumerate(FINGER_JOINT_NAMES)},
                "left_link_origin_distance_m": left_distance,
                "right_link_origin_distance_m": right_distance,
            }
            rows.append(row)
            print(json.dumps(row, separators=(",", ":")))

            rgb, *_ = camera.render(rgb=True)
            annotated = annotate(as_numpy(rgb), phase, command, target_q, actual)
            writer.write(annotated)
            if phase_frame == frames_per_command - 1:
                cv2.imwrite(str(screenshot_path), annotated)
            global_frame += 1
    writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    facts = {
        "demo_json": str(args.demo_json.resolve()),
        "trajectory": str(trajectory_path),
        "urdf": str(urdf_path),
        "source_gripper_representation": {
            "packed_action_indices": [6, 13],
            "packed_formula": "packed=-3.24*(q-0.001)/(0.044-0.001)",
            "decode_formula": "openness=clip(packed/-3.24,0,1)",
            "genesis_target_formula": "q=0.001+0.043*openness",
        },
        "urdf_finger_joints": urdf_facts,
        "genesis_runtime_finger_joints": runtime_facts,
        "commands": {
            "A": {"normalized": 1.0, "target_q": SIM1_OPEN_Q, "measured_state": "OPEN"},
            "B": {"normalized": 0.0, "target_q": SIM1_CLOSED_Q, "measured_state": "CLOSED"},
        },
        "genesis_drive": {
            "mode": "direct",
            "target_api": "robot.set_qpos(full_19_dof_q, zero_velocity=True)",
            "pd_controller_enabled": False,
        },
        "final_measurements": {
            phase: {
                "actual_q": [rows[index][f"actual_{name}"] for name in FINGER_JOINT_NAMES],
                "left_link_origin_distance_m": rows[index]["left_link_origin_distance_m"],
                "right_link_origin_distance_m": rows[index]["right_link_origin_distance_m"],
            }
            for phase, index in (
                ("A_OPEN", frames_per_command - 1),
                ("B_CLOSED", 2 * frames_per_command - 1),
            )
        },
        "artifacts": {
            "video": str(video_path),
            "frame_log": str(csv_path),
            "open_screenshot": str(open_screenshot),
            "closed_screenshot": str(closed_screenshot),
        },
    }
    facts_path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(f"Facts: {facts_path}")
    print(f"Frames: {csv_path}")
    print(f"Video: {video_path}")
    print(f"OPEN screenshot: {open_screenshot}")
    print(f"CLOSED screenshot: {closed_screenshot}")


if __name__ == "__main__":
    main()
