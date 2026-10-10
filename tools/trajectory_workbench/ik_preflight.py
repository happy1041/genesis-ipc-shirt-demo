#!/usr/bin/env python3
"""Run independent causal IK preflight for a compiled Workbench stage edit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.trajectory_workbench.server import WorkbenchData


SOURCE_JOINT_NAMES = (
    "joint1", "joint2", "joint3",
    "left_joint11", "left_joint12", "left_joint13", "left_joint14",
    "left_joint15", "left_joint16", "left_joint17", "left_joint18",
    "right_joint21", "right_joint22", "right_joint23", "right_joint24",
    "right_joint25", "right_joint26", "right_joint27", "right_joint28",
)
TCP_LOCAL = np.array([0.155, 0.001786, 0.014], dtype=np.float64)


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cosine = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(cosine)))


def quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(first, dtype=np.float64)
    bw, bx, by, bz = np.asarray(second, dtype=np.float64)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def apply_local_y_rotation(quaternion: np.ndarray, angle_deg: float) -> np.ndarray:
    half = np.radians(float(angle_deg)) * 0.5
    delta = np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float64)
    result = quaternion_multiply(quaternion, delta)
    return result / np.linalg.norm(result)


def pair_set(value: np.ndarray) -> set[tuple[int, int]]:
    array = as_numpy(value).reshape(-1, 2)
    return {tuple(sorted((int(first), int(second)))) for first, second in array}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--edits", type=Path, help="JSON containing {'edits': [...]}; omit for baseline")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = WorkbenchData(args.config.expanduser().resolve())
    requested_edits = []
    orientation_assist_request = None
    hand = str(data.editor_config.get("default_hand", "right"))
    if args.edits:
        loaded = json.loads(args.edits.expanduser().resolve().read_text(encoding="utf-8"))
        requested_edits = list(loaded.get("edits", loaded if isinstance(loaded, list) else []))
        if isinstance(loaded, dict):
            hand = str(loaded.get("hand", hand))
            orientation_assist_request = loaded.get("orientation_assist")
    compiled = data.compile_stage(requested_edits, hand)
    if not compiled["accepted"]:
        raise RuntimeError(
            "Position trajectory failed speed/acceleration gates before IK: "
            f"{compiled['resolved_metrics']}"
        )

    orientation_mode = str(data.config.get("ik_orientation_mode", "full"))
    plan_value = data.config.get("second_fold_material_plan")
    if plan_value and "ik_orientation_mode" not in data.config:
        plan_path = Path(plan_value).expanduser().resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_hand = plan["hands"][0 if hand == "left" else 1]
        orientation_mode = str(plan_hand.get("ik_orientation_mode", "full"))
    if orientation_mode == "full":
        aligned_axis = 2
        rotation_mask = [True, True, True]
    elif orientation_mode == "closing_y":
        aligned_axis = 1
        rotation_mask = [False, True, False]
    elif orientation_mode == "length_x":
        aligned_axis = 0
        rotation_mask = [True, False, False]
    elif orientation_mode == "tool_z":
        aligned_axis = 2
        rotation_mask = [False, False, True]
    else:
        raise ValueError(f"Unsupported IK orientation mode: {orientation_mode}")

    assist_enabled = bool(
        orientation_assist_request is True
        or (
            isinstance(orientation_assist_request, dict)
            and orientation_assist_request.get("enabled", True)
        )
    )
    assist_gain_deg_per_mm = -0.181379499
    assist_max_abs_angle_deg = 15.0
    if isinstance(orientation_assist_request, dict):
        assist_gain_deg_per_mm = float(
            orientation_assist_request.get("gain_deg_per_mm", assist_gain_deg_per_mm)
        )
        assist_max_abs_angle_deg = float(
            orientation_assist_request.get("max_abs_angle_deg", assist_max_abs_angle_deg)
        )

    import genesis as gs

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, substeps=1, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(Path(data.config["robot_urdf"]).expanduser().resolve()),
            pos=tuple(data.config["robot_base_pos_m"]),
            fixed=True,
            visualization=False,
            collision=True,
            convexify=False,
            decimate=False,
        ),
        material=gs.materials.Rigid(),
    )
    for joint in robot.joints:
        if joint.n_qs:
            joint._init_qpos = np.zeros(joint.n_qs, dtype=np.float64)
    scene.build()

    genesis_joint_names = tuple(joint.name for joint in robot.joints if joint.n_qs)
    if genesis_joint_names != SOURCE_JOINT_NAMES:
        raise RuntimeError(
            "Workbench replay joint order is not the current Genesis URDF order: "
            f"genesis={genesis_joint_names}"
        )
    index_by_name = {name: index for index, name in enumerate(genesis_joint_names)}
    arm_prefix = "left_joint1" if hand == "left" else "right_joint2"
    arm_indices = np.array(
        [index_by_name[f"{arm_prefix}{joint}"] for joint in range(1, 7)], dtype=np.int64
    )
    finger_indices = np.array(
        [index_by_name[name] for name in ("left_joint17", "left_joint18", "right_joint27", "right_joint28")],
        dtype=np.int64,
    )
    link_prefix = "left_link1" if hand == "left" else "right_link2"
    tcp_link = robot.get_link(name=f"{link_prefix}6")
    tool_links = [robot.get_link(name=f"{link_prefix}{suffix}") for suffix in (6, 7, 8)]
    lower, upper = robot.get_dofs_limit(dofs_idx_local=arm_indices)
    lower = as_numpy(lower).reshape(-1)
    upper = as_numpy(upper).reshape(-1)
    geom_names = {
        int(geom.idx): f"{geom.link.name}" for geom in robot.geoms
    }

    frames = np.array([item["frame"] for item in compiled["resolved_world_path"]], dtype=np.int64)
    targets = np.array([item["world_m"] for item in compiled["resolved_world_path"]], dtype=np.float64)
    replay_indices = np.array([data.source_to_index[int(frame)] for frame in frames], dtype=np.int64)
    baseline_q = data.robot_q[replay_indices]
    baseline_quat = []
    baseline_pairs = []
    baseline_clearances = []
    for qpos in baseline_q:
        robot.set_qpos(qpos, zero_velocity=True)
        baseline_quat.append(as_numpy(tcp_link.get_quat(relative=False)).reshape(4))
        baseline_pairs.append(pair_set(robot.detect_collision()))
        baseline_clearances.append(
            min(float(as_numpy(link.get_AABB()).reshape(2, 3)[0, 2] - 0.8) for link in tool_links)
        )
    baseline_quat = np.asarray(baseline_quat)
    baseline_arm_steps = np.zeros(len(frames), dtype=np.float64)
    if len(frames) > 1:
        baseline_arm_steps[1:] = np.max(
            np.abs(np.diff(baseline_q[:, arm_indices], axis=0)), axis=1
        )
    assist_angles_deg = np.zeros(len(frames), dtype=np.float64)
    target_quat_wxyz = baseline_quat.copy()
    effective_orientation_mode = orientation_mode
    if assist_enabled:
        base_positions = np.asarray(
            [item["actual_world_m"] for item in compiled["series"]], dtype=np.float64
        )
        correction_norm_mm = np.linalg.norm(targets - base_positions, axis=1) * 1000.0
        assist_angles_deg = np.clip(
            assist_gain_deg_per_mm * correction_norm_mm,
            -abs(assist_max_abs_angle_deg),
            abs(assist_max_abs_angle_deg),
        )
        target_quat_wxyz = np.asarray(
            [
                apply_local_y_rotation(quaternion, angle)
                for quaternion, angle in zip(baseline_quat, assist_angles_deg)
            ],
            dtype=np.float64,
        )
        effective_orientation_mode = "bounded_local_y"
        aligned_axis = 2
        rotation_mask = [True, True, True]

    previous_q = baseline_q[0].copy()
    resolved_q = []
    samples = []
    max_position_error = 0.0
    max_joint_step = 0.0
    max_raw_joint_step = 0.0
    max_joint_step_excess_vs_baseline = 0.0
    max_joint_step_over_allowed = 0.0
    slew_limited_frames = []
    min_limit_margin = float("inf")
    min_table_clearance = float("inf")
    min_table_clearance_delta = float("inf")
    introduced_pairs: set[tuple[str, str]] = set()
    first_position_failure_frame = None
    first_joint_step_failure_frame = None
    position_only_diagnostic = None
    worst_position_ordinal = None
    worst_target_pos = None
    worst_target_quat = None
    for ordinal, (source_frame, target_pos, target_quat, source_q) in enumerate(
        zip(frames, targets, target_quat_wxyz, baseline_q)
    ):
        seed = previous_q.copy()
        non_arm = np.ones(len(seed), dtype=bool)
        non_arm[arm_indices] = False
        seed[non_arm] = source_q[non_arm]
        solved, _ = robot.inverse_kinematics(
            link=tcp_link,
            pos=target_pos,
            quat=target_quat,
            local_point=TCP_LOCAL,
            init_qpos=seed,
            dofs_idx_local=arm_indices,
            max_samples=1,
            max_solver_iters=50,
            damping=0.05,
            max_step_size=0.1,
            pos_tol=1.0e-4,
            rot_tol=1.0e-4,
            rot_mask=rotation_mask,
            respect_joint_limit=True,
            return_error=True,
        )
        candidate = as_numpy(solved).reshape(-1)
        candidate[finger_indices] = source_q[finger_indices]
        raw_joint_step = float(
            np.max(np.abs(candidate[arm_indices] - previous_q[arm_indices]))
        )
        max_raw_joint_step = max(max_raw_joint_step, raw_joint_step)
        # Preserve an inherited source step at the same frame. Otherwise the
        # fixed 0.05 rad cap makes a zero-edit trajectory lag its own baseline.
        # The absolute step remains a separately reported hardware warning.
        allowed_joint_step = max(0.05, float(baseline_arm_steps[ordinal]) + 1.0e-6)
        if raw_joint_step > allowed_joint_step:
            scale = allowed_joint_step / raw_joint_step
            candidate[arm_indices] = previous_q[arm_indices] + scale * (
                candidate[arm_indices] - previous_q[arm_indices]
            )
            slew_limited_frames.append(int(source_frame))
        candidate[arm_indices] = np.clip(candidate[arm_indices], lower, upper)
        robot.set_qpos(candidate, zero_velocity=True)
        link_pos = as_numpy(tcp_link.get_pos(relative=False)).reshape(3)
        link_quat = as_numpy(tcp_link.get_quat(relative=False)).reshape(4)
        fk_tcp = link_pos + quat_to_matrix(link_quat) @ TCP_LOCAL
        position_error = float(np.linalg.norm(fk_tcp - target_pos))
        joint_step = float(np.max(np.abs(candidate[arm_indices] - previous_q[arm_indices])))
        joint_step_excess = max(
            0.0, joint_step - float(baseline_arm_steps[ordinal])
        )
        max_joint_step_excess_vs_baseline = max(
            max_joint_step_excess_vs_baseline, joint_step_excess
        )
        joint_step_over_allowed = max(0.0, joint_step - allowed_joint_step)
        max_joint_step_over_allowed = max(
            max_joint_step_over_allowed, joint_step_over_allowed
        )
        if position_error > max_position_error:
            max_position_error = position_error
            worst_position_ordinal = ordinal
            worst_target_pos = target_pos.copy()
            worst_target_quat = target_quat.copy()
        if position_error > 0.003 and first_position_failure_frame is None:
            first_position_failure_frame = int(source_frame)
        if joint_step > allowed_joint_step + 1.0e-9 and first_joint_step_failure_frame is None:
            first_joint_step_failure_frame = int(source_frame)
        max_joint_step = max(max_joint_step, joint_step)
        margin = float(
            np.min(
                np.minimum(
                    candidate[arm_indices] - lower,
                    upper - candidate[arm_indices],
                )
            )
        )
        tool_clearance = min(
            float(as_numpy(link.get_AABB()).reshape(2, 3)[0, 2] - 0.8) for link in tool_links
        )
        new_pairs = pair_set(robot.detect_collision()) - baseline_pairs[ordinal]
        for first, second in new_pairs:
            introduced_pairs.add((geom_names.get(first, str(first)), geom_names.get(second, str(second))))
        min_limit_margin = min(min_limit_margin, margin)
        min_table_clearance = min(min_table_clearance, tool_clearance)
        min_table_clearance_delta = min(
            min_table_clearance_delta,
            tool_clearance - baseline_clearances[ordinal],
        )
        samples.append(
            {
                "source_frame": int(source_frame),
                "position_error_mm": position_error * 1000.0,
                "fk_tcp_world_m": fk_tcp.tolist(),
                "max_arm_joint_step_rad": joint_step,
                "baseline_arm_joint_step_rad": float(baseline_arm_steps[ordinal]),
                "joint_step_excess_vs_baseline_rad": joint_step_excess,
                "allowed_arm_joint_step_rad": allowed_joint_step,
                "joint_step_over_allowed_rad": joint_step_over_allowed,
                "raw_solver_joint_step_rad": raw_joint_step,
                "arm_joint_limit_margin_rad": margin,
                "tool_table_clearance_mm": tool_clearance * 1000.0,
                "tool_table_clearance_delta_vs_baseline_mm": (
                    tool_clearance - baseline_clearances[ordinal]
                ) * 1000.0,
            }
        )
        resolved_q.append(candidate.copy())
        previous_q = candidate

    if worst_position_ordinal is not None and max_position_error > 0.003:
        ordinal = int(worst_position_ordinal)
        target_pos = np.asarray(worst_target_pos, dtype=np.float64)
        target_quat = np.asarray(worst_target_quat, dtype=np.float64)
        source_q = baseline_q[ordinal]
        try:
            position_only, _ = robot.inverse_kinematics(
                link=tcp_link,
                pos=target_pos,
                local_point=TCP_LOCAL,
                init_qpos=source_q,
                dofs_idx_local=arm_indices,
                max_samples=1,
                max_solver_iters=100,
                damping=0.05,
                max_step_size=0.1,
                pos_tol=1.0e-4,
                respect_joint_limit=True,
                return_error=True,
            )
            position_only_q = as_numpy(position_only).reshape(-1)
            position_only_q[finger_indices] = source_q[finger_indices]
            robot.set_qpos(position_only_q, zero_velocity=True)
            free_pos = as_numpy(tcp_link.get_pos(relative=False)).reshape(3)
            free_quat = as_numpy(tcp_link.get_quat(relative=False)).reshape(4)
            free_tcp = free_pos + quat_to_matrix(free_quat) @ TCP_LOCAL
            position_only_diagnostic = {
                "source_frame": int(frames[ordinal]),
                "position_error_mm": float(np.linalg.norm(free_tcp - target_pos) * 1000.0),
                "orientation_relaxation_deg": (
                    quaternion_distance_deg(free_quat, target_quat)
                    if effective_orientation_mode in ("full", "bounded_local_y")
                    else float(np.degrees(np.arccos(np.clip(np.dot(
                        quat_to_matrix(free_quat)[:, aligned_axis],
                        quat_to_matrix(target_quat)[:, aligned_axis],
                    ), -1.0, 1.0))))
                ),
            }
        except Exception as diagnostic_error:
            position_only_diagnostic = {
                "source_frame": int(frames[ordinal]),
                "error": str(diagnostic_error),
            }

    gates = {
        "position_error_mm": 3.0,
        "joint_step_rad": 0.05,
        "joint_step_over_allowed_rad": 1.0e-6,
        "joint_limit_margin_rad": -1.0e-6,
        "table_clearance_mm": 0.5,
        "table_clearance_warning_mm": 2.0,
        "table_clearance_delta_vs_baseline_warning_mm": -0.5,
        "introduced_collision_pairs": 0,
    }
    summary = {
        "source_frame_start": int(frames[0]),
        "source_frame_end": int(frames[-1]),
        "frames": int(len(frames)),
        "hand": hand,
        "orientation_mode": effective_orientation_mode,
        "source_orientation_mode": orientation_mode,
        "rotation_mask": rotation_mask,
        "orientation_assist": {
            "enabled": assist_enabled,
            "axis_local": [0.0, 1.0, 0.0],
            "gain_deg_per_mm": assist_gain_deg_per_mm,
            "max_abs_angle_deg": assist_max_abs_angle_deg,
            "resolved_peak_angle_deg": float(np.max(np.abs(assist_angles_deg), initial=0.0)),
        },
        "position_path_accepted": bool(compiled["accepted"]),
        "max_position_error_mm": max_position_error * 1000.0,
        "max_position_error_frame": (
            None if worst_position_ordinal is None else int(frames[worst_position_ordinal])
        ),
        "max_arm_joint_step_rad": max_joint_step,
        "max_raw_solver_joint_step_rad": max_raw_joint_step,
        "baseline_max_arm_joint_step_rad": float(
            np.max(baseline_arm_steps, initial=0.0)
        ),
        "max_joint_step_excess_vs_baseline_rad": max_joint_step_excess_vs_baseline,
        "max_joint_step_over_allowed_rad": max_joint_step_over_allowed,
        "slew_limited_frame_count": len(slew_limited_frames),
        "slew_limited_frames": slew_limited_frames,
        "min_arm_joint_limit_margin_rad": min_limit_margin,
        "min_tool_table_clearance_mm": min_table_clearance * 1000.0,
        "min_tool_table_clearance_delta_vs_baseline_mm": (
            min_table_clearance_delta * 1000.0
        ),
        "table_clearance_warning": bool(min_table_clearance * 1000.0 < gates["table_clearance_warning_mm"]),
        "inherited_hardware_timing_warning": bool(
            np.max(baseline_arm_steps, initial=0.0) > gates["joint_step_rad"]
        ),
        "absolute_joint_step_gate_pass": bool(
            max_joint_step <= gates["joint_step_rad"] + 1.0e-9
        ),
        "absolute_table_clearance_gate_pass": bool(
            min_table_clearance * 1000.0 >= gates["table_clearance_mm"]
        ),
        "introduced_collision_pairs": sorted([list(value) for value in introduced_pairs]),
        "first_position_failure_frame": first_position_failure_frame,
        "first_joint_step_failure_frame": first_joint_step_failure_frame,
        "position_only_diagnostic": position_only_diagnostic,
        "gates": gates,
    }
    summary["accepted"] = bool(
        summary["max_position_error_mm"] <= gates["position_error_mm"]
        and summary["max_joint_step_over_allowed_rad"]
        <= gates["joint_step_over_allowed_rad"]
        and summary["min_arm_joint_limit_margin_rad"] >= gates["joint_limit_margin_rad"]
        and summary["min_tool_table_clearance_delta_vs_baseline_mm"]
        >= gates["table_clearance_delta_vs_baseline_warning_mm"]
        and len(summary["introduced_collision_pairs"]) == 0
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "resolved_joint_q.npz",
        source_frames=frames.astype(np.int32),
        robot_q=np.asarray(resolved_q, dtype=np.float32),
        target_tcp_position=np.asarray(targets, dtype=np.float32),
        target_tcp_quat_wxyz=np.asarray(target_quat_wxyz, dtype=np.float32),
        orientation_assist_angle_deg=np.asarray(assist_angles_deg, dtype=np.float32),
        fk_tcp_position=np.asarray([sample["fk_tcp_world_m"] for sample in samples], dtype=np.float32),
    )
    report = {
        "config": str(args.config.expanduser().resolve()),
        "edits": requested_edits,
        "position_compile": {
            key: compiled[key]
            for key in (
                "baseline_metrics", "resolved_metrics", "limits", "accepted", "required_duration_scale"
            )
        },
        "ik_summary": summary,
        "samples": samples,
    }
    report_path = output_dir / "ik_preflight.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report={report_path}")
    print(f"resolved_joint_q={output_dir / 'resolved_joint_q.npz'}")
    if not summary["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
