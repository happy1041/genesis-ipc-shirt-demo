#!/usr/bin/env python3
"""CPU causal dual-X5 IK for an explicit, independently timed action plan.

Only an accepted plan produces robot_q.npz. Failure output is diagnostic, never
a runnable replay. This checks rigid kinematics, not cloth grasp success.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.trajectory_workbench.ik_preflight import (
    SOURCE_JOINT_NAMES, TCP_LOCAL, as_numpy, pair_set, quat_to_matrix,
    quaternion_distance_deg,
)


def read_inputs(compiled: dict, seed_path: Path) -> tuple[dict, np.ndarray]:
    if compiled.get("accepted") is False:
        raise ValueError("Compiled action plan failed its trajectory gates")
    frames = np.asarray(compiled["source_frames"])
    if frames.ndim != 1 or not len(frames) or not np.all(np.isfinite(frames)):
        raise ValueError("source_frames must be a nonempty finite vector")
    if not np.all(frames == frames.astype(np.int64)) or np.any(np.diff(frames) != 1):
        raise ValueError("source_frames must be consecutive integer simulation frames")
    arrays = {"source_frames": frames.astype(np.int64)}
    for hand in ("left", "right"):
        for suffix, shape in (("pos", (len(frames), 3)), ("quat", (len(frames), 4)),
                              ("opening", (len(frames),))):
            key = f"{hand}_{suffix}"
            value = np.asarray(compiled[key], dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{key} must have finite shape {shape}")
            if suffix == "quat":
                norms = np.linalg.norm(value, axis=1)
                if np.any(np.abs(norms - 1) > 1e-3):
                    raise ValueError(f"{key} requires normalized WXYZ quaternions")
                value = value / norms[:, None]
            if suffix == "opening" and np.any((value < 0) | (value > .044)):
                raise ValueError(f"{key} outside per-finger range 0..0.044 m")
            arrays[key] = value
    with np.load(seed_path, allow_pickle=False) as seed:
        q = np.asarray(seed["robot_q"], dtype=np.float64)
    if q.shape == (1, 19):
        q = q[0]
    if q.shape != (19,) or not np.all(np.isfinite(q)):
        raise ValueError("seed robot_q must be one finite 19-joint state")
    for hand, fingers in (("left", [9, 10]), ("right", [17, 18])):
        # The physical checkpoint can contain micron-scale asymmetric finger
        # positions, while the editor represents a symmetric per-finger command.
        # Validate that command against the measured mean without changing the
        # exact seed; do not mistake normal solver residuals for a new opening.
        if np.ptp(q[fingers]) > 2e-5 or abs(float(np.mean(q[fingers])) - arrays[f"{hand}_opening"][0]) > 2e-5:
            raise ValueError(f"First {hand} opening must match checkpoint finger mean within 20 um; initial finger asymmetry must also be <=20 um")
    return arrays, q.copy()


def solve(config: dict, arrays: dict, initial: np.ndarray) -> tuple[dict, np.ndarray]:
    import genesis as gs

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 60, substeps=1),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True),
        show_viewer=False, show_FPS=False,
    )
    robot = scene.add_entity(morph=gs.morphs.URDF(
        file=str(Path(config["robot_urdf"]).expanduser().resolve()),
        pos=tuple(config["robot_base_pos_m"]), fixed=True, visualization=False,
        collision=True, convexify=False, decimate=False,
    ), material=gs.materials.Rigid())
    for joint in robot.joints:
        if joint.n_qs:
            joint._init_qpos = np.zeros(joint.n_qs)
    scene.build()
    if tuple(j.name for j in robot.joints if j.n_qs) != SOURCE_JOINT_NAMES:
        raise ValueError("Current URDF joint order does not match 19-joint X5 replay")
    arms = {"left": np.arange(3, 9), "right": np.arange(11, 17)}
    fingers = {"left": [9, 10], "right": [17, 18]}
    links = {h: robot.get_link(name=f"{h}_link{p}6") for h, p in (("left", 1), ("right", 2))}
    tools = [robot.get_link(name=f"{h}_link{p}{s}")
             for h, p in (("left", 1), ("right", 2)) for s in (6, 7, 8)]
    table_z = float(config.get("action_plan_ik", {}).get("table_z_m", .8))
    settings = config.get("action_plan_ik", {})
    absolute_floor = float(settings.get("min_tool_table_clearance_m", -.0035))
    relative_floor = float(settings.get("min_clearance_delta_m", -.0035))
    limits_lo, limits_hi = (as_numpy(x).reshape(-1) for x in robot.get_dofs_limit())
    def clearance():
        return min(float(as_numpy(link.get_AABB()).reshape(2, 3)[0, 2] - table_z) for link in tools)
    robot.set_qpos(initial, zero_velocity=True)
    baseline_pairs = pair_set(robot.detect_collision())
    initial_clearance = clearance()
    geom_names = {int(g.idx): g.link.name for g in robot.geoms}
    samples, trajectory = [], []
    previous = initial.copy()
    introduced = set()
    for ordinal, frame in enumerate(arrays["source_frames"]):
        q = previous.copy()
        if ordinal:
            # Only copy the solved arm's DOFs: the other arm and root remain causal.
            for hand in ("left", "right"):
                robot.set_qpos(q, zero_velocity=True)
                solved, _ = robot.inverse_kinematics(
                    link=links[hand], pos=arrays[f"{hand}_pos"][ordinal],
                    quat=arrays[f"{hand}_quat"][ordinal], local_point=TCP_LOCAL,
                    init_qpos=q, dofs_idx_local=arms[hand], max_samples=1,
                    max_solver_iters=80, damping=.05, max_step_size=.1,
                    pos_tol=1e-4, rot_tol=1e-4, rot_mask=[True, True, True],
                    respect_joint_limit=True, return_error=True,
                )
                candidate = as_numpy(solved).reshape(-1)
                q[arms[hand]] = candidate[arms[hand]]
                q[fingers[hand]] = arrays[f"{hand}_opening"][ordinal]
        if not np.all(np.isfinite(q)):
            raise ValueError(f"IK returned non-finite joints at source frame {frame}")
        robot.set_qpos(q, zero_velocity=True)
        sample = {"source_frame": int(frame)}
        for hand in ("left", "right"):
            quat = as_numpy(links[hand].get_quat(relative=False)).reshape(4)
            tcp = as_numpy(links[hand].get_pos(relative=False)).reshape(3) + quat_to_matrix(quat) @ TCP_LOCAL
            sample[f"{hand}_position_error_m"] = float(np.linalg.norm(tcp - arrays[f"{hand}_pos"][ordinal]))
            sample[f"{hand}_orientation_error_deg"] = quaternion_distance_deg(quat.copy(), arrays[f"{hand}_quat"][ordinal].copy())
            sample[f"{hand}_fk_tcp_world_m"] = tcp.tolist()
        sample["arm_step_rad"] = float(max(np.max(np.abs(q[idx] - previous[idx])) for idx in arms.values()))
        sample["joint_limit_margin"] = float(np.min(np.minimum(q - limits_lo, limits_hi - q)))
        sample["tool_table_clearance_m"] = clearance()
        for hand in arms:
            sample[f"{hand}_tool_table_clearance_m"] = min(
                float(as_numpy(link.get_AABB()).reshape(2,3)[0,2]-table_z)
                for link in tools if link.name.startswith(hand+'_'))
        # Allow normal descent from a hovering seed, but not worsened inherited
        # table penetration. The absolute floor remains a separate hard gate.
        sample["clearance_delta_m"] = sample["tool_table_clearance_m"] - min(initial_clearance, 0.)
        pairs = pair_set(robot.detect_collision()) - baseline_pairs
        introduced.update(pairs)
        sample["new_collision_pairs"] = [list(p) for p in sorted(pairs)]
        samples.append(sample)
        trajectory.append(q.copy())
        previous = q
    gates = {
        "position": all(s[f"{h}_position_error_m"] <= .005 for s in samples for h in arms),
        "orientation": all(s[f"{h}_orientation_error_deg"] <= 2 for s in samples for h in arms),
        "joint_step": all(s["arm_step_rad"] <= .05 + 1e-9 for s in samples),
        "joint_limits": all(s["joint_limit_margin"] >= -1e-6 for s in samples),
        "no_new_self_collision": not introduced,
        "table_clearance": all(s["tool_table_clearance_m"] >= absolute_floor and s["clearance_delta_m"] >= relative_floor for s in samples),
        "initial_seed_exact": bool(np.array_equal(trajectory[0], initial)),
    }
    result = {
        "accepted": all(gates.values()), "gates": gates, "samples": samples,
        "backend": "genesis_cpu", "orientation": "full_wxyz", "frames": len(samples),
        "limits": {"position_error_m": .005, "orientation_error_deg": 2,
                   "arm_step_rad": .05, "table_clearance_m": absolute_floor,
                   "clearance_delta_m": relative_floor},
        "initial_tool_table_clearance_m": initial_clearance,
        "clearance_reference_m": min(initial_clearance, 0.),
        "inherited_collision_pairs": [[geom_names.get(a, str(a)), geom_names.get(b, str(b))] for a, b in sorted(baseline_pairs)],
        "warnings": (["Seed has inherited collision pairs; these are exempted from the new-pair gate. This preflight does not certify their penetration depth or safety."] if baseline_pairs else []),
        "new_collision_pairs": [[geom_names.get(a, str(a)), geom_names.get(b, str(b))] for a, b in sorted(introduced)],
        "scope": "Rigid kinematic preflight only; does not establish cloth grasp success.",
    }
    return result, np.asarray(trajectory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "compiled", "seed", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        config = json.loads(args.config.read_text())
        compiled = json.loads(args.compiled.read_text())
        arrays, seed = read_inputs(compiled, args.seed)
        result, q = solve(config, arrays, seed)
        name = "robot_q.npz" if result["accepted"] else "diagnostic_rejected_robot_q.npz"
        np.savez_compressed(args.output_dir / name, robot_q=q,
                            source_frames=arrays["source_frames"],
                            joint_names=np.asarray(SOURCE_JOINT_NAMES),
                            accepted=np.asarray(result["accepted"]))
        result["trajectory_file"] = name
    except Exception as error:
        result = {"accepted": False, "error": f"{type(error).__name__}: {error}"}
    result["inputs"] = {k: str(getattr(args, k).resolve()) for k in ("config", "compiled", "seed")}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"accepted": result["accepted"], "result": str(args.output_dir / "result.json")}))
    if not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
