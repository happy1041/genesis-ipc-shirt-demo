"""Prepare explicit-action physics runs and bind completed stages to Workbench.

No GPU work is started here. A prepared run is intentionally not a successful
physics/IK result. Old fold checkpoints are never relabelled as action states.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT
JOINT_NAMES = (
    "joint1", "joint2", "joint3",
    *[f"left_joint1{i}" for i in range(1, 9)],
    *[f"right_joint2{i}" for i in range(1, 9)],
)
FORMAT = "genesis-ipc-action-plan-prefix-v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prefix_hash(joint_q, openness):
    digest = hashlib.sha256()
    for values in (joint_q, openness):
        digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def checkpoint_meta(checkpoint):
    path = Path(checkpoint).resolve().with_suffix(".pkl")
    for suffix in (".pkl", ".ipc_state.npz", ".meta.json"):
        if not path.with_suffix(suffix).is_file():
            raise ValueError(f"Missing checkpoint component: {path.with_suffix(suffix)}")
    meta = read_json(path.with_suffix(".meta.json"))
    if meta.get("signature", {}).get("format") != FORMAT:
        raise ValueError("Only checkpoints produced by explicit action-plan mode can continue this chain")
    return path, meta


def export_workbench_bundle(output_dir, config_path, joint_path, checkpoint=None, initial_position=None, visualize_proxies=False, contact_dump_frames=None, replay_closeup_hand='right', replay_six_view=False, no_shadows=False):
    """Export safe, manually launchable run; robot_q includes the shared initial row.

    Config ``action_manifest`` binds a continuation to its *completed* parent.
    q is SOURCE_JOINT_NAMES order, in metres/radians, including four fingers.
    """
    output = Path(output_dir).resolve()
    if replay_closeup_hand not in ('left','right'):raise ValueError('Invalid replay closeup hand')
    config_path, joint_path = Path(config_path).resolve(), Path(joint_path).resolve()
    config = read_json(config_path)
    report = read_json(joint_path.parent / "result.json")
    if report.get("accepted") is not True or report.get("trajectory_file") != joint_path.name:
        raise ValueError("Export requires the accepted IK trajectory artifact")
    compiled = read_json(report["inputs"]["compiled"])
    if compiled.get("accepted") is not True:
        raise ValueError("Export requires accepted PATH preflight")
    if Path(report["inputs"]["config"]).resolve() != config_path:
        raise ValueError("IK result was solved against another Workbench config")
    with np.load(joint_path, allow_pickle=False) as data:
        if "accepted" not in data or not bool(np.asarray(data["accepted"]).item()):
            raise ValueError("Rejected/unvalidated joint file cannot be exported")
        q = np.asarray(data["robot_q"], dtype=np.float64)
        frames = np.asarray(data["source_frames"], dtype=np.int64)
        if not np.array_equal(frames, np.arange(int(compiled["start_frame"]), int(compiled["start_frame"]) + len(q))):
            raise ValueError("IK source_frames must be contiguous and start at the compiled plan anchor")
        if "joint_names" in data and tuple(data["joint_names"].tolist()) != JOINT_NAMES:
            raise ValueError("Resolved joint order does not match the native SIM1/X5 order")
        fps = float(compiled["fps"])
    if q.ndim != 2 or q.shape[1] != 19 or len(q) < 2 or not np.all(np.isfinite(q)):
        raise ValueError("Need finite (T>=2,19) robot_q including an initial anchor")
    if not 1 <= fps <= 240:
        raise ValueError("action_fps must be in [1,240]")
    fingers = q[:, [9, 10, 17, 18]]
    if np.any(fingers < -1e-9) or np.any(fingers > .044 + 1e-9):
        raise ValueError("X5 finger targets must remain within [0,.044] metres")
    openness = np.clip(np.column_stack((fingers[:, :2].mean(1), fingers[:, 2:].mean(1))) / .044, 0, 1)
    initial_q = q[0].copy()
    start = 0
    parent_path = config.get("action_manifest")
    parent = read_json(parent_path) if parent_path else None
    physics = dict(parent["physics"] if parent else {
        "action_fps": fps, "cloth_bending": 40., "cloth_E": 20000.,
        "cloth_self_friction": 2., "initial_shirt_x": .667,
        "initial_shirt_y": .015, "initial_shirt_z": .93,
    })
    if initial_position is not None:
        if parent or checkpoint:
            raise ValueError('Cannot translate an existing cloth checkpoint')
        if set(initial_position) != {'initial_shirt_x', 'initial_shirt_y', 'initial_shirt_z'}:
            raise ValueError('Initial position requires exactly initial_shirt_x/y/z')
        if not all(np.isfinite(float(v)) for v in initial_position.values()):
            raise ValueError('Initial position must be finite')
        physics.update({k:float(v) for k,v in initial_position.items()})
    if fps != physics["action_fps"]:
        raise ValueError("Continuation action_fps must match the parent physics timestep")
    pair_override = config.get('cloth_table_pair_friction_override')
    if pair_override is not None:
        if not np.isfinite(pair_override) or pair_override < 0:
            raise ValueError('cloth/table pair override must be finite and nonnegative')
        physics['cloth_table_pair_friction'] = float(pair_override)
    if parent:
        if checkpoint and Path(checkpoint).resolve() != Path(parent["checkpoint"]).resolve():
            raise ValueError("Checkpoint must match the config's parent stage")
        checkpoint, meta = checkpoint_meta(checkpoint or parent["checkpoint"])
        if not np.allclose(initial_q, meta["previous_q"], rtol=0, atol=2e-7):
            raise ValueError("New stage initial robot_q differs from checkpoint previous_q")
        with np.load(parent["trajectory"], allow_pickle=False) as data:
            prefix_q, prefix_open = data["joint_q"], data["openness"]
        start = int(meta["source_frame"]) + 1
        if len(prefix_q) != start or int(meta["trajectory_index"]) != start - 1:
            raise ValueError("Parent checkpoint is not at the terminal native trajectory frame")
        if prefix_hash(prefix_q, prefix_open) != meta["signature"]["action_prefix_sha256"]:
            raise ValueError("Parent trajectory prefix has changed since checkpoint creation")
        q = np.concatenate((prefix_q, q[1:]))
        openness = np.concatenate((prefix_open, openness[1:]))
    elif checkpoint:
        raise ValueError("Checkpoint continuation requires imported config.action_manifest")
    if int(compiled["start_frame"]) != (start - 1 if parent else 0):
        raise ValueError("Stage must start at frame 0 or the imported terminal checkpoint frame")
    for key in ("cloth_obj", "robot_urdf"):
        if not Path(config[key]).is_file():
            raise ValueError(f"Missing {key}: {config[key]}")
    output.mkdir(parents=True, exist_ok=False)
    trajectory = output / "trajectory.npz"
    np.savez_compressed(trajectory, joint_q=q, openness=openness, joint_names=JOINT_NAMES)
    checkpoint_out = output / "checkpoint.pkl"
    replay = output / "stage.replay_states.npz"
    metrics = output / "stage.json"
    # Load the same physical preset as the established 8k baseline; the new
    # runner mode explicitly bypasses its historical frame-specific corrections.
    argv = ["bash", "scripts/run_demo.sh", "--physics-only", "--action-plan-trajectory",
            "--frames", str(len(q)), "--action-fps", str(fps), "--viewer",
            "--save-checkpoint-source-frame", str(len(q) - 1),
            "--save-third-fold-checkpoint", str(checkpoint_out)]
    for key in ('expected_cloth_vertices','expected_cloth_faces'):
        if key in config:argv += ['--'+key.replace('_','-'),str(int(config[key]))]
    for key, value in physics.items():
        if key != "action_fps":
            argv += ["--" + key.replace("_", "-"), str(value)]
    if visualize_proxies:
        argv.append('--visualize-ipc-proxies')
    if no_shadows:
        argv.append('--no-shadows')
    if pair_override is not None:
        argv.append('--allow-material-change-on-checkpoint')
    if contact_dump_frames:
        argv += ['--contact-dump-frames'] + [str(int(frame)) for frame in contact_dump_frames]
    if checkpoint:
        argv += ["--load-third-fold-checkpoint", str(checkpoint), "--allow-checkpoint-path-relocation"]
    render_argv = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in ("--save-checkpoint-source-frame", "--save-third-fold-checkpoint", "--load-third-fold-checkpoint"):
            skip_next = True
        elif token == "--physics-only":
            render_argv.append("--render-only")
        elif token != "--viewer":
            render_argv.append(token)
    if replay_closeup_hand!='right':render_argv+=['--replay-closeup-hand',replay_closeup_hand]
    if replay_six_view:render_argv.append('--replay-six-view')
    env = {"TRAJECTORY": str(trajectory), "SHIRT_OBJ": config["cloth_obj"],
           "ROBOT_URDF": config["robot_urdf"], "OUTPUT_ROOT": str(output),
           "RUN_LABEL": "stage", "DEMO_CONFIG": str(PROJECT / "configs/dhat_1p5_pure_friction.args")}
    # run_demo adds _physics to outputs; record exactly those generated names.
    replay = output / "stage_physics.replay_states.npz"
    metrics = output / "stage_physics.json"
    manifest = {
        "schema": "workbench-action-run-v1", "status": "PREPARED_NOT_RUN",
        "config": str(config_path), "joint_input": str(joint_path),
        "trajectory": str(trajectory), "checkpoint": str(checkpoint_out),
        "replay": str(replay), "metrics": str(metrics),
        "tcp_csv": str(output / "stage_physics_tcp_trajectory.csv"),
        "video": str(output / "stage_overhead.mp4"),
        "frame_range": [start, len(q) - 1], "frames_to_execute": len(q) - start,
        "parent_manifest": str(parent_path) if parent else None,
        "physics": physics, "cwd": str(PROJECT), "argv": argv, "render_argv": render_argv, "env": env,
        "replay_closeup_hand": replay_closeup_hand,
        "replay_six_view": replay_six_view,
        "source_hashes": {str(p): sha(p) for p in (config_path, joint_path, Path(report["inputs"]["compiled"]), joint_path.parent / "result.json", PROJECT / "src/run_genesis_ipc.py", Path(__file__).resolve())},
        "notes": ["No simulation launched by export", "Viewer enabled; keep desktop session open", "Checkpoint continuation rejects changed command prefixes", "No legacy fold IK or staged gripper events applied"],
    }
    write_json(output / "manifest.json", manifest)
    command = "env " + " ".join(shlex.quote(f"{k}={v}") for k, v in env.items()) + " " + shlex.join(argv)
    render_command = "env " + " ".join(shlex.quote(f"{k}={v}") for k, v in env.items()) + " " + shlex.join(render_argv)
    guard = "\n".join(
        "if [[ -e " + shlex.quote(str(path)) + " ]]; then echo 'Run output already exists; export a new bundle' >&2; exit 2; fi"
        for path in (output / "physics.log", metrics, replay, checkpoint_out)
    )
    (output / "run.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(PROJECT)) + "\n" + guard + "\n"
        + command + " 2>&1 | tee " + shlex.quote(str(output / "physics.log")) + "\n"
        + render_command + " 2>&1 | tee " + shlex.quote(str(output / "render.log")) + "\n",
        encoding="utf-8",
    )
    return manifest


def import_completed_stage(base_config_path, stage_manifest_path, output_path):
    """Bind actual replay/terminal checkpoint for editing the next stage."""
    manifest_path = Path(stage_manifest_path).resolve()
    manifest = read_json(manifest_path)
    config = copy.deepcopy(read_json(base_config_path))
    original = read_json(manifest["config"])
    if any(Path(config[k]).resolve() != Path(original[k]).resolve() for k in ("robot_urdf", "cloth_obj")):
        raise ValueError("Import config cloth and robot must match the completed stage")
    checkpoint, meta = checkpoint_meta(manifest["checkpoint"])
    last = int(manifest["frame_range"][1])
    metrics = read_json(manifest["metrics"])
    if int(metrics.get("executed_frames", -1)) != manifest["frames_to_execute"]:
        raise ValueError("Stage physics did not execute the full exported range")
    if int(meta["source_frame"]) != last:
        raise ValueError("Terminal checkpoint frame does not match the stage")
    with np.load(manifest["trajectory"], allow_pickle=False) as trajectory:
        if prefix_hash(trajectory["joint_q"], trajectory["openness"]) != meta["signature"]["action_prefix_sha256"]:
            raise ValueError("Completed trajectory does not match checkpoint prefix hash")
        if not np.allclose(trajectory['joint_q'][-1], meta['previous_q'], atol=2e-7, rtol=0):
            raise ValueError('Checkpoint command differs from terminal trajectory command')
    with np.load(manifest["replay"], allow_pickle=False) as replay:
        frames = np.asarray(replay["source_frames"])
        expected = np.arange(manifest["frame_range"][0], last + 1)
        if not np.array_equal(frames, expected) or not np.all(np.isfinite(replay["cloth_pos"])):
            raise ValueError("Completed replay is incomplete or non-finite")
        if not np.isfinite(replay['robot_q']).all():
            raise ValueError('Replay robot state is non-finite')
        # Replay is measured rigid state; previous_q is the commanded target.
        # Contact can produce a small difference. Never replace the saved
        # physical state with targets, nor require measurement == command.
        command_tracking_error = (replay['robot_q'][-1] - np.asarray(meta['previous_q'])).tolist()
    if not Path(manifest["tcp_csv"]).is_file():
        raise ValueError("Missing TCP telemetry for completed stage")
    if config.get('background_mode')!='replay_cpu' and not Path(manifest["video"]).is_file():
        raise ValueError("Render the stage overhead video before importing into Workbench")
    first = int(frames[0])
    config.update({"name": config.get("name", "Workbench") + " · next action stage",
                   "replay": manifest["replay"], "tcp_csvs": [manifest["tcp_csv"]],
                   "default_frame": last, "display_frame_range": [int(frames[0]), last],
                   "video_segments": [{"source_range": [first, last], "video": manifest["video"], "video_frame_start": 0}],
                   "action_manifest": str(manifest_path),
                   "planning_command_trajectory": manifest['trajectory'],
                   "action_checkpoint": str(checkpoint), "start_checkpoint": str(checkpoint),
                   "checkpoint_measured_minus_commanded_q": command_tracking_error,
                   "active_segment": "action_continuation"})
    config.pop("checkpoint_catalog", None)
    config.pop("action_plan", None)
    if config.get('regrasp_edit_binding'):
        binding=config['regrasp_edit_binding']
        binding['frame']=last
        binding.pop('placement_defaults',None)
        binding.pop('reference_markers',None)
    config["trajectory_editor"] = {
        "stage": "action_continuation", "label": "上一动作末态 → 下一动作", "default_hand": "left",
        "frame_range": [first, last], "limits": {"max_step_mm": 12.5, "max_speed_mps": .75, "max_accel_mps2": 20.},
        "phases": [{"id": "completed", "label": "已完成动作", "range": [first, last], "color": "#357d83"}],
        "events": [{"frame": last, "label": "新动作起点 / checkpoint", "kind": "boundary"}],
        "hands": {hand: {"label": hand, "color": color, "material_contact_frame": last,
                   "nodes": [{"frame": first, "name": "completed start", "draggable": False, "support_radius": 2},
                             {"frame": last, "name": "checkpoint / next start", "draggable": False, "support_radius": 2}],
                   "semantic_path": [{"frame": first, "zero": True}, {"frame": last, "zero": True}]}
                  for hand, color in (("left", "#48d58a"), ("right", "#ff6b78"))},
        "material_anchors": [],
    }
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--config", required=True)
    export.add_argument("--joints", required=True)
    export.add_argument("--output", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("--config", required=True)
    imp.add_argument("--manifest", required=True)
    imp.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "export":
        result = export_workbench_bundle(args.output, args.config, args.joints)
    else:
        result = import_completed_stage(args.config, args.manifest, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
