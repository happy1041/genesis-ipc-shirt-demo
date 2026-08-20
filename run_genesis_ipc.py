#!/usr/bin/env python3
"""Replay a SIM1 Acone trajectory against the same shirt mesh using Genesis IPC."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import genesis as gs
import genesis.utils.geom as gu
import trimesh
import uipc
from PIL import Image, ImageDraw

from genesis.engine.couplers.ipc_coupler.utils import (
    compute_link_to_link_transform,
    find_target_link_for_fixed_merge,
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
ARM_JOINT_NAMES = tuple(f"left_joint1{i}" for i in range(1, 7)) + tuple(
    f"right_joint2{i}" for i in range(1, 7)
)
FINGER_JOINT_NAMES = (
    ("left_joint17", "left_joint18"),
    ("right_joint27", "right_joint28"),
)
FINGER_JOINT_NAME_SET = frozenset(name for pair in FINGER_JOINT_NAMES for name in pair)
FINGER_URDF_LOWER = 0.0
FINGER_URDF_UPPER = 0.044
IPC_ROBOT_LINKS = (
    "left_link15",
    "left_link16",
    "left_link17",
    "left_link18",
    "right_link25",
    "right_link26",
    "right_link27",
    "right_link28",
)
TCP_LOCAL = np.array([0.155, 0.001786, 0.014], dtype=np.float64)
EPISODE0_FIRST_GRASP_CANDIDATES = np.array(
    [6275, 3846, 88, 3564, 2038, 3813, 6010, 6442, 5855, 2583, 887, 5410], dtype=np.int64
)


def as_numpy(value) -> np.ndarray:
    """Convert a Genesis tensor (or ndarray) to a detached NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def quat_wxyz_to_matrix(quat) -> np.ndarray:
    w, x, y, z = as_numpy(quat).reshape(4)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(matrix) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized wxyz quaternion."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ]
        )
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat = np.array(
                [
                    (m[2, 1] - m[1, 2]) / s,
                    0.25 * s,
                    (m[0, 1] + m[1, 0]) / s,
                    (m[0, 2] + m[2, 0]) / s,
                ]
            )
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat = np.array(
                [
                    (m[0, 2] - m[2, 0]) / s,
                    (m[0, 1] + m[1, 0]) / s,
                    0.25 * s,
                    (m[1, 2] + m[2, 1]) / s,
                ]
            )
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat = np.array(
                [
                    (m[1, 0] - m[0, 1]) / s,
                    (m[0, 2] + m[2, 0]) / s,
                    (m[1, 2] + m[2, 1]) / s,
                    0.25 * s,
                ]
            )
    return quat / np.linalg.norm(quat)


def slerp_quat_wxyz(start, end, weight: float) -> np.ndarray:
    """Shortest-path spherical interpolation between normalized quaternions."""
    q0 = np.asarray(start, dtype=np.float64).reshape(4)
    q1 = np.asarray(end, dtype=np.float64).reshape(4)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        mixed = q0 + float(weight) * (q1 - q0)
        return mixed / np.linalg.norm(mixed)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    return (
        np.sin((1.0 - float(weight)) * angle) / sin_angle * q0
        + np.sin(float(weight) * angle) / sin_angle * q1
    )


def table_parallel_quat(quat) -> np.ndarray:
    """Level the TCP length axis while retaining the gripper closing direction."""
    raw_rot = quat_wxyz_to_matrix(quat)
    closing_y = raw_rot[:, 1].copy()
    closing_y[2] = 0.0
    norm = np.linalg.norm(closing_y)
    if norm < 1.0e-8:
        length_x = raw_rot[:, 0].copy()
        length_x[2] = 0.0
        length_x /= np.linalg.norm(length_x)
        closing_y = np.cross(np.array([0.0, 0.0, 1.0]), length_x)
    else:
        closing_y /= norm
    world_z = np.array([0.0, 0.0, 1.0])
    length_x = np.cross(closing_y, world_z)
    length_x /= np.linalg.norm(length_x)
    leveled_rot = np.column_stack((length_x, closing_y, world_z))
    return matrix_to_quat_wxyz(leveled_rot)


def rotate_quat_about_world_x(quat, angle_rad: float) -> np.ndarray:
    """Rotate a TCP pose only inside the robot-front (world Y-Z) plane."""
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    world_x_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )
    return matrix_to_quat_wxyz(
        world_x_rotation @ quat_wxyz_to_matrix(quat)
    )


class IPCVirtualGrasp:
    """Attach a small cloth patch to each SIM1 TCP using libuipc SPCs."""

    def __init__(
        self,
        robot,
        cloth,
        openness,
        radius: float,
        points: int,
        final_right_points: int,
        first_candidates: tuple[np.ndarray, np.ndarray] | None,
        mode: str,
    ):
        self.cloth = cloth
        self.openness = openness
        self.radius = radius
        self.mode = mode
        self.coupler = cloth.sim.coupler
        self.hands = [
            {
                "name": "left",
                "link": robot.get_link(name="left_link16"),
                "counts": (points, points),
                "closed": False,
                "grasp_index": 0,
                "ids": np.empty(0, dtype=np.int64),
                "local": np.empty((0, 3), dtype=np.float64),
                "targets": np.empty((0, 3), dtype=np.float64),
                "attach_center": None,
                "first_candidates": first_candidates[0] if first_candidates is not None else None,
            },
            {
                "name": "right",
                "link": robot.get_link(name="right_link26"),
                "counts": (points, points, final_right_points),
                "closed": False,
                "grasp_index": 0,
                "ids": np.empty(0, dtype=np.int64),
                "local": np.empty((0, 3), dtype=np.float64),
                "targets": np.empty((0, 3), dtype=np.float64),
                "attach_center": None,
                "first_candidates": first_candidates[1] if first_candidates is not None else None,
            },
        ]

    @staticmethod
    def _link_pose(hand) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = as_numpy(hand["link"].get_pos(relative=False)).reshape(3).astype(np.float64)
        rot = quat_wxyz_to_matrix(hand["link"].get_quat(relative=False))
        tcp = pos + rot @ TCP_LOCAL
        return pos, rot, tcp

    def _cloth_positions(self) -> np.ndarray:
        pos = as_numpy(self.cloth.get_state().pos).astype(np.float64)
        return pos.reshape((-1, 3))

    def _release(self, hand, frame: int) -> None:
        if len(hand["ids"]):
            if self.mode == "hard":
                self.coupler.clear_fem_vertex_hard_bindings(self.cloth, hand["ids"])
            else:
                self.coupler.clear_fem_vertex_constraints(self.cloth, hand["ids"])
            print(f"grasp_release frame={frame} hand={hand['name']} points={len(hand['ids'])}")
        hand["ids"] = np.empty(0, dtype=np.int64)
        hand["local"] = np.empty((0, 3), dtype=np.float64)
        hand["targets"] = np.empty((0, 3), dtype=np.float64)
        hand["attach_center"] = None

    def _attach(self, hand, frame: int, cloth_pos: np.ndarray) -> None:
        pos, rot, tcp = self._link_pose(hand)
        distances = np.linalg.norm(cloth_pos - tcp, axis=1)
        if hand["grasp_index"] == 0 and hand["first_candidates"] is not None:
            candidate_mask = np.ones(len(distances), dtype=bool)
            candidate_mask[hand["first_candidates"]] = False
            distances[candidate_mask] = np.inf
        held_by_other = np.concatenate(
            [other["ids"] for other in self.hands if other is not hand and len(other["ids"])]
        ) if any(other is not hand and len(other["ids"]) for other in self.hands) else np.empty(0, dtype=np.int64)
        distances[held_by_other] = np.inf

        grasp_index = hand["grasp_index"]
        counts = hand["counts"]
        count = counts[min(grasp_index, len(counts) - 1)]
        nearest = np.argsort(distances)[:count]
        nearest_distance = float(distances[nearest[0]])
        if not np.isfinite(nearest_distance) or nearest_distance > self.radius:
            print(
                f"grasp_miss frame={frame} hand={hand['name']} "
                f"nearest={nearest_distance:.4f}m radius={self.radius:.4f}m"
            )
            hand["grasp_index"] += 1
            return

        # Only retain vertices inside the capture radius. The initial target is
        # their current position, avoiding a one-frame snap to the TCP center.
        nearest = nearest[distances[nearest] <= self.radius]
        selected_distance = float(distances[nearest].max())
        selected = cloth_pos[nearest]
        hand["ids"] = nearest.astype(np.int64)
        hand["local"] = (selected - pos) @ rot
        hand["targets"] = selected.copy()
        hand["attach_center"] = selected.mean(axis=0)
        if self.mode == "hard":
            self.coupler.set_fem_vertex_hard_bindings(self.cloth, hand["ids"], selected)
        else:
            self.coupler.set_fem_vertex_constraints(self.cloth, hand["ids"], selected)
        print(
            f"grasp_attach frame={frame} hand={hand['name']} points={len(nearest)} "
            f"nearest={nearest_distance:.4f}m selected_max={selected_distance:.4f}m "
            f"ids={nearest.tolist()}"
        )
        hand["grasp_index"] += 1

    def update(self, frame: int, source_frame: int | None = None) -> None:
        frame_label = frame if source_frame is None else source_frame
        cloth_pos = None
        for hand_index, hand in enumerate(self.hands):
            value = float(self.openness[frame, hand_index])
            if hand["closed"] and value >= 0.60:
                self._release(hand, frame_label)
                hand["closed"] = False
            elif not hand["closed"] and value <= 0.50:
                hand["closed"] = True
                if cloth_pos is None:
                    cloth_pos = self._cloth_positions()
                self._attach(hand, frame_label, cloth_pos)

        # Targets use the current kinematic link pose and are consumed by the
        # IPC animator in the immediately following scene.step().
        for hand in self.hands:
            if len(hand["ids"]):
                pos, rot, _ = self._link_pose(hand)
                targets = pos + hand["local"] @ rot.T
                hand["targets"] = targets
                if self.mode == "hard":
                    self.coupler.set_fem_vertex_hard_bindings(self.cloth, hand["ids"], targets)
                else:
                    self.coupler.set_fem_vertex_constraints(self.cloth, hand["ids"], targets)

    def report_tracking(self, source_frame: int) -> None:
        held = [hand for hand in self.hands if len(hand["ids"])]
        if not held:
            return
        cloth_pos = self._cloth_positions()
        for hand in held:
            actual = cloth_pos[hand["ids"]]
            error = np.linalg.norm(actual - hand["targets"], axis=1)
            target_travel = np.linalg.norm(hand["targets"].mean(axis=0) - hand["attach_center"])
            actual_travel = np.linalg.norm(actual.mean(axis=0) - hand["attach_center"])
            print(
                f"grasp_track frame={source_frame} hand={hand['name']} "
                f"target_travel={target_travel:.4f}m actual_travel={actual_travel:.4f}m "
                f"mean_error={error.mean():.4f}m max_error={error.max():.4f}m"
            )

    def summary(self) -> dict:
        return {
            hand["name"]: {"grasp_events": hand["grasp_index"], "still_held": int(len(hand["ids"]))}
            for hand in self.hands
        }


class ContactGraspDiagnostics:
    """Track each close/lift/release event without applying attachment forces.

    A nearby cloth vertex at the instant of closing is not evidence of a grasp.
    Each event therefore follows the same cloth patch until release and compares
    its displacement with the TCP displacement.  This produces an explicit
    PASS/FAIL verdict that complements (but never replaces) rendered keyframes.
    """

    def __init__(
        self,
        robot,
        cloth,
        openness,
        finger_indices: tuple[np.ndarray, np.ndarray],
        expected_candidates: tuple[np.ndarray, np.ndarray] | None = None,
        radius: float = 0.06,
        points: int = 12,
    ):
        self.cloth = cloth
        self.openness = openness
        self.radius = radius
        self.points = points
        self.hands = [
            {
                "name": "left",
                "link": robot.get_link(name="left_link16"),
                "finger_indices": finger_indices[0],
                "expected_ids": None if expected_candidates is None else expected_candidates[0],
                "closed": False,
                "ids": np.empty(0, dtype=np.int64),
                "event": None,
                "events": [],
            },
            {
                "name": "right",
                "link": robot.get_link(name="right_link26"),
                "finger_indices": finger_indices[1],
                "expected_ids": None if expected_candidates is None else expected_candidates[1],
                "closed": False,
                "ids": np.empty(0, dtype=np.int64),
                "event": None,
                "events": [],
            },
        ]

    @staticmethod
    def _tcp(hand) -> np.ndarray:
        pos = as_numpy(hand["link"].get_pos(relative=False)).reshape(3).astype(np.float64)
        rot = quat_wxyz_to_matrix(hand["link"].get_quat(relative=False))
        return pos + rot @ TCP_LOCAL

    def _cloth_positions(self) -> np.ndarray:
        return as_numpy(self.cloth.get_state().pos).astype(np.float64).reshape((-1, 3))

    @staticmethod
    def _event_measurement(hand, cloth_pos: np.ndarray, tcp: np.ndarray) -> dict:
        event = hand["event"]
        patch = cloth_pos[hand["ids"]].mean(axis=0)
        patch_delta = patch - event["initial_patch"]
        tcp_delta = tcp - event["initial_tcp"]
        tcp_travel = float(np.linalg.norm(tcp_delta))
        patch_travel = float(np.linalg.norm(patch_delta))
        if tcp_travel > 1.0e-6:
            follow_projection = float(np.dot(patch_delta, tcp_delta) / (tcp_travel**2))
            follow_cosine = float(
                np.dot(patch_delta, tcp_delta) / max(patch_travel * tcp_travel, 1.0e-9)
            )
        else:
            follow_projection = 0.0
            follow_cosine = 0.0
        return {
            "frame": None,
            "tcp_travel_m": tcp_travel,
            "patch_travel_m": patch_travel,
            "patch_lift_m": float(patch_delta[2]),
            "tcp_lift_m": float(tcp_delta[2]),
            "patch_tcp_distance_m": float(np.linalg.norm(patch - tcp)),
            "patch_tcp_distance_growth_m": float(
                np.linalg.norm(patch - tcp) - event["initial_patch_tcp_distance_m"]
            ),
            "follow_projection": follow_projection,
            "follow_cosine": follow_cosine,
        }

    @staticmethod
    def _verdict(event: dict) -> tuple[str, str]:
        samples = [sample for sample in event["samples"] if sample["tcp_travel_m"] >= 0.02]
        if not event["tracked_points"]:
            return "FAIL", "no cloth vertex was close enough at close"
        if not samples:
            return "INCONCLUSIVE", "TCP did not travel 20 mm before release/end"
        sample = max(samples, key=lambda value: value["tcp_travel_m"])
        follows = sample["follow_projection"] >= 0.55 and sample["follow_cosine"] >= 0.65
        stays_near = sample["patch_tcp_distance_growth_m"] <= 0.035
        if follows and stays_near:
            return "PASS", "cloth patch co-moved with the lifting TCP"
        return (
            "FAIL",
            "cloth patch did not co-move with TCP "
            f"(projection={sample['follow_projection']:.2f}, "
            f"cos={sample['follow_cosine']:.2f}, "
            f"separation_growth={sample['patch_tcp_distance_growth_m'] * 1000:.1f} mm)",
        )

    def _finish_event(self, hand, source_frame: int) -> None:
        event = hand["event"]
        if event is None:
            return
        event["end_frame"] = int(source_frame)
        event["verdict"], event["reason"] = self._verdict(event)
        hand["events"].append(event)
        print(
            f"grasp_verdict hand={hand['name']} event={len(hand['events'])} "
            f"close_frame={event['close_frame']} verdict={event['verdict']} "
            f"reason={event['reason']}"
        )
        hand["event"] = None
        hand["ids"] = np.empty(0, dtype=np.int64)

    def update(self, frame: int, source_frame: int, robot, finger_targets: np.ndarray) -> None:
        cloth_pos = None
        robot_q = as_numpy(robot.get_qpos()).reshape(-1)
        for hand_index, hand in enumerate(self.hands):
            openness = float(self.openness[frame, hand_index])
            if source_frame % 30 == 0:
                q_actual = robot_q[hand["finger_indices"]]
                q_target = finger_targets[hand["finger_indices"]]
                print(
                    f"finger_track frame={source_frame} hand={hand['name']} openness={openness:.3f} "
                    f"q_actual={q_actual.tolist()} q_target={q_target.tolist()}"
                )
            if not hand["closed"] and openness <= 0.50:
                hand["closed"] = True
                if cloth_pos is None:
                    cloth_pos = self._cloth_positions()
                tcp = self._tcp(hand)
                distances = np.linalg.norm(cloth_pos - tcp, axis=1)
                ids = np.argsort(distances)[: self.points]
                ids = ids[distances[ids] <= self.radius]
                hand["ids"] = ids.astype(np.int64)
                initial_patch = cloth_pos[ids].mean(axis=0) if len(ids) else None
                hand["event"] = {
                    "close_frame": int(source_frame),
                    "end_frame": None,
                    "tracked_points": int(len(ids)),
                    "initial_patch": initial_patch,
                    "initial_tcp": tcp,
                    "initial_patch_tcp_distance_m": (
                        float(np.linalg.norm(initial_patch - tcp)) if initial_patch is not None else None
                    ),
                    "samples": [],
                    "verdict": "INCONCLUSIVE",
                    "reason": "run ended before verification",
                }
                expected_text = ""
                # These public IDs describe only episode 0's first grasp.  Reusing
                # them for later folds produced a plausible-looking but meaningless
                # distance in old logs, so never print them after event 1.
                if (
                    not hand["events"]
                    and hand["expected_ids"] is not None
                    and len(hand["expected_ids"])
                ):
                    expected = cloth_pos[hand["expected_ids"]]
                    expected_center = expected.mean(axis=0)
                    expected_text = (
                        f" expected_center={expected_center.tolist()}"
                        f" expected_tcp_distance={float(np.linalg.norm(expected_center - tcp)):.4f}m"
                    )
                print(
                    f"contact_probe frame={source_frame} hand={hand['name']} points={len(ids)} "
                    f"nearest={float(distances.min()):.4f}m ids={ids.tolist()}"
                    f"{expected_text}"
                )
            elif hand["closed"] and openness >= 0.60:
                hand["closed"] = False
                self._finish_event(hand, source_frame)

            if hand["event"] is None or hand["event"]["initial_patch"] is None or not len(hand["ids"]):
                continue
            if cloth_pos is None:
                cloth_pos = self._cloth_positions()
            tcp = self._tcp(hand)
            sample = self._event_measurement(hand, cloth_pos, tcp)
            sample["frame"] = int(source_frame)
            hand["event"]["samples"].append(sample)
            if source_frame % 30 == 0:
                print(
                    f"contact_track frame={source_frame} hand={hand['name']} "
                    f"patch_lift={sample['patch_lift_m']:.4f}m "
                    f"tcp_lift={sample['tcp_lift_m']:.4f}m "
                    f"patch_tcp_distance={sample['patch_tcp_distance_m']:.4f}m "
                    f"follow_projection={sample['follow_projection']:.3f}"
                )

    def summary(self) -> dict:
        # Finalize still-closed events so truncated second-fold runs get a verdict.
        for hand in self.hands:
            if hand["event"] is not None:
                last_frame = hand["event"]["samples"][-1]["frame"] if hand["event"]["samples"] else -1
                self._finish_event(hand, last_frame)
        return {
            hand["name"]: {
                "events": [
                    {
                        key: value
                        for key, value in event.items()
                        if key not in {"initial_patch", "initial_tcp", "samples"}
                    }
                    | {
                        "verification_sample": (
                            max(event["samples"], key=lambda sample: sample["tcp_travel_m"])
                            if event["samples"]
                            else None
                        )
                    }
                    for event in hand["events"]
                ]
            }
            for hand in self.hands
        }


class IPCProxyVisualizer:
    """Overlay the exact rigid meshes and transforms used by libuipc.

    The IPC coupler builds one affine body per non-fixed robot link.  Its mesh
    is the union of the Genesis collision geoms expressed in that link's local
    frame, including fixed-joint children.  Reconstructing that same union and
    driving it with ``_abd_data_by_link`` makes the debug overlay represent the
    collision body used by IPC, rather than the separate visual mesh.
    """

    COLOR_RGBA = np.array((255, 76, 24, 105), dtype=np.uint8)
    TABLE_TOP_Z = 0.8
    # A shallow soft-constraint target below the contact surface is intentional:
    # it creates the normal force needed to pinch cloth against the table.  The
    # actual IPC proxy remains subject to the much stricter collision gate.
    TARGET_TABLE_PENETRATION_TOLERANCE_M = 0.0035

    def __init__(
        self,
        scene,
        robot,
        cloth,
        coupler,
        enabled: bool,
        output_dir: Path | None,
    ):
        self.scene = scene
        self.robot = robot
        self.cloth = cloth
        self.coupler = coupler
        self.enabled = bool(enabled)
        self.output_dir = None if output_dir is None else output_dir.expanduser().resolve()
        self.links = []
        self.nodes = []
        self.local_vertices = []
        self.local_meshes = []
        self.local_edges = []
        self.current_transforms = []
        self.samples: dict[int, dict[str, dict]] = {}
        self.running_summary = {
            "min_proxy_table_clearance_m": float("inf"),
            "min_target_table_clearance_m": float("inf"),
            "max_proxy_target_translation_error_m": 0.0,
            "max_proxy_target_rotation_error_deg": 0.0,
            "min_proxy_table_clearance_at": None,
            "min_target_table_clearance_at": None,
            "max_proxy_target_translation_error_at": None,
            "max_proxy_target_rotation_error_at": None,
            "proxy_below_table_frames": 0,
            "target_below_table_frames": 0,
            "sampled_frames": 0,
        }

    @staticmethod
    def _link_transform(link) -> np.ndarray:
        pos = as_numpy(link.get_pos(relative=False)).reshape(3)
        quat = as_numpy(link.get_quat(relative=False)).reshape(4)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = quat_wxyz_to_matrix(quat)
        transform[:3, 3] = pos
        return transform

    @staticmethod
    def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
        relative = first[:3, :3].T @ second[:3, :3]
        cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def initialize(self) -> None:
        if not self.enabled:
            return
        coupled_sources = []
        for name in IPC_ROBOT_LINKS:
            link = self.robot.get_link(name=name)
            if link in self.coupler._abd_slots_by_link:
                coupled_sources.append(link)

        target_groups = {}
        for source_link in coupled_sources:
            target = find_target_link_for_fixed_merge(source_link)
            target_groups.setdefault(target, []).append(source_link)

        for target_link, source_links in target_groups.items():
            vertices = []
            faces = []
            vertex_offset = 0
            for source_link in source_links:
                for geom in source_link.geoms:
                    if geom.type == gs.GEOM_TYPE.PLANE or not geom.n_verts:
                        continue
                    geom_vertices = gu.transform_by_trans_quat(
                        geom.init_verts, geom.init_pos, geom.init_quat
                    )
                    if source_link is not target_link:
                        geom_vertices = gu.transform_by_trans_quat(
                            geom_vertices,
                            *compute_link_to_link_transform(source_link, target_link),
                        )
                    vertices.append(np.asarray(geom_vertices, dtype=np.float64))
                    geom_faces = np.asarray(geom.init_faces, dtype=np.int64)
                    faces.append(geom_faces + vertex_offset)
                    vertex_offset += len(geom_vertices)
            if not vertices:
                continue
            proxy_mesh = trimesh.Trimesh(
                vertices=np.concatenate(vertices, axis=0),
                faces=np.concatenate(faces, axis=0),
                process=False,
            )
            proxy_mesh.visual.face_colors = np.tile(
                self.COLOR_RGBA, (len(proxy_mesh.faces), 1)
            )
            ipc_entries = self.coupler._abd_data_by_link.get(target_link)
            initial_transform = (
                np.asarray(ipc_entries[0].transform, dtype=np.float64).reshape(4, 4)
                if ipc_entries
                else self._link_transform(target_link)
            )
            # Before the first IPC retrieval the entry is initialized to I.
            # Use the built Genesis pose for this one render-only instant.
            if np.allclose(initial_transform, np.eye(4), atol=1.0e-10):
                initial_transform = self._link_transform(target_link)
            node = self.scene.draw_debug_mesh(proxy_mesh, T=initial_transform)
            self.links.append(target_link)
            self.nodes.append(node)
            self.local_vertices.append(np.asarray(proxy_mesh.vertices, dtype=np.float64))
            self.local_meshes.append(proxy_mesh.copy())
            # A decimated wireframe is much clearer than a translucent solid in
            # the offline diagnostic cameras, and keeps PNG generation cheap.
            edges = np.asarray(proxy_mesh.edges_unique, dtype=np.int64)
            if len(edges) > 400:
                edges = edges[np.linspace(0, len(edges) - 1, 400, dtype=np.int64)]
            self.local_edges.append(edges)
            self.current_transforms.append(initial_transform)
        print(
            "ipc_proxy_visualization initialized "
            f"links={[link.name for link in self.links]} color=orange"
        )

    def update(self, source_frame: int | None = None, record: bool = False) -> dict:
        if not self.enabled or not self.nodes:
            return {}
        ipc_transforms = []
        metrics = {}
        for link, local_vertices, local_mesh in zip(
            self.links, self.local_vertices, self.local_meshes
        ):
            entries = self.coupler._abd_data_by_link.get(link)
            ipc_transform = (
                np.asarray(entries[0].transform, dtype=np.float64).reshape(4, 4).copy()
                if entries
                else self._link_transform(link)
            )
            genesis_transform = self._link_transform(link)
            ipc_transforms.append(ipc_transform)
            world_vertices = (
                ipc_transform[:3, :3] @ local_vertices.T
            ).T + ipc_transform[:3, 3]
            target_world_vertices = (
                genesis_transform[:3, :3] @ local_vertices.T
            ).T + genesis_transform[:3, 3]
            table_clearances = world_vertices[:, 2] - self.TABLE_TOP_Z
            target_table_clearances = target_world_vertices[:, 2] - self.TABLE_TOP_Z
            metrics[link.name] = {
                "translation_error_m": float(
                    np.linalg.norm(ipc_transform[:3, 3] - genesis_transform[:3, 3])
                ),
                "rotation_error_deg": self._rotation_error_deg(
                    ipc_transform, genesis_transform
                ),
                "min_proxy_world_z_m": float(np.min(world_vertices[:, 2])),
                "table_clearance_m": float(np.min(table_clearances)),
                "below_table_vertex_fraction": float(np.mean(table_clearances < 0.0)),
                "target_min_proxy_world_z_m": float(
                    np.min(target_world_vertices[:, 2])
                ),
                "target_table_clearance_m": float(np.min(target_table_clearances)),
                "target_below_table_vertex_fraction": float(
                    np.mean(target_table_clearances < 0.0)
                ),
                "ipc_transform": ipc_transform.tolist(),
                "genesis_transform": genesis_transform.tolist(),
            }
            if record:
                # IPC collision is three-dimensional; a camera-space overlap is
                # not proof of penetration.  Evaluate the cloth vertices in the
                # exact local frame of each affine collision body.  trimesh uses
                # positive signed distance for points inside a watertight mesh.
                cloth_world = as_numpy(self.cloth.get_state().pos).astype(
                    np.float64
                ).reshape((-1, 3))
                cloth_local = (
                    cloth_world - ipc_transform[:3, 3]
                ) @ ipc_transform[:3, :3]
                bounds = np.asarray(local_mesh.bounds, dtype=np.float64)
                # A point inside a closed mesh must also be inside its AABB.
                # Prefiltering avoids thousands of expensive ray tests against
                # robot links on the opposite side of the shirt.
                aabb_candidate = np.all(
                    (cloth_local >= bounds[0] - 1.0e-3)
                    & (cloth_local <= bounds[1] + 1.0e-3),
                    axis=1,
                )
                signed_distance = np.full(len(cloth_local), -np.inf, dtype=np.float64)
                if np.any(aabb_candidate):
                    signed_distance[aabb_candidate] = trimesh.proximity.signed_distance(
                        local_mesh, cloth_local[aabb_candidate]
                    )
                inside_depth = np.maximum(signed_distance, 0.0)
                metrics[link.name].update(
                    {
                        "proxy_mesh_watertight": bool(local_mesh.is_watertight),
                        "cloth_vertices_inside_count": int(
                            np.count_nonzero(inside_depth > 0.0)
                        ),
                        "cloth_vertices_inside_over_0_2mm_count": int(
                            np.count_nonzero(inside_depth > 2.0e-4)
                        ),
                        "max_cloth_vertex_inside_depth_m": float(
                            np.max(inside_depth, initial=0.0)
                        ),
                    }
                )
        summary = self.running_summary
        minimum_proxy_clearance = min(
            value["table_clearance_m"] for value in metrics.values()
        )
        minimum_target_clearance = min(
            value["target_table_clearance_m"] for value in metrics.values()
        )
        proxy_clearance_link = min(metrics, key=lambda name: metrics[name]["table_clearance_m"])
        target_clearance_link = min(
            metrics, key=lambda name: metrics[name]["target_table_clearance_m"]
        )
        translation_link = max(
            metrics, key=lambda name: metrics[name]["translation_error_m"]
        )
        rotation_link = max(metrics, key=lambda name: metrics[name]["rotation_error_deg"])
        if minimum_proxy_clearance < summary["min_proxy_table_clearance_m"]:
            summary["min_proxy_table_clearance_m"] = minimum_proxy_clearance
            summary["min_proxy_table_clearance_at"] = {
                "frame": source_frame,
                "link": proxy_clearance_link,
            }
        if minimum_target_clearance < summary["min_target_table_clearance_m"]:
            summary["min_target_table_clearance_m"] = minimum_target_clearance
            summary["min_target_table_clearance_at"] = {
                "frame": source_frame,
                "link": target_clearance_link,
            }
        maximum_translation = metrics[translation_link]["translation_error_m"]
        if maximum_translation > summary["max_proxy_target_translation_error_m"]:
            summary["max_proxy_target_translation_error_m"] = maximum_translation
            summary["max_proxy_target_translation_error_at"] = {
                "frame": source_frame,
                "link": translation_link,
            }
        maximum_rotation = metrics[rotation_link]["rotation_error_deg"]
        if maximum_rotation > summary["max_proxy_target_rotation_error_deg"]:
            summary["max_proxy_target_rotation_error_deg"] = maximum_rotation
            summary["max_proxy_target_rotation_error_at"] = {
                "frame": source_frame,
                "link": rotation_link,
            }
        summary["proxy_below_table_frames"] += int(minimum_proxy_clearance < -2.0e-4)
        summary["target_below_table_frames"] += int(minimum_target_clearance < 0.0)
        summary["sampled_frames"] += 1
        self.scene.update_debug_objects(tuple(self.nodes), tuple(ipc_transforms))
        self.current_transforms = ipc_transforms
        if record and source_frame is not None:
            self.samples[int(source_frame)] = metrics
            worst_translation = max(
                value["translation_error_m"] for value in metrics.values()
            )
            worst_rotation = max(value["rotation_error_deg"] for value in metrics.values())
            minimum_clearance = min(value["table_clearance_m"] for value in metrics.values())
            print(
                f"ipc_proxy_pose frame={source_frame} "
                f"worst_translation_mm={worst_translation * 1000.0:.3f} "
                f"worst_rotation_deg={worst_rotation:.3f} "
                f"min_table_clearance_mm={minimum_clearance * 1000.0:.3f}"
            )
        return metrics

    def draw_camera_overlay(
        self,
        image: Image.Image,
        camera_pos,
        camera_lookat,
        camera_up,
        vertical_fov_deg: float = 42.0,
    ) -> None:
        """Project IPC proxy edges into an offline camera image."""
        if not self.enabled or not self.current_transforms:
            return
        width, height = image.size
        position = np.asarray(camera_pos, dtype=np.float64)
        forward = np.asarray(camera_lookat, dtype=np.float64) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(camera_up, dtype=np.float64))
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        focal = height / (2.0 * np.tan(np.deg2rad(vertical_fov_deg) * 0.5))
        draw = ImageDraw.Draw(image, "RGBA")
        for link, vertices, edges, transform in zip(
            self.links,
            self.local_vertices,
            self.local_edges,
            self.current_transforms,
        ):
            world = (
                transform[:3, :3] @ vertices.T
            ).T + transform[:3, 3]
            relative = world - position
            depth = relative @ forward
            projected = np.empty((len(world), 2), dtype=np.float64)
            projected[:, 0] = width * 0.5 + focal * (relative @ right) / np.maximum(depth, 1.0e-8)
            projected[:, 1] = height * 0.5 - focal * (relative @ true_up) / np.maximum(depth, 1.0e-8)
            for first, second in edges:
                if depth[first] <= 0.02 or depth[second] <= 0.02:
                    continue
                p0 = projected[first]
                p1 = projected[second]
                if (
                    max(p0[0], p1[0]) < 0
                    or min(p0[0], p1[0]) >= width
                    or max(p0[1], p1[1]) < 0
                    or min(p0[1], p1[1]) >= height
                ):
                    continue
                draw.line(
                    (float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])),
                    fill=(255, 78, 24, 190),
                    width=1,
                )

    def finish(self) -> Path | None:
        if not self.enabled or self.output_dir is None or not self.samples:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "ipc_proxy_pose_errors.json"
        cloth_inside_over_tolerance = sum(
            value.get("cloth_vertices_inside_over_0_2mm_count", 0)
            for frame in self.samples.values()
            for value in frame.values()
        )
        gate = {
            "proxy_table_clearance_pass": (
                self.running_summary["min_proxy_table_clearance_m"] >= -2.0e-4
            ),
            "target_table_clearance_pass": (
                self.running_summary["min_target_table_clearance_m"]
                >= -self.TARGET_TABLE_PENETRATION_TOLERANCE_M
            ),
            "proxy_target_pose_pass": (
                self.running_summary["max_proxy_target_translation_error_m"] <= 5.0e-3
                and self.running_summary["max_proxy_target_rotation_error_deg"] <= 2.0
            ),
            "cloth_proxy_vertex_penetration_pass": cloth_inside_over_tolerance == 0,
        }
        gate["pass"] = all(gate.values())
        path.write_text(
            json.dumps(
                {
                    "overlay": "orange meshes are IPC affine-body collision proxies",
                    "running_summary": self.running_summary,
                    "cloth_inside_over_0_2mm_total": cloth_inside_over_tolerance,
                    "gate": gate,
                    "frames": self.samples,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"IPC proxy pose diagnostics: {path}")
        print("ipc_collision_gate " + json.dumps(gate, sort_keys=True))
        return path


class IPCActualVisualSynchronizer:
    """Render coupled robot visuals at the actual IPC affine-body poses."""

    def __init__(self, robot, coupler, enabled: bool):
        self.robot = robot
        self.coupler = coupler
        self.enabled = bool(enabled)
        self.entries = []

    def initialize(self) -> None:
        if not self.enabled:
            return
        for name in IPC_ROBOT_LINKS:
            source_link = self.robot.get_link(name=name)
            target_link = find_target_link_for_fixed_merge(source_link)
            if target_link not in self.coupler._abd_slots_by_link:
                continue
            source_to_target = np.eye(4, dtype=np.float64)
            if source_link is not target_link:
                merge_pos, merge_quat = compute_link_to_link_transform(
                    source_link, target_link
                )
                source_to_target[:3, :3] = quat_wxyz_to_matrix(merge_quat)
                source_to_target[:3, 3] = np.asarray(merge_pos, dtype=np.float64)
            for vgeom in source_link.vgeoms:
                visual_origin = np.eye(4, dtype=np.float64)
                visual_origin[:3, :3] = quat_wxyz_to_matrix(vgeom.init_quat)
                visual_origin[:3, 3] = np.asarray(vgeom.init_pos, dtype=np.float64)
                self.entries.append(
                    (target_link, vgeom, source_to_target @ visual_origin)
                )
        if not self.entries:
            raise RuntimeError(
                "--render-ipc-actual-visuals found no coupled robot visual geometry"
            )
        print(
            "ipc_actual_visual_sync initialized "
            f"vgeoms={len(self.entries)} "
            f"links={sorted({target.name for target, _, _ in self.entries})}"
        )

    def update_render_transforms(self) -> None:
        if not self.enabled:
            return
        render_transforms = self.robot._solver._vgeoms_render_T
        for target_link, vgeom, target_from_visual in self.entries:
            ipc_entries = self.coupler._abd_data_by_link.get(target_link)
            transform = (
                np.asarray(ipc_entries[0].transform, dtype=np.float64).reshape(4, 4)
                if ipc_entries
                else IPCProxyVisualizer._link_transform(target_link)
            )
            if np.allclose(transform, np.eye(4), atol=1.0e-10):
                transform = IPCProxyVisualizer._link_transform(target_link)
            render_transforms[vgeom.idx, 0] = (
                transform @ target_from_visual
            ).astype(np.float32)


class KeyframeVisualDiagnostics:
    """Automatically render mandatory multi-view grasp keyframes and contact sheets."""

    # Capture every fold from approach through release.  Previously the first
    # fold had no mandatory images, so a run could never become a fully audited
    # G2 checkpoint even when its contact logs looked healthy.
    FIRST_FOLD_FRAMES = (0, 45, 54, 90, 117, 150, 210, 270, 300, 332, 340)
    # Include the complete second-fold lifecycle.  The old sheet stopped at
    # frame 480 and could prove that both hands had grasped, but it could not
    # reveal the much larger geometric failure at placement: the two side
    # flaps were landing next to one another instead of on the same footprint.
    SECOND_FOLD_FRAMES = (
        332, 385, 393, 430, 439, 455, 480, 505, 530, 550, 570, 583, 600, 619
    )
    THIRD_FOLD_FRAMES = (
        650, 680, 683, 686, 690, 700, 720, 750, 780, 840, 900, 920, 941, 960, 979
    )
    DIAGNOSTIC_FRAMES = frozenset(
        FIRST_FOLD_FRAMES + SECOND_FOLD_FRAMES + THIRD_FOLD_FRAMES
    )

    def __init__(self, scene, output_dir: Path | None):
        self.output_dir = None if output_dir is None else output_dir.expanduser().resolve()
        self.camera = None
        self.images: dict[int, dict[str, Path]] = {}
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.camera = scene.add_camera(
                res=(800, 600),
                pos=(1.50, -1.35, 1.55),
                lookat=(0.68, 0.0, 0.82),
                fov=42,
                GUI=False,
            )

    @staticmethod
    def _rgb_image(rgb) -> Image.Image:
        array = as_numpy(rgb)
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
        return Image.fromarray(array[..., :3])

    def capture(
        self,
        source_frame: int,
        right_tcp: np.ndarray,
        ipc_proxy_visuals: IPCProxyVisualizer | None = None,
    ) -> None:
        if self.camera is None or source_frame not in self.DIAGNOSTIC_FRAMES:
            return
        views = {
            "overview": ((1.50, -1.35, 1.55), (0.68, 0.0, 0.82), (0.0, 0.0, 1.0)),
            "overhead": ((1.48, 0.0, 2.45), (0.68, 0.0, 0.80), (0.0, 1.0, 0.0)),
            "shirt_bottom": ((1.55, -0.05, 1.04), (0.68, 0.0, 0.82), (0.0, 0.0, 1.0)),
            "right_grasp": (
                tuple(right_tcp + np.array((0.26, -0.32, 0.18))),
                tuple(right_tcp),
                (0.0, 0.0, 1.0),
            ),
        }
        frame_images = {}
        for view_name, (pos, lookat, up) in views.items():
            self.camera.set_pose(pos=pos, lookat=lookat, up=up)
            rgb = self.camera.render(
                rgb=True,
                depth=False,
                segmentation=False,
                normal=False,
                force_render=True,
            )
            if isinstance(rgb, tuple):
                rgb = rgb[0]
            image = self._rgb_image(rgb)
            if ipc_proxy_visuals is not None:
                ipc_proxy_visuals.draw_camera_overlay(image, pos, lookat, up)
            draw = ImageDraw.Draw(image)
            label_width = 470 if ipc_proxy_visuals is not None else 250
            draw.rectangle((0, 0, label_width, 30), fill=(0, 0, 0))
            draw.text((8, 8), f"frame {source_frame} | {view_name}", fill=(255, 255, 255))
            if ipc_proxy_visuals is not None:
                draw.text((245, 8), "orange = IPC collision proxy", fill=(255, 110, 50))
            path = self.output_dir / f"frame_{source_frame:04d}_{view_name}.png"
            image.save(path)
            frame_images[view_name] = path
        self.images[source_frame] = frame_images
        print(f"keyframe_visual frame={source_frame} views={list(frame_images)}")

    def finish(self) -> Path | None:
        if self.output_dir is None or not self.images:
            return None
        ordered_views = ("overview", "overhead", "shirt_bottom", "right_grasp")
        thumb_size = (400, 300)

        def make_sheet(name: str, requested_frames: tuple[int, ...]) -> tuple[Path | None, list[int]]:
            ordered_frames = [frame for frame in requested_frames if frame in self.images]
            if not ordered_frames:
                return None, []
            sheet = Image.new(
                "RGB",
                (thumb_size[0] * len(ordered_views), thumb_size[1] * len(ordered_frames)),
                "black",
            )
            for row, frame in enumerate(ordered_frames):
                for column, view in enumerate(ordered_views):
                    with Image.open(self.images[frame][view]) as image:
                        sheet.paste(
                            image.resize(thumb_size),
                            (column * thumb_size[0], row * thumb_size[1]),
                        )
            path = self.output_dir / f"{name}_contact_sheet.png"
            sheet.save(path)
            return path, ordered_frames

        first_path, first_frames = make_sheet("first_fold", self.FIRST_FOLD_FRAMES)
        second_path, second_frames = make_sheet("second_fold", self.SECOND_FOLD_FRAMES)
        third_path, third_frames = make_sheet("third_fold", self.THIRD_FOLD_FRAMES)
        index = {
            "purpose": (
                "Mandatory visual verification of every fold's approach, "
                "grasp, transport, placement, release and retreat"
            ),
            "mesh_warning": "Do not compare grasp validity across different mesh resolutions.",
            "first_fold_frames": first_frames,
            "second_fold_frames": second_frames,
            "third_fold_frames": third_frames,
            "views": list(ordered_views),
            "first_fold_contact_sheet": str(first_path) if first_path else None,
            "second_fold_contact_sheet": str(second_path) if second_path else None,
            "third_fold_contact_sheet": str(third_path) if third_path else None,
        }
        (self.output_dir / "visual_index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"First-fold contact sheet: {first_path}")
        print(f"Second-fold contact sheet: {second_path}")
        print(f"Third-fold contact sheet: {third_path}")
        return third_path or second_path or first_path


def summarize_second_fold_motion(debug_dir: Path | None) -> dict | None:
    """Separate intended flap transport from unintended stationary-body drag."""
    if debug_dir is None:
        return None
    frames = (0, 393, 439, 480, 505, 530, 550, 570, 583)
    paths = {frame: debug_dir / f"frame_{frame:04d}.npz" for frame in frames}
    if not all(path.is_file() for path in paths.values()):
        return None
    states = {frame: np.load(path) for frame, path in paths.items()}
    initial = np.asarray(states[0]["cloth_pos"])
    # The public task folds the initial low-X and high-X outer panels onto the
    # middle panel.  The initial middle 30% is therefore the table-supported
    # body that should not follow the second-fold grippers horizontally.
    low, high = np.quantile(initial[:, 0], (0.35, 0.65))
    stationary_mask = (initial[:, 0] > low) & (initial[:, 0] < high)
    # A stricter inner-core diagnostic is useful when the fold boundary itself
    # is intentionally pulled into the stack.  Keep the wider mask as the
    # verdict guardrail, but also report this core so a valid fold is not
    # confused with whole-shirt translation.
    deep_low, deep_high = np.quantile(initial[:, 0], (0.45, 0.55))
    y_low, y_high = np.quantile(initial[:, 1], (0.30, 0.70))
    deep_core_mask = (
        (initial[:, 0] > deep_low)
        & (initial[:, 0] < deep_high)
        & (initial[:, 1] > y_low)
        & (initial[:, 1] < y_high)
    )
    intervals = {}
    for start, end in zip(frames[1:-1], frames[2:]):
        displacement = np.asarray(states[end]["cloth_pos"]) - np.asarray(
            states[start]["cloth_pos"]
        )
        horizontal = np.linalg.norm(displacement[stationary_mask, :2], axis=1)
        deep_core_horizontal = np.linalg.norm(
            displacement[deep_core_mask, :2], axis=1
        )
        intervals[f"{start}_to_{end}"] = {
            "stationary_body_median_xy_m": float(np.median(horizontal)),
            "stationary_body_p90_xy_m": float(np.quantile(horizontal, 0.90)),
            "deep_core_median_xy_m": float(np.median(deep_core_horizontal)),
            "deep_core_p90_xy_m": float(np.quantile(deep_core_horizontal, 0.90)),
            "all_vertices_centroid_xy_m": np.mean(displacement[:, :2], axis=0).tolist(),
        }
    displacement = np.asarray(states[583]["cloth_pos"]) - np.asarray(
        states[393]["cloth_pos"]
    )
    horizontal = np.linalg.norm(displacement[stationary_mask, :2], axis=1)
    deep_core_horizontal = np.linalg.norm(displacement[deep_core_mask, :2], axis=1)
    intervals["393_to_583"] = {
        "stationary_body_median_xy_m": float(np.median(horizontal)),
        "stationary_body_p90_xy_m": float(np.quantile(horizontal, 0.90)),
        "deep_core_median_xy_m": float(np.median(deep_core_horizontal)),
        "deep_core_p90_xy_m": float(np.quantile(deep_core_horizontal, 0.90)),
        "all_vertices_centroid_xy_m": np.mean(displacement[:, :2], axis=0).tolist(),
    }
    full = intervals["393_to_583"]
    verdict = (
        "PASS"
        if full["stationary_body_median_xy_m"] <= 0.015
        and full["stationary_body_p90_xy_m"] <= 0.030
        else "FAIL"
    )
    summary = {
        "definition": "initial world-X 35th--65th percentile; central base panel",
        "tracked_vertices": int(np.count_nonzero(stationary_mask)),
        "deep_core_definition": (
            "initial world-X 45th--55th and world-Y 30th--70th percentiles; "
            "reported for diagnosis but not used to weaken the verdict"
        ),
        "deep_core_tracked_vertices": int(np.count_nonzero(deep_core_mask)),
        "intervals": intervals,
        "verdict": verdict,
        "thresholds_m": {"median": 0.015, "p90": 0.030},
    }
    output_path = debug_dir / "second_fold_motion_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"second_fold_motion_verdict={verdict} path={output_path}")
    return summary


def _xy_grid_overlap(first_xy: np.ndarray, second_xy: np.ndarray, cell_m: float = 0.01) -> dict:
    """Measure planar footprint overlap without requiring scipy."""
    if len(first_xy) == 0 or len(second_xy) == 0:
        return {"cells_first": 0, "cells_second": 0, "intersection": 0, "coverage": 0.0, "iou": 0.0}
    origin = np.minimum(np.min(first_xy, axis=0), np.min(second_xy, axis=0))

    def occupied(points: np.ndarray) -> set[tuple[int, int]]:
        cells = np.floor((points - origin) / cell_m).astype(np.int64)
        return {tuple(cell) for cell in cells}

    first_cells = occupied(first_xy)
    second_cells = occupied(second_xy)
    intersection = len(first_cells & second_cells)
    union = len(first_cells | second_cells)
    return {
        "cell_m": cell_m,
        "cells_first": len(first_cells),
        "cells_second": len(second_cells),
        "intersection": intersection,
        # Coverage is the useful stack metric: 1 means the smaller footprint
        # lies entirely on the larger one, even when their areas differ.
        "coverage": intersection / max(1, min(len(first_cells), len(second_cells))),
        "iou": intersection / max(1, union),
    }


def summarize_fold_layering(debug_dir: Path | None) -> dict | None:
    """Check whether fold one, fold two and the base occupy one stacked footprint."""
    if debug_dir is None:
        return None
    required = (0, 332, 393, 583, 600)
    paths = {frame: debug_dir / f"frame_{frame:04d}.npz" for frame in required}
    if not all(path.is_file() for path in paths.values()):
        return None
    positions = {
        frame: np.asarray(np.load(path)["cloth_pos"], dtype=np.float64)
        for frame, path in paths.items()
    }
    initial_x = positions[0][:, 0]
    low, high = np.quantile(initial_x, (0.35, 0.65))
    # Mesh topology and vertex IDs are stable throughout IPC.  In the initial
    # shirt frame, low-X is the first outer panel, high-X is the second outer
    # panel, and the middle is the base both panels must cover.  This semantic
    # split is stable even if an entire panel is dragged or badly folded.
    first_mask = initial_x <= low
    second_mask = initial_x >= high
    base_mask = (initial_x > low) & (initial_x < high)

    final = positions[583]

    def region(mask: np.ndarray) -> dict:
        points = final[mask]
        return {
            "vertices": int(np.count_nonzero(mask)),
            "centroid_xyz_m": np.mean(points, axis=0).tolist() if len(points) else None,
            "xy_min_m": np.min(points[:, :2], axis=0).tolist() if len(points) else None,
            "xy_max_m": np.max(points[:, :2], axis=0).tolist() if len(points) else None,
            "z_q10_q50_q90_m": np.quantile(points[:, 2], (0.10, 0.50, 0.90)).tolist()
            if len(points)
            else None,
        }

    first_xy = final[first_mask, :2]
    second_xy = final[second_mask, :2]
    base_xy = final[base_mask, :2]
    first_second = _xy_grid_overlap(first_xy, second_xy)
    first_base = _xy_grid_overlap(first_xy, base_xy)
    second_base = _xy_grid_overlap(second_xy, base_xy)
    first_centroid = np.mean(first_xy, axis=0)
    second_centroid = np.mean(second_xy, axis=0)
    centroid_delta = second_centroid - first_centroid

    # This is deliberately a geometry verdict, independent of gripper logs.
    # The thresholds are permissive for wrinkled cloth but reject visibly
    # side-by-side panels.
    verdict = (
        "PASS"
        if first_second["coverage"] >= 0.45
        and first_base["coverage"] >= 0.45
        and second_base["coverage"] >= 0.45
        and abs(float(centroid_delta[0])) <= 0.06
        else "FAIL"
    )
    summary = {
        "definition": (
            "initial world-X low 35%, middle 30%, and high 35% define fold one, base, "
            "and fold two; their frame-583 10 mm planar occupancy is compared"
        ),
        "initial_x_region_bounds_m": {"p35": float(low), "p65": float(high)},
        "regions_at_frame_583": {
            "first_fold": region(first_mask),
            "second_fold": region(second_mask),
            "base": region(base_mask),
        },
        "second_minus_first_centroid_xy_m": centroid_delta.tolist(),
        "overlap": {
            "first_vs_second": first_second,
            "first_vs_base": first_base,
            "second_vs_base": second_base,
        },
        "verdict": verdict,
        "warning": "Automatic segmentation is a guardrail; always inspect the multi-view placement frames.",
    }
    output_path = debug_dir / "fold_layering_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"fold_layering_verdict={verdict} path={output_path}")
    return summary


def summarize_third_fold_motion(debug_dir: Path | None) -> dict | None:
    """Measure whether the shirt-top support moves while the bottom is folded."""
    if debug_dir is None:
        return None
    # Frame 941 is the last cloth placement frame required by the tuned demo.
    # Release/retreat/settle may be appended outside the source trajectory, so
    # requiring source frames 960 and 979 suppressed this summary for otherwise
    # complete runs that intentionally hand control to the post-release phase.
    frames = (0, 690, 720, 750, 780, 810, 840, 870, 900, 920, 941)
    paths = {frame: debug_dir / f"frame_{frame:04d}.npz" for frame in frames}
    if not all(path.is_file() for path in paths.values()):
        return None
    positions = {
        frame: np.asarray(np.load(path)["cloth_pos"], dtype=np.float64)
        for frame, path in paths.items()
    }

    # Episode-0 shirt-centric convention: initial +Y is collar/top. The top
    # half should remain supported by the table while the -Y waist/bottom half
    # is transported toward it during fold three.
    initial_y = positions[0][:, 1]
    median_y = float(np.median(initial_y))
    stationary_mask = initial_y >= median_y
    moving_mask = ~stationary_mask

    intervals = {}
    for start, end in zip(frames[1:], frames[2:]):
        displacement = positions[end] - positions[start]
        stationary_xy = np.linalg.norm(displacement[stationary_mask, :2], axis=1)
        intervals[f"{start}_to_{end}"] = {
            "stationary_top_median_xy_m": float(np.median(stationary_xy)),
            "stationary_top_p90_xy_m": float(np.quantile(stationary_xy, 0.90)),
            "all_vertices_centroid_xy_m": np.mean(displacement[:, :2], axis=0).tolist(),
        }

    base = positions[690]
    final = positions[941]
    stationary_total = np.linalg.norm(
        final[stationary_mask, :2] - base[stationary_mask, :2], axis=1
    )
    overlap = _xy_grid_overlap(final[moving_mask, :2], final[stationary_mask, :2])
    verdict = (
        "PASS"
        if float(np.median(stationary_total)) <= 0.030 and overlap["coverage"] >= 0.45
        else "FAIL"
    )
    summary = {
        "definition": (
            "initial +Y half is shirt top/collar support; initial -Y half is the "
            "waist/bottom panel transported in fold three"
        ),
        "initial_y_median_m": median_y,
        "stationary_top_vertices": int(np.count_nonzero(stationary_mask)),
        "moving_bottom_vertices": int(np.count_nonzero(moving_mask)),
        "intervals": intervals,
        "690_to_941_stationary_top_median_xy_m": float(np.median(stationary_total)),
        "690_to_941_stationary_top_p90_xy_m": float(np.quantile(stationary_total, 0.90)),
        "final_bottom_vs_top_overlap": overlap,
        "verdict": verdict,
        "thresholds": {"stationary_median_m": 0.030, "overlap_coverage": 0.45},
    }
    output_path = debug_dir / "third_fold_motion_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"third_fold_motion_verdict={verdict} path={output_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim1-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--shirt-obj", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=0, help="0 means the full trajectory")
    parser.add_argument(
        "--trajectory-stride",
        type=int,
        default=1,
        help="Preview every Nth source frame; physics dt and output fps are adjusted to preserve action time",
    )
    parser.add_argument("--action-fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--record-fps", type=int, default=60)
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=None,
        help="Optional Acone URDF override, for example one with simplified collision-only finger meshes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-frames", type=int, default=120)
    parser.add_argument("--initial-shirt-x", type=float, default=0.660)
    parser.add_argument("--initial-shirt-y", type=float, default=0.0)
    parser.add_argument("--initial-shirt-z", type=float, default=0.93)
    parser.add_argument("--contact-d-hat", type=float, default=0.0002)
    parser.add_argument(
        "--contact-constitution",
        choices=("ipc", "al-ipc"),
        default="ipc",
        help="libuIPC contact pipeline used for cloth, rigid, and self contact",
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--visualize-ipc-proxies",
        action="store_true",
        help=(
            "Overlay the orange rigid proxy meshes at the transforms actually "
            "used by IPC and save their pose error at visual keyframes."
        ),
    )
    parser.add_argument(
        "--render-ipc-actual-visuals",
        action="store_true",
        help=(
            "Render coupled robot visual meshes at the actual IPC affine-body "
            "poses while leaving Genesis FK as the soft-constraint target."
        ),
    )
    parser.add_argument("--camera-view", choices=("overhead", "oblique"), default="overhead")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument(
        "--trajectory-preflight-only",
        action="store_true",
        help=(
            "Build the scene, solve all Cartesian trajectory corrections and "
            "report the worst IK frame, then exit before IPC stepping."
        ),
    )
    parser.add_argument(
        "--record-multi-view",
        action="store_true",
        help=(
            "Record synchronized overview, overhead, shirt-bottom and moving "
            "right-grasp cameras, then compose a four-panel video with ffmpeg"
        ),
    )
    parser.add_argument(
        "--drive-mode",
        choices=("direct", "hybrid", "pd"),
        default="direct",
        help="Hybrid teleports arm joints but drives finger joints with PD so contact can create pinch pressure",
    )
    parser.add_argument("--two-way-coupling", action="store_true")
    parser.add_argument(
        "--robot-coup-type",
        choices=("two_way_soft_constraint", "external_articulation"),
        default="two_way_soft_constraint",
        help=(
            "IPC coupling for the fixed-base Acone robot. External articulation "
            "keeps the joint topology inside IPC; soft constraint preserves the "
            "legacy six-link setup."
        ),
    )
    parser.add_argument(
        "--ipc-constraint-strength-translation",
        type=float,
        default=100.0,
        help=(
            "Soft-transform coupling strength used to keep IPC rigid collision bodies "
            "on their Genesis link translation targets"
        ),
    )
    parser.add_argument(
        "--ipc-constraint-strength-rotation",
        type=float,
        default=100.0,
        help=(
            "Soft-transform coupling strength used to keep IPC rigid collision bodies "
            "on their Genesis link rotation targets"
        ),
    )
    parser.add_argument(
        "--ipc-rigid-rigid-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable IPC collision between robot rigid proxies and the rigid table. "
            "Enabled by default for physical demos; use the explicit --no form only "
            "for legacy SIM1 replay diagnostics."
        ),
    )
    parser.add_argument("--log-level", default="info", choices=("debug", "info", "warning"))
    parser.add_argument("--cloth-E", type=float, default=2.0e4)
    parser.add_argument("--cloth-rho", type=float, default=280.0)
    parser.add_argument("--cloth-thickness", type=float, default=0.0002)
    parser.add_argument("--cloth-bending", type=float, default=10.0)
    parser.add_argument("--cloth-friction", type=float, default=0.5)
    parser.add_argument("--robot-friction", type=float, default=1.2)
    parser.add_argument("--table-friction", type=float, default=0.3)
    parser.add_argument(
        "--fast-preview",
        action="store_true",
        help="Use a cheaper IPC nonlinear solve for iteration; keep disabled for final comparison",
    )
    parser.add_argument("--virtual-grasp", action="store_true")
    parser.add_argument(
        "--grasp-mode",
        choices=("hard", "soft"),
        default="soft",
        help="Soft is CCD-safe; hard is an experimental state override and may destabilize IPC contact",
    )
    parser.add_argument("--grasp-radius", type=float, default=0.06)
    parser.add_argument("--grasp-points", type=int, default=6)
    parser.add_argument("--final-right-grasp-points", type=int, default=6)
    parser.add_argument("--grasp-strength", type=float, default=20.0)
    parser.add_argument("--exact-finger-collision", action="store_true")
    parser.add_argument("--prepin-first-grasp", action="store_true")
    parser.add_argument(
        "--contact-grasp-test",
        action="store_true",
        help="Enable passive patch tracking for a no-anchor friction-grasp test",
    )
    parser.add_argument(
        "--finger-overclose",
        type=float,
        default=0.0,
        help=(
            "Advance closed finger targets by this many metres to generate pinch pressure; "
            "targets are clamped to the Acone URDF lower limit (0 m)"
        ),
    )
    parser.add_argument(
        "--right-finger-overclose-extra",
        type=float,
        default=0.0,
        help=(
            "Additional right-hand close advance in metres. This is useful because "
            "the recorded right first-grasp target remains slightly more open than "
            "the left target; the final command is still clamped to the URDF limit."
        ),
    )
    parser.add_argument(
        "--first-grasp-clearance-lift",
        type=float,
        default=0.0,
        help=(
            "Raise both TCPs while approaching, closing and initially lifting the "
            "first grasp. This keeps the exact finger collision meshes above the "
            "table when rigid-rigid IPC is enabled. The correction ramps over "
            "source frames 45--60, stays through frame 150, and fades by 210."
        ),
    )
    parser.add_argument(
        "--first-grasp-right-depth",
        type=float,
        default=0.0,
        help=(
            "Lower only the robot-right TCP during the first-grasp clearance "
            "window. This compensates asymmetric cloth/finger capture without "
            "changing the already successful robot-left grasp."
        ),
    )
    parser.add_argument(
        "--first-fold-tcp-lift",
        type=float,
        default=0.0,
        help=(
            "Raise both TCPs during first-fold placement. The correction ramps "
            "over source frames 240--260, stays through release at 332, and "
            "fades out by frame 350."
        ),
    )
    parser.add_argument(
        "--first-fold-transfer-lift",
        type=float,
        default=0.0,
        help=(
            "Additional clearance during the horizontal part of the first fold. "
            "It ramps in over source frames 240--260, stays through frame 295, "
            "then fades by frame 325 so placement is predominantly vertical."
        ),
    )
    parser.add_argument(
        "--first-fold-stack-overlap",
        type=float,
        default=0.0,
        help=(
            "Move both closed TCPs along world +X during first-fold placement so "
            "the initial low-X outer shirt panel covers the central base panel. "
            "It ramps over frames 260--295, stays through release at 332, and "
            "fades by 350. Positive values deepen overlap; negative values "
            "leave a narrower first-fold panel. The value is in metres."
        ),
    )
    parser.add_argument(
        "--second-fold-left-approach-lift",
        type=float,
        default=0.0,
        help=(
            "Raise only the robot-left TCP while it travels toward the shirt-right "
            "sleeve for the second fold. The correction ramps over source frames "
            "340--355, stays through 370, then descends gradually while retaining "
            "40%% of the clearance at close, rising to 48%% as the raw path "
            "continues descending. The retained lift "
            "is held through 415 and fades only after the fingers have closed."
        ),
    )
    parser.add_argument(
        "--second-fold-right-approach-lift",
        type=float,
        default=0.0,
        help=(
            "Raise only the robot-right TCP during its later approach for the "
            "second fold. The correction ramps over source frames 400--415, "
            "starts its descent at 420, and keeps 42.5%% of the requested clearance "
            "through close at frame 439. The retained lift is held through 455 "
            "and fades only after the fingers have closed."
        ),
    )
    parser.add_argument(
        "--second-fold-transport-lift",
        type=float,
        default=0.0,
        help=(
            "Additional lift after each hand closes for the second fold. The "
            "robot-left correction ramps over source frames 393--415 and the "
            "robot-right correction over 439--455; both stay high through 535 "
            "and fade by 570. This separates the horizontal transfer from the "
            "final vertical placement instead of dragging the stack near the table."
        ),
    )
    parser.add_argument(
        "--second-fold-lift-first-planar-hold",
        action="store_true",
        help=(
            "After each second-fold gripper closes, temporarily hold that TCP's "
            "world X/Y at its close pose while the Z transport lift develops. "
            "The robot-left hold is released over frames 415--475 and the "
            "robot-right hold over frames 465--515. This makes the transfer "
            "genuinely lift-first instead of adding Z while the public path is "
            "already sweeping horizontally through the folded cloth."
        ),
    )
    parser.add_argument(
        "--second-fold-roll-arc-height",
        type=float,
        default=0.0,
        help=(
            "Replace the public second-fold TCP sweep with the path selected by "
            "--second-fold-roll-path. This value is the clearance/arc height in "
            "metres."
        ),
    )
    parser.add_argument(
        "--second-fold-roll-path",
        choices=("staged", "smooth_arc"),
        default="staged",
        help=(
            "Shape used when --second-fold-roll-arc-height is positive. 'staged' "
            "keeps the legacy vertical-lift / level-transfer / vertical-place "
            "path. 'smooth_arc' moves X/Y and Z together on one smooth arch from "
            "frames 439--570, avoiding the high-tension corner created by lifting "
            "the grasped edge vertically before any fold motion."
        ),
    )
    parser.add_argument(
        "--second-fold-placement-relax",
        type=float,
        default=0.0,
        help=(
            "Move both closed TCPs along world +X (away from the robot) near "
            "the end of the second fold to unload horizontal cloth tension. "
            "The correction ramps over source frames 505--545, stays through "
            "release, and fades according to the correction-release frame "
            "arguments."
        ),
    )
    parser.add_argument(
        "--second-fold-placement-lift",
        type=float,
        default=0.0,
        help=(
            "Keep both second-fold TCPs above the unmodified terminal Z while "
            "the transported panel lands and is released. The correction ramps "
            "over source frames 535--570, stays through release, and then fades "
            "according to the correction-release frame arguments. This prevents "
            "a dense/thick cloth stack from being "
            "compressed into a horizontal push during final placement."
        ),
    )
    parser.add_argument(
        "--second-fold-stack-overlap",
        type=float,
        default=0.0,
        help=(
            "Move both closed TCPs along world -X during the elevated second-fold "
            "transfer so "
            "the initial high-X outer shirt panel covers the same central base "
            "panel as fold one. It ramps over frames 455--505, stays through "
            "release, and then fades according to the correction-release frame "
            "arguments. The value is a nonnegative distance."
        ),
    )
    parser.add_argument(
        "--second-fold-correction-release-start",
        type=int,
        default=600,
        help=(
            "Source frame at which the second-fold placement/stack corrections "
            "start fading. Keep this at or after the grippers have opened to "
            "avoid dragging the released flap during correction withdrawal."
        ),
    )
    parser.add_argument(
        "--second-fold-correction-release-end",
        type=int,
        default=619,
        help=(
            "Source frame at which the second-fold placement/stack correction "
            "fade completes. Must be greater than the release-start frame."
        ),
    )
    parser.add_argument(
        "--third-fold-right-grasp-lift",
        type=float,
        default=0.0,
        help=(
            "Raise the robot-right TCP during the third-fold approach and "
            "close. It ramps over source frames 620--650, stays through "
            "660, and fades through 720 while the post-close lift takes over."
        ),
    )
    parser.add_argument(
        "--third-fold-right-grasp-depth",
        type=float,
        default=0.0,
        help=(
            "Lower the robot-right TCP locally around the third-fold close. "
            "It ramps in over source frames 670--690, stays through the raw "
            "post-close dip at 720, and fades by 750. Use only after the high "
            "third-fold transit lift has recovered before closing."
        ),
    )
    parser.add_argument(
        "--third-fold-right-grasp-lateral",
        type=float,
        default=0.0,
        help=(
            "Shift the robot-right TCP along its local +Y closing axis for the "
            "third fold. This centers the front/back shirt layers between the "
            "two fingers; it ramps over source frames 650--680, stays through "
            "release at 941, and fades by 960."
        ),
    )
    parser.add_argument(
        "--third-fold-right-grasp-world-x",
        type=float,
        default=0.0,
        help=(
            "Signed world-X correction for the robot-right TCP during the third "
            "fold. It follows the same 650--680 ramp, stays through release at "
            "941, and fades by 960. This is intended to compensate measured "
            "whole-shirt translation after fold two; negative values move the "
            "grasp toward the robot in the current scene."
        ),
    )
    parser.add_argument(
        "--third-fold-right-grasp-world-y",
        type=float,
        default=0.0,
        help=(
            "Signed world-Y correction for the robot-right TCP during the third "
            "fold, using the same timing as --third-fold-right-grasp-world-x."
        ),
    )
    parser.add_argument(
        "--third-fold-placement-depth",
        type=float,
        default=0.0,
        help=(
            "Lower the closed robot-right TCP near the end of the third fold so "
            "the transported flap rests on the folded stack before the recorded "
            "release. The correction ramps over source frames 890--920, stays "
            "through frame 945, and fades as the arm retreats by frame 970."
        ),
    )
    parser.add_argument(
        "--third-fold-post-close-lift",
        type=float,
        default=0.0,
        help=(
            "Add a lift-first arc after the robot-right gripper closes at frame 690. "
            "The correction rises through frame 735, stays through frame 850, "
            "and returns to the public placement height by frame 925."
        ),
    )
    parser.add_argument(
        "--third-fold-smooth-rotation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reparameterize the public robot-right wrist quaternion path from "
            "source frames 690--780 to constant angular speed. The original "
            "orientation path and both endpoint poses remain unchanged."
        ),
    )
    parser.add_argument(
        "--third-fold-outward-pull-cancel",
        type=float,
        default=0.0,
        help=(
            "Cancel the public third-fold path's short pull toward the shirt "
            "waist immediately after closing. The +Y correction grows from "
            "frames 690--750 and fades by frame 780, before the main folding "
            "arc. This keeps the garment body from being tensioned and dragged."
        ),
    )
    parser.add_argument(
        "--third-fold-shirt-top-offset",
        type=float,
        default=0.0,
        help=(
            "Move the third-fold placement toward the shirt top/collar. In the "
            "episode-0 shirt frame this is world +Y. The correction begins only "
            "after lift, reaches full value at frame 880, stays through release, "
            "and fades by frame 960."
        ),
    )
    parser.add_argument(
        "--third-fold-placement-level",
        type=float,
        default=0.0,
        help=(
            "Blend the robot-right gripper toward a table-parallel placement "
            "orientation after the third-fold flap is airborne. Zero preserves "
            "the public SIM1 quaternion and one makes the TCP length axis fully "
            "horizontal. The blend ramps over source frames 800--885, stays "
            "through release, and fades during retreat by frame 979."
        ),
    )
    parser.add_argument(
        "--third-fold-front-plane-roll-deg",
        type=float,
        default=0.0,
        help=(
            "After the waist flap is airborne, rotate the robot-right gripper "
            "only about world X, i.e. inside the robot-front/world-YZ plane. "
            "The correction is applied on top of each public SIM1 pose, ramps "
            "from source frame 780 through frame 920, stays through release at "
            "frame 945, then fades by frame 979. Positive angles turn the "
            "finger direction in the shirt-bottom view toward table-horizontal."
        ),
    )
    parser.add_argument(
        "--third-fold-release-hold",
        action="store_true",
        help=(
            "Hold the corrected robot-right TCP fixed at source frame 920 while "
            "the gripper opens through frame 945, then retreat vertically through "
            "frame 979. This avoids dragging the released fold across the table."
        ),
    )
    parser.add_argument(
        "--post-release-settle-frames",
        type=int,
        default=0,
        help=(
            "After the optional open-gripper retreat finishes, keep the robot "
            "stationary and continue stepping/recording for this many frames "
            "so the released cloth can visibly settle."
        ),
    )
    parser.add_argument(
        "--post-release-open-hold-frames",
        type=int,
        default=0,
        help=(
            "Hold the final pose with the gripper open for this many frames "
            "before retreating, so release is complete before the wrist moves."
        ),
    )
    parser.add_argument(
        "--post-release-retreat-frames",
        type=int,
        default=0,
        help=(
            "Move the open robot-right gripper smoothly away from the released "
            "cloth over this many frames before the free-settle observation."
        ),
    )
    parser.add_argument(
        "--post-release-retreat-height",
        type=float,
        default=0.0,
        help="World-Z lift, in metres, during the open-gripper retreat.",
    )
    parser.add_argument(
        "--post-release-retreat-top-offset",
        type=float,
        default=0.0,
        help=(
            "World-Y motion toward the shirt top, in metres, during the "
            "open-gripper retreat."
        ),
    )
    parser.add_argument(
        "--debug-third-fold-dir",
        type=Path,
        default=None,
        help=(
            "Write cloth vertices, TCP pose, finger-link poses and finger q at "
            "selected source frames around the third-fold grasp."
        ),
    )
    parser.add_argument(
        "--debug-second-fold-dir",
        type=Path,
        default=None,
        help=(
            "Write cloth vertices and both TCP poses at selected source frames "
            "around the second-fold grasp, transport and release."
        ),
    )
    parser.add_argument(
        "--keyframe-diagnostics-dir",
        type=Path,
        default=None,
        help=(
            "Render mandatory second-fold pre-close/close/lift keyframes from "
            "four views and assemble a visual contact sheet."
        ),
    )
    parser.add_argument(
        "--verify-third-fold-checkpoint",
        action="store_true",
        help=(
            "Run to the third-fold checkpoint once, finish the requested frames, "
            "restore Genesis+IPC in the same process, replay only the suffix, and "
            "report the final cloth-state reconstruction error"
        ),
    )
    parser.add_argument(
        "--third-fold-checkpoint-source-frame",
        type=int,
        default=619,
        help=(
            "Legacy checkpoint source frame used by in-process verification and, "
            "unless --save-checkpoint-source-frame is supplied, persistent saves"
        ),
    )
    parser.add_argument(
        "--save-checkpoint-source-frame",
        type=int,
        default=None,
        help=(
            "Source frame for a persistent checkpoint save. This is intentionally "
            "separate from the source frame of --load-third-fold-checkpoint, so a "
            "G2 checkpoint at frame 332 can produce a G3 checkpoint at frame 583 "
            "in one suffix run."
        ),
    )
    parser.add_argument(
        "--save-third-fold-checkpoint",
        type=Path,
        default=None,
        help=(
            "Persist a reusable pre-third-fold checkpoint. The requested .pkl "
            "is accompanied by .ipc_state.npz and .meta.json sidecars."
        ),
    )
    parser.add_argument(
        "--load-third-fold-checkpoint",
        type=Path,
        default=None,
        help=(
            "Restore a checkpoint written by --save-third-fold-checkpoint and "
            "start directly after its source frame. Third-fold trajectory "
            "parameters may change; mesh, physics and first-two-fold parameters may not."
        ),
    )
    parser.add_argument("--finger-kp", type=float, default=1000.0)
    parser.add_argument("--finger-kv", type=float, default=50.0)
    return parser.parse_args()


def summarize_settled_cloth(
    cloth_pos: np.ndarray, shirt_obj: Path, table_top_z: float = 0.8
) -> dict:
    """Measure whether the initial garment actually settled onto the table.

    The global maximum alone is a poor diagnostic because the collar and a
    single folded triangle can dominate it.  When a garment atlas matching the
    active mesh exists, also report both sleeve-tip bands independently.
    """
    positions = np.asarray(cloth_pos, dtype=np.float64).reshape((-1, 3))
    height = positions[:, 2] - float(table_top_z)

    def region_summary(mask: np.ndarray) -> dict:
        sample = height[np.asarray(mask, dtype=bool)]
        if sample.size == 0:
            return {"vertices": 0}
        quantiles = np.quantile(sample, [0.0, 0.1, 0.5, 0.9, 0.99, 1.0])
        return {
            "vertices": int(sample.size),
            "height_quantiles_m": {
                key: float(value)
                for key, value in zip(
                    ("min", "p10", "median", "p90", "p99", "max"), quantiles
                )
            },
            "fraction_above_5mm": float(np.mean(sample > 0.005)),
            "fraction_above_20mm": float(np.mean(sample > 0.020)),
            "fraction_above_40mm": float(np.mean(sample > 0.040)),
        }

    summary = {
        "table_top_z_m": float(table_top_z),
        "global": region_summary(np.ones(len(positions), dtype=bool)),
        "atlas": None,
    }
    atlas_path = shirt_obj.with_suffix("").with_name(
        f"{shirt_obj.stem}.garment_atlas.npz"
    )
    if atlas_path.is_file():
        with np.load(atlas_path) as atlas:
            raw_xyz = np.asarray(atlas["rest_raw_xyz"], dtype=np.float64)
            surface_layer = np.asarray(atlas["surface_layer"], dtype=np.int8)
        if len(raw_xyz) == len(positions):
            raw_x = raw_xyz[:, 0]
            negative_tip = raw_x <= np.quantile(raw_x, 0.08)
            positive_tip = raw_x >= np.quantile(raw_x, 0.92)
            summary["atlas"] = str(atlas_path.resolve())
            summary["negative_x_sleeve_tip"] = region_summary(negative_tip)
            summary["positive_x_sleeve_tip"] = region_summary(positive_tip)
            summary["table_up_surface"] = region_summary(surface_layer == 0)
            summary["table_facing_surface"] = region_summary(surface_layer == 1)
        else:
            summary["atlas_error"] = (
                f"vertex count mismatch: cloth={len(positions)} atlas={len(raw_xyz)}"
            )
    return summary


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def checkpoint_sidecars(path: Path) -> tuple[Path, Path, Path]:
    """Return the Genesis, native-IPC-state and metadata checkpoint paths."""
    scene_path = path.expanduser().resolve()
    if scene_path.suffix != ".pkl":
        scene_path = scene_path.with_suffix(".pkl")
    return (
        scene_path,
        scene_path.with_suffix(".ipc_state.npz"),
        scene_path.with_suffix(".meta.json"),
    )


def snapshot_ipc_state(coupler) -> dict[str, np.ndarray]:
    """Copy the public libuipc FEM and ABD state-accessor buffers."""
    arrays: dict[str, np.ndarray] = {}
    coupler._fem_state_feature.copy_to(coupler._fem_state_geom)
    fem_pos = coupler._fem_state_geom.vertices().find(uipc.builtin.position)
    fem_vel = coupler._fem_state_geom.vertices().find(uipc.builtin.velocity)
    if fem_pos is None or fem_vel is None:
        raise RuntimeError("IPC FEM checkpoint state has no position/velocity fields")
    arrays["fem_position"] = np.asarray(fem_pos.view(), dtype=np.float64).copy()
    arrays["fem_velocity"] = np.asarray(fem_vel.view(), dtype=np.float64).copy()

    if coupler._abd_state_feature is not None and coupler._abd_state_geom is not None:
        coupler._abd_state_feature.copy_to(coupler._abd_state_geom)
        abd_transform = coupler._abd_state_geom.instances().find(uipc.builtin.transform)
        abd_velocity = coupler._abd_state_geom.instances().find(uipc.builtin.velocity)
        if abd_transform is None or abd_velocity is None:
            raise RuntimeError("IPC ABD checkpoint state has no transform/velocity fields")
        arrays["abd_transform"] = np.asarray(abd_transform.view(), dtype=np.float64).copy()
        arrays["abd_velocity"] = np.asarray(abd_velocity.view(), dtype=np.float64).copy()
    return arrays


def restore_ipc_state(coupler, arrays: dict[str, np.ndarray]) -> None:
    """Restore state accessor buffers into a newly initialized libuipc world."""
    coupler._fem_state_feature.copy_to(coupler._fem_state_geom)
    fem_pos = coupler._fem_state_geom.vertices().find(uipc.builtin.position)
    fem_vel = coupler._fem_state_geom.vertices().find(uipc.builtin.velocity)
    if fem_pos is None or fem_vel is None:
        raise RuntimeError("IPC FEM restore state has no position/velocity fields")
    if fem_pos.view().shape != arrays["fem_position"].shape:
        raise RuntimeError(
            "IPC FEM checkpoint topology mismatch: "
            f"current={fem_pos.view().shape}, saved={arrays['fem_position'].shape}"
        )
    fem_pos.view()[...] = arrays["fem_position"]
    fem_vel.view()[...] = arrays["fem_velocity"]
    coupler._fem_state_feature.copy_from(coupler._fem_state_geom)

    if "abd_transform" in arrays:
        if coupler._abd_state_feature is None or coupler._abd_state_geom is None:
            raise RuntimeError("Saved checkpoint has ABD state but current scene does not")
        coupler._abd_state_feature.copy_to(coupler._abd_state_geom)
        abd_transform = coupler._abd_state_geom.instances().find(uipc.builtin.transform)
        abd_velocity = coupler._abd_state_geom.instances().find(uipc.builtin.velocity)
        if abd_transform is None or abd_velocity is None:
            raise RuntimeError("IPC ABD restore state has no transform/velocity fields")
        if abd_transform.view().shape != arrays["abd_transform"].shape:
            raise RuntimeError(
                "IPC ABD checkpoint topology mismatch: "
                f"current={abd_transform.view().shape}, saved={arrays['abd_transform'].shape}"
            )
        abd_transform.view()[...] = arrays["abd_transform"]
        abd_velocity.view()[...] = arrays["abd_velocity"]
        coupler._abd_state_feature.copy_from(coupler._abd_state_geom)

    # Pull the restored native state back into Genesis' FEM/rigid buffers so
    # rendering and the first suffix step start from the same configuration.
    coupler._ipc_world.retrieve()
    coupler._retrieve_fem_states()
    coupler._retrieve_rigid_states()


def main() -> None:
    args = parse_args()
    if args.contact_grasp_test:
        # Visual verification is a mandatory artifact for grasp development.
        # Logs can quantify co-motion, but cannot prove that the intended layer
        # entered the finger gap.
        if args.keyframe_diagnostics_dir is None:
            args.keyframe_diagnostics_dir = args.output.with_suffix("").with_name(
                f"{args.output.stem}_keyframes"
            )
        if args.debug_second_fold_dir is None:
            args.debug_second_fold_dir = args.output.with_suffix("").with_name(
                f"{args.output.stem}_second_fold_state"
            )
    if args.substeps < 1:
        raise ValueError("--substeps must be at least 1")
    if args.post_release_settle_frames < 0:
        raise ValueError("--post-release-settle-frames must be non-negative")
    if args.post_release_open_hold_frames < 0:
        raise ValueError("--post-release-open-hold-frames must be non-negative")
    if args.post_release_retreat_frames < 0:
        raise ValueError("--post-release-retreat-frames must be non-negative")
    if args.trajectory_stride < 1:
        raise ValueError("--trajectory-stride must be at least 1")
    if args.contact_grasp_test and args.virtual_grasp:
        raise ValueError("--contact-grasp-test must run without --virtual-grasp")
    if args.ipc_constraint_strength_translation <= 0.0:
        raise ValueError("--ipc-constraint-strength-translation must be positive")
    if args.ipc_constraint_strength_rotation <= 0.0:
        raise ValueError("--ipc-constraint-strength-rotation must be positive")
    if args.verify_third_fold_checkpoint and not args.no_record:
        raise ValueError("--verify-third-fold-checkpoint currently requires --no-record")
    if args.load_third_fold_checkpoint is not None and args.verify_third_fold_checkpoint:
        raise ValueError(
            "--load-third-fold-checkpoint cannot be combined with "
            "--verify-third-fold-checkpoint"
        )
    if args.robot_coup_type == "external_articulation" and args.drive_mode == "direct":
        raise ValueError(
            "--robot-coup-type external_articulation does not support the direct "
            "set_qpos trajectory replay; select --drive-mode pd for an experimental "
            "dynamics-controlled run"
        )
    urdf = require_file(
        args.robot_urdf
        if args.robot_urdf is not None
        else args.sim1_root / "assets/acone/acone.urdf"
    )
    trajectory_path = require_file(args.trajectory)
    shirt_obj = require_file(args.shirt_obj)
    first_grasp_candidates = None
    if args.prepin_first_grasp or args.contact_grasp_test:
        candidates = EPISODE0_FIRST_GRASP_CANDIDATES.copy()
        vertex_map_path = shirt_obj.with_suffix(".vertex_map.npz")
        if vertex_map_path.is_file():
            with np.load(vertex_map_path) as mapping_data:
                source_to_lowres = np.asarray(mapping_data["source_to_lowres"], dtype=np.int64)
            candidates = source_to_lowres[candidates]
            print(f"Mapped first-grasp candidates through {vertex_map_path}")
        first_grasp_candidates = (np.unique(candidates[:6]), np.unique(candidates[6:]))
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with np.load(trajectory_path) as data:
        joint_q = np.asarray(data["joint_q"], dtype=np.float64)
        openness = np.asarray(data["openness"], dtype=np.float64)
    if joint_q.ndim != 2 or joint_q.shape[1] != 19:
        raise ValueError(f"Expected trajectory shape (T, 19), got {joint_q.shape}")
    source_frame_count = len(joint_q) if args.frames <= 0 else min(args.frames, len(joint_q))
    source_frames = np.arange(0, source_frame_count, args.trajectory_stride, dtype=np.int64)
    joint_q = joint_q[source_frames]
    openness = openness[source_frames]
    frame_count = len(source_frames)
    if openness.shape != (frame_count, 2):
        raise ValueError(f"Expected openness shape ({frame_count}, 2), got {openness.shape}")

    physics_dt = args.trajectory_stride / (args.action_fps * args.substeps)
    effective_record_fps = max(1, round(args.record_fps / args.trajectory_stride))
    kinematic_grasp_demo = args.virtual_grasp and args.grasp_mode == "hard"
    if args.fast_preview:
        strict_contact_test = args.virtual_grasp or args.contact_grasp_test
        newton_max_iterations = 20 if args.contact_grasp_test else 12
        linesearch_iterations = 3
        newton_tolerance = 1.0e-2 if strict_contact_test else 5.0e-2
        newton_translation_tolerance = 1.0e-3 if strict_contact_test else 2.0e-2
        linear_system_tolerance = 1.0e-2
    else:
        newton_max_iterations = 50
        linesearch_iterations = 8
        newton_tolerance = 1.0e-2
        newton_translation_tolerance = 1.0e-3
        linear_system_tolerance = 1.0e-3
    # Cartesian trajectory preflight does not step IPC. Keep it independent of
    # CUDA so a wedged viewer/driver cannot block reachability diagnostics.
    genesis_backend = gs.cpu if args.trajectory_preflight_only else gs.gpu
    gs.init(backend=genesis_backend, logging_level=args.log_level, seed=args.seed)
    camera_pos = (1.50, 0.0, 2.50) if args.camera_view == "overhead" else (1.55, -1.45, 1.65)
    camera_lookat = (0.64, 0.0, 0.78) if args.camera_view == "overhead" else (0.64, 0.0, 0.86)
    coupler_options = (
        gs.options.LegacyCouplerOptions()
        if args.trajectory_preflight_only
        else gs.options.IPCCouplerOptions(
            constraint_strength_translation=args.ipc_constraint_strength_translation,
            constraint_strength_rotation=args.ipc_constraint_strength_rotation,
            newton_max_iterations=newton_max_iterations,
            n_linesearch_iterations=linesearch_iterations,
            linesearch_report_energy=False,
            newton_tolerance=newton_tolerance,
            newton_translation_tolerance=newton_translation_tolerance,
            newton_semi_implicit_enable=False,
            linear_system_tolerance=linear_system_tolerance,
            contact_enable=not kinematic_grasp_demo,
            enable_rigid_rigid_contact=args.ipc_rigid_rigid_contact,
            two_way_coupling=args.two_way_coupling,
            contact_d_hat=args.contact_d_hat,
            contact_constitution=args.contact_constitution,
            contact_resistance=1.0e7,
        )
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=physics_dt,
            gravity=(0.0, 0.0, 0.0) if kinematic_grasp_demo else (0.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True),
        coupler_options=coupler_options,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=42,
            max_FPS=effective_record_fps,
        ),
        # A strong ambient term flattens centimeter-scale cloth folds and makes
        # the uniformly colored surface read like molded rubber.  Keep enough
        # fill light for the dark fabric while restoring directional shading.
        vis_options=gs.options.VisOptions(ambient_light=(0.26, 0.26, 0.26)),
        show_viewer=args.viewer,
    )

    table = None
    if not args.trajectory_preflight_only:
        table = scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.65, 0.0, 0.4), size=(1.0, 2.0, 0.8), fixed=True
            ),
            material=gs.materials.Rigid(
                coup_type="ipc_only",
                coup_friction=args.table_friction,
                contact_resistance=1.0e7,
            ),
            surface=gs.surfaces.Plastic(color=(0.78, 0.82, 0.86, 1.0)),
        )
    robot_material_kwargs = {
        "coup_type": args.robot_coup_type,
        "coup_friction": args.robot_friction,
        "contact_resistance": 1.0e7,
        # State-level scripted grasp and finger contact fight over the same
        # vertices and can invalidate CCD history. In hard mode the virtual
        # attachment replaces robot/cloth contact altogether.
        "enable_coup_collision": not (args.virtual_grasp and args.grasp_mode == "hard"),
    }
    if args.robot_coup_type == "two_way_soft_constraint":
        robot_material_kwargs["coup_links"] = IPC_ROBOT_LINKS
    else:
        # External articulation must register the complete kinematic tree, but
        # only the same wrist/finger links as the legacy setup need cloth contact.
        robot_material_kwargs["coup_collision_links"] = IPC_ROBOT_LINKS
    robot_material = (
        gs.materials.Rigid(needs_coup=False)
        if args.trajectory_preflight_only
        else gs.materials.Rigid(**robot_material_kwargs)
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(urdf),
            pos=(0.0, 0.0, 0.17),
            fixed=True,
            visualization=not args.trajectory_preflight_only,
            # Keep collision geometry in CPU-only preflight as well: the
            # post-release IK scan uses tool-link AABBs to reject arm/hand
            # clearance paths that are reachable but geometrically unsafe.
            collision=True,
            convexify=not args.exact_finger_collision,
            decimate=not args.exact_finger_collision,
        ),
        material=robot_material,
        surface=gs.surfaces.Plastic(color=(0.56, 0.61, 0.66, 1.0)),
    )

    # SIM1/Newton stores the 19 coordinates as one complete left chain followed
    # by one complete right chain. Resolve columns by name instead of assuming a
    # parser-specific ordering; the current legacy parser happens to preserve it.
    genesis_joint_names = tuple(joint.name for joint in robot.joints if joint.n_qs)
    source_index_by_name = {name: index for index, name in enumerate(SOURCE_JOINT_NAMES)}
    if set(genesis_joint_names) != set(SOURCE_JOINT_NAMES):
        raise RuntimeError(
            "SIM1/Genesis actuated joint names differ: "
            f"source={SOURCE_JOINT_NAMES}, genesis={genesis_joint_names}"
        )
    genesis_from_source = np.array(
        [source_index_by_name[name] for name in genesis_joint_names], dtype=np.int64
    )
    joint_q = joint_q[:, genesis_from_source]
    genesis_index_by_name = {name: index for index, name in enumerate(genesis_joint_names)}
    arm_dof_indices = np.array(
        [genesis_index_by_name[name] for name in ARM_JOINT_NAMES], dtype=np.int64
    )
    finger_dof_indices = tuple(
        np.array([genesis_index_by_name[name] for name in names], dtype=np.int64)
        for names in FINGER_JOINT_NAMES
    )
    all_finger_dof_indices = np.concatenate(finger_dof_indices)
    print(f"SIM1 source joint order: {SOURCE_JOINT_NAMES}")
    print(f"Genesis joint order: {genesis_joint_names}")
    print(f"Genesis <- source columns: {genesis_from_source.tolist()}")
    cloth = None
    if not args.trajectory_preflight_only:
        cloth = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(shirt_obj),
                scale=0.001,
                pos=(args.initial_shirt_x, args.initial_shirt_y, args.initial_shirt_z),
                euler=(-90.0, 0.0, 0.0),
            ),
            material=gs.materials.FEM.Cloth(
                E=args.cloth_E,
                nu=0.49,
                rho=args.cloth_rho,
                thickness=args.cloth_thickness,
                bending_stiffness=args.cloth_bending,
                friction_mu=args.cloth_friction,
                contact_resistance=1.0e7,
            ),
            surface=gs.surfaces.Plastic(
                color=(0.10, 0.34, 0.68, 1.0),
                roughness=1.0,
                smooth=True,
                # Do not average normals across very sharp fold ridges.  This
                # preserves broad smooth drape but gives creases a readable edge.
                normal_diff_clamp=42.0,
            ),
        )
    canonical_camera_specs = {
        "overview": ((1.50, -1.35, 1.55), (0.68, 0.0, 0.82), (0.0, 0.0, 1.0)),
        "overhead": ((1.48, 0.0, 2.45), (0.68, 0.0, 0.80), (0.0, 1.0, 0.0)),
        "shirt_bottom": ((1.55, -0.05, 1.04), (0.68, 0.0, 0.82), (0.0, 0.0, 1.0)),
        # Updated to follow the right TCP before every physics step.
        "right_grasp": ((1.20, -0.32, 1.10), (0.94, 0.0, 0.92), (0.0, 0.0, 1.0)),
    }
    record_cameras = {}
    if args.record_multi_view and not args.trajectory_preflight_only:
        for view_name, (pos, lookat, up) in canonical_camera_specs.items():
            record_cameras[view_name] = scene.add_camera(
                res=(800, 600),
                pos=pos,
                lookat=lookat,
                up=up,
                fov=42,
                GUI=False,
            )
        primary_record_view = "overhead" if args.camera_view == "overhead" else "overview"
        camera = record_cameras[primary_record_view]
    elif not args.trajectory_preflight_only:
        primary_record_view = args.camera_view
        camera = scene.add_camera(
            res=(960, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=42,
            GUI=False,
        )
    keyframe_visuals = KeyframeVisualDiagnostics(
        scene,
        None if args.trajectory_preflight_only else args.keyframe_diagnostics_dir,
    )

    # ``_init_qpos`` is part of the imported joint-frame definition, not merely
    # the initial runtime state. Baking the first trajectory pose into it changes
    # FK even when get_qpos() later reports the requested absolute values. Keep
    # every movable joint at the URDF zero and apply the recorded pose only after
    # build via set_qpos(). The neutral Acone pose is clear of the shirt here.
    q_cursor = 0
    for joint in robot.joints:
        if joint.n_qs:
            joint._init_qpos = np.zeros(joint.n_qs, dtype=np.float64)
            q_cursor += joint.n_qs
    if q_cursor != joint_q.shape[1]:
        raise RuntimeError(f"URDF exposes {q_cursor} q coordinates, trajectory has {joint_q.shape[1]}")

    if args.virtual_grasp and not args.trajectory_preflight_only:
        if args.grasp_mode == "soft":
            cloth.sim.coupler.set_fem_vertex_constraint_strength_rate(args.grasp_strength)
        if args.grasp_mode == "soft" and args.prepin_first_grasp:
            cloth.sim.coupler.register_fem_vertex_constraint_candidates(
                cloth, np.concatenate(first_grasp_candidates)
            )

    build_start = time.perf_counter()
    scene.build()
    ipc_proxy_visuals = None
    ipc_actual_visuals = None
    if not args.trajectory_preflight_only:
        ipc_proxy_visuals = IPCProxyVisualizer(
            scene,
            robot,
            cloth,
            cloth.sim.coupler,
            args.visualize_ipc_proxies,
            args.keyframe_diagnostics_dir,
        )
        ipc_proxy_visuals.initialize()
        ipc_actual_visuals = IPCActualVisualSynchronizer(
            robot, cloth.sim.coupler, args.render_ipc_actual_visuals
        )
        ipc_actual_visuals.initialize()
        if args.render_ipc_actual_visuals:
            scene.register_post_visual_state_callback(
                ipc_actual_visuals.update_render_transforms
            )

    if (
        args.first_grasp_clearance_lift < 0.0
        or args.first_grasp_right_depth < 0.0
        or args.first_fold_tcp_lift < 0.0
        or args.first_fold_transfer_lift < 0.0
        or args.second_fold_left_approach_lift < 0.0
        or args.second_fold_right_approach_lift < 0.0
        or args.second_fold_transport_lift < 0.0
        or args.second_fold_roll_arc_height < 0.0
        or args.second_fold_placement_relax < 0.0
        or args.second_fold_placement_lift < 0.0
        or args.third_fold_right_grasp_lift < 0.0
        or args.third_fold_right_grasp_depth < 0.0
        or args.third_fold_right_grasp_lateral < 0.0
        or args.third_fold_post_close_lift < 0.0
        or args.third_fold_outward_pull_cancel < 0.0
        or args.third_fold_shirt_top_offset < 0.0
        or not 0.0 <= args.third_fold_placement_level <= 1.0
        or not -180.0 <= args.third_fold_front_plane_roll_deg <= 180.0
    ):
        raise ValueError("trajectory lift corrections must be non-negative")
    if (
        args.second_fold_correction_release_start < 583
        or args.second_fold_correction_release_end
        <= args.second_fold_correction_release_start
        or args.second_fold_correction_release_end > 619
    ):
        raise ValueError(
            "second-fold correction release must start at/after frame 583 and "
            "end strictly after it but no later than frame 619"
        )
    if (
        args.third_fold_placement_level > 0.0
        and args.third_fold_front_plane_roll_deg != 0.0
    ):
        raise ValueError(
            "third-fold table leveling and front-plane roll are alternative "
            "orientation strategies; enable only one"
        )
    if (
        args.first_grasp_clearance_lift > 0.0
        or args.first_grasp_right_depth > 0.0
        or args.first_fold_tcp_lift > 0.0
        or args.first_fold_transfer_lift > 0.0
        or args.second_fold_left_approach_lift > 0.0
        or args.second_fold_right_approach_lift > 0.0
        or args.second_fold_transport_lift > 0.0
        or args.second_fold_roll_arc_height > 0.0
        or args.second_fold_placement_relax > 0.0
        or args.second_fold_placement_lift > 0.0
        or args.third_fold_right_grasp_lift > 0.0
        or args.third_fold_right_grasp_depth > 0.0
        or args.third_fold_right_grasp_lateral > 0.0
        or args.third_fold_right_grasp_world_x != 0.0
        or args.third_fold_right_grasp_world_y != 0.0
        or args.third_fold_post_close_lift > 0.0
        or args.third_fold_smooth_rotation
        or args.third_fold_outward_pull_cancel > 0.0
        or args.third_fold_shirt_top_offset > 0.0
        or args.third_fold_placement_level > 0.0
        or args.third_fold_front_plane_roll_deg != 0.0
    ):
        # This narrow Cartesian correction compensates for the higher folded
        # cloth stack in Genesis. Without it the still-closed grippers descend
        # into the stack and turn the final placement into a horizontal push.
        # The initial grasp, lift, x/y path, and tool orientation stay intact.
        lift_links = (
            robot.get_link(name="left_link16"),
            robot.get_link(name="right_link26"),
        )
        lift_arm_indices = tuple(
            np.array([genesis_index_by_name[name] for name in names], dtype=np.int64)
            for names in (
                tuple(f"left_joint1{i}" for i in range(1, 7)),
                tuple(f"right_joint2{i}" for i in range(1, 7)),
            )
        )

        def smoothstep(value: float) -> float:
            value = float(np.clip(value, 0.0, 1.0))
            return value * value * (3.0 - 2.0 * value)

        def first_fold_lift_weight(source_frame: int) -> float:
            if source_frame < 240 or source_frame > 350:
                return 0.0
            if source_frame < 260:
                return smoothstep((source_frame - 240) / 20.0)
            if source_frame <= 332:
                return 1.0
            return 1.0 - smoothstep((source_frame - 332) / 18.0)

        def first_grasp_clearance_weight(source_frame: int) -> float:
            # The public replay lowers the pointed finger collision meshes about
            # 33 mm through the tabletop at the first close. That only appeared
            # to work while rigid-rigid IPC was disabled. Keep the fingertips
            # above the tabletop during close and the initial vertical lift,
            # then smoothly rejoin the accepted fold trajectory.
            if source_frame < 45 or source_frame > 210:
                return 0.0
            if source_frame < 60:
                return smoothstep((source_frame - 45) / 15.0)
            if source_frame <= 150:
                return 1.0
            return 1.0 - smoothstep((source_frame - 150) / 60.0)

        def first_fold_transfer_weight(source_frame: int) -> float:
            # Delay most of the descent until x/y are near their destination.
            # This changes the diagonal push into high transfer + vertical place.
            if source_frame < 240 or source_frame > 325:
                return 0.0
            if source_frame < 260:
                return smoothstep((source_frame - 240) / 20.0)
            if source_frame <= 295:
                return 1.0
            return 1.0 - smoothstep((source_frame - 295) / 30.0)

        def first_fold_stack_weight(source_frame: int) -> float:
            if source_frame < 260 or source_frame > 350:
                return 0.0
            if source_frame < 295:
                return smoothstep((source_frame - 260) / 35.0)
            if source_frame <= 332:
                return 1.0
            return 1.0 - smoothstep((source_frame - 332) / 18.0)

        def second_fold_left_approach_weight(source_frame: int) -> float:
            # Shirt-centric description: the robot-left arm crosses the first
            # folded flap on its way to the shirt-right sleeve. The public TCP
            # descends below the table-top height before arriving. Recovering
            # the raw pose exactly at close put the pointed collision mesh below
            # the table.  The old 30% retention still left the corrected target
            # as much as 9.5 mm below the tabletop at frames 393--439.  IPC then
            # correctly stopped the physical proxy at the table while the gray
            # Genesis visual continued through it, producing an 8--12 mm / 5--8
            # degree pose split.  At the close itself, however, 50% put the
            # physical proxy above the cloth and missed the grasp.  Retain 40%
            # (=24 mm) at frame 393, then ramp to 48% (=28.8 mm) by frame 425
            # as the public target continues descending.  This keeps the close
            # near the cloth and the later target at least 1 mm above the table.
            if source_frame < 340 or source_frame > 455:
                return 0.0
            if source_frame < 355:
                return smoothstep((source_frame - 340) / 15.0)
            if source_frame <= 370:
                return 1.0
            if source_frame < 393:
                return 1.0 - 0.6 * smoothstep((source_frame - 370) / 23.0)
            if source_frame < 425:
                return 0.4 + 0.08 * smoothstep((source_frame - 393) / 32.0)
            if source_frame <= 439:
                return 0.48
            return 0.48 * (1.0 - smoothstep((source_frame - 439) / 16.0))

        def second_fold_right_approach_weight(source_frame: int) -> float:
            # The robot-right arm begins its second-fold traverse later than
            # the robot-left arm. Its raw TCP moves from x=0.622, z=0.891 m at
            # frame 400 to x=0.777, z=0.774 m at the close crossing (frame
            # 439). Keep this diagonal traverse above the first folded flap,
            # then descend over the intended grasp. The previous 430--439 ramp
            # dropped 56 mm in nine samples and still left the TCP centre about
            # 9 mm below the cloth at close. Start the descent ten frames
            # earlier.  The old 42% retention still left right_link28 about
            # 3.4 mm below the table at frame 439.  Retain 42.5% (=34 mm with
            # the 80 mm baseline): the target is only about 3 mm into the soft
            # contact surface, while the actual IPC proxy remains above the
            # table and close enough to the cloth to establish the grasp.
            if source_frame < 400 or source_frame > 475:
                return 0.0
            if source_frame < 415:
                return smoothstep((source_frame - 400) / 15.0)
            if source_frame <= 420:
                return 1.0
            if source_frame < 439:
                return 1.0 - 0.575 * smoothstep((source_frame - 420) / 19.0)
            if source_frame <= 455:
                return 0.425
            return 0.425 * (1.0 - smoothstep((source_frame - 455) / 20.0))

        def second_fold_left_transport_weight(source_frame: int) -> float:
            # The left hand closes first at frame 393. The public TCP stays near
            # the table until about frame 450, so its initial x/y motion pulls
            # the whole garment. Lift the grasped edge first, then recover the
            # original placement well before the fingers open around frame 570.
            if source_frame < 393 or source_frame > 570:
                return 0.0
            if source_frame < 415:
                return smoothstep((source_frame - 393) / 22.0)
            if source_frame <= 535:
                return 1.0
            return 1.0 - smoothstep((source_frame - 535) / 35.0)

        def second_fold_right_transport_weight(source_frame: int) -> float:
            # The right hand closes later at frame 439. Cross-fade its approach
            # clearance into the same lift-first transfer.
            if source_frame < 439 or source_frame > 570:
                return 0.0
            if source_frame < 455:
                return smoothstep((source_frame - 439) / 16.0)
            if source_frame <= 535:
                return 1.0
            return 1.0 - smoothstep((source_frame - 535) / 35.0)

        def second_fold_left_planar_hold_weight(source_frame: int) -> float:
            if not args.second_fold_lift_first_planar_hold:
                return 0.0
            if source_frame < 393 or source_frame > 475:
                return 0.0
            if source_frame <= 415:
                return 1.0
            return 1.0 - smoothstep((source_frame - 415) / 60.0)

        def second_fold_right_planar_hold_weight(source_frame: int) -> float:
            if not args.second_fold_lift_first_planar_hold:
                return 0.0
            if source_frame < 439 or source_frame > 515:
                return 0.0
            if source_frame <= 465:
                return 1.0
            return 1.0 - smoothstep((source_frame - 465) / 50.0)

        def second_fold_placement_relax_weight(source_frame: int) -> float:
            # Both hands end the public second-fold path under horizontal
            # tension toward the robot (world -X). Once most of the transfer is
            # complete, give that tension back while the flap is close to the
            # table, then release without changing the post-release arm path.
            release_start = args.second_fold_correction_release_start
            release_end = args.second_fold_correction_release_end
            if source_frame < 505 or source_frame > release_end:
                return 0.0
            if source_frame < 545:
                return smoothstep((source_frame - 505) / 40.0)
            if source_frame <= release_start:
                return 1.0
            return 1.0 - smoothstep(
                (source_frame - release_start) / (release_end - release_start)
            )

        def second_fold_placement_lift_weight(source_frame: int) -> float:
            # The 13,767-face mesh forms a thicker, rougher stack than the
            # reduced debug mesh. Preserve clearance during the final descent
            # and let gravity complete the last few millimetres after opening,
            # instead of driving the closed fingers into the folded layers.
            release_start = args.second_fold_correction_release_start
            release_end = args.second_fold_correction_release_end
            if source_frame < 535 or source_frame > release_end:
                return 0.0
            if source_frame < 570:
                return smoothstep((source_frame - 535) / 35.0)
            if source_frame <= release_start:
                return 1.0
            return 1.0 - smoothstep(
                (source_frame - release_start) / (release_end - release_start)
            )

        def second_fold_stack_weight(source_frame: int) -> float:
            # Add the final overlap while both grasp points are still elevated.
            # Applying this correction during descent made the transported panel
            # scrape the first folded panel and pull the entire garment toward
            # the robot, destroying the three-layer stack needed by fold three.
            release_start = args.second_fold_correction_release_start
            release_end = args.second_fold_correction_release_end
            if source_frame < 455 or source_frame > release_end:
                return 0.0
            if source_frame < 505:
                return smoothstep((source_frame - 455) / 50.0)
            if source_frame <= release_start:
                return 1.0
            return 1.0 - smoothstep(
                (source_frame - release_start) / (release_end - release_start)
            )

        def third_fold_right_grasp_weight(source_frame: int) -> float:
            # Keep the approach clear of the table through close. Fading the lift
            # out by frame 688 made the requested rigid target penetrate the table;
            # IPC correctly stopped its proxy, but that split the visual and physical
            # gripper poses. Extend the fade until the post-close lift takes over.
            if source_frame < 620 or source_frame > 720:
                return 0.0
            if source_frame < 650:
                return smoothstep((source_frame - 620) / 30.0)
            if source_frame <= 660:
                return 1.0
            return 1.0 - smoothstep((source_frame - 660) / 60.0)

        def third_fold_right_depth_weight(source_frame: int) -> float:
            if source_frame < 670 or source_frame > 750:
                return 0.0
            if source_frame < 690:
                return smoothstep((source_frame - 670) / 20.0)
            if source_frame <= 720:
                return 1.0
            return 1.0 - smoothstep((source_frame - 720) / 30.0)

        def third_fold_right_lateral_weight(source_frame: int) -> float:
            if source_frame < 650 or source_frame > 960:
                return 0.0
            if source_frame < 680:
                return smoothstep((source_frame - 650) / 30.0)
            if source_frame <= 941:
                return 1.0
            return 1.0 - smoothstep((source_frame - 941) / 19.0)

        def third_fold_placement_depth_weight(source_frame: int) -> float:
            if source_frame < 890 or source_frame > 970:
                return 0.0
            if source_frame < 920:
                return smoothstep((source_frame - 890) / 30.0)
            if source_frame <= 945:
                return 1.0
            return 1.0 - smoothstep((source_frame - 945) / 25.0)

        def third_fold_post_close_lift_weight(source_frame: int) -> float:
            # The public trajectory dips after closing. That is acceptable in
            # SIM1's own settled cloth state, but in Genesis the folded stack is
            # higher and more wrinkled. Cross-fade the dip into a vertical lift,
            # keep the flap clear during horizontal transport, then descend only
            # after most of the planar motion is complete.
            if source_frame < 690 or source_frame > 925:
                return 0.0
            if source_frame < 735:
                return smoothstep((source_frame - 690) / 45.0)
            if source_frame <= 850:
                return 1.0
            return 1.0 - smoothstep((source_frame - 850) / 75.0)

        def third_fold_outward_pull_cancel_weight(source_frame: int) -> float:
            # The public path moves the closed TCP about 69 mm back toward the
            # waist between frames 690 and 750 before beginning the main fold.
            # That straightens the flap under tension and drags the stationary
            # shirt body. A quadratic rise matches the observed pull better
            # than a linear/smoothstep ramp (the first half moves only ~19 mm),
            # then blends back to the public arc by frame 780.
            if source_frame < 690 or source_frame > 780:
                return 0.0
            if source_frame <= 750:
                progress = (source_frame - 690) / 60.0
                return progress * progress
            return 1.0 - smoothstep((source_frame - 750) / 30.0)

        def third_fold_shirt_top_weight(source_frame: int) -> float:
            # Shirt-centric convention: +Y points from the waist/bottom toward
            # the collar/top for episode 0. Do not perturb the grasp; establish
            # this placement correction only while the flap is already lifted.
            if source_frame < 780 or source_frame > 960:
                return 0.0
            if source_frame < 880:
                return smoothstep((source_frame - 780) / 100.0)
            if source_frame <= 941:
                return 1.0
            return 1.0 - smoothstep((source_frame - 941) / 19.0)

        def third_fold_level_weight(source_frame: int) -> float:
            # Reorient only after the waist flap is safely airborne. Keeping the
            # public grasp quaternion through close avoids sweeping a finger
            # through the folded stack; leveling before the final descent lets
            # the flap arrive broadside instead of being planted vertically.
            if source_frame < 800 or source_frame > 979:
                return 0.0
            if source_frame < 885:
                return smoothstep((source_frame - 800) / 85.0)
            if source_frame <= 945:
                return 1.0
            return 1.0 - smoothstep((source_frame - 945) / 34.0)

        def third_fold_front_plane_roll_weight(source_frame: int) -> float:
            # The shirt-bottom diagnostic camera looks approximately along
            # world -X, so rotating about world X is a pure rotation in that
            # front view. This avoids the previous table-leveling correction,
            # which swung the TCP tool axis from +X toward +Z and forced the
            # wrist into a low, sideways-twisted IK branch.
            if source_frame < 750 or source_frame > 979:
                return 0.0
            # Begin as soon as the grasped waist edge has visibly cleared the
            # table and finish before the final descent.  The earlier 780--920
            # ramp kept the panel almost vertical through most of transport,
            # so it arrived as a compact hanging bundle even though the final
            # tool orientation was correct.
            if source_frame < 870:
                return smoothstep((source_frame - 750) / 120.0)
            if source_frame <= 945:
                return 1.0
            return 1.0 - smoothstep((source_frame - 945) / 34.0)

        def raw_tcp_at_source(source_frame: int, hand_index: int) -> np.ndarray:
            matches = np.flatnonzero(source_frames == source_frame)
            if len(matches) != 1:
                raise RuntimeError(
                    "Second-fold Cartesian correction requires exactly one trajectory sample "
                    f"for source frame {source_frame}, got {len(matches)}"
                )
            robot.set_qpos(joint_q[int(matches[0])], zero_velocity=True)
            link = lift_links[hand_index]
            link_pos = as_numpy(link.get_pos()).reshape(3)
            link_quat = as_numpy(link.get_quat()).reshape(4)
            return link_pos + quat_wxyz_to_matrix(link_quat) @ TCP_LOCAL

        def raw_quat_at_source(source_frame: int, hand_index: int) -> np.ndarray:
            matches = np.flatnonzero(source_frames == source_frame)
            if len(matches) != 1:
                raise RuntimeError(
                    "Third-fold rotation smoothing requires exactly one trajectory "
                    f"sample for source frame {source_frame}, got {len(matches)}"
                )
            robot.set_qpos(joint_q[int(matches[0])], zero_velocity=True)
            return as_numpy(lift_links[hand_index].get_quat()).reshape(4).copy()

        second_fold_close_tcp = None
        if (
            args.second_fold_lift_first_planar_hold
            and int(np.max(source_frames)) >= 439
        ):
            second_fold_close_tcp = (
                raw_tcp_at_source(393, 0),
                raw_tcp_at_source(439, 1),
            )

        second_fold_roll_endpoints = None
        if args.second_fold_roll_arc_height > 0.0 and int(np.max(source_frames)) >= 570:
            roll_start = [raw_tcp_at_source(439, hand) for hand in range(2)]
            roll_end = [raw_tcp_at_source(570, hand) for hand in range(2)]
            for hand in range(2):
                start_lift = (
                    args.second_fold_transport_lift
                    * (
                        second_fold_left_transport_weight(439)
                        if hand == 0
                        else second_fold_right_transport_weight(439)
                    )
                )
                if hand == 0:
                    start_lift += (
                        args.second_fold_left_approach_lift
                        * second_fold_left_approach_weight(439)
                    )
                else:
                    start_lift += (
                        args.second_fold_right_approach_lift
                        * second_fold_right_approach_weight(439)
                    )
                end_lift = (
                    args.second_fold_transport_lift
                    * (
                        second_fold_left_transport_weight(570)
                        if hand == 0
                        else second_fold_right_transport_weight(570)
                    )
                )
                roll_start[hand] = roll_start[hand] + np.array([0.0, 0.0, start_lift])
                roll_end[hand] = roll_end[hand] + np.array(
                    [
                        args.second_fold_placement_relax
                        * second_fold_placement_relax_weight(570)
                        - args.second_fold_stack_overlap * second_fold_stack_weight(570),
                        0.0,
                        end_lift
                        + args.second_fold_placement_lift
                        * second_fold_placement_lift_weight(570),
                    ]
                )
            second_fold_roll_endpoints = (tuple(roll_start), tuple(roll_end))

        third_fold_rotation_quats = None
        if args.third_fold_smooth_rotation:
            if int(np.max(source_frames)) < 780:
                raise RuntimeError(
                    "Third-fold rotation smoothing requires source frame 780"
                )
            raw_rotation_quats = np.asarray(
                [raw_quat_at_source(frame, 1) for frame in range(690, 781)],
                dtype=np.float64,
            )
            # Keep quaternion signs continuous before measuring the path.
            for quat_index in range(1, len(raw_rotation_quats)):
                if np.dot(
                    raw_rotation_quats[quat_index - 1],
                    raw_rotation_quats[quat_index],
                ) < 0.0:
                    raw_rotation_quats[quat_index] *= -1.0
            raw_step_angles = 2.0 * np.arccos(
                np.clip(
                    np.sum(raw_rotation_quats[:-1] * raw_rotation_quats[1:], axis=1),
                    -1.0,
                    1.0,
                )
            )
            cumulative_angle = np.concatenate(([0.0], np.cumsum(raw_step_angles)))
            desired_angles = np.linspace(
                0.0, cumulative_angle[-1], len(raw_rotation_quats)
            )
            resampled_quats = []
            for desired_angle in desired_angles:
                segment = min(
                    int(np.searchsorted(cumulative_angle, desired_angle, side="right") - 1),
                    len(raw_rotation_quats) - 2,
                )
                segment = max(segment, 0)
                segment_angle = cumulative_angle[segment + 1] - cumulative_angle[segment]
                segment_progress = (
                    0.0
                    if segment_angle <= 1.0e-12
                    else (desired_angle - cumulative_angle[segment]) / segment_angle
                )
                resampled_quats.append(
                    slerp_quat_wxyz(
                        raw_rotation_quats[segment],
                        raw_rotation_quats[segment + 1],
                        segment_progress,
                    )
                )
            third_fold_rotation_quats = np.asarray(
                resampled_quats, dtype=np.float64
            )

        source_joint_q = joint_q.copy()
        corrected_joint_q = joint_q.copy()
        corrected_frames = 0
        max_ik_error = 0.0
        max_ik_error_source_frame = -1
        max_ik_error_hand = "none"
        third_fold_release_target = None
        for frame, source_frame in enumerate(source_frames):
            first_grasp_clearance = (
                args.first_grasp_clearance_lift
                * first_grasp_clearance_weight(int(source_frame))
            )
            first_fold_lift = (
                args.first_fold_tcp_lift * first_fold_lift_weight(int(source_frame))
                + args.first_fold_transfer_lift
                * first_fold_transfer_weight(int(source_frame))
            )
            per_hand_lift = (
                first_grasp_clearance
                + first_fold_lift
                + args.second_fold_left_approach_lift
                * second_fold_left_approach_weight(int(source_frame))
                + args.second_fold_transport_lift
                * second_fold_left_transport_weight(int(source_frame)),
                first_grasp_clearance
                - args.first_grasp_right_depth
                * first_grasp_clearance_weight(int(source_frame))
                + first_fold_lift
                + args.second_fold_right_approach_lift
                * second_fold_right_approach_weight(int(source_frame))
                + args.second_fold_transport_lift
                * second_fold_right_transport_weight(int(source_frame))
                + args.third_fold_right_grasp_lift
                * third_fold_right_grasp_weight(int(source_frame))
                + args.third_fold_post_close_lift
                * third_fold_post_close_lift_weight(int(source_frame))
                - args.third_fold_right_grasp_depth
                * third_fold_right_depth_weight(int(source_frame))
                - args.third_fold_placement_depth
                * third_fold_placement_depth_weight(int(source_frame)),
            )
            right_lateral = (
                args.third_fold_right_grasp_lateral
                * third_fold_right_lateral_weight(int(source_frame))
            )
            third_fold_world_xy = np.array(
                [
                    args.third_fold_right_grasp_world_x,
                    args.third_fold_right_grasp_world_y,
                    0.0,
                ]
            ) * third_fold_right_lateral_weight(int(source_frame))
            third_fold_placement_xy = np.array(
                [0.0, args.third_fold_shirt_top_offset, 0.0]
            ) * third_fold_shirt_top_weight(int(source_frame))
            third_fold_pull_cancel_xy = np.array(
                [0.0, args.third_fold_outward_pull_cancel, 0.0]
            ) * third_fold_outward_pull_cancel_weight(int(source_frame))
            second_fold_relax = (
                args.second_fold_placement_relax
                * second_fold_placement_relax_weight(int(source_frame))
            )
            second_fold_placement_lift = (
                args.second_fold_placement_lift
                * second_fold_placement_lift_weight(int(source_frame))
            )
            first_fold_stack = (
                args.first_fold_stack_overlap
                * first_fold_stack_weight(int(source_frame))
            )
            second_fold_stack = (
                args.second_fold_stack_overlap
                * second_fold_stack_weight(int(source_frame))
            )
            second_fold_planar_hold = (
                second_fold_left_planar_hold_weight(int(source_frame)),
                second_fold_right_planar_hold_weight(int(source_frame)),
            )
            second_fold_roll_active = (
                second_fold_roll_endpoints is not None
                and 439 <= int(source_frame) <= 570
            )
            if (
                max(per_hand_lift) <= 0.0
                and right_lateral <= 0.0
                and not np.any(third_fold_world_xy)
                and not np.any(third_fold_placement_xy)
                and not np.any(third_fold_pull_cancel_xy)
                and args.third_fold_placement_level
                * third_fold_level_weight(int(source_frame)) <= 0.0
                and abs(args.third_fold_front_plane_roll_deg)
                * third_fold_front_plane_roll_weight(int(source_frame)) <= 0.0
                and not (
                    args.third_fold_smooth_rotation
                    and 690 <= int(source_frame) <= 780
                )
                and second_fold_relax <= 0.0
                and second_fold_placement_lift <= 0.0
                and first_fold_stack <= 0.0
                and second_fold_stack <= 0.0
                and max(second_fold_planar_hold) <= 0.0
                and not second_fold_roll_active
                # Frames 920--979 deliberately hold the corrected third-fold
                # release pose and then retreat vertically.  At frame 960 the
                # ordinary third-fold correction weights have all faded to
                # zero, but the release-hold trajectory is still active.  Do
                # not fall back to the raw public trajectory here: that caused
                # a one-frame TCP jump of roughly 25 cm at release.
                and not (
                    args.third_fold_release_hold
                    and 920 <= int(source_frame) <= 979
                )
            ):
                continue
            robot.set_qpos(joint_q[frame], zero_velocity=True)
            targets = []
            for link, lift in zip(lift_links, per_hand_lift):
                link_pos = as_numpy(link.get_pos()).reshape(3)
                link_quat = as_numpy(link.get_quat()).reshape(4)
                link_rot = quat_wxyz_to_matrix(link_quat)
                tcp = link_pos + link_rot @ TCP_LOCAL
                targets.append((tcp + np.array([0.0, 0.0, lift]), link_quat))

            if second_fold_relax > 0.0:
                relax_offset = np.array([second_fold_relax, 0.0, 0.0])
                targets = [(tcp + relax_offset, quat) for tcp, quat in targets]

            # The staged roll endpoint already contains this lift while frames
            # 439--570 are active. Apply it only to the post-roll hold/release
            # segment here to avoid double-counting at frame 570.
            if second_fold_placement_lift > 0.0 and not second_fold_roll_active:
                placement_lift_offset = np.array(
                    [0.0, 0.0, second_fold_placement_lift]
                )
                targets = [
                    (tcp + placement_lift_offset, quat) for tcp, quat in targets
                ]

            stack_offset_x = first_fold_stack - second_fold_stack
            if stack_offset_x != 0.0:
                stack_offset = np.array([stack_offset_x, 0.0, 0.0])
                targets = [(tcp + stack_offset, quat) for tcp, quat in targets]

            if second_fold_close_tcp is not None:
                held_targets = []
                for hand_index, ((target_tcp, quat), hold_weight) in enumerate(
                    zip(targets, second_fold_planar_hold)
                ):
                    if hold_weight > 0.0:
                        target_tcp = target_tcp.copy()
                        target_tcp[:2] += hold_weight * (
                            second_fold_close_tcp[hand_index][:2] - target_tcp[:2]
                        )
                    held_targets.append((target_tcp, quat))
                targets = held_targets

            if second_fold_roll_active:
                roll_start, roll_end = second_fold_roll_endpoints
                staged_targets = []
                for hand_index, (_, quat) in enumerate(targets):
                    start_tcp = roll_start[hand_index]
                    end_tcp = roll_end[hand_index]
                    if args.second_fold_roll_path == "smooth_arc":
                        # Move in-plane and upward together. The legacy staged
                        # path first pulled the grasped edge straight up while
                        # its crease remained on the table, then translated an
                        # already taut panel and dragged the garment body.
                        progress = smoothstep(
                            (int(source_frame) - 439) / float(570 - 439)
                        )
                        staged_tcp = (
                            (1.0 - progress) * start_tcp + progress * end_tcp
                        )
                        staged_tcp[2] += (
                            args.second_fold_roll_arc_height
                            * np.sin(np.pi * progress)
                        )
                    else:
                        # Legacy lift / level-transfer / place path retained so
                        # previous experiments remain exactly reproducible.
                        clearance_z = max(
                            start_tcp[2] + args.second_fold_roll_arc_height,
                            end_tcp[2] + 0.020,
                        )
                        staged_tcp = start_tcp.copy()
                        if int(source_frame) <= 465:
                            lift_progress = smoothstep(
                                (int(source_frame) - 439) / 26.0
                            )
                            staged_tcp[2] = (
                                (1.0 - lift_progress) * start_tcp[2]
                                + lift_progress * clearance_z
                            )
                        elif int(source_frame) <= 535:
                            transfer_progress = smoothstep(
                                (int(source_frame) - 465) / 70.0
                            )
                            staged_tcp[:2] = (
                                (1.0 - transfer_progress) * start_tcp[:2]
                                + transfer_progress * end_tcp[:2]
                            )
                            staged_tcp[2] = clearance_z
                        else:
                            place_progress = smoothstep(
                                (int(source_frame) - 535) / 35.0
                            )
                            staged_tcp[:2] = end_tcp[:2]
                            staged_tcp[2] = (
                                (1.0 - place_progress) * clearance_z
                                + place_progress * end_tcp[2]
                            )
                    staged_targets.append((staged_tcp, quat))
                targets = staged_targets

            if right_lateral > 0.0:
                right_rot = quat_wxyz_to_matrix(targets[1][1])
                targets[1] = (
                    targets[1][0] + right_rot @ np.array([0.0, right_lateral, 0.0]),
                    targets[1][1],
                )

            if np.any(third_fold_world_xy):
                targets[1] = (
                    targets[1][0] + third_fold_world_xy,
                    targets[1][1],
                )

            if np.any(third_fold_placement_xy):
                targets[1] = (
                    targets[1][0] + third_fold_placement_xy,
                    targets[1][1],
                )

            if np.any(third_fold_pull_cancel_xy):
                targets[1] = (
                    targets[1][0] + third_fold_pull_cancel_xy,
                    targets[1][1],
                )

            if (
                third_fold_rotation_quats is not None
                and 690 <= int(source_frame) <= 780
            ):
                targets[1] = (
                    targets[1][0],
                    third_fold_rotation_quats[int(source_frame) - 690],
                )

            level_weight = (
                args.third_fold_placement_level
                * third_fold_level_weight(int(source_frame))
            )
            if level_weight > 0.0:
                raw_quat = targets[1][1]
                targets[1] = (
                    targets[1][0],
                    slerp_quat_wxyz(
                        raw_quat, table_parallel_quat(raw_quat), level_weight
                    ),
                )

            front_plane_roll_weight = third_fold_front_plane_roll_weight(
                int(source_frame)
            )
            if front_plane_roll_weight > 0.0:
                targets[1] = (
                    targets[1][0],
                    rotate_quat_about_world_x(
                        targets[1][1],
                        np.deg2rad(args.third_fold_front_plane_roll_deg)
                        * front_plane_roll_weight,
                    ),
                )

            if args.third_fold_release_hold:
                source_frame_int = int(source_frame)
                if source_frame_int == 920:
                    third_fold_release_target = (
                        targets[1][0].copy(),
                        targets[1][1].copy(),
                    )
                elif 921 <= source_frame_int <= 979:
                    if third_fold_release_target is None:
                        raise RuntimeError(
                            "Third-fold release hold did not capture source frame 920"
                        )
                    held_tcp, held_quat = third_fold_release_target
                    if source_frame_int <= 945:
                        targets[1] = (held_tcp.copy(), held_quat.copy())
                    else:
                        retreat = 0.04 * smoothstep(
                            (source_frame_int - 945) / float(979 - 945)
                        )
                        targets[1] = (
                            held_tcp + np.array([0.0, 0.0, retreat]),
                            held_quat.copy(),
                        )

            if (
                args.third_fold_release_hold
                and int(source_frame) > 920
                and frame > 0
            ):
                # The public trajectory retreats laterally after release and is
                # a poor IK seed for the deliberately stationary/vertical path.
                # Seed from the previous corrected pose to stay on one smooth
                # reachable branch.
                corrected = corrected_joint_q[frame - 1].copy()
            else:
                corrected = corrected_joint_q[frame].copy()
            for hand_name, link, indices, (target_tcp, target_quat) in zip(
                ("left", "right"), lift_links, lift_arm_indices, targets
            ):
                corrected, error = robot.inverse_kinematics(
                    link=link,
                    pos=target_tcp,
                    quat=target_quat,
                    local_point=TCP_LOCAL,
                    init_qpos=corrected,
                    dofs_idx_local=indices,
                    max_samples=1,
                    max_solver_iters=50,
                    pos_tol=1.0e-4,
                    rot_tol=1.0e-4,
                    return_error=True,
                )
                corrected = as_numpy(corrected).reshape(-1)
                ik_error = float(np.linalg.norm(as_numpy(error)))
                if ik_error > max_ik_error:
                    max_ik_error = ik_error
                    max_ik_error_source_frame = int(source_frame)
                    max_ik_error_hand = hand_name
            # The previous corrected pose is only an IK seed for arm continuity.
            # Preserve the source finger command for this frame so release-hold
            # cannot accidentally freeze the gripper at its frame-920 close value.
            corrected[all_finger_dof_indices] = source_joint_q[
                frame, all_finger_dof_indices
            ]
            corrected_joint_q[frame] = corrected
            corrected_frames += 1
        joint_q = corrected_joint_q
        print(
            f"first_grasp_clearance_lift={args.first_grasp_clearance_lift:.4f}m "
            f"first_grasp_right_depth={args.first_grasp_right_depth:.4f}m "
            f"first_fold_tcp_lift={args.first_fold_tcp_lift:.4f}m "
            f"first_fold_transfer_lift={args.first_fold_transfer_lift:.4f}m "
            f"first_fold_stack_overlap={args.first_fold_stack_overlap:.4f}m "
            f"second_fold_left_approach_lift={args.second_fold_left_approach_lift:.4f}m "
            f"second_fold_right_approach_lift={args.second_fold_right_approach_lift:.4f}m "
            f"second_fold_transport_lift={args.second_fold_transport_lift:.4f}m "
            f"second_fold_roll_arc_height={args.second_fold_roll_arc_height:.4f}m "
            f"second_fold_roll_path={args.second_fold_roll_path} "
            f"second_fold_lift_first_planar_hold={args.second_fold_lift_first_planar_hold} "
            f"second_fold_placement_relax={args.second_fold_placement_relax:.4f}m "
            f"second_fold_placement_lift={args.second_fold_placement_lift:.4f}m "
            f"second_fold_stack_overlap={args.second_fold_stack_overlap:.4f}m "
            f"second_fold_correction_release="
            f"{args.second_fold_correction_release_start}:"
            f"{args.second_fold_correction_release_end} "
            f"third_fold_right_grasp_lift={args.third_fold_right_grasp_lift:.4f}m "
            f"third_fold_right_grasp_depth={args.third_fold_right_grasp_depth:.4f}m "
            f"third_fold_right_grasp_lateral={args.third_fold_right_grasp_lateral:.4f}m "
            f"third_fold_right_grasp_world_x={args.third_fold_right_grasp_world_x:.4f}m "
            f"third_fold_right_grasp_world_y={args.third_fold_right_grasp_world_y:.4f}m "
            f"third_fold_placement_depth={args.third_fold_placement_depth:.4f}m "
            f"third_fold_post_close_lift={args.third_fold_post_close_lift:.4f}m "
            f"third_fold_outward_pull_cancel={args.third_fold_outward_pull_cancel:.4f}m "
            f"third_fold_shirt_top_offset={args.third_fold_shirt_top_offset:.4f}m "
            f"third_fold_placement_level={args.third_fold_placement_level:.3f} "
            f"third_fold_front_plane_roll_deg={args.third_fold_front_plane_roll_deg:.1f} "
            f"corrected_frames={corrected_frames} max_ik_error={max_ik_error:.6f} "
            f"max_ik_error_source_frame={max_ik_error_source_frame} "
            f"max_ik_error_hand={max_ik_error_hand}"
        )
        if args.trajectory_preflight_only:
            # A prescribed Genesis target that intersects the table is
            # impossible for the contact-enabled IPC proxy to follow.  That
            # conflict previously looked like coupling lag during the second
            # fold: the gray target passed through the table while IPC
            # correctly kept the orange finger above it.  Scan every corrected
            # target pose before any expensive cloth simulation and reject the
            # trajectory at the source.
            target_collision_links = tuple(
                robot.get_link(name=name) for name in IPC_ROBOT_LINKS
            )
            minimum_target_clearance = float("inf")
            minimum_target_clearance_at = None
            target_below_table_frames = set()
            smoothed_right_quats = []
            for trajectory_index, source_frame in enumerate(source_frames):
                robot.set_qpos(joint_q[trajectory_index], zero_velocity=True)
                if (
                    args.third_fold_smooth_rotation
                    and 690 <= int(source_frame) <= 780
                ):
                    smoothed_right_quats.append(
                        as_numpy(lift_links[1].get_quat(relative=False))
                        .reshape(4)
                        .astype(np.float64)
                    )
                frame_minimum = float("inf")
                for link in target_collision_links:
                    link_aabb = as_numpy(link.get_AABB()).reshape(2, 3)
                    clearance = float(
                        link_aabb[0, 2] - IPCProxyVisualizer.TABLE_TOP_Z
                    )
                    frame_minimum = min(frame_minimum, clearance)
                    if clearance < minimum_target_clearance:
                        minimum_target_clearance = clearance
                        minimum_target_clearance_at = {
                            "source_frame": int(source_frame),
                            "trajectory_index": int(trajectory_index),
                            "link": link.name,
                        }
                if frame_minimum < 0.0:
                    target_below_table_frames.add(int(source_frame))
            target_clearance_summary = {
                "min_clearance_m": minimum_target_clearance,
                "min_clearance_at": minimum_target_clearance_at,
                "below_table_frame_count": len(target_below_table_frames),
                "allowed_soft_target_penetration_m": (
                    IPCProxyVisualizer.TARGET_TABLE_PENETRATION_TOLERANCE_M
                ),
            }
            print(
                "corrected_target_table_clearance="
                f"{json.dumps(target_clearance_summary, sort_keys=True)}"
            )
            if minimum_target_clearance < (
                -IPCProxyVisualizer.TARGET_TABLE_PENETRATION_TOLERANCE_M
            ):
                raise RuntimeError(
                    "Corrected robot collision target exceeds the shallow "
                    "soft-contact table allowance; "
                    f"summary={target_clearance_summary}"
                )

            if args.third_fold_smooth_rotation:
                smoothed_right_quats = np.asarray(
                    smoothed_right_quats, dtype=np.float64
                )
                adjacent_dots = np.abs(
                    np.sum(
                        smoothed_right_quats[:-1] * smoothed_right_quats[1:],
                        axis=1,
                    )
                )
                angular_steps_deg = np.degrees(
                    2.0 * np.arccos(np.clip(adjacent_dots, -1.0, 1.0))
                )
                rotation_smoothing_summary = {
                    "frames": int(len(smoothed_right_quats)),
                    "max_step_deg": float(np.max(angular_steps_deg)),
                    "mean_step_deg": float(np.mean(angular_steps_deg)),
                }
                print(
                    "third_fold_rotation_smoothing="
                    f"{json.dumps(rotation_smoothing_summary, sort_keys=True)}"
                )
                if rotation_smoothing_summary["max_step_deg"] > 0.75:
                    raise RuntimeError(
                        "Third-fold smoothed wrist rotation still has an excessive "
                        f"single-frame step; summary={rotation_smoothing_summary}"
                    )

            if args.third_fold_release_hold:
                release_indices = np.flatnonzero(
                    (source_frames >= 920) & (source_frames <= 979)
                )
                expected_release_frames = np.arange(920, 980, dtype=np.int64)
                actual_release_frames = source_frames[release_indices].astype(np.int64)
                if not np.array_equal(actual_release_frames, expected_release_frames):
                    raise RuntimeError(
                        "Third-fold release continuity gate requires every source "
                        "frame from 920 through 979"
                    )

                release_tcp = []
                for trajectory_index in release_indices:
                    robot.set_qpos(joint_q[trajectory_index], zero_velocity=True)
                    right_link = lift_links[1]
                    right_pos = as_numpy(
                        right_link.get_pos(relative=False)
                    ).reshape(3)
                    right_quat = as_numpy(
                        right_link.get_quat(relative=False)
                    ).reshape(4)
                    release_tcp.append(
                        right_pos
                        + quat_wxyz_to_matrix(right_quat) @ TCP_LOCAL
                    )
                release_tcp = np.asarray(release_tcp, dtype=np.float64)
                release_arm_q = joint_q[release_indices][
                    :, lift_arm_indices[1]
                ]
                release_finger_q = joint_q[release_indices][
                    :, finger_dof_indices[1]
                ]
                source_release_finger_q = source_joint_q[release_indices][
                    :, finger_dof_indices[1]
                ]

                max_tcp_step = float(
                    np.max(np.linalg.norm(np.diff(release_tcp, axis=0), axis=1))
                )
                max_arm_joint_step = float(
                    np.max(np.abs(np.diff(release_arm_q, axis=0)))
                )
                hold_end = int(np.searchsorted(actual_release_frames, 946))
                max_hold_drift = float(
                    np.max(
                        np.linalg.norm(
                            release_tcp[:hold_end] - release_tcp[0], axis=1
                        )
                    )
                )
                retreat_tcp = release_tcp[hold_end:]
                max_retreat_xy_drift = float(
                    np.max(
                        np.linalg.norm(
                            retreat_tcp[:, :2] - release_tcp[0, :2], axis=1
                        )
                    )
                )
                min_retreat_z_step = float(np.min(np.diff(retreat_tcp[:, 2])))
                retreat_rise = float(release_tcp[-1, 2] - release_tcp[hold_end - 1, 2])
                max_finger_source_error = float(
                    np.max(np.abs(release_finger_q - source_release_finger_q))
                )
                release_open_mask = openness[release_indices, 1] >= 0.5
                opening_finger_q = release_finger_q[release_open_mask]
                min_opening_finger_step = float(
                    np.min(np.diff(opening_finger_q, axis=0))
                )
                final_finger_min = float(np.min(release_finger_q[-1]))

                release_gate = {
                    "max_tcp_step_m": max_tcp_step,
                    "max_arm_joint_step_rad": max_arm_joint_step,
                    "max_hold_drift_m": max_hold_drift,
                    "max_retreat_xy_drift_m": max_retreat_xy_drift,
                    "min_retreat_z_step_m": min_retreat_z_step,
                    "retreat_rise_m": retreat_rise,
                    "max_finger_source_error_m": max_finger_source_error,
                    "min_opening_finger_step_m": min_opening_finger_step,
                    "final_finger_min_m": final_finger_min,
                }
                print(
                    "third_fold_release_continuity="
                    + json.dumps(release_gate, sort_keys=True)
                )
                if (
                    max_tcp_step > 0.005
                    or max_arm_joint_step > 0.05
                    or max_hold_drift > 0.0005
                    or max_retreat_xy_drift > 0.001
                    or min_retreat_z_step < -5.0e-5
                    or not 0.039 <= retreat_rise <= 0.041
                    or max_finger_source_error > 1.0e-9
                    or min_opening_finger_step < -1.0e-9
                    or final_finger_min < 0.0439
                ):
                    raise RuntimeError(
                        "Third-fold release continuity gate failed: "
                        + json.dumps(release_gate, sort_keys=True)
                    )

            # Scan the post-release clearance motion from the *corrected* final
            # fold pose.  This is deliberately a two-leg Cartesian plan:
            # vertical first, then shirt-top.  Besides IK residual and joint
            # margin, report a conservative AABB distance between the two tool
            # assemblies along both legs so a reachable retreat is not mistaken
            # for a safe one.
            if args.post_release_retreat_frames:
                release_q = corrected_joint_q[-1].copy()
                release_q[all_finger_dof_indices] = FINGER_URDF_UPPER
                robot.set_qpos(release_q, zero_velocity=True)
                right_link = lift_links[1]
                right_pos = as_numpy(right_link.get_pos(relative=False)).reshape(3)
                right_quat = as_numpy(right_link.get_quat(relative=False)).reshape(4)
                right_tcp = right_pos + quat_wxyz_to_matrix(right_quat) @ TCP_LOCAL
                left_tool_links = tuple(
                    robot.get_link(name=name)
                    for name in ("left_link16", "left_link17", "left_link18")
                )
                right_tool_links = tuple(
                    robot.get_link(name=name)
                    for name in ("right_link26", "right_link27", "right_link28")
                )

                def tool_aabb_clearance() -> float:
                    clearance = np.inf
                    for left_tool_link in left_tool_links:
                        left_aabb = as_numpy(left_tool_link.get_AABB()).reshape(2, 3)
                        for right_tool_link in right_tool_links:
                            right_aabb = as_numpy(right_tool_link.get_AABB()).reshape(2, 3)
                            axis_gap = np.maximum(
                                np.maximum(
                                    left_aabb[0] - right_aabb[1],
                                    right_aabb[0] - left_aabb[1],
                                ),
                                0.0,
                            )
                            clearance = min(clearance, float(np.linalg.norm(axis_gap)))
                    return clearance

                lower, upper = robot.get_dofs_limit(
                    dofs_idx_local=lift_arm_indices[1]
                )
                lower = as_numpy(lower).reshape(-1)
                upper = as_numpy(upper).reshape(-1)
                heights = sorted(
                    {
                        0.05,
                        0.075,
                        0.08,
                        0.085,
                        0.09,
                        0.095,
                        0.10,
                        0.125,
                        0.15,
                        0.175,
                        0.20,
                        float(args.post_release_retreat_height),
                    }
                )
                print(
                    "post_release_ik_scan_start "
                    f"tcp={right_tcp.tolist()} "
                    f"requested_top={args.post_release_retreat_top_offset:.4f}m"
                )
                for scan_height in heights:
                    vertical_tcp = right_tcp + np.array([0.0, 0.0, scan_height])
                    vertical_q, vertical_error = robot.inverse_kinematics(
                        link=right_link,
                        pos=vertical_tcp,
                        quat=right_quat,
                        local_point=TCP_LOCAL,
                        init_qpos=release_q,
                        dofs_idx_local=lift_arm_indices[1],
                        max_samples=1,
                        max_solver_iters=100,
                        pos_tol=1.0e-4,
                        rot_tol=1.0e-4,
                        return_error=True,
                    )
                    vertical_q = as_numpy(vertical_q).reshape(-1)
                    vertical_error_m = float(np.linalg.norm(as_numpy(vertical_error)))
                    final_tcp = vertical_tcp + np.array(
                        [0.0, args.post_release_retreat_top_offset, 0.0]
                    )
                    final_q, final_error = robot.inverse_kinematics(
                        link=right_link,
                        pos=final_tcp,
                        quat=right_quat,
                        local_point=TCP_LOCAL,
                        init_qpos=vertical_q,
                        dofs_idx_local=lift_arm_indices[1],
                        max_samples=1,
                        max_solver_iters=100,
                        pos_tol=1.0e-4,
                        rot_tol=1.0e-4,
                        return_error=True,
                    )
                    final_q = as_numpy(final_q).reshape(-1)
                    final_error_m = float(np.linalg.norm(as_numpy(final_error)))
                    min_tool_clearance = np.inf
                    for start_q, end_q in (
                        (release_q, vertical_q),
                        (vertical_q, final_q),
                    ):
                        for blend in np.linspace(0.0, 1.0, 11):
                            robot.set_qpos(
                                start_q + float(blend) * (end_q - start_q),
                                zero_velocity=True,
                            )
                            min_tool_clearance = min(
                                min_tool_clearance, tool_aabb_clearance()
                            )
                    right_arm_q = final_q[lift_arm_indices[1]]
                    finite_margin = np.minimum(
                        right_arm_q - lower, upper - right_arm_q
                    )
                    finite_margin = finite_margin[np.isfinite(finite_margin)]
                    min_joint_margin = (
                        float(np.min(finite_margin)) if len(finite_margin) else np.inf
                    )
                    print(
                        "post_release_ik_scan "
                        f"height={scan_height:.4f}m "
                        f"vertical_error={vertical_error_m:.6f}m "
                        f"final_error={final_error_m:.6f}m "
                        f"min_joint_margin={min_joint_margin:.6f}rad "
                        f"min_tool_aabb_clearance={min_tool_clearance:.6f}m"
                    )
                robot.set_qpos(release_q, zero_velocity=True)
            print("trajectory_preflight=complete; exiting before IPC stepping")
            return

    # A persistent suffix checkpoint is valid only when everything that formed
    # the cloth state through its source frame is identical. Corrections whose
    # first active frame is later than the checkpoint are deliberately omitted,
    # so a frame-332 checkpoint can be reused while tuning fold two and a
    # frame-583 checkpoint can be reused while tuning fold three.
    def build_checkpoint_signature(prefix_source_frame: int) -> dict:
        signature = {
            "format": "genesis-ipc-fold-prefix-v2",
            "prefix_source_frame": int(prefix_source_frame),
            "shirt_obj": str(shirt_obj),
            "shirt_obj_size": shirt_obj.stat().st_size,
            "trajectory": str(trajectory_path),
            "trajectory_size": trajectory_path.stat().st_size,
            "cloth_vertices": int(cloth.n_vertices),
            "trajectory_stride": args.trajectory_stride,
            "action_fps": args.action_fps,
            "physics_dt": physics_dt,
            "substeps": args.substeps,
            "drive_mode": args.drive_mode,
            "robot_coup_type": args.robot_coup_type,
            "two_way_coupling": args.two_way_coupling,
            "contact_d_hat": args.contact_d_hat,
            "cloth_E": args.cloth_E,
            "cloth_rho": args.cloth_rho,
            "cloth_thickness": args.cloth_thickness,
            "cloth_bending": args.cloth_bending,
            "cloth_friction": args.cloth_friction,
            "table_friction": args.table_friction,
            "robot_friction": args.robot_friction,
            "initial_shirt": [args.initial_shirt_x, args.initial_shirt_y, args.initial_shirt_z],
            "settle_frames": args.settle_frames,
            "finger_overclose": args.finger_overclose,
            "right_finger_overclose_extra": args.right_finger_overclose_extra,
            "finger_kp": args.finger_kp,
            "finger_kv": args.finger_kv,
            "exact_finger_collision": args.exact_finger_collision,
            "ipc_constraint_strength_translation": args.ipc_constraint_strength_translation,
            "ipc_constraint_strength_rotation": args.ipc_constraint_strength_rotation,
            "ipc_rigid_rigid_contact": args.ipc_rigid_rigid_contact,
            "first_grasp_clearance_lift": args.first_grasp_clearance_lift,
            "first_grasp_right_depth": args.first_grasp_right_depth,
            "first_fold_tcp_lift": args.first_fold_tcp_lift,
            "first_fold_transfer_lift": args.first_fold_transfer_lift,
            "first_fold_stack_overlap": args.first_fold_stack_overlap,
        }
        if prefix_source_frame >= 340:
            signature.update(
            {
                "second_fold_left_approach_lift": args.second_fold_left_approach_lift,
                "second_fold_right_approach_lift": args.second_fold_right_approach_lift,
                "second_fold_transport_lift": args.second_fold_transport_lift,
                "second_fold_roll_arc_height": args.second_fold_roll_arc_height,
                "second_fold_roll_path": args.second_fold_roll_path,
                "second_fold_lift_first_planar_hold": (
                    args.second_fold_lift_first_planar_hold
                ),
                "second_fold_placement_relax": args.second_fold_placement_relax,
                "second_fold_placement_lift": args.second_fold_placement_lift,
                "second_fold_stack_overlap": args.second_fold_stack_overlap,
                "second_fold_correction_release_start": (
                    args.second_fold_correction_release_start
                ),
                "second_fold_correction_release_end": (
                    args.second_fold_correction_release_end
                ),
            }
        )
        return signature

    persistent_save_source_frame = (
        args.save_checkpoint_source_frame
        if args.save_checkpoint_source_frame is not None
        else args.third_fold_checkpoint_source_frame
    )

    robot.set_qpos(joint_q[0])
    robot.set_dofs_kp(np.full(19, 500.0))
    robot.set_dofs_kv(np.full(19, 50.0))
    robot.set_dofs_kp(
        np.full(len(all_finger_dof_indices), args.finger_kp),
        dofs_idx_local=all_finger_dof_indices,
    )
    robot.set_dofs_kv(
        np.full(len(all_finger_dof_indices), args.finger_kv),
        dofs_idx_local=all_finger_dof_indices,
    )
    if args.drive_mode != "direct":
        robot.control_dofs_position(joint_q[0])
    build_seconds = time.perf_counter() - build_start

    virtual_grasp = None
    if args.virtual_grasp:
        virtual_grasp = IPCVirtualGrasp(
            robot,
            cloth,
            openness,
            radius=args.grasp_radius,
            points=args.grasp_points,
            final_right_points=args.final_right_grasp_points,
            first_candidates=first_grasp_candidates,
            mode=args.grasp_mode,
        )

    contact_diagnostics = None
    if args.contact_grasp_test:
        contact_diagnostics = ContactGraspDiagnostics(
            robot,
            cloth,
            openness,
            finger_indices=finger_dof_indices,
            expected_candidates=first_grasp_candidates,
        )

    third_fold_debug_dir = None
    third_fold_debug_frames = {
        0, 650, 660, 670, 680, 688, 690, 700, 720, 735, 750,
        780, 810, 840, 870, 890, 900, 920, 941, 960, 979,
    }
    right_debug_links = {
        name: robot.get_link(name=name)
        for name in ("right_link26", "right_link27", "right_link28")
    }
    if args.debug_third_fold_dir is not None:
        third_fold_debug_dir = args.debug_third_fold_dir.expanduser().resolve()
        third_fold_debug_dir.mkdir(parents=True, exist_ok=True)

    second_fold_debug_dir = None
    second_fold_debug_frames = {
        0, 332, 385, 393, 405, 415, 430, 439, 455, 480, 505, 530, 550, 570, 583, 600, 619
    }
    both_debug_links = {
        name: robot.get_link(name=name) for name in ("left_link16", "right_link26")
    }
    if args.debug_second_fold_dir is not None:
        second_fold_debug_dir = args.debug_second_fold_dir.expanduser().resolve()
        second_fold_debug_dir.mkdir(parents=True, exist_ok=True)

    def save_second_fold_debug(
        source_frame: int, robot_q: np.ndarray, finger_command: np.ndarray
    ) -> None:
        if second_fold_debug_dir is None or source_frame not in second_fold_debug_frames:
            return
        cloth_pos = as_numpy(cloth.get_state().pos).astype(np.float64).reshape((-1, 3))
        payload: dict[str, np.ndarray] = {
            "cloth_pos": cloth_pos,
            "cloth_centroid": np.mean(cloth_pos, axis=0),
            "cloth_xy_q10": np.quantile(cloth_pos[:, :2], 0.10, axis=0),
            "cloth_xy_q90": np.quantile(cloth_pos[:, :2], 0.90, axis=0),
            "robot_q": as_numpy(robot_q).astype(np.float64).reshape(-1),
            "finger_command": as_numpy(finger_command).astype(np.float64).reshape(-1),
        }
        for name, link in both_debug_links.items():
            pos = as_numpy(link.get_pos(relative=False)).reshape(3).astype(np.float64)
            quat = as_numpy(link.get_quat(relative=False)).reshape(4).astype(np.float64)
            payload[f"{name}_tcp"] = pos + quat_wxyz_to_matrix(quat) @ TCP_LOCAL
            payload[f"{name}_quat_wxyz"] = quat
        np.savez_compressed(second_fold_debug_dir / f"frame_{source_frame:04d}.npz", **payload)
        print(
            f"second_fold_snapshot frame={source_frame} "
            f"centroid={payload['cloth_centroid'].tolist()}"
        )

    def save_third_fold_debug(
        source_frame: int, robot_q: np.ndarray, finger_command: np.ndarray
    ) -> None:
        if third_fold_debug_dir is None or source_frame not in third_fold_debug_frames:
            return
        payload: dict[str, np.ndarray] = {
            "cloth_pos": as_numpy(cloth.get_state().pos).astype(np.float64).reshape((-1, 3)),
            "robot_q": as_numpy(robot_q).astype(np.float64).reshape(-1),
            "finger_command": as_numpy(finger_command).astype(np.float64).reshape(-1),
        }
        payload["cloth_centroid"] = np.mean(payload["cloth_pos"], axis=0)
        tcp_link = right_debug_links["right_link26"]
        tcp_pos = as_numpy(tcp_link.get_pos(relative=False)).reshape(3).astype(np.float64)
        tcp_quat = as_numpy(tcp_link.get_quat(relative=False)).reshape(4).astype(np.float64)
        tcp_rot = quat_wxyz_to_matrix(tcp_quat)
        payload["tcp"] = tcp_pos + tcp_rot @ TCP_LOCAL
        payload["tcp_quat_wxyz"] = tcp_quat
        for name, link in right_debug_links.items():
            payload[f"{name}_pos"] = as_numpy(link.get_pos(relative=False)).reshape(3)
            payload[f"{name}_quat_wxyz"] = as_numpy(link.get_quat(relative=False)).reshape(4)
            # The interactive Viewer renders the Genesis rigid-solver pose above,
            # while cloth contact is resolved against the ABD pose maintained by
            # libuipc. With two_way_soft_constraint these need not coincide under
            # load, especially when reverse force coupling is disabled. Persist
            # both transforms so a visually correct pinch can be distinguished
            # from a physically misaligned IPC collision body.
            ipc_entry = cloth.sim.coupler._abd_data_by_link.get(link)
            if ipc_entry:
                payload[f"{name}_ipc_transform"] = np.asarray(
                    ipc_entry[0].transform, dtype=np.float64
                ).reshape(4, 4)
        np.savez_compressed(third_fold_debug_dir / f"frame_{source_frame:04d}.npz", **payload)
        print(
            f"third_fold_snapshot frame={source_frame} vertices={len(payload['cloth_pos'])} "
            f"tcp={payload['tcp'].tolist()}"
        )

    def finger_targets(frame: int, q: np.ndarray) -> np.ndarray:
        targets = q.copy()
        if args.finger_overclose <= 0.0 and args.right_finger_overclose_extra <= 0.0:
            return targets
        close_weight = np.clip((0.5 - openness[frame]) / 0.5, 0.0, 1.0)
        for hand_index, indices in enumerate(finger_dof_indices):
            overclose = args.finger_overclose
            if hand_index == 1:
                overclose += args.right_finger_overclose_extra
            targets[indices] = np.maximum(
                targets[indices] - overclose * close_weight[hand_index],
                FINGER_URDF_LOWER,
            )
        return targets

    def set_hybrid_kinematic_dofs(frame: int, q: np.ndarray) -> None:
        # Keep the arm trajectory exact. Fingers also follow the recorded qpos
        # while open; each hand is released to its PD actuator only after its
        # recorded openness crosses the close threshold. This avoids accumulating
        # unconstrained finger-controller drift during the long approach phase.
        robot.set_dofs_position(
            q[arm_dof_indices], dofs_idx_local=arm_dof_indices, zero_velocity=True
        )
        for hand_index, indices in enumerate(finger_dof_indices):
            if openness[frame, hand_index] > 0.50:
                robot.set_dofs_position(q[indices], dofs_idx_local=indices, zero_velocity=True)

    def save_reference_cloth_snapshot(
        debug_dir: Path | None, source_frame: int, cloth_pos: np.ndarray
    ) -> None:
        """Write the immutable prefix reference needed by suffix metrics.

        A checkpoint-only run used to start with no frame-0 cloth evidence, so
        its movement metric became null even when all suffix frames existed.
        Future checkpoints carry this diagnostic-only array alongside the
        native IPC state; it is never fed back into the solver.
        """
        if debug_dir is None:
            return
        output = debug_dir / f"frame_{source_frame:04d}.npz"
        if output.is_file():
            return
        positions = np.asarray(cloth_pos, dtype=np.float64).reshape((-1, 3))
        np.savez_compressed(
            output,
            cloth_pos=positions,
            cloth_centroid=np.mean(positions, axis=0),
            diagnostic_reference_only=np.array(True),
        )
        print(f"diagnostic_reference_snapshot frame={source_frame} path={output}")

    def capture_diagnostic_state(source_frame: int, commanded_q: np.ndarray) -> None:
        """Write numeric and visual evidence from one synchronized scene state."""
        current_q = robot.get_qpos()
        save_second_fold_debug(source_frame, current_q, commanded_q)
        save_third_fold_debug(source_frame, current_q, commanded_q)
        if ipc_proxy_visuals is not None:
            ipc_proxy_visuals.update(
                source_frame,
                record=source_frame in KeyframeVisualDiagnostics.DIAGNOSTIC_FRAMES,
            )
        right_link = both_debug_links["right_link26"]
        right_pos = as_numpy(right_link.get_pos(relative=False)).reshape(3)
        right_quat = as_numpy(right_link.get_quat(relative=False)).reshape(4)
        keyframe_visuals.capture(
            source_frame,
            right_pos + quat_wxyz_to_matrix(right_quat) @ TCP_LOCAL,
            ipc_proxy_visuals if args.visualize_ipc_proxies else None,
        )

    start_frame = 0
    previous_q = joint_q[0].copy()
    diagnostic_initial_cloth_pos = None
    settled_cloth_summary = None
    settled_cloth_snapshot_path = None
    loaded_checkpoint_meta = None
    if args.load_third_fold_checkpoint is not None:
        scene_ckpt, ipc_ckpt, meta_ckpt = checkpoint_sidecars(
            args.load_third_fold_checkpoint
        )
        for required_path in (scene_ckpt, ipc_ckpt, meta_ckpt):
            if not required_path.is_file():
                raise FileNotFoundError(required_path)
        loaded_checkpoint_meta = json.loads(meta_ckpt.read_text(encoding="utf-8"))
        saved_signature = loaded_checkpoint_meta.get("signature")
        load_source_frame = int(loaded_checkpoint_meta["source_frame"])
        expected_load_signature = build_checkpoint_signature(load_source_frame)
        if saved_signature != expected_load_signature:
            differing = sorted(
                key
                for key in set(saved_signature or {}) | set(expected_load_signature)
                if (saved_signature or {}).get(key) != expected_load_signature.get(key)
            )
            difference_values = {
                key: {
                    "saved": (saved_signature or {}).get(key),
                    "current": expected_load_signature.get(key),
                }
                for key in differing
            }
            raise RuntimeError(
                "Third-fold checkpoint is incompatible with this prefix configuration; "
                f"differing fields={differing} values={difference_values}"
            )
        scene.load_checkpoint(scene_ckpt)
        scene._t = int(loaded_checkpoint_meta["scene_t"])
        scene._sim._cur_substep_global = int(
            loaded_checkpoint_meta["sim_substep_global"]
        )
        checkpoint_reference_cloth = None
        with np.load(ipc_ckpt) as saved_ipc:
            if "diagnostic_initial_cloth_position" in saved_ipc.files:
                checkpoint_reference_cloth = np.asarray(
                    saved_ipc["diagnostic_initial_cloth_position"], dtype=np.float64
                ).copy()
            restore_ipc_state(
                cloth.sim.coupler,
                {name: np.asarray(saved_ipc[name]).copy() for name in saved_ipc.files},
            )
        start_frame = int(loaded_checkpoint_meta["trajectory_index"]) + 1
        previous_q = np.asarray(
            loaded_checkpoint_meta["previous_q"], dtype=np.float64
        )
        print(
            "persistent_third_fold_checkpoint_loaded "
            f"source_frame={loaded_checkpoint_meta['source_frame']} "
            f"start_index={start_frame} scene={scene_ckpt} ipc={ipc_ckpt}"
        )
        if checkpoint_reference_cloth is not None:
            diagnostic_initial_cloth_pos = checkpoint_reference_cloth
            save_reference_cloth_snapshot(
                second_fold_debug_dir, 0, checkpoint_reference_cloth
            )
            save_reference_cloth_snapshot(
                third_fold_debug_dir, 0, checkpoint_reference_cloth
            )
        # Capture the restored prefix endpoint before the first suffix step.
        # This makes frame 332/583 visible even when no prefix frames execute.
        capture_diagnostic_state(
            int(loaded_checkpoint_meta["source_frame"]), previous_q
        )
    else:
        for settle_frame in range(args.settle_frames):
            if args.drive_mode == "direct":
                robot.set_qpos(joint_q[0], zero_velocity=True)
            elif args.drive_mode == "hybrid":
                set_hybrid_kinematic_dofs(0, joint_q[0])
            if args.drive_mode != "direct":
                robot.control_dofs_position(joint_q[0])
            scene.step()
            if settle_frame == 0 or (settle_frame + 1) % 60 == 0:
                print(f"settle={settle_frame + 1}/{args.settle_frames}")

        settled_positions = (
            as_numpy(cloth.get_state().pos).astype(np.float64).reshape((-1, 3)).copy()
        )
        settled_cloth_summary = summarize_settled_cloth(
            settled_positions, shirt_obj, table_top_z=0.8
        )
        settled_cloth_snapshot_path = args.output.with_name(
            f"{args.output.stem}_settled_cloth.npz"
        )
        settled_cloth_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            settled_cloth_snapshot_path,
            cloth_pos=settled_positions,
            table_top_z=np.array(0.8, dtype=np.float64),
        )
        global_height = settled_cloth_summary["global"]["height_quantiles_m"]
        print(
            "settled_cloth "
            f"median_height_mm={global_height['median'] * 1000.0:.2f} "
            f"p90_height_mm={global_height['p90'] * 1000.0:.2f} "
            f"max_height_mm={global_height['max'] * 1000.0:.2f} "
            f"snapshot={settled_cloth_snapshot_path}"
        )

    recording_paths: dict[str, Path] = {}
    multiview_grid_path = None
    if not args.no_record:
        if args.record_multi_view:
            for view_name, view_camera in record_cameras.items():
                if view_name == primary_record_view:
                    output_path = args.output
                else:
                    output_path = args.output.with_name(
                        f"{args.output.stem}_{view_name}{args.output.suffix}"
                    )
                recording_paths[view_name] = output_path
                view_camera.start_recording(
                    save_to_filename=str(output_path), fps=effective_record_fps
                )
        else:
            recording_paths[primary_record_view] = args.output
            camera.start_recording(save_to_filename=str(args.output), fps=effective_record_fps)

    # Persist the actually executed Cartesian TCP trajectory, not merely the
    # source joint commands.  This file is flushed every frame so it remains
    # useful when an interactive run is paused or interrupted before metrics
    # and videos are finalized.
    tcp_telemetry_path = args.output.with_name(f"{args.output.stem}_tcp_trajectory.csv")
    tcp_telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    tcp_telemetry_file = tcp_telemetry_path.open("w", encoding="utf-8")
    tcp_telemetry_file.write(
        "source_frame,trajectory_index,hand,tcp_x_m,tcp_y_m,tcp_z_m,"
        "step_dx_mm,step_dy_mm,step_dz_mm,step_dxy_mm,openness,"
        "finger_q0_m,finger_q1_m\n"
    )
    previous_tcp_by_hand: dict[str, np.ndarray] = {}

    def write_tcp_telemetry(trajectory_index: int, source_frame: int) -> None:
        actual_q = as_numpy(robot.get_qpos()).reshape(-1)
        for hand_index, (hand_name, link_name) in enumerate(
            (("left", "left_link16"), ("right", "right_link26"))
        ):
            link = both_debug_links[link_name]
            pos = as_numpy(link.get_pos(relative=False)).reshape(3).astype(np.float64)
            quat = as_numpy(link.get_quat(relative=False)).reshape(4).astype(np.float64)
            tcp = pos + quat_wxyz_to_matrix(quat) @ TCP_LOCAL
            previous_tcp = previous_tcp_by_hand.get(hand_name)
            if previous_tcp is None:
                delta_mm = np.full(3, np.nan, dtype=np.float64)
                dxy_mm = float("nan")
            else:
                delta_mm = (tcp - previous_tcp) * 1000.0
                dxy_mm = float(np.linalg.norm(delta_mm[:2]))
            finger_indices = finger_dof_indices[hand_index]
            tcp_telemetry_file.write(
                f"{source_frame},{trajectory_index},{hand_name},"
                f"{tcp[0]:.9f},{tcp[1]:.9f},{tcp[2]:.9f},"
                f"{delta_mm[0]:.6f},{delta_mm[1]:.6f},{delta_mm[2]:.6f},"
                f"{dxy_mm:.6f},{openness[trajectory_index, hand_index]:.6f},"
                f"{actual_q[finger_indices[0]]:.9f},"
                f"{actual_q[finger_indices[1]]:.9f}\n"
            )
            previous_tcp_by_hand[hand_name] = tcp.copy()
        tcp_telemetry_file.flush()

    if loaded_checkpoint_meta is not None:
        restored_index = int(loaded_checkpoint_meta["trajectory_index"])
        write_tcp_telemetry(
            restored_index, int(loaded_checkpoint_meta["source_frame"])
        )

    step_times: list[float] = []
    checkpoint = None
    checkpoint_path = args.output.with_suffix(".third_fold_checkpoint.pkl")
    persistent_checkpoint_saved = False
    for frame in range(start_frame, len(joint_q)):
        target_q = joint_q[frame]
        tick = time.perf_counter()
        for substep in range(args.substeps):
            alpha = (substep + 1) / args.substeps
            interpolated_q = previous_q + alpha * (target_q - previous_q)
            commanded_q = finger_targets(frame, interpolated_q)
            if args.drive_mode == "direct":
                robot.set_qpos(commanded_q, zero_velocity=True)
            elif args.drive_mode == "hybrid":
                set_hybrid_kinematic_dofs(frame, interpolated_q)
            if args.drive_mode != "direct":
                robot.control_dofs_position(commanded_q)
            if virtual_grasp is not None:
                virtual_grasp.update(frame, int(source_frames[frame]))
            if args.record_multi_view:
                right_link = both_debug_links["right_link26"]
                right_pos = as_numpy(right_link.get_pos(relative=False)).reshape(3)
                right_quat = as_numpy(right_link.get_quat(relative=False)).reshape(4)
                right_tcp = right_pos + quat_wxyz_to_matrix(right_quat) @ TCP_LOCAL
                record_cameras["right_grasp"].set_pose(
                    pos=tuple(right_tcp + np.array((0.26, -0.32, 0.18))),
                    lookat=tuple(right_tcp),
                    up=(0.0, 0.0, 1.0),
                )
            scene.step()
            if contact_diagnostics is not None and substep + 1 == args.substeps:
                contact_diagnostics.update(
                    frame, int(source_frames[frame]), robot, commanded_q
                )
            if substep + 1 == args.substeps:
                capture_diagnostic_state(int(source_frames[frame]), commanded_q)
                write_tcp_telemetry(frame, int(source_frames[frame]))
                if int(source_frames[frame]) == 0:
                    diagnostic_initial_cloth_pos = as_numpy(
                        cloth.get_state().pos
                    ).astype(np.float64).reshape((-1, 3)).copy()
            if (
                virtual_grasp is not None
                and substep + 1 == args.substeps
                and int(source_frames[frame]) % 60 < args.trajectory_stride
            ):
                virtual_grasp.report_tracking(int(source_frames[frame]))
        step_times.append(time.perf_counter() - tick)
        previous_q = target_q
        if (
            args.save_third_fold_checkpoint is not None
            and not persistent_checkpoint_saved
            and int(source_frames[frame]) == persistent_save_source_frame
        ):
            scene_ckpt, ipc_ckpt, meta_ckpt = checkpoint_sidecars(
                args.save_third_fold_checkpoint
            )
            scene_ckpt.parent.mkdir(parents=True, exist_ok=True)
            scene.save_checkpoint(scene_ckpt)
            checkpoint_arrays = snapshot_ipc_state(cloth.sim.coupler)
            if diagnostic_initial_cloth_pos is not None:
                checkpoint_arrays["diagnostic_initial_cloth_position"] = (
                    diagnostic_initial_cloth_pos
                )
            np.savez_compressed(ipc_ckpt, **checkpoint_arrays)
            save_signature = build_checkpoint_signature(persistent_save_source_frame)
            persistent_meta = {
                "format": save_signature["format"],
                "source_frame": int(source_frames[frame]),
                "trajectory_index": frame,
                "scene_t": int(scene._t),
                "sim_substep_global": int(scene._sim._cur_substep_global),
                "previous_q": target_q.tolist(),
                "signature": save_signature,
                "diagnostic_reference_source_frame": (
                    0 if diagnostic_initial_cloth_pos is not None else None
                ),
            }
            meta_ckpt.write_text(
                json.dumps(persistent_meta, indent=2), encoding="utf-8"
            )
            persistent_checkpoint_saved = True
            print(
                "persistent_third_fold_checkpoint_saved "
                f"source_frame={int(source_frames[frame])} scene={scene_ckpt} "
                f"ipc={ipc_ckpt} meta={meta_ckpt}"
            )
        if (
            args.verify_third_fold_checkpoint
            and checkpoint is None
            and int(source_frames[frame]) == args.third_fold_checkpoint_source_frame
        ):
            ipc_world = cloth.sim.coupler._ipc_world
            if ipc_world is None:
                raise RuntimeError("IPC world is unavailable at checkpoint time")
            if not ipc_world.dump():
                raise RuntimeError("libuipc failed to dump the third-fold checkpoint")
            scene.save_checkpoint(checkpoint_path)
            checkpoint = {
                "trajectory_index": frame,
                "ipc_frame": int(ipc_world.frame()),
                "scene_t": int(scene._t),
                "sim_substep_global": int(scene._sim._cur_substep_global),
                "previous_q": target_q.copy(),
            }
            print(
                "third_fold_checkpoint_saved "
                f"source_frame={int(source_frames[frame])} "
                f"ipc_frame={checkpoint['ipc_frame']} path={checkpoint_path}"
            )
        if frame == 0 or (frame + 1) % 60 == 0 or frame + 1 == frame_count:
            print(
                f"frame={frame + 1}/{frame_count} "
                f"wall_fps={1.0 / max(np.mean(step_times[-60:]), 1e-9):.2f}"
            )

    # Release first, then retreat along a single seeded IK branch before the
    # free-settle observation.  The public post-release path is intentionally
    # not used: near this reach limit it can switch branches and sweep the wrist
    # through the folded stack.  Camera recording remains active throughout.
    post_release_frames = (
        args.post_release_open_hold_frames
        + args.post_release_retreat_frames
        + args.post_release_settle_frames
    )
    if post_release_frames:
        release_start_q = finger_targets(frame_count - 1, joint_q[-1].copy())
        release_q = release_start_q.copy()
        release_q[all_finger_dof_indices] = FINGER_URDF_UPPER
        retreat_q = release_q.copy()
        retreat_available = False
        if args.post_release_retreat_frames:
            right_link = lift_links[1]
            right_pos = as_numpy(right_link.get_pos(relative=False)).reshape(3)
            right_quat = as_numpy(right_link.get_quat(relative=False)).reshape(4)
            right_tcp = right_pos + quat_wxyz_to_matrix(right_quat) @ TCP_LOCAL
            # The held release pose is close to this arm's reach boundary.
            # Prefer the requested shirt-top/up diagonal, but shrink the
            # horizontal component first and finally fall back to a vertical
            # clearance move.  A failed candidate must never cancel release.
            retreat_candidates = [
                (
                    args.post_release_retreat_top_offset,
                    args.post_release_retreat_height,
                ),
                (
                    0.5 * args.post_release_retreat_top_offset,
                    args.post_release_retreat_height,
                ),
                (
                    0.25 * args.post_release_retreat_top_offset,
                    args.post_release_retreat_height,
                ),
                (0.0, args.post_release_retreat_height),
                (0.0, 0.6 * args.post_release_retreat_height),
            ]
            for candidate_top, candidate_height in retreat_candidates:
                candidate_tcp = right_tcp + np.array(
                    [0.0, candidate_top, candidate_height], dtype=np.float64
                )
                candidate_q, candidate_error = robot.inverse_kinematics(
                    link=right_link,
                    pos=candidate_tcp,
                    quat=right_quat,
                    local_point=TCP_LOCAL,
                    init_qpos=release_q,
                    dofs_idx_local=lift_arm_indices[1],
                    max_samples=1,
                    max_solver_iters=80,
                    pos_tol=1.0e-4,
                    rot_tol=1.0e-4,
                    return_error=True,
                )
                candidate_error_m = float(
                    np.linalg.norm(as_numpy(candidate_error))
                )
                print(
                    "post_release_retreat_candidate "
                    f"top={candidate_top:.4f}m height={candidate_height:.4f}m "
                    f"ik_error={candidate_error_m:.6f}m"
                )
                if candidate_error_m <= 0.005:
                    retreat_q = as_numpy(candidate_q).reshape(-1)
                    retreat_q[all_finger_dof_indices] = FINGER_URDF_UPPER
                    retreat_available = True
                    print(
                        "post_release_retreat_plan "
                        f"top={candidate_top:.4f}m "
                        f"height={candidate_height:.4f}m "
                        f"ik_error={candidate_error_m:.6f}m"
                    )
                    break
            if not retreat_available:
                print(
                    "post_release_retreat_plan unavailable; "
                    "continuing with full release and stationary free settle"
                )

        def command_post_release(q: np.ndarray) -> None:
            if args.drive_mode == "direct":
                robot.set_qpos(q, zero_velocity=True)
            elif args.drive_mode == "hybrid":
                set_hybrid_kinematic_dofs(frame_count - 1, q)
            if args.drive_mode != "direct":
                robot.control_dofs_position(q)

        for hold_frame in range(args.post_release_open_hold_frames):
            blend = smoothstep(
                (hold_frame + 1) / float(args.post_release_open_hold_frames)
            )
            commanded_release_q = release_start_q + blend * (
                release_q - release_start_q
            )
            command_post_release(commanded_release_q)
            scene.step()
            if hold_frame == 0 or hold_frame + 1 == args.post_release_open_hold_frames:
                print(
                    "post_release_open_hold="
                    f"{hold_frame + 1}/{args.post_release_open_hold_frames}"
                )

        active_retreat_frames = (
            args.post_release_retreat_frames if retreat_available else 0
        )
        for retreat_frame in range(active_retreat_frames):
            blend = smoothstep(
                (retreat_frame + 1) / float(active_retreat_frames)
            )
            commanded_retreat_q = release_q + blend * (retreat_q - release_q)
            command_post_release(commanded_retreat_q)
            scene.step()
            if retreat_frame == 0 or retreat_frame + 1 == args.post_release_retreat_frames:
                print(
                    "post_release_retreat="
                    f"{retreat_frame + 1}/{active_retreat_frames}"
                )

        for settle_frame in range(args.post_release_settle_frames):
            command_post_release(retreat_q)
            scene.step()
            if settle_frame == 0 or (settle_frame + 1) % 60 == 0:
                print(
                    "post_release_settle="
                    f"{settle_frame + 1}/{args.post_release_settle_frames}"
                )

    checkpoint_verify = None
    if (
        args.save_third_fold_checkpoint is not None and not persistent_checkpoint_saved
    ):
        raise RuntimeError(
            "Requested persistent checkpoint source frame was not present: "
            f"{persistent_save_source_frame}"
        )
    tcp_telemetry_file.close()
    if args.verify_third_fold_checkpoint:
        if checkpoint is None:
            raise RuntimeError(
                "Requested checkpoint source frame was not present in this trajectory: "
                f"{args.third_fold_checkpoint_source_frame}"
            )
        first_final_cloth = as_numpy(cloth.get_state().pos).astype(np.float64).reshape((-1, 3)).copy()
        scene.load_checkpoint(checkpoint_path)
        scene._t = checkpoint["scene_t"]
        scene._sim._cur_substep_global = checkpoint["sim_substep_global"]
        ipc_world = cloth.sim.coupler._ipc_world
        if not ipc_world.recover(checkpoint["ipc_frame"]):
            raise RuntimeError(
                f"libuipc failed to recover checkpoint frame {checkpoint['ipc_frame']}"
            )
        ipc_world.retrieve()
        replay_previous_q = checkpoint["previous_q"].copy()
        replay_start = checkpoint["trajectory_index"] + 1
        print(
            "third_fold_checkpoint_recovered "
            f"source_frame={args.third_fold_checkpoint_source_frame} "
            f"replaying={frame_count - replay_start} frames"
        )
        replay_times: list[float] = []
        for frame in range(replay_start, frame_count):
            target_q = joint_q[frame]
            tick = time.perf_counter()
            for substep in range(args.substeps):
                alpha = (substep + 1) / args.substeps
                interpolated_q = replay_previous_q + alpha * (target_q - replay_previous_q)
                commanded_q = finger_targets(frame, interpolated_q)
                if args.drive_mode == "direct":
                    robot.set_qpos(commanded_q, zero_velocity=True)
                elif args.drive_mode == "hybrid":
                    set_hybrid_kinematic_dofs(frame, interpolated_q)
                if args.drive_mode != "direct":
                    robot.control_dofs_position(commanded_q)
                scene.step(update_visualizer=False)
            replay_times.append(time.perf_counter() - tick)
            replay_previous_q = target_q
        replay_final_cloth = as_numpy(cloth.get_state().pos).astype(np.float64).reshape((-1, 3))
        vertex_error = np.linalg.norm(replay_final_cloth - first_final_cloth, axis=1)
        checkpoint_verify = {
            "checkpoint_source_frame": args.third_fold_checkpoint_source_frame,
            "ipc_frame": checkpoint["ipc_frame"],
            "replayed_frames": frame_count - replay_start,
            "rms_vertex_error_m": float(np.sqrt(np.mean(vertex_error**2))),
            "max_vertex_error_m": float(np.max(vertex_error)),
            "mean_replay_wall_fps": float(1.0 / np.mean(replay_times)),
        }
        print(f"third_fold_checkpoint_verify={checkpoint_verify}")

    if not args.no_record:
        if args.record_multi_view:
            for view_camera in record_cameras.values():
                view_camera.stop_recording()
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                try:
                    import imageio_ffmpeg

                    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                except (ImportError, RuntimeError):
                    ffmpeg = None
            if ffmpeg is None:
                print("Multi-view grid skipped: ffmpeg is not installed")
            else:
                ordered_views = ("overview", "overhead", "shirt_bottom", "right_grasp")
                multiview_grid_path = args.output.with_name(
                    f"{args.output.stem}_multiview{args.output.suffix}"
                )
                command = [ffmpeg, "-y"]
                for view_name in ordered_views:
                    command.extend(("-i", str(recording_paths[view_name])))
                # Fixed layout: top-left overview, top-right overhead,
                # bottom-left shirt-bottom, bottom-right moving right-grasp.
                filter_graph = (
                    "[0:v][1:v][2:v][3:v]"
                    "xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:fill=black[out]"
                )
                command.extend(
                    (
                        "-filter_complex",
                        filter_graph,
                        "-map",
                        "[out]",
                        "-c:v",
                        "libx264",
                        "-crf",
                        "20",
                        "-preset",
                        "medium",
                        "-pix_fmt",
                        "yuv420p",
                        str(multiview_grid_path),
                    )
                )
                try:
                    subprocess.run(command, check=True)
                    print(f"Multi-view video: {multiview_grid_path}")
                except subprocess.CalledProcessError as error:
                    print(f"Multi-view grid failed (individual videos are intact): {error}")
                    multiview_grid_path = None
        else:
            camera.stop_recording()

    keyframe_contact_sheet = keyframe_visuals.finish()
    ipc_proxy_pose_diagnostics = (
        ipc_proxy_visuals.finish() if ipc_proxy_visuals is not None else None
    )
    second_fold_motion_summary = summarize_second_fold_motion(second_fold_debug_dir)
    fold_layering_summary = summarize_fold_layering(second_fold_debug_dir)
    third_fold_motion_summary = summarize_third_fold_motion(third_fold_debug_dir)

    # Viewer recording is independent of Camera recording. The stock Viewer
    # deletes its temporary video when the window closes if the user started
    # recording but did not press "Stop Rec" before the script ended. Preserve
    # that recording automatically instead of silently losing it.
    viewer_recording_path = None
    if args.viewer and scene.viewer.recording:
        viewer_recording_path = args.output.with_name(f"{args.output.stem}_viewer.mp4")
        pyrender_viewer = scene.viewer._pyrender_viewer
        with pyrender_viewer.render_lock:
            pyrender_viewer.save_video(str(viewer_recording_path))
            pyrender_viewer.viewer_flags["record"] = False
            pyrender_viewer.set_caption(pyrender_viewer.viewer_flags["window_title"])
        print(f"Viewer video: {viewer_recording_path}")

    metrics = {
        "backend": "Genesis IPC / FEM.Cloth",
        "sim1_root": str(args.sim1_root.resolve()),
        "urdf": str(urdf),
        "shirt_obj": str(shirt_obj),
        "trajectory": str(trajectory_path),
        "source_joint_order": SOURCE_JOINT_NAMES,
        "genesis_joint_order": genesis_joint_names,
        "genesis_from_source_columns": genesis_from_source.tolist(),
        "frames": frame_count,
        "executed_frames": frame_count - start_frame,
        "start_trajectory_index": start_frame,
        "source_frames": source_frame_count,
        "trajectory_stride": args.trajectory_stride,
        "action_fps": args.action_fps,
        "record_fps": effective_record_fps,
        "robot_urdf": str(urdf),
        "seed": args.seed,
        "physics_dt": physics_dt,
        "substeps": args.substeps,
        "drive_mode": args.drive_mode,
        "two_way_coupling": args.two_way_coupling,
        "robot_coup_type": args.robot_coup_type,
        "ipc_constraint_strength_translation": args.ipc_constraint_strength_translation,
        "ipc_constraint_strength_rotation": args.ipc_constraint_strength_rotation,
        "ipc_rigid_rigid_contact": args.ipc_rigid_rigid_contact,
        "first_grasp_clearance_lift": args.first_grasp_clearance_lift,
        "first_grasp_right_depth": args.first_grasp_right_depth,
        "settle_frames": args.settle_frames,
        "settled_cloth_summary": settled_cloth_summary,
        "settled_cloth_snapshot": (
            str(settled_cloth_snapshot_path)
            if settled_cloth_snapshot_path is not None
            else None
        ),
        "initial_shirt_x": args.initial_shirt_x,
        "initial_shirt_y": args.initial_shirt_y,
        "initial_shirt_z": args.initial_shirt_z,
        "contact_d_hat": args.contact_d_hat,
        "contact_constitution": args.contact_constitution,
        "visualize_ipc_proxies": args.visualize_ipc_proxies,
        "render_ipc_actual_visuals": args.render_ipc_actual_visuals,
        "ipc_proxy_pose_diagnostics": (
            str(ipc_proxy_pose_diagnostics) if ipc_proxy_pose_diagnostics else None
        ),
        "cloth_E": args.cloth_E,
        "cloth_rho": args.cloth_rho,
        "cloth_thickness": args.cloth_thickness,
        "cloth_areal_density_kg_m2": args.cloth_rho * args.cloth_thickness,
        "cloth_bending": args.cloth_bending,
        "cloth_friction": args.cloth_friction,
        "table_friction": args.table_friction,
        "robot_friction": args.robot_friction,
        "fast_preview": args.fast_preview,
        "solver": {
            "newton_max_iterations": newton_max_iterations,
            "linesearch_iterations": linesearch_iterations,
            "newton_tolerance": newton_tolerance,
            "newton_translation_tolerance": newton_translation_tolerance,
            "linear_system_tolerance": linear_system_tolerance,
        },
        "virtual_grasp": args.virtual_grasp,
        "grasp_mode": args.grasp_mode,
        "kinematic_grasp_demo": kinematic_grasp_demo,
        "contact_enable": not kinematic_grasp_demo,
        "camera_view": args.camera_view,
        "record_multi_view": args.record_multi_view,
        "multi_view_recordings": {
            view_name: str(path) for view_name, path in recording_paths.items()
        },
        "multi_view_grid": str(multiview_grid_path) if multiview_grid_path else None,
        "tcp_trajectory_csv": str(tcp_telemetry_path),
        "grasp_radius": args.grasp_radius,
        "grasp_points": args.grasp_points,
        "final_right_grasp_points": args.final_right_grasp_points,
        "grasp_strength": args.grasp_strength,
        "exact_finger_collision": args.exact_finger_collision,
        "grasp_summary": virtual_grasp.summary() if virtual_grasp is not None else None,
        "contact_grasp_test": args.contact_grasp_test,
        "finger_overclose": args.finger_overclose,
        "right_finger_overclose_extra": args.right_finger_overclose_extra,
        "first_fold_tcp_lift": args.first_fold_tcp_lift,
        "first_fold_transfer_lift": args.first_fold_transfer_lift,
        "first_fold_stack_overlap": args.first_fold_stack_overlap,
        "second_fold_left_approach_lift": args.second_fold_left_approach_lift,
        "second_fold_right_approach_lift": args.second_fold_right_approach_lift,
        "second_fold_transport_lift": args.second_fold_transport_lift,
        "second_fold_roll_arc_height": args.second_fold_roll_arc_height,
        "second_fold_roll_path": args.second_fold_roll_path,
        "second_fold_lift_first_planar_hold": args.second_fold_lift_first_planar_hold,
        "second_fold_placement_relax": args.second_fold_placement_relax,
        "second_fold_placement_lift": args.second_fold_placement_lift,
        "second_fold_stack_overlap": args.second_fold_stack_overlap,
        "second_fold_correction_release_start": (
            args.second_fold_correction_release_start
        ),
        "second_fold_correction_release_end": args.second_fold_correction_release_end,
        "third_fold_right_grasp_lift": args.third_fold_right_grasp_lift,
        "third_fold_right_grasp_depth": args.third_fold_right_grasp_depth,
        "third_fold_right_grasp_lateral": args.third_fold_right_grasp_lateral,
        "third_fold_right_grasp_world_x": args.third_fold_right_grasp_world_x,
        "third_fold_right_grasp_world_y": args.third_fold_right_grasp_world_y,
        "third_fold_placement_depth": args.third_fold_placement_depth,
        "third_fold_post_close_lift": args.third_fold_post_close_lift,
        "third_fold_outward_pull_cancel": args.third_fold_outward_pull_cancel,
        "third_fold_shirt_top_offset": args.third_fold_shirt_top_offset,
        "third_fold_placement_level": args.third_fold_placement_level,
        "third_fold_front_plane_roll_deg": args.third_fold_front_plane_roll_deg,
        "third_fold_smooth_rotation": args.third_fold_smooth_rotation,
        "debug_third_fold_dir": (
            str(third_fold_debug_dir) if third_fold_debug_dir is not None else None
        ),
        "debug_second_fold_dir": (
            str(second_fold_debug_dir) if second_fold_debug_dir is not None else None
        ),
        "keyframe_diagnostics_dir": (
            str(args.keyframe_diagnostics_dir.expanduser().resolve())
            if args.keyframe_diagnostics_dir is not None
            else None
        ),
        "keyframe_contact_sheet": (
            str(keyframe_contact_sheet) if keyframe_contact_sheet is not None else None
        ),
        "second_fold_motion_summary": second_fold_motion_summary,
        "fold_layering_summary": fold_layering_summary,
        "third_fold_motion_summary": third_fold_motion_summary,
        "viewer_recording": str(viewer_recording_path) if viewer_recording_path else None,
        "third_fold_checkpoint_verify": checkpoint_verify,
        "persistent_third_fold_checkpoint_saved": (
            str(checkpoint_sidecars(args.save_third_fold_checkpoint)[0])
            if persistent_checkpoint_saved
            else None
        ),
        "persistent_checkpoint_saved_source_frame": (
            persistent_save_source_frame if persistent_checkpoint_saved else None
        ),
        "persistent_third_fold_checkpoint_loaded": (
            str(checkpoint_sidecars(args.load_third_fold_checkpoint)[0])
            if loaded_checkpoint_meta is not None
            else None
        ),
        "persistent_checkpoint_loaded_source_frame": (
            int(loaded_checkpoint_meta["source_frame"])
            if loaded_checkpoint_meta is not None
            else None
        ),
        "finger_kp": args.finger_kp,
        "finger_kv": args.finger_kv,
        "contact_grasp_summary": (
            contact_diagnostics.summary() if contact_diagnostics is not None else None
        ),
        "first_grasp_candidates": (
            [part.tolist() for part in first_grasp_candidates]
            if first_grasp_candidates is not None
            else None
        ),
        "build_seconds": build_seconds,
        "wall_seconds": float(np.sum(step_times)),
        "mean_wall_fps": float(1.0 / np.mean(step_times)),
        "cloth": {
            "E": args.cloth_E,
            "rho": args.cloth_rho,
            "thickness": args.cloth_thickness,
            "bending_stiffness": args.cloth_bending,
            "friction_mu": args.cloth_friction,
        },
        "entities": {"table": str(table.uid), "robot": str(robot.uid), "cloth": str(cloth.uid)},
    }
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics: {metrics_path}")
    if not args.no_record:
        print(f"Video: {args.output}")


if __name__ == "__main__":
    main()
