#!/usr/bin/env python3
"""Local click-to-measure TCP trajectory workbench."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import webbrowser
from collections import Counter, OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw

try:
    from .trajectory_compiler import PositionEdit, PositionLimits, compile_position_edits
except ImportError:
    from trajectory_compiler import PositionEdit, PositionLimits, compile_position_edits


def require_file(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_drop_controls(value, binding):
    """Validate preview metadata without treating it as executable trajectory."""
    if not binding or not isinstance(value, dict):
        raise ValueError('当前配置不支持独立释放高度调试')
    if value.get('schema') != 'workbench-drop-height-controls-v1':
        raise ValueError('Unknown drop-controls schema')
    table=float(binding['table_z_m'])
    if value.get('table_z_m') != table or value.get('click_projection') not in ('table_plane','cloth_surface'):
        raise ValueError('Invalid reference plane/projection mode')
    result=dict(schema=value['schema'],table_z_m=table,click_projection=value['click_projection'])
    for key in ('transport_z_world_m','release_z_world_m'):
        values=value.get(key)
        if not isinstance(values,dict) or set(values)!={'left','right'}:
            raise ValueError('Independent heights require left/right values')
        if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) or not table<=v<=table+.5 for v in values.values()):
            raise ValueError('Heights must be0..500mm above table')
        result[key]={h:float(values[h]) for h in ('left','right')}
    result.update(target_semantics='tcp_release_xy_and_independent_heights_not_final_cloth_position',validation='preview_only_not_IK_or_physics')
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def majority(values: np.ndarray) -> int:
    return int(Counter(int(value) for value in values).most_common(1)[0][0])


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    pos: np.ndarray
    right: np.ndarray
    true_up: np.ndarray
    forward: np.ndarray
    focal: float

    @classmethod
    def from_config(cls, config: dict) -> "CameraModel":
        width, height = (int(value) for value in config["resolution"])
        pos = np.asarray(config["pos_m"], dtype=np.float64)
        lookat = np.asarray(config["lookat_m"], dtype=np.float64)
        nominal_up = unit(np.asarray(config["up"], dtype=np.float64))
        forward = unit(lookat - pos)
        right = unit(np.cross(forward, nominal_up))
        true_up = unit(np.cross(right, forward))
        focal = 0.5 * height / math.tan(math.radians(float(config["vertical_fov_deg"])) * 0.5)
        return cls(width, height, pos, right, true_up, forward, focal)

    def project(self, point: np.ndarray) -> tuple[float, float] | None:
        relative = np.asarray(point, dtype=np.float64) - self.pos
        depth = float(np.dot(relative, self.forward))
        if depth <= 1.0e-9:
            return None
        x_camera = float(np.dot(relative, self.right))
        y_camera = float(np.dot(relative, self.true_up))
        u = self.width * 0.5 + self.focal * x_camera / depth - 0.5
        v = self.height * 0.5 - self.focal * y_camera / depth - 0.5
        return u, v

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        x_camera = (float(u) + 0.5 - self.width * 0.5) / self.focal
        y_camera = (self.height * 0.5 - (float(v) + 0.5)) / self.focal
        direction = unit(self.right * x_camera + self.true_up * y_camera + self.forward)
        return self.pos.copy(), direction


class WorkbenchData:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        checkpoint_catalog_value = self.config.get("checkpoint_catalog")
        self.checkpoint_catalog_path = (
            require_file(checkpoint_catalog_value) if checkpoint_catalog_value else None
        )
        self.checkpoint_catalog = (
            json.loads(self.checkpoint_catalog_path.read_text(encoding="utf-8"))
            if self.checkpoint_catalog_path is not None
            else None
        )
        if self.checkpoint_catalog is not None:
            segment_ids = {
                str(segment["id"]) for segment in self.checkpoint_catalog.get("segments", [])
            }
            active_segment = str(self.config.get("active_segment", ""))
            if active_segment not in segment_ids:
                raise ValueError(
                    f"active_segment {active_segment!r} is not in checkpoint catalog {sorted(segment_ids)}"
                )
        self.replay_path = require_file(self.config["replay"])
        self.cloth_obj_path = require_file(self.config["cloth_obj"])
        self.atlas_path = require_file(self.config["atlas"])
        self.patch_output_dir = Path(self.config["patch_output_dir"]).expanduser().resolve()
        self.tcp_paths = [require_file(value) for value in self.config["tcp_csvs"]]
        self.camera = CameraModel.from_config(self.config["camera"])
        self.editor_config = self.config.get("trajectory_editor")
        self.display_frame_range = tuple(
            int(value) for value in self.config.get(
                "display_frame_range", [0, 2**31 - 1]
            )
        )

        with np.load(self.replay_path) as replay:
            self.source_frames = np.asarray(replay["source_frames"], dtype=np.int32)
            self.cloth_pos = np.asarray(replay["cloth_pos"], dtype=np.float64)
            self.robot_q = np.asarray(replay["robot_q"], dtype=np.float64)
        if self.cloth_pos.shape[0] != len(self.source_frames):
            raise ValueError("Replay source_frames and cloth_pos length mismatch")
        self.source_to_index = {int(frame): index for index, frame in enumerate(self.source_frames)}
        if len(self.source_to_index) != len(self.source_frames):
            raise ValueError("Replay source frames are not unique")

        loaded = trimesh.load(str(self.cloth_obj_path), process=False, maintain_order=True)
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"Expected one cloth Trimesh, got {type(loaded)!r}")
        self.faces = np.asarray(loaded.faces, dtype=np.int64)
        n_vertices=len(loaded.vertices)
        if self.faces.ndim!=2 or self.faces.shape[1]!=3 or self.cloth_pos.shape[1:]!=(n_vertices,3):
            raise ValueError(f"Cloth OBJ/replay topology mismatch: faces={self.faces.shape}, pos={self.cloth_pos.shape}")
        for key,actual in [('expected_cloth_vertices',n_vertices),('expected_cloth_faces',len(self.faces))]:
            if key in self.config and int(self.config[key])!=actual:
                raise ValueError(f'{key} mismatch: {actual}')

        with np.load(self.atlas_path) as atlas:
            self.canonical_uv = np.asarray(atlas["canonical_uv"], dtype=np.float64)
            self.width_band = np.asarray(atlas["width_band"], dtype=np.int8)
            self.length_zone = np.asarray(atlas["length_zone"], dtype=np.int8)
            self.surface_layer = np.asarray(atlas["surface_layer"], dtype=np.int8)
            if 'rest_raw_xyz' in atlas and not np.allclose(atlas['rest_raw_xyz'],loaded.vertices,rtol=0,atol=1e-8):
                raise ValueError('Atlas source vertices differ from selected OBJ; do not reuse another mesh atlas')
        if self.canonical_uv.shape!=(n_vertices,2) or any(a.shape!=(n_vertices,) for a in (self.width_band,self.length_zone,self.surface_layer)):
            raise ValueError('Atlas topology differs from replay/OBJ')
        atlas_metadata=self.atlas_path.with_suffix('.json')
        if atlas_metadata.is_file():
            atlas_meta=json.loads(atlas_metadata.read_text())
            if atlas_meta.get('mesh_sha256') and atlas_meta['mesh_sha256']!=sha256(self.cloth_obj_path):
                raise ValueError('Atlas mesh hash differs from selected OBJ')
        self.tcp = self._load_tcp()
        self._frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._mesh_cache: OrderedDict[int, trimesh.Trimesh] = OrderedDict()
        self._video_lock = threading.Lock()
        self._ik_lock = threading.Lock()

        self.background_mode=self.config.get('background_mode','video')
        if self.background_mode=='replay_cpu':
            self.video_segments=[]
            self.video_fps=float(self.config.get('replay_fps',60.))
            return
        if self.background_mode!='video':raise ValueError('Unknown Workbench background mode')

        raw_segments = self.config.get("video_segments")
        if raw_segments is None:
            raw_segments = [{
                "source_range": [int(self.source_frames.min()), int(self.source_frames.max())],
                "video": self.config["video"],
                "video_frame_start": 0,
            }]
        self.video_segments = []
        fps_values = []
        for raw in raw_segments:
            path = require_file(raw["video"])
            source_start, source_end = (int(value) for value in raw["source_range"])
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            capture.release()
            if (width, height) != (self.camera.width, self.camera.height):
                raise ValueError(
                    f"Video and configured camera resolution differ: {path} "
                    f"has {width}x{height}, config has {self.camera.width}x{self.camera.height}"
                )
            video_frame_start = int(raw.get("video_frame_start", 0))
            required = source_end - source_start + 1
            if video_frame_start < 0 or video_frame_start + required > frame_count:
                raise ValueError(
                    f"Video segment cannot cover source frames {source_start}..{source_end}: "
                    f"{path} has {frame_count} frames"
                )
            png_dir = raw.get("png_dir")
            self.video_segments.append({
                "source_start": source_start,
                "source_end": source_end,
                "video_frame_start": video_frame_start,
                "path": path,
                "png_dir": None if not png_dir else Path(png_dir).expanduser().resolve(),
            })
            fps_values.append(fps)
        if not fps_values or max(fps_values) - min(fps_values) > 1.0e-6:
            raise ValueError(f"Video segment FPS values differ: {fps_values}")
        self.video_fps = fps_values[0]

    def _load_tcp(self) -> dict[int, dict[str, dict]]:
        result: dict[int, dict[str, dict]] = {}
        for path in self.tcp_paths:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    frame = int(row["source_frame"])
                    hand = str(row["hand"])
                    result.setdefault(frame, {})[hand] = {
                        "world_m": [float(row["tcp_x_m"]), float(row["tcp_y_m"]), float(row["tcp_z_m"])],
                        "openness": float(row["openness"]),
                    }
        return result

    def _video_location(self, source_frame: int) -> tuple[dict, int]:
        for segment in self.video_segments:
            if segment["source_start"] <= source_frame <= segment["source_end"]:
                local = segment["video_frame_start"] + source_frame - segment["source_start"]
                return segment, int(local)
        raise ValueError(f"No video segment covers source frame {source_frame}")

    def frame_image(self, source_frame: int) -> np.ndarray:
        if source_frame in self._frame_cache:
            frame = self._frame_cache.pop(source_frame)
            self._frame_cache[source_frame] = frame
            return frame.copy()
        if self.background_mode=='replay_cpu':
            try:
                from .replay_cpu_background import render_frame
            except ImportError:
                from replay_cpu_background import render_frame
            with self._video_lock:
                frame=render_frame(self,source_frame)
            self._frame_cache[source_frame]=frame
            while len(self._frame_cache)>6:self._frame_cache.popitem(last=False)
            return frame.copy()
        segment, local_frame = self._video_location(source_frame)
        with self._video_lock:
            capture = cv2.VideoCapture(str(segment["path"]))
            capture.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
            ok, frame = capture.read()
            capture.release()
        if not ok:
            raise RuntimeError(
                f"Could not decode source frame {source_frame} "
                f"(video frame {local_frame} in {segment['path']})"
            )
        self._frame_cache[source_frame] = frame
        while len(self._frame_cache) > 6:
            self._frame_cache.popitem(last=False)
        return frame.copy()

    def frame_png_path(self, source_frame: int) -> Path | None:
        if self.background_mode=='replay_cpu':return None
        segment, _ = self._video_location(source_frame)
        png_dir = segment["png_dir"]
        if png_dir is None:
            return None
        path = png_dir / f"source_{source_frame:04d}.png"
        return path if path.is_file() else None

    def mesh(self, source_frame: int) -> trimesh.Trimesh:
        if source_frame in self._mesh_cache:
            mesh = self._mesh_cache.pop(source_frame)
            self._mesh_cache[source_frame] = mesh
            return mesh
        index = self.source_to_index.get(source_frame)
        if index is None:
            raise ValueError(f"Replay has no source frame {source_frame}")
        mesh = trimesh.Trimesh(vertices=self.cloth_pos[index], faces=self.faces, process=False)
        self._mesh_cache[source_frame] = mesh
        while len(self._mesh_cache) > 4:
            self._mesh_cache.popitem(last=False)
        return mesh

    def stage_hint(self, frame: int) -> str:
        if self.editor_config:
            for phase in self.editor_config.get("phases", []):
                start, end = (int(value) for value in phase["range"])
                if start <= frame <= end:
                    return f"{self.editor_config.get('label', self.editor_config['stage'])} / {phase['label']}"
        if frame < 45:
            return "initial / settle"
        if frame <= 339:
            return "first_fold"
        if frame <= 392:
            return "second_fold / approach"
        if frame <= 505:
            return "second_fold / grasp_peel"
        if frame <= 570:
            return "second_fold / transport_place"
        if frame <= 619:
            return "second_fold / release"
        if frame <= 689:
            return "third_fold / approach"
        if frame <= 780:
            return "third_fold / grasp_lift"
        if frame <= 920:
            return "third_fold / transport_place"
        return "third_fold / release_retreat"

    def frame_info(self, source_frame: int) -> dict:
        hands = {}
        for hand, value in self.tcp.get(source_frame, {}).items():
            pixel = self.camera.project(np.asarray(value["world_m"], dtype=np.float64))
            hands[hand] = {**value, "pixel_uv": list(pixel) if pixel else None}
        return {
            "source_frame": source_frame,
            "stage_hint": self.stage_hint(source_frame),
            "hands": hands,
            "video_fps": self.video_fps,
            "resolution": [self.camera.width, self.camera.height],
        }

    def _hand_config(self, hand: str) -> dict:
        if not self.editor_config:
            raise ValueError("No trajectory_editor configured")
        hands = self.editor_config.get("hands")
        if hands is None:
            configured_hand = str(self.editor_config["hand"])
            if hand != configured_hand:
                raise ValueError(f"Stage only configures hand {configured_hand}")
            return {
                "label": configured_hand,
                "color": "#ff6b78" if hand == "right" else "#48d58a",
                "nodes": self.editor_config["nodes"],
            }
        if hand not in hands:
            raise ValueError(f"Stage has no hand {hand}; available={list(hands)}")
        return hands[hand]

    def _stage_base(self, hand: str) -> tuple[np.ndarray, np.ndarray]:
        if not self.editor_config:
            raise ValueError("No trajectory_editor configured")
        start, end = (int(value) for value in self.editor_config["frame_range"])
        frames = np.arange(start, end + 1, dtype=np.int64)
        missing = [int(frame) for frame in frames if hand not in self.tcp.get(int(frame), {})]
        if missing:
            raise ValueError(f"TCP telemetry missing {hand} frames: {missing[:8]}")
        positions = np.array([self.tcp[int(frame)][hand]["world_m"] for frame in frames])
        return frames, positions

    def compile_stage(self, requested_edits: list[dict], hand: str | None = None) -> dict:
        hand = str(hand or self.editor_config.get("default_hand", self.editor_config.get("hand", "right")))
        hand_config = self._hand_config(hand)
        frames, base = self._stage_base(hand)
        node_by_frame = {
            int(node["frame"]): node for node in hand_config["nodes"]
        }
        edits = []
        all_edits = []
        semantic_values: dict[int, np.ndarray] = {}
        configured_semantic_path = hand_config.get("semantic_path")
        semantic_node_frames = {
            int(item["node_frame"])
            for item in (configured_semantic_path or [])
            if "node_frame" in item
        }
        for request in requested_edits:
            frame = int(request["frame"])
            node = node_by_frame.get(frame)
            if node is None or not bool(node["draggable"]):
                raise ValueError(f"Frame is not an editable trajectory node: {frame}")
            edit = PositionEdit(
                frame=frame,
                delta_m=np.asarray(request["delta_m"], dtype=np.float64),
                support_radius=int(node["support_radius"]),
                mode=str(node.get("mode", "local")),
            )
            all_edits.append(edit)
            if frame in semantic_node_frames or (
                configured_semantic_path is None
                and self.editor_config["stage"] == "second_fold"
                and any(
                    token in str(node["name"])
                    for token in ("pregrasp", "material", "apex", "placement")
                )
            ):
                semantic_values[frame] = edit.delta_m
            else:
                edits.append(edit)
        semantic_keyframes = None
        if configured_semantic_path is not None:
            fallback_by_node = {
                int(item["node_frame"]): int(item["fallback_node_frame"])
                for item in configured_semantic_path
                if "node_frame" in item and "fallback_node_frame" in item
            }

            def semantic_value(node_frame: int, seen: set[int] | None = None) -> np.ndarray:
                node_frame = int(node_frame)
                if node_frame in semantic_values:
                    return semantic_values[node_frame]
                seen = set() if seen is None else set(seen)
                if node_frame in seen:
                    raise ValueError("semantic_path fallback cycle")
                seen.add(node_frame)
                fallback = fallback_by_node.get(node_frame)
                if fallback is None:
                    return np.zeros(3, dtype=np.float64)
                return semantic_value(fallback, seen)

            semantic_keyframes = []
            for item in configured_semantic_path:
                frame = int(item["frame"])
                if item.get("zero", False):
                    value = np.zeros(3, dtype=np.float64)
                elif "node_frame" in item:
                    value = semantic_value(int(item["node_frame"]))
                else:
                    raise ValueError(
                        "semantic_path entries require zero=true or node_frame"
                    )
                semantic_keyframes.append((frame, value))
        elif self.editor_config["stage"] == "second_fold":
            by_name = {str(node["name"]): node for node in hand_config["nodes"]}
            pregrasp = next(node for name, node in by_name.items() if "pregrasp" in name)
            material = next(node for name, node in by_name.items() if "material" in name)
            pinched = next(node for name, node in by_name.items() if "pinched" in name)
            apex = next(node for name, node in by_name.items() if "apex" in name)
            placement = next(node for name, node in by_name.items() if "placement" in name)
            pregrasp_frame = int(pregrasp["frame"])
            material_frame = int(material["frame"])
            apex_frame = int(apex["frame"])
            placement_frame = int(placement["frame"])
            pregrasp_delta = semantic_values.get(pregrasp_frame, np.zeros(3, dtype=np.float64))
            target_delta = semantic_values.get(material_frame, np.zeros(3, dtype=np.float64))
            apex_delta = semantic_values.get(apex_frame, np.zeros(3, dtype=np.float64))
            placement_delta = semantic_values.get(placement_frame, np.zeros(3, dtype=np.float64))
            semantic_keyframes = [
                (int(self.editor_config["frame_range"][0]), np.zeros(3, dtype=np.float64)),
                (pregrasp_frame, pregrasp_delta),
                (material_frame, target_delta),
                (int(pinched["frame"]), target_delta),
                (apex_frame, apex_delta),
                (placement_frame, placement_delta),
                (int(self.editor_config["frame_range"][1]), placement_delta),
            ]
        limit_config = self.editor_config["limits"]
        result = compile_position_edits(
            frames,
            base,
            edits,
            stage_range=tuple(int(value) for value in self.editor_config["frame_range"]),
            fps=float(self.video_fps),
            limits=PositionLimits(
                max_step_mm=float(limit_config["max_step_mm"]),
                max_speed_mps=float(limit_config["max_speed_mps"]),
                max_accel_mps2=float(limit_config["max_accel_mps2"]),
            ),
            semantic_keyframes=semantic_keyframes,
        )
        baseline_pixels = [self.camera.project(point) for point in result["base_position_m"]]
        resolved_pixels = [self.camera.project(point) for point in result["resolved_position_m"]]
        node_results = []
        edit_by_frame = {edit.frame: edit for edit in all_edits}
        for node in hand_config["nodes"]:
            frame = int(node["frame"])
            index = frame - int(frames[0])
            edit = edit_by_frame.get(frame)
            node_results.append(
                {
                    **node,
                    "base_world_m": result["base_position_m"][index].tolist(),
                    "resolved_world_m": result["resolved_position_m"][index].tolist(),
                    "delta_m": (edit.delta_m.tolist() if edit else [0.0, 0.0, 0.0]),
                    "base_pixel_uv": list(baseline_pixels[index]) if baseline_pixels[index] else None,
                    "resolved_pixel_uv": list(resolved_pixels[index]) if resolved_pixels[index] else None,
                }
            )
        step_mm = np.zeros(len(frames), dtype=np.float64)
        if len(frames) > 1:
            step_mm[1:] = np.linalg.norm(np.diff(result["resolved_position_m"], axis=0), axis=1) * 1000.0
        speed_mm_s = step_mm * float(self.video_fps)
        return {
            "stage": self.editor_config["stage"],
            "stage_label": self.editor_config.get("label", self.editor_config["stage"]),
            "hand": hand,
            "hand_label": hand_config.get("label", hand),
            "color": hand_config.get("color", "#ffffff"),
            "frame_range": self.editor_config["frame_range"],
            "nodes": node_results,
            "baseline_path": [
                {"frame": int(frames[index]), "pixel_uv": list(baseline_pixels[index])}
                for index in range(len(frames))
                if baseline_pixels[index] is not None
            ],
            "resolved_path": [
                {"frame": int(frames[index]), "pixel_uv": list(resolved_pixels[index])}
                for index in range(len(frames))
                if resolved_pixels[index] is not None
            ],
            "resolved_world_path": [
                {"frame": int(frame), "world_m": point.tolist()}
                for frame, point in zip(frames, result["resolved_position_m"])
            ],
            "series": [
                {
                    "frame": int(frame),
                    "actual_world_m": result["base_position_m"][index].tolist(),
                    "planned_world_m": result["resolved_position_m"][index].tolist(),
                    "actual_pixel_uv": list(baseline_pixels[index]) if baseline_pixels[index] else None,
                    "planned_pixel_uv": list(resolved_pixels[index]) if resolved_pixels[index] else None,
                    "openness": float(self.tcp[int(frame)][hand]["openness"]),
                    "step_mm": float(step_mm[index]),
                    "speed_mm_s": float(speed_mm_s[index]),
                }
                for index, frame in enumerate(frames)
            ],
            "baseline_metrics": result["baseline_metrics"],
            "resolved_metrics": result["resolved_metrics"],
            "introduced_metrics": result["introduced_metrics"],
            "limits": result["limits"],
            "accepted": result["accepted"],
            "required_duration_scale": result["required_duration_scale"],
            "ik_status": result["ik_status"],
        }

    def stage_info(self) -> dict:
        if not self.editor_config:
            raise ValueError("No trajectory_editor configured")
        configured_hands = self.editor_config.get("hands")
        hand_names = list(configured_hands) if configured_hands is not None else [str(self.editor_config["hand"])]
        return {
            "stage": self.editor_config["stage"],
            "label": self.editor_config.get("label", self.editor_config["stage"]),
            "frame_range": self.editor_config["frame_range"],
            "default_hand": self.editor_config.get("default_hand", hand_names[0]),
            "default_node_frame": self.editor_config.get("default_node_frame"),
            "phases": self.editor_config.get("phases", []),
            "events": self.editor_config.get("events", []),
            "hands": {hand: self.compile_stage([], hand) for hand in hand_names},
        }

    def material_anchors(self, source_frame: int) -> dict:
        if source_frame not in self.source_to_index:
            raise ValueError(f"Replay has no source frame {source_frame}")
        index = self.source_to_index[source_frame]
        results = []
        for anchor in self.editor_config.get("material_anchors", []):
            vertex_ids = np.asarray(anchor["vertex_ids"], dtype=np.int64)
            material_world = np.median(self.cloth_pos[index, vertex_ids], axis=0)
            normal = unit(np.asarray(anchor["normal_world"], dtype=np.float64))
            target = material_world + normal * float(anchor.get("tcp_normal_offset_mm", 0.0)) * 0.001
            hand = str(anchor["hand"])
            tcp = np.asarray(self.tcp[source_frame][hand]["world_m"], dtype=np.float64)
            pixel = self.camera.project(material_world)
            delta = target - tcp
            results.append({
                **anchor,
                "source_frame": source_frame,
                "material_world_m": material_world.tolist(),
                "target_tcp_world_m": target.tolist(),
                "material_pixel_uv": list(pixel) if pixel else None,
                "actual_tcp_world_m": tcp.tolist(),
                "delta_mm": (delta * 1000.0).tolist(),
                "distance_mm": float(np.linalg.norm(delta) * 1000.0),
            })
        return {"source_frame": source_frame, "anchors": results}

    def pick(self, source_frame: int, u: float, v: float, clearance_mm: float) -> dict:
        mesh = self.mesh(source_frame)
        origin, direction = self.camera.ray(u, v)
        locations, ray_ids, triangle_ids = mesh.ray.intersects_location(
            ray_origins=origin.reshape(1, 3),
            ray_directions=direction.reshape(1, 3),
            multiple_hits=True,
        )
        if not len(locations):
            return {"source_frame": source_frame, "pixel_uv": [u, v], "hits": []}
        distances = np.linalg.norm(locations - origin, axis=1)
        order = np.argsort(distances)
        hits = []
        previous_distance = None
        for location, triangle_id, distance in zip(locations[order], triangle_ids[order], distances[order]):
            if previous_distance is not None and abs(float(distance) - previous_distance) < 0.0001:
                continue
            previous_distance = float(distance)
            triangle_id = int(triangle_id)
            vertex_ids = self.faces[triangle_id]
            triangle = mesh.vertices[vertex_ids]
            barycentric = trimesh.triangles.points_to_barycentric(
                triangle.reshape(1, 3, 3), location.reshape(1, 3)
            )[0]
            normal = np.array(mesh.face_normals[triangle_id], dtype=np.float64, copy=True)
            if float(np.dot(normal, self.camera.pos - location)) < 0.0:
                normal *= -1.0
            target = location + normal * (float(clearance_mm) * 0.001)
            canonical = barycentric @ self.canonical_uv[vertex_ids]
            hand_deltas = {}
            for hand, tcp in self.tcp.get(source_frame, {}).items():
                tcp_world = np.asarray(tcp["world_m"], dtype=np.float64)
                delta = target - tcp_world
                hand_deltas[hand] = {
                    "actual_tcp_world_m": tcp_world.tolist(),
                    "delta_world_m": delta.tolist(),
                    "delta_world_mm": (delta * 1000.0).tolist(),
                    "delta_xy_mm": float(np.linalg.norm(delta[:2]) * 1000.0),
                    "delta_norm_mm": float(np.linalg.norm(delta) * 1000.0),
                }
            hits.append(
                {
                    "hit_index": len(hits),
                    "distance_from_camera_m": float(distance),
                    "world_m": location.tolist(),
                    "target_tcp_world_m": target.tolist(),
                    "triangle_id": triangle_id,
                    "vertex_ids": vertex_ids.tolist(),
                    "barycentric": barycentric.tolist(),
                    "normal_world": normal.tolist(),
                    "canonical_uv": canonical.tolist(),
                    "width_band": majority(self.width_band[vertex_ids]),
                    "length_zone": majority(self.length_zone[vertex_ids]),
                    "surface_layer": majority(self.surface_layer[vertex_ids]),
                    "vertex_width_bands": self.width_band[vertex_ids].astype(int).tolist(),
                    "vertex_length_zones": self.length_zone[vertex_ids].astype(int).tolist(),
                    "vertex_surface_layers": self.surface_layer[vertex_ids].astype(int).tolist(),
                    "mixed_surface": len(set(int(value) for value in self.surface_layer[vertex_ids])) > 1,
                    "hand_deltas": hand_deltas,
                }
            )
            if len(hits) >= 12:
                break
        return {
            "source_frame": source_frame,
            "pixel_uv": [float(u), float(v)],
            "clearance_mm": float(clearance_mm),
            "stage_hint": self.stage_hint(source_frame),
            "hits": hits,
        }

    def save_regrasp_points(self, request: dict) -> dict:
        """Save material identities only; no TCP calibration, IK or physics."""
        binding = self.config.get("regrasp_edit_binding")
        if not binding:
            raise ValueError("当前配置不是末态再抓点选择模式")
        frame = int(request['source_frame'])
        if frame not in self.source_to_index:
            raise ValueError('保存帧不在当前回放中')
        points = {}
        active_hands = binding.get('hands', ['left', 'right'])
        if not active_hands or any(h not in ('left', 'right') for h in active_hands):
            raise ValueError('Invalid configured hands')
        supplied = request.get('points', {})
        drops = request.get('placement_targets_world_m', {})
        if not isinstance(supplied, dict) or not isinstance(drops, dict):
            raise ValueError('抓点/落点必须是按手标记的对象')
        if (set(supplied) | set(drops)) - set(active_hands):
            # Legacy UI sends null entries for inactive hands; reject actual data only.
            if any(v is not None for items in (supplied,drops) for k,v in items.items() if k not in active_hands):
                raise ValueError('包含未启用的手')
        if not binding.get('placement_enabled') and any(v is not None for v in drops.values()):
            raise ValueError('当前模式不支持落点')
        for hand in active_hands:
            item = supplied.get(hand)
            if item is None:
                continue
            bary = np.asarray(item["barycentric"], dtype=float)
            if bary.shape != (3,) or not np.all(np.isfinite(bary)) or np.any(bary < -1e-7) or np.any(bary > 1 + 1e-7):
                raise ValueError(f"{hand}: 无效三角面重心坐标")
            resolved = self.resolve_material_anchor(
                triangle_id=int(item["triangle_id"]), barycentric=bary.tolist(),
                target_frame=frame, hand=hand, clearance_mm=0.0,
            )
            ids = self.faces[resolved["triangle_id"]]
            points[hand] = {key: resolved[key] for key in (
                "triangle_id", "vertex_ids", "barycentric", "material_world_m",
                "material_pixel_uv", "normal_world",
            )}
            points[hand]["surface_layer"] = majority(self.surface_layer[ids])
            points[hand]["vertex_surface_layers"] = self.surface_layer[ids].astype(int).tolist()
            points[hand]["selected_hit_index"] = int(item.get("selected_hit_index", 0))
            selected_frame = int(item.get('selected_source_frame', frame))
            if selected_frame not in self.source_to_index:
                raise ValueError('材料点选取帧不在当前回放中')
            points[hand]['selected_source_frame'] = selected_frame
        output = Path(binding["output_dir"]).expanduser().resolve()
        placement = None
        if binding.get('placement_enabled'):
            placement = {}
            for hand in active_hands:
                if drops.get(hand) is None:
                    continue
                value = np.asarray(drops[hand], dtype=float)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError(f'{hand}: 落点必须是完整有限XYZ')
                placement[hand] = value.tolist()
        if not points and not placement:
            raise ValueError('请至少指定一个抓点或落点')
        controls = None
        if 'placement_debug_controls' in request:
            controls = validate_drop_controls(request['placement_debug_controls'], self.config.get('drop_debug_binding'))
        output.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = output / f"regrasp_points_{stamp}.json"
        payload = {"schema": "workbench-material-regrasp-v1", "created_at": stamp,
                   "source_frame": frame, "config": str(self.config_path),
                   "replay": str(self.replay_path), "cloth_obj": str(self.cloth_obj_path),
                   "points": points, "point_semantics": "material_surface_not_calibrated_tcp",
                   "ik_started": False, "physics_started": False, "path": str(path)}
        payload['active_hands'] = active_hands
        payload['planning_reference_frame'] = int(binding['frame'])
        payload['frame_semantics'] = 'source_frame is save/display frame; material anchors resolved there; selected_source_frame records pick provenance; planning reference does not force a UI jump'
        missing = {'points': [h for h in active_hands if h not in points]}
        if placement is not None:
            missing['placement_targets_world_m'] = [h for h in active_hands if h not in placement]
        payload['selection_complete'] = not any(missing.values())
        payload['missing_selections'] = missing
        payload['save_semantics'] = 'explicit_selections_only; omitted_fields_are_unspecified; no_implicit_merge_or_clear'
        payload['grasp_intent'] = binding.get('grasp_intent')
        if binding.get('waypoint_slots'):
            payload['waypoint_slots'] = binding['waypoint_slots']
            payload['execution_hand'] = binding.get('execution_hand')
        if binding.get('interaction_mode') == 'closed_gripper_push':
            payload['interaction_mode'] = 'closed_gripper_push'
        if controls is not None:
            payload['placement_debug_controls'] = controls
        encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        if placement is not None:
            payload['placement_targets_world_m'] = placement
            payload['placement_semantics'] = 'world_space_placement_reference_not_calibrated_tcp'
            if binding.get('interaction_mode') == 'closed_gripper_push':
                payload['placement_semantics'] = 'world_space_push_endpoint_reference_not_predicted_cloth_position'
            payload['motion_intent'] = binding.get('motion_intent', 'smooth_lift_transport_lower; release existing grip before regrasp; agent validates IK and contact')
            encoded = json.dumps(payload, indent=2, ensure_ascii=False) + '\n'
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        temporary = output / f".latest_{stamp}.tmp"
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output / "regrasp_latest.json")
        return {"saved": True, **payload, "latest": str(output / "regrasp_latest.json")}

    def resolve_material_anchor(
        self,
        *,
        triangle_id: int,
        barycentric: list[float],
        target_frame: int,
        hand: str,
        clearance_mm: float,
    ) -> dict:
        if target_frame not in self.source_to_index:
            raise ValueError(f"Replay has no source frame {target_frame}")
        if hand not in self.tcp.get(target_frame, {}):
            raise ValueError(f"TCP telemetry has no {hand} at frame {target_frame}")
        triangle_id = int(triangle_id)
        if triangle_id < 0 or triangle_id >= len(self.faces):
            raise ValueError(f"triangle_id out of range: {triangle_id}")
        barycentric_array = np.asarray(barycentric, dtype=np.float64)
        if barycentric_array.shape != (3,) or not np.isclose(barycentric_array.sum(), 1.0, atol=1.0e-5):
            raise ValueError(f"invalid barycentric coordinate: {barycentric}")
        index = self.source_to_index[target_frame]
        vertex_ids = self.faces[triangle_id]
        triangle = self.cloth_pos[index, vertex_ids]
        world = barycentric_array @ triangle
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal = unit(normal)
        if float(np.dot(normal, self.camera.pos - world)) < 0.0:
            normal *= -1.0
        target = world + normal * (float(clearance_mm) * 0.001)
        tcp_world = np.asarray(self.tcp[target_frame][hand]["world_m"], dtype=np.float64)
        delta = target - tcp_world
        return {
            "target_frame": target_frame,
            "hand": hand,
            "triangle_id": triangle_id,
            "vertex_ids": vertex_ids.tolist(),
            "barycentric": barycentric_array.tolist(),
            "material_world_m": world.tolist(),
            "material_pixel_uv": list(self.camera.project(world)) if self.camera.project(world) else None,
            "target_tcp_world_m": target.tolist(),
            "normal_world": normal.tolist(),
            "actual_tcp_world_m": tcp_world.tolist(),
            "delta_m": delta.tolist(),
            "delta_mm": (delta * 1000.0).tolist(),
        }

    def smoke_test(self, source_frame: int, preview_output: Path | None) -> dict:
        mesh = self.mesh(source_frame)
        visible = []
        for triangle_id, triangle in enumerate(mesh.triangles):
            centroid = triangle.mean(axis=0)
            pixel = self.camera.project(centroid)
            if pixel and 0 <= pixel[0] < self.camera.width and 0 <= pixel[1] < self.camera.height:
                visible.append((triangle_id, centroid, pixel))
        if not visible:
            raise RuntimeError("No cloth triangle projects into the camera")
        _, _, pixel = visible[len(visible) // 2]
        result = self.pick(source_frame, pixel[0], pixel[1], 0.0)
        if not result["hits"]:
            raise RuntimeError("Projected cloth smoke-test point did not raycast back to cloth")
        if preview_output is not None:
            image_bgr = self.frame_image(source_frame)
            image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            stage_bundle = self.stage_info()
            stage = stage_bundle["hands"][stage_bundle["default_hand"]]
            baseline_path = [tuple(item["pixel_uv"]) for item in stage["baseline_path"]]
            if len(baseline_path) >= 2:
                draw.line(baseline_path, fill=(190, 205, 220), width=2)
            for node in stage["nodes"]:
                if node.get("show_handle", True) is False or node["resolved_pixel_uv"] is None:
                    continue
                u, v = node["resolved_pixel_uv"]
                radius = 7 if node["draggable"] else 5
                color = (75, 211, 223) if node["draggable"] else (120, 132, 150)
                draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color, outline=(18, 24, 34), width=2)
                if node.get("show_label", True):
                    offset = node.get("label_offset_px", [9, -8])
                    draw.text((u + offset[0], v + offset[1]), f"{node['frame']} {node['name']}", fill=(235, 240, 246))
            colors = {"left": (60, 220, 120), "right": (255, 90, 90)}
            for hand, value in self.frame_info(source_frame)["hands"].items():
                if value["pixel_uv"]:
                    u, v = value["pixel_uv"]
                    draw.ellipse((u - 9, v - 9, u + 9, v + 9), outline=colors[hand], width=4)
                    draw.text((u + 12, v - 8), f"{hand} TCP", fill=colors[hand])
            hit_u, hit_v = pixel
            draw.ellipse((hit_u - 8, hit_v - 8, hit_u + 8, hit_v + 8), fill=(255, 220, 30))
            right_pixel = self.frame_info(source_frame)["hands"].get("right", {}).get("pixel_uv")
            if right_pixel:
                draw.line((right_pixel[0], right_pixel[1], hit_u, hit_v), fill=(255, 220, 30), width=3)
            preview_output.parent.mkdir(parents=True, exist_ok=True)
            image.save(preview_output)
        return result

    def save_patch(self, patch: dict) -> Path:
        required = ("schema_version", "source_frame", "hand", "target", "delta_world_m", "application")
        missing = [key for key in required if key not in patch]
        if missing:
            raise ValueError(f"Patch missing fields: {missing}")
        frame = int(patch["source_frame"])
        if frame not in self.source_to_index:
            raise ValueError(f"Patch source frame is not in replay: {frame}")
        hand = str(patch["hand"])
        if hand not in ("left", "right"):
            raise ValueError(f"Unsupported hand: {hand}")
        stage = str(patch["application"].get("stage", "unassigned"))
        safe_stage = re.sub(r"[^0-9A-Za-z]+", "_", stage).strip("_")
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.patch_output_dir.mkdir(parents=True, exist_ok=True)
        path = self.patch_output_dir / f"frame_{frame:04d}_{hand}_{safe_stage or 'unassigned'}_{timestamp}.json"
        path.write_text(json.dumps(patch, indent=2), encoding="utf-8")
        return path

    def save_trajectory_draft(self, request: dict) -> dict:
        edits_by_hand = request.get("edits", {})
        if not isinstance(edits_by_hand, dict):
            raise ValueError("draft edits must be an object keyed by hand")
        compiled = {
            hand: self.compile_stage(list(edits_by_hand.get(hand, [])), hand)
            for hand in self.editor_config.get("hands", {})
        }
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        root = self.patch_output_dir.parent / "trajectory_workbench_drafts"
        draft_dir = root / timestamp
        draft_dir.mkdir(parents=True, exist_ok=False)
        arrays = {}
        source_frames = None
        for hand, result in compiled.items():
            frames = np.asarray(
                [item["frame"] for item in result["resolved_world_path"]], dtype=np.int32
            )
            if source_frames is None:
                source_frames = frames
            elif not np.array_equal(source_frames, frames):
                raise ValueError("left/right draft frame ranges differ")
            arrays[f"{hand}_actual_tcp_m"] = np.asarray(
                [item["actual_world_m"] for item in result["series"]], dtype=np.float32
            )
            arrays[f"{hand}_planned_tcp_m"] = np.asarray(
                [item["planned_world_m"] for item in result["series"]], dtype=np.float32
            )
        resolved_path = draft_dir / "resolved_tcp.npz"
        np.savez_compressed(resolved_path, source_frames=source_frames, **arrays)
        ik_runs = request.get("ik_runs", {})
        combined_joint_path = None
        combined_joint_summary = None
        if all(bool(ik_runs.get(hand, {}).get("accepted")) for hand in ("left", "right")):
            arm_columns = {
                "left": np.arange(3, 9, dtype=np.int64),
                "right": np.arange(11, 17, dtype=np.int64),
            }
            replay_rows = np.array(
                [self.source_to_index[int(frame)] for frame in source_frames], dtype=np.int64
            )
            combined_q = self.robot_q[replay_rows].copy()
            ik_sources = {}
            for hand in ("left", "right"):
                joint_path = require_file(ik_runs[hand]["resolved_joint_q"])
                with np.load(joint_path, allow_pickle=False) as data:
                    ik_frames = np.asarray(data["source_frames"], dtype=np.int64)
                    ik_q = np.asarray(data["robot_q"], dtype=np.float64)
                if not np.array_equal(ik_frames, source_frames):
                    raise ValueError(f"{hand} IK source frames differ from draft")
                if ik_q.shape != combined_q.shape:
                    raise ValueError(
                        f"{hand} IK shape {ik_q.shape} differs from {combined_q.shape}"
                    )
                combined_q[:, arm_columns[hand]] = ik_q[:, arm_columns[hand]]
                ik_sources[hand] = {
                    "path": str(joint_path),
                    "sha256": sha256(joint_path),
                }
            if not np.all(np.isfinite(combined_q)):
                raise ValueError("Combined joint trajectory contains non-finite values")
            combined_joint_path = draft_dir / "resolved_joint_q_combined.npz"
            np.savez_compressed(
                combined_joint_path,
                source_frames=source_frames.astype(np.int32),
                robot_q=combined_q.astype(np.float32),
            )
            combined_joint_summary = {
                "path": str(combined_joint_path),
                "sha256": sha256(combined_joint_path),
                "shape": list(combined_q.shape),
                "left_max_arm_step_rad": float(
                    np.max(np.abs(np.diff(combined_q[:, arm_columns["left"]], axis=0)), initial=0.0)
                ),
                "right_max_arm_step_rad": float(
                    np.max(np.abs(np.diff(combined_q[:, arm_columns["right"]], axis=0)), initial=0.0)
                ),
                "finger_columns_preserved_from_replay": [9, 10, 17, 18],
                "ik_sources": ik_sources,
                "validation": "shape/frames/finite only; combined-arm collision and IPC physics still required",
            }
        payload = {
            "schema_version": 1,
            "kind": "trajectory_workbench_draft",
            "status": "path_checked_not_physics_executed",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config": str(self.config_path),
            "config_sha256": sha256(self.config_path),
            "replay": str(self.replay_path),
            "replay_sha256": sha256(self.replay_path),
            "stage": self.editor_config["stage"],
            "frame_range": self.editor_config["frame_range"],
            "selected_frame": int(request.get("selected_frame", self.config["default_frame"])),
            "selected_hand": str(request.get("selected_hand", self.editor_config.get("default_hand", "left"))),
            "edits": edits_by_hand,
            "compile": {
                hand: {
                    "accepted": bool(result["accepted"]),
                    "baseline_metrics": result["baseline_metrics"],
                    "resolved_metrics": result["resolved_metrics"],
                    "limits": result["limits"],
                    "required_duration_scale": result["required_duration_scale"],
                }
                for hand, result in compiled.items()
            },
            "ik_runs": ik_runs,
            "combined_joint_trajectory": combined_joint_summary,
            "orientation_assist": request.get("orientation_assist", {}),
            "resolved_tcp_npz": str(resolved_path),
            "resolved_joint_q_combined": (
                str(combined_joint_path) if combined_joint_path is not None else None
            ),
            "note": (
                "Workbench draft only. An agent must review the semantic nodes, "
                "compile them into the runner, run causal IK, and validate IPC physics/video."
            ),
        }
        draft_path = draft_dir / "draft.json"
        draft_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        latest_path = root / "latest.json"
        latest_path.write_text(
            json.dumps({
                "draft": str(draft_path),
                "resolved_tcp_npz": str(resolved_path),
                "resolved_joint_q_combined": (
                    str(combined_joint_path) if combined_joint_path is not None else None
                ),
                "created_at_utc": payload["created_at_utc"],
            }, indent=2),
            encoding="utf-8",
        )
        return {
            "saved": True,
            "draft": str(draft_path),
            "resolved_tcp_npz": str(resolved_path),
            "resolved_joint_q_combined": (
                str(combined_joint_path) if combined_joint_path is not None else None
            ),
            "latest": str(latest_path),
            "path_accepted": all(value["accepted"] for value in compiled.values()),
        }

    def run_ik_preflight(
        self, requested_edits: list[dict], hand: str, orientation_assist: dict | bool | None = None
    ) -> dict:
        if not self._ik_lock.acquire(blocking=False):
            raise RuntimeError("Another IK preflight is already running; wait for it to finish")
        try:
            return self._run_ik_preflight_locked(requested_edits, hand, orientation_assist)
        finally:
            self._ik_lock.release()

    def _run_ik_preflight_locked(
        self, requested_edits: list[dict], hand: str, orientation_assist: dict | bool | None
    ) -> dict:
        compiled = self.compile_stage(requested_edits, hand)
        if not compiled["accepted"]:
            raise ValueError("Position path is rejected; fix speed/acceleration before IK")
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = self.patch_output_dir.parent / "trajectory_workbench_ik_runs" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        edits_path = output_dir / "edits.json"
        edits_path.write_text(
            json.dumps(
                {
                    "hand": hand,
                    "edits": requested_edits,
                    "orientation_assist": orientation_assist,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        project_root = Path(__file__).resolve().parents[2]
        command = [
            str(project_root / "scripts" / "run_trajectory_ik_preflight.sh"),
            "--edits",
            str(edits_path),
        ]
        environment = os.environ.copy()
        environment["TRAJECTORY_WORKBENCH_CONFIG"] = str(self.config_path)
        environment["TRAJECTORY_IK_OUTPUT_DIR"] = str(output_dir)
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
        report_path = output_dir / "ik_preflight.json"
        if not report_path.is_file():
            raise RuntimeError(
                f"IK preflight produced no report (exit={completed.returncode}):\n"
                + completed.stdout[-4000:]
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        fk_path = []
        for sample in report.get("samples", []):
            world = np.asarray(sample["fk_tcp_world_m"], dtype=np.float64)
            pixel = self.camera.project(world)
            fk_path.append({
                "frame": int(sample["source_frame"]),
                "world_m": world.tolist(),
                "pixel_uv": list(pixel) if pixel else None,
            })
        return {
            "accepted": bool(report["ik_summary"]["accepted"]),
            "summary": report["ik_summary"],
            "report": str(report_path),
            "resolved_joint_q": str(output_dir / "resolved_joint_q.npz"),
            "fk_path": fk_path,
            "exit_code": int(completed.returncode),
        }


class WorkbenchHandler(BaseHTTPRequestHandler):
    data: WorkbenchData
    index_path: Path

    def log_message(self, format: str, *args) -> None:
        print(f"workbench_http {self.address_string()} {format % args}")

    def send_bytes(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            # The user may navigate/reload while a long IK request is running.
            # The computation result remains on disk; do not turn a closed
            # client connection into a second failing HTTP response.
            return

    def send_json(self, value: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(value).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_bytes(self.index_path.read_bytes(), "text/html; charset=utf-8")
                return
            if parsed.path == "/action_panel.js":
                self.send_bytes(self.index_path.with_name('action_panel.js').read_bytes(), 'application/javascript; charset=utf-8')
                return
            if parsed.path == '/orientation_panel.js':
                self.send_bytes(self.index_path.with_name('orientation_panel.js').read_bytes(), 'application/javascript; charset=utf-8')
                return
            if parsed.path == '/release_debug.js':
                self.send_bytes(self.index_path.with_name('release_debug.js').read_bytes(), 'application/javascript; charset=utf-8')
                return
            if parsed.path == '/api/gripper-pose':
                try:
                    from .orientation_preview import pose_preview
                except ImportError:
                    from orientation_preview import pose_preview
                query = parse_qs(parsed.query)
                frame = int(query.get('frame', [self.data.config['default_frame']])[0])
                self.send_json(pose_preview(self.data, frame, query.get('hand', ['right'])[0]))
                return
            if parsed.path == "/api/config":
                frame_min, frame_max = self.data.display_frame_range
                self.send_json(
                    {
                        "name": self.data.config["name"],
                        "default_frame": int(self.data.config["default_frame"]),
                        "frame_min": int(frame_min),
                        "frame_max": int(frame_max),
                        "video_fps": self.data.video_fps,
                        "camera": self.data.config["camera"],
                        "active_segment": self.data.config.get("active_segment"),
                        "release_edit_binding": self.data.config.get("release_edit_binding"),
                        "regrasp_edit_binding": self.data.config.get("regrasp_edit_binding"),
                        "measurement_only": self.data.config.get("measurement_only",False),
                        "drop_debug_binding": self.data.config.get("drop_debug_binding"),
                        "orientation_reference_frames": self.data.config.get("orientation_reference_frames"),
                        "checkpoint_catalog": (
                            str(self.data.checkpoint_catalog_path)
                            if self.data.checkpoint_catalog_path is not None
                            else None
                        ),
                    }
                )
                return
            if parsed.path == "/api/checkpoints":
                if self.data.checkpoint_catalog is None:
                    self.send_json({"error": "no checkpoint catalog configured"}, HTTPStatus.NOT_FOUND)
                else:
                    self.send_json(self.data.checkpoint_catalog)
                return
            if parsed.path == "/api/state":
                frame = int(parse_qs(parsed.query).get("frame", [self.data.config["default_frame"]])[0])
                self.send_json(self.data.frame_info(frame))
                return
            if parsed.path == "/api/stage":
                self.send_json(self.data.stage_info())
                return
            if parsed.path == "/api/anchors":
                frame = int(parse_qs(parsed.query).get("frame", [self.data.config["default_frame"]])[0])
                self.send_json(self.data.material_anchors(frame))
                return
            if parsed.path == "/api/frame.jpg":
                frame = int(parse_qs(parsed.query).get("frame", [self.data.config["default_frame"]])[0])
                ok, encoded = cv2.imencode(".jpg", self.data.frame_image(frame), [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                self.send_bytes(encoded.tobytes(), "image/jpeg")
                return
            if parsed.path == "/api/frame.png":
                frame = int(parse_qs(parsed.query).get("frame", [self.data.config["default_frame"]])[0])
                png_path = self.data.frame_png_path(frame)
                if png_path is not None:
                    self.send_bytes(png_path.read_bytes(), "image/png")
                else:
                    ok, encoded = cv2.imencode(".png", self.data.frame_image(frame))
                    if not ok:
                        raise RuntimeError("PNG encoding failed")
                    self.send_bytes(encoded.tobytes(), "image/png")
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        try:
            endpoint = urlparse(self.path).path
            if endpoint == '/api/save-gripper-pose':
                try:
                    from .orientation_preview import save_pose
                except ImportError:
                    from orientation_preview import save_pose
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    raise ValueError('Pose request is empty or too large')
                self.send_json(save_pose(self.data, json.loads(self.rfile.read(length).decode('utf-8'))))
                return
            if endpoint.startswith('/api/action-'):
                try:
                    from .action_service import handle
                except ImportError:
                    from action_service import handle
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 2_000_000:
                    raise ValueError('动作请求为空或过大')
                request = json.loads(self.rfile.read(length).decode('utf-8'))
                if endpoint == '/api/action-activate':
                    target = Path(request['config']).resolve()
                    binding_root = (self.data.patch_output_dir.parent/'action_bindings').resolve()
                    if not target.is_relative_to(binding_root):
                        raise ValueError('只能打开当前Workbench核验生成的子阶段绑定')
                    replacement = WorkbenchData(target)
                    type(self).data = replacement
                    self.send_json({'activated':True, 'config':str(target)})
                    return
                self.send_json(handle(self.data, endpoint, request))
                return
            if endpoint not in (
                "/api/pick",
                "/api/save-patch",
                "/api/save-draft",
                "/api/save-regrasp",
                "/api/compile-stage",
                "/api/resolve-anchor",
                "/api/run-ik",
            ):
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            if endpoint == "/api/save-regrasp":
                self.send_json(self.data.save_regrasp_points(request))
                return
            if endpoint == "/api/compile-stage":
                self.send_json(
                    self.data.compile_stage(
                        list(request.get("edits", [])),
                        str(request.get("hand", self.data.editor_config.get("default_hand", "right"))),
                    )
                )
                return
            if endpoint == "/api/save-draft":
                self.send_json(self.data.save_trajectory_draft(request))
                return
            if endpoint == "/api/resolve-anchor":
                self.send_json(
                    self.data.resolve_material_anchor(
                        triangle_id=int(request["triangle_id"]),
                        barycentric=list(request["barycentric"]),
                        target_frame=int(request["target_frame"]),
                        hand=str(request["hand"]),
                        clearance_mm=float(request.get("clearance_mm", 0.0)),
                    )
                )
                return
            if endpoint == "/api/run-ik":
                self.send_json(
                    self.data.run_ik_preflight(
                        list(request.get("edits", [])),
                        str(request.get("hand", self.data.editor_config.get("default_hand", "right"))),
                        request.get("orientation_assist"),
                    )
                )
                return
            if endpoint == "/api/save-patch":
                path = self.data.save_patch(request)
                self.send_json({"saved": True, "path": str(path)})
                return
            result = self.data.pick(
                int(request["source_frame"]),
                float(request["u"]),
                float(request["v"]),
                float(request.get("clearance_mm", 0.0)),
            )
            self.send_json(result)
        except Exception as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--preview-output", type=Path)
    args = parser.parse_args()
    data = WorkbenchData(args.config.expanduser().resolve())
    if args.smoke_test:
        result = data.smoke_test(int(data.config["default_frame"]), args.preview_output)
        print(json.dumps({"hits": len(result["hits"]), "first_hit": result["hits"][0]}, indent=2))
        return
    WorkbenchHandler.data = data
    WorkbenchHandler.index_path = Path(__file__).with_name("index.html")
    server = ThreadingHTTPServer((args.host, args.port), WorkbenchHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Trajectory Workbench: {url}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
