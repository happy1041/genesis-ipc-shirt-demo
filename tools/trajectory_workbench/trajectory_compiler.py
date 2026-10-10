"""Compile sparse TCP node edits into a compact, C2-continuous correction field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PositionEdit:
    frame: int
    delta_m: np.ndarray
    support_radius: int
    mode: str = "local"


@dataclass(frozen=True)
class PositionLimits:
    max_step_mm: float = 6.5
    max_speed_mps: float = 0.40
    max_accel_mps2: float = 15.0


def wendland_c2(frames: np.ndarray, center: int, radius: int) -> np.ndarray:
    if radius < 2:
        raise ValueError("support_radius must be at least 2 frames")
    distance = np.abs(np.asarray(frames, dtype=np.float64) - float(center)) / float(radius)
    value = np.zeros_like(distance)
    inside = distance < 1.0
    r = distance[inside]
    value[inside] = (1.0 - r) ** 4 * (4.0 * r + 1.0)
    value[distance == 0.0] = 1.0
    return value


def minimum_jerk(progress: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _local_correction(frames: np.ndarray, edits: list[PositionEdit]) -> np.ndarray:
    if not edits:
        return np.zeros((len(frames), 3), dtype=np.float64)
    centers = np.array([edit.frame for edit in edits], dtype=np.int64)
    phi = np.column_stack(
        [wendland_c2(frames, edit.frame, edit.support_radius) for edit in edits]
    )
    kernel_at_nodes = np.column_stack(
        [wendland_c2(centers, edit.frame, edit.support_radius) for edit in edits]
    )
    condition = float(np.linalg.cond(kernel_at_nodes))
    if not np.isfinite(condition) or condition > 1.0e8:
        raise ValueError(f"Node supports are ill-conditioned: cond={condition:.3e}")
    values = np.stack([np.asarray(edit.delta_m, dtype=np.float64) for edit in edits])
    coefficients = np.linalg.solve(kernel_at_nodes, values)
    return phi @ coefficients


def _semantic_correction(
    frames: np.ndarray,
    keyframes: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    """Interpolate semantic correction anchors with zero endpoint velocity/acceleration."""
    correction = np.zeros((len(frames), 3), dtype=np.float64)
    if not keyframes:
        return correction
    ordered = sorted(
        (int(frame), np.asarray(value, dtype=np.float64)) for frame, value in keyframes
    )
    if len({frame for frame, _ in ordered}) != len(ordered):
        raise ValueError("semantic keyframe frames must be unique")
    if any(value.shape != (3,) for _, value in ordered):
        raise ValueError("semantic keyframe values must have shape (3,)")
    correction[frames <= ordered[0][0]] = ordered[0][1]
    correction[frames >= ordered[-1][0]] = ordered[-1][1]
    for (start, start_value), (end, end_value) in zip(ordered[:-1], ordered[1:]):
        if end <= start:
            raise ValueError("semantic keyframes must be strictly increasing")
        mask = (frames >= start) & (frames <= end)
        progress = (frames[mask].astype(np.float64) - float(start)) / float(end - start)
        weight = minimum_jerk(progress)[:, None]
        correction[mask] = start_value + (end_value - start_value) * weight
    return correction


def compile_position_edits(
    frames: np.ndarray,
    base_position_m: np.ndarray,
    edits: list[PositionEdit],
    *,
    stage_range: tuple[int, int],
    fps: float,
    limits: PositionLimits,
    semantic_keyframes: list[tuple[int, np.ndarray]] | None = None,
) -> dict:
    frames = np.asarray(frames, dtype=np.int64)
    base = np.asarray(base_position_m, dtype=np.float64)
    if base.shape != (len(frames), 3):
        raise ValueError(f"base position shape mismatch: {base.shape}")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    start, end = (int(value) for value in stage_range)
    if start > end:
        raise ValueError("invalid stage_range")
    if not np.all(np.diff(frames) == 1):
        raise ValueError("compiler currently requires consecutive source frames")
    for edit in edits:
        if edit.frame < start or edit.frame > end:
            raise ValueError(f"edit frame outside stage: {edit.frame}")
        if np.asarray(edit.delta_m).shape != (3,):
            raise ValueError(f"delta must have shape (3,), got {np.asarray(edit.delta_m).shape}")
        if edit.mode not in ("local", "hold_after"):
            raise ValueError(f"unsupported edit mode: {edit.mode}")

    correction = np.zeros_like(base)
    local_edits = [edit for edit in edits if edit.mode == "local"]
    correction += _local_correction(frames, local_edits)
    correction += _semantic_correction(frames, semantic_keyframes or [])
    for edit in (value for value in edits if value.mode == "hold_after"):
        ramp_start = edit.frame - edit.support_radius
        if ramp_start < start:
            raise ValueError(f"hold_after ramp crosses stage start: frame={edit.frame}")
        progress = (frames.astype(np.float64) - float(ramp_start)) / float(edit.support_radius)
        weight = minimum_jerk(progress)
        weight[frames < ramp_start] = 0.0
        weight[frames >= edit.frame] = 1.0
        correction += weight[:, None] * np.asarray(edit.delta_m, dtype=np.float64)

    outside = (frames < start) | (frames > end)
    correction[outside] = 0.0
    resolved = base + correction

    def metrics(position: np.ndarray) -> dict:
        step_m = np.linalg.norm(np.diff(position, axis=0), axis=1)
        velocity = step_m * fps
        acceleration = np.linalg.norm(np.diff(position, n=2, axis=0), axis=1) * fps**2
        return {
            "max_step_mm": float(step_m.max(initial=0.0) * 1000.0),
            "max_speed_mps": float(velocity.max(initial=0.0)),
            "max_accel_mps2": float(acceleration.max(initial=0.0)),
        }

    baseline_metrics = metrics(base)
    resolved_metrics = metrics(resolved)
    ratios = {
        "step": resolved_metrics["max_step_mm"] / limits.max_step_mm,
        "speed": resolved_metrics["max_speed_mps"] / limits.max_speed_mps,
        "accel": resolved_metrics["max_accel_mps2"] / limits.max_accel_mps2,
    }
    introduced = {
        "step": max(0.0, resolved_metrics["max_step_mm"] - baseline_metrics["max_step_mm"]),
        "speed": max(0.0, resolved_metrics["max_speed_mps"] - baseline_metrics["max_speed_mps"]),
        "accel": max(0.0, resolved_metrics["max_accel_mps2"] - baseline_metrics["max_accel_mps2"]),
    }
    accepted = all(value <= 1.0 + 1.0e-9 for value in ratios.values())
    required_scale = max(
        ratios["step"],
        ratios["speed"],
        np.sqrt(ratios["accel"]),
        1.0,
    ) * (1.0 if accepted else 1.05)
    return {
        "frames": frames,
        "base_position_m": base,
        "resolved_position_m": resolved,
        "correction_m": correction,
        "baseline_metrics": baseline_metrics,
        "resolved_metrics": resolved_metrics,
        "introduced_metrics": introduced,
        "limits": {
            "max_step_mm": limits.max_step_mm,
            "max_speed_mps": limits.max_speed_mps,
            "max_accel_mps2": limits.max_accel_mps2,
        },
        "accepted": accepted,
        "required_duration_scale": float(required_scale),
        "ik_status": "not_run",
    }
