"""Independent action timelines; positions in metres, quaternions in wxyz.

Durations count intervals: N intervals produce N+1 samples including the initial
state. Actions stop at their explicit endpoints by default; opt-in endpoint
velocities allow a smooth translational pass-through. No legacy replay correction
or implicit return-to-baseline is applied. These are TCP kinematic gates only,
not IK, collision, grasp, or physical-success certification.
"""
from __future__ import annotations

import copy
import math
import numbers
import re

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _scalar(value, name):
    if isinstance(value, (bool, str)) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _vector(value, size, name):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != size:
        raise ValueError(f"{name} must have {size} components")
    return np.array([_scalar(v, name) for v in value], dtype=float)


def _state(value, name, previous=None):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    pos = _vector(value.get("pos"), 3, name + ".pos")
    quat = _vector(value.get("quat", previous[1] if previous is not None else None), 4, name + ".quat")
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"{name}.quat must have nonzero finite norm")
    opening = _scalar(value.get("opening", previous[2] if previous is not None else None), name + ".opening")
    if not 0 <= opening <= .044:
        raise ValueError(f"{name}.opening must be in [0, 0.044] metres per finger")
    return pos, quat / norm, opening


def _slerp(a, b, t):
    if np.dot(a, b) < 0:
        b = -b
    dot = float(np.clip(np.dot(a, b), -1, 1))
    if dot > .9995:
        result = (1 - t[:, None]) * a + t[:, None] * b
    else:
        angle = math.acos(dot)
        result = (np.sin((1-t[:, None])*angle)*a + np.sin(t[:, None]*angle)*b) / math.sin(angle)
    return result / np.linalg.norm(result, axis=1)[:, None]


def _quat_product(a,b):
    aw,ax,ay,az=np.moveaxis(np.asarray(a),-1,0)
    bw,bx,by,bz=np.moveaxis(np.asarray(b),-1,0)
    return np.stack((aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw),axis=-1)


def parse_gripper_events(text: str) -> list:
    """Parse absolute frame commands, e.g. ``t=120 close both duration=18``."""
    if not isinstance(text, str):
        raise ValueError("gripper event text must be a string")
    result = []
    pattern = r"t=(\d+)\s+(open|close)\s+(both|left|right)(?:\s+duration=(\d+))?"
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(pattern, line)
        if not match:
            raise ValueError(f"invalid gripper event on line {line_number}; expected t=120 close both duration=18")
        frame, command, hand, duration = match.groups()
        result.append({"frame": int(frame), "command": command, "hand": hand,
                       "duration_frames": int(duration or 18)})
    return result


def _apply_gripper_events(request, result):
    if "gripper_events" not in request:
        result["gripper_events"] = []
        return
    events = request["gripper_events"]
    if isinstance(events, str):
        events = parse_gripper_events(events)
    if not isinstance(events, list):
        raise ValueError("gripper_events must be a list or event text")
    if not events:
        # An explicitly empty event list leaves endpoint opening trajectories
        # intact (e.g. a small release gap); it must not freeze the seed opening.
        result['gripper_events'] = []
        return
    normalized = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("gripper events must be objects")
        frame = _integer(event.get("frame"), "gripper frame")
        duration = _integer(event.get("duration_frames", 18), "gripper duration_frames", 1)
        command, hand = event.get("command"), event.get("hand", "both")
        if command not in ("open", "close") or hand not in ("both", "left", "right"):
            raise ValueError("gripper command must be open/close and hand both/left/right")
        if frame < result["start_frame"] or frame + duration > result["end_frame"]:
            raise ValueError("gripper transition must fit entirely within plan frame range")
        normalized.append({"frame": frame, "command": command, "hand": hand, "duration_frames": duration})
    normalized.sort(key=lambda event: event["frame"])
    for hand in ("left", "right"):
        values = np.full(len(result["source_frames"]), result[hand + "_opening"][0])
        previous_end = result["start_frame"]
        for event in normalized:
            if event["hand"] not in ("both", hand):
                continue
            if event["frame"] < previous_end:
                raise ValueError(f"overlapping gripper transitions for {hand}")
            index = event["frame"] - result["start_frame"]
            duration = event["duration_frames"]
            target = .044 if event["command"] == "open" else 0.0
            u = np.arange(duration + 1, dtype=float) / duration
            t = 10*u**3 - 15*u**4 + 6*u**5
            values[index:index + duration + 1] = values[index] + t * (target - values[index])
            values[index + duration:] = target
            previous_end = event["frame"] + duration
        result[hand + "_opening"] = values.tolist()
    result["gripper_events"] = normalized
    for endpoint in result["events"]:
        for hand in ("left", "right"):
            endpoint[hand + "_opening"] = result[hand + "_opening"][endpoint["sample_index"]]


def compile_action_plan(request: dict) -> dict:
    """Compile explicit two-hand endpoints into JSON-friendly sample arrays.

    Optional ``boundaries`` is a list of action IDs at which a downstream runner
    should save a checkpoint. Optional ``limits`` has max_speed_mps (0.75) and
    max_accel_mps2 (20). Actions may omit quat to retain previous orientation.
    Optional action ``position_end_velocity_mps`` maps left/right to world XYZ
    velocity vectors. The next action inherits that start velocity; unspecified
    ends stop. Nonzero velocities use quintic Hermite position interpolation with
    zero endpoint accelerations. They cannot accompany a hold; an upward arc
    requires explicit arc_with_endpoint_velocity=True (the arc bump has zero
    endpoint velocity/acceleration). The final action must stop.
    Invalid inputs raise ValueError; valid but too-fast plans return accepted=False.
    """
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    first = _integer(request.get("start_frame", 0), "start_frame")
    fps = _scalar(request.get("fps", 60), "fps")
    if not 0 < fps <= 1000:
        raise ValueError("fps must be in (0, 1000]")
    limits = request.get("limits", {})
    if not isinstance(limits, dict):
        raise ValueError("limits must be an object")
    limits = {key: _scalar(limits.get(key, default), key) for key, default in
              (("max_speed_mps", .75), ("max_accel_mps2", 20.0))}
    if any(v <= 0 for v in limits.values()):
        raise ValueError("limits must be positive")
    start = request.get("start")
    if not isinstance(start, dict):
        raise ValueError("start must contain left and right states")
    states = {hand: _state(start.get(hand), "start." + hand) for hand in ("left", "right")}
    actions = request.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("actions must be a nonempty list")
    boundaries = request.get("boundaries", [])
    if not isinstance(boundaries, list) or any(not isinstance(b, str) for b in boundaries):
        raise ValueError("boundaries must be a list of action IDs")
    arrays = {hand: [[state[0]], [state[1]], [state[2]]] for hand, state in states.items()}
    endpoint_velocities = {hand: np.zeros(3) for hand in states}
    endpoint_speed_max = {hand: 0. for hand in states}
    total, seen, events = 0, set(), []
    for action in actions:
        if not isinstance(action, dict):
            raise ValueError("each action must be an object")
        identifier = action.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("action IDs must be nonempty and unique")
        seen.add(identifier)
        duration = _integer(action.get("duration_frames"), identifier + ".duration_frames", 1)
        if total + duration > 10000:
            raise ValueError("total action duration exceeds 10000 frames")
        u = np.arange(1, duration + 1, dtype=float) / duration
        t = 10*u**3 - 15*u**4 + 6*u**5
        velocity_spec = action.get('position_end_velocity_mps', {})
        allow_arc_velocity=action.get('arc_with_endpoint_velocity',False)
        if not isinstance(allow_arc_velocity,bool):
            raise ValueError('arc_with_endpoint_velocity must be a boolean')
        if not isinstance(velocity_spec, dict) or set(velocity_spec) - set(states):
            raise ValueError('position_end_velocity_mps must map left/right to XYZ vectors')
        for hand in states:
            old = states[hand]
            new = _state(action.get(hand), identifier + "." + hand, old)
            positions = old[0] + t[:, None] * (new[0] - old[0])
            # Optional smooth upward bow, with zero endpoint slope. A per-hand
            # mapping permits one arm to remain exactly stationary.
            arc = action.get('arc_height_m', 0.)
            height = _scalar(arc.get(hand, 0.) if isinstance(arc, dict) else arc, identifier+'.arc_height_m')
            if height < 0:
                raise ValueError('arc_height_m must be nonnegative')
            start_velocity = endpoint_velocities[hand]
            end_velocity = _vector(velocity_spec.get(hand, [0., 0., 0.]), 3,
                                   identifier+'.position_end_velocity_mps.'+hand)
            if np.any(start_velocity) or np.any(end_velocity):
                if np.array_equal(old[0], new[0]):
                    raise ValueError('nonzero endpoint velocity cannot accompany a position hold')
                if height != 0 and not allow_arc_velocity:
                    raise ValueError('nonzero endpoint velocity cannot accompany arc_height_m')
                # Quintic Hermite basis: exact positions, specified velocities,
                # and zero acceleration at both ends. Keep the default arithmetic
                # untouched so historic plans compile to exactly the same samples.
                seconds = duration / fps
                h0 = u - 6*u**3 + 8*u**4 - 3*u**5
                h1 = -4*u**3 + 7*u**4 - 3*u**5
                positions += seconds * (h0[:, None]*start_velocity + h1[:, None]*end_velocity)
            endpoint_velocities[hand] = end_velocity
            endpoint_speed_max[hand] = max(endpoint_speed_max[hand], float(np.linalg.norm(end_velocity)))
            positions[:, 2] += height * 16 * t**2 * (1-t)**2
            positions[-1] = new[0]
            quats = _slerp(old[1], new[1], t)
            profile=action.get('rotation_profile',{}).get(hand)
            if profile is not None:
                world=np.radians(_scalar(profile['world_x_deg'],identifier+'.world_x_deg'))*t/2
                local=np.radians(_scalar(profile['tool_x_deg'],identifier+'.tool_x_deg'))*t/2
                zeros=np.zeros_like(t)
                qa=np.column_stack((np.cos(world),np.sin(world),zeros,zeros))
                qb=np.column_stack((np.cos(local),np.sin(local),zeros,zeros))
                quats=_quat_product(_quat_product(qa,old[1]),qb)
                if abs(float(np.dot(quats[-1],new[1])))<1-1e-6:
                    raise ValueError('rotation_profile endpoint differs from target quaternion')
                quats[-1]=new[1]
            openings = old[2] + t * (new[2] - old[2])
            openings[-1] = new[2]
            for destination, values in zip(arrays[hand], (positions, quats, openings)):
                destination.extend(values)
            states[hand] = (new[0], quats[-1], new[2])
        total += duration
        events.append({"id": identifier, "frame": first + total, "sample_index": total,
                       "duration_frames": duration, "checkpoint": identifier in boundaries,
                       "left_opening": states["left"][2], "right_opening": states["right"][2]})
    if set(boundaries) - seen:
        raise ValueError("boundaries refer to unknown action IDs")
    if any(np.any(v) for v in endpoint_velocities.values()):
        raise ValueError('final action position endpoint velocity must be zero')
    result = {"start_frame": first, "end_frame": first + total, "fps": fps,
              "duration_frames": total, "duration_seconds": total / fps,
              "source_frames": list(range(first, first + total + 1)), "events": events,
              "checkpoint_frames": [e["frame"] for e in events if e["checkpoint"]],
              "limits": limits, "metrics": {}, "accepted": True,
              "gate_scope": "TCP translation only; IK/collision/grasp require separate validation"}
    for hand, values in arrays.items():
        pos = np.asarray(values[0])
        velocity = np.diff(pos, axis=0) * fps
        # Include stopped endpoints so even one-interval moves have acceleration gates.
        accel = np.diff(np.vstack((np.zeros(3), velocity, np.zeros(3))), axis=0) * fps
        speed_max = max(float(np.max(np.linalg.norm(velocity, axis=1))), endpoint_speed_max[hand])
        accel_max = float(np.max(np.linalg.norm(accel, axis=1)))
        result["metrics"][hand] = {"max_speed_mps": speed_max, "max_accel_mps2": accel_max}
        result["accepted"] &= speed_max <= limits["max_speed_mps"] and accel_max <= limits["max_accel_mps2"]
        for suffix, value in zip(("pos", "quat", "opening"), values):
            result[hand + "_" + suffix] = np.asarray(value).tolist()
    _apply_gripper_events(request, result)
    return result


def default_plan(start: dict, start_frame: int = 0, pull_axis: str = "x", pull_sign: int = -1) -> dict:
    """Draft A only: already grasped -> lift 5 cm -> pull 5 cm -> hold.

    Does NOT assume/perform a grasp: the caller must supply the actual grasped
    start state and opening. Review pull axis against the scene before running.
    """
    if pull_axis not in ("x", "y") or pull_sign not in (-1, 1):
        raise ValueError("pull_axis must be x/y and pull_sign must be -1/+1")
    current = copy.deepcopy(start)
    actions = []
    for identifier, duration, axis, distance in (("A_lift", 60, 2, .05),
                                                ("A_pull", 60, "xyz".index(pull_axis), .05*pull_sign),
                                                ("A_hold", 30, 2, 0)):
        for hand in ("left", "right"):
            current[hand]["pos"] = list(current[hand]["pos"])
            current[hand]["pos"][axis] += distance
        actions.append({"id": identifier, "duration_frames": duration, **copy.deepcopy(current)})
    request = {"start_frame": start_frame, "fps": 60, "start": copy.deepcopy(start),
               "actions": actions, "boundaries": ["A_hold"]}
    compile_action_plan(request)
    return request
