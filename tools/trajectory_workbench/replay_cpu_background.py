"""CPU-only, camera-aligned inspection images of actual saved replay geometry.

This is a diagnostic view, not an Isaac render or another physics simulation.
Cloth triangles use their original indexing and perspective-correct z buffering;
there is no topology reduction, smoothing, coordinate translation or deformation.
Optional robot visuals are the existing export's actual world transforms.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda fn: fn


@njit(cache=False)  # CLI and package imports use different module names; avoid incompatible disk cache.
def _rasterize(xyz, triangles, rgb, image, depth, owners, owner_offset):
    """xyz=(pixel x, pixel y, positive camera depth); integer pixel centres."""
    height, width = depth.shape
    for face_id in range(len(triangles)):
        a, b, c = triangles[face_id]
        x0, y0, z0 = xyz[a]
        x1, y1, z1 = xyz[b]
        x2, y2, z2 = xyz[c]
        if min(z0, z1, z2) <= 1.e-6:
            continue
        den = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(den) < 1.e-12:
            continue
        xmin = max(0, int(np.ceil(min(x0, x1, x2))))
        xmax = min(width-1, int(np.floor(max(x0, x1, x2))))
        ymin = max(0, int(np.ceil(min(y0, y1, y2))))
        ymax = min(height-1, int(np.floor(max(y0, y1, y2))))
        for y in range(ymin, ymax+1):
            for x in range(xmin, xmax+1):
                u = ((y1-y2)*(x-x2)+(x2-x1)*(y-y2))/den
                v = ((y2-y0)*(x-x2)+(x0-x2)*(y-y2))/den
                w = 1-u-v
                if min(u, v, w) < -1.e-9:
                    continue
                z = 1/(u/z0+v/z1+w/z2)
                if z < depth[y, x]:
                    depth[y, x] = z
                    owners[y, x] = face_id+owner_offset
                    image[y, x, 0] = rgb[face_id, 0]
                    image[y, x, 1] = rgb[face_id, 1]
                    image[y, x, 2] = rgb[face_id, 2]


def _project(camera, vertices):
    relative = vertices-camera.pos
    z = relative@camera.forward
    safe_z = np.where(z > 1.e-9, z, 1.e-9)
    return np.column_stack((camera.width*.5 + camera.focal*(relative@camera.right)/safe_z-.5,
                            camera.height*.5-camera.focal*(relative@camera.true_up)/safe_z-.5, z))


def _colors(vertices, triangles, bgr):
    normals = np.cross(vertices[triangles[:, 1]]-vertices[triangles[:, 0]],
                       vertices[triangles[:, 2]]-vertices[triangles[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.e-15)
    light = np.array([-.35, -.2, 1.0])
    light /= np.linalg.norm(light)
    # Two-sided shading: winding/seam conventions must not make cloth invisible.
    brightness = .48+.52*np.abs(normals@light)
    return np.clip(brightness[:, None]*np.asarray(bgr), 0, 255).astype(np.uint8)


@lru_cache(maxsize=2)
def _load_robot(path, frames_path):
    with np.load(path, allow_pickle=False) as arrays:
        transforms = np.asarray(arrays['transforms'], dtype=np.float64)
        flags = arrays.get('is_ipc_proxy', np.zeros(len(arrays['names']), dtype=bool))
        meshes = [(i, np.asarray(arrays[f'verts_{i}'], dtype=np.float64),
                   np.asarray(arrays[f'faces_{i}'], dtype=np.int64))
                  for i in range(len(arrays['names'])) if not flags[i]]
    with np.load(frames_path, allow_pickle=False) as replay:
        frames = replay['original_source_frames'] if 'original_source_frames' in replay.files else replay['source_frames']
        frame_to_index = {int(frame): i for i, frame in enumerate(frames)}
    if len(frame_to_index) != len(transforms):
        raise ValueError('Robot visuals must have unique source frames matching transforms')
    return transforms, meshes, frame_to_index


def render_geometry(data, source_frame):
    """Return BGR image, camera-depth map, and visible cloth face IDs for QA.

    owner=-1 means table/background; robot IDs begin at len(data.faces).
    Data must expose camera, config, faces, cloth_pos and source_to_index.
    """
    source_frame = int(source_frame)
    if source_frame not in data.source_to_index:
        raise ValueError(f'No saved replay for source frame {source_frame}')
    camera = data.camera
    vertices = np.asarray(data.cloth_pos[data.source_to_index[source_frame]], dtype=np.float64)
    triangles = np.asarray(data.faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError('Cloth replay/topology mismatch')
    if not np.isfinite(vertices).all():
        raise ValueError('Non-finite cloth replay positions')
    h, w = camera.height, camera.width
    image = np.full((h, w, 3), (43, 47, 51), dtype=np.uint8)
    depth = np.full((h, w), np.inf)
    owners = np.full((h, w), -1, dtype=np.int64)
    y, x = np.indices((h, w))
    directions = (camera.forward + ((x+.5-w*.5)/camera.focal)[..., None]*camera.right
                  + ((h*.5-y-.5)/camera.focal)[..., None]*camera.true_up)
    table_z = float(data.config.get('table_z_m', .8))
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (table_z-camera.pos[2])/directions[..., 2]
    valid = np.isfinite(t) & (t > 0)
    depth[valid] = t[valid]
    image[valid] = (224, 228, 231)
    world = camera.pos + directions*np.where(valid, t, 0)[..., None]
    # Reference grid only, not invented table dimensions or texture.
    grid = ((np.abs((world[..., 0]+.05) % .1-.05) < .0007)
            | (np.abs((world[..., 1]+.05) % .1-.05) < .0007)) & valid
    image[grid] = (208, 213, 218)
    _rasterize(_project(camera, vertices), triangles,
               _colors(vertices, triangles, (185, 123, 51)), image, depth, owners, 0)
    robot_path = data.config.get('robot_visuals_npz')
    if robot_path:
        robot_path = Path(robot_path).expanduser().resolve()
        frames_path = Path(data.config.get('robot_visuals_source_frames_npz', robot_path.parent/'replay.npz')).resolve()
        transforms, meshes, frame_map = _load_robot(str(robot_path), str(frames_path))
        if source_frame not in frame_map:
            raise ValueError(f'Robot visuals do not cover source frame {source_frame}')
        row = transforms[frame_map[source_frame]]
        offset = len(triangles)
        for i, local_vertices, faces in meshes:
            matrix = row[i]
            points = local_vertices@matrix[:3, :3].T+matrix[:3, 3]
            _rasterize(_project(camera, points), faces, _colors(points, faces, (137, 129, 120)),
                       image, depth, owners, offset)
            offset += len(faces)
    return image, depth, owners


def render_frame(data, source_frame):
    """Render a camera-aligned saved replay frame to BGR uint8, entirely on CPU."""
    image, _, _ = render_geometry(data, source_frame)
    mode = 'cloth + actual robot visuals' if data.config.get('robot_visuals_npz') else 'cloth only (robot omitted)'
    label = f'CPU replay | frame {int(source_frame)} | {mode} | grid 100mm'
    cv2.rectangle(image, (0, 0), (image.shape[1], 24), (32, 36, 39), -1)
    cv2.putText(image, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1, cv2.LINE_AA)
    return image
