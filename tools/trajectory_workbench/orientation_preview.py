"""Read-only measured-pose preview and explicit quaternion intent exports.

No IK, collision acceptance or physics is performed by this module.
"""
import datetime as dt
from functools import lru_cache
import json
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

TCP_LOCAL = np.array([.155, .001786, .014])
BOX_EDGES = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
             [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]


def validate_frame_hand(data, frame, hand):
    if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
        raise ValueError('source_frame must be an integer')
    if hand not in ('left', 'right'):
        raise ValueError('hand must be left or right')
    if int(frame) not in data.source_to_index:
        raise ValueError('source_frame is absent from the current replay')
    lo, hi = data.display_frame_range
    if not lo <= int(frame) <= hi:
        raise ValueError('source_frame is outside the displayed range')
    return data.source_to_index[int(frame)]


def origin_transform(element):
    tf = np.eye(4)
    if element is not None:
        tf[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        tf[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return tf


def fk(config, q):
    root = ET.parse(config['robot_urdf']).getroot()
    pending = list(root.findall('joint'))
    active = [j.get('name') for j in pending if j.get('type') != 'fixed']
    if len(active) != len(q):
        raise ValueError('URDF active-joint order does not match replay robot_q')
    result = {'base_link': np.eye(4)}
    result['base_link'][:3, 3] = config['robot_base_pos_m']
    while pending:
        ready = [j for j in pending if j.find('parent').get('link') in result]
        if not ready:
            raise ValueError('URDF joint tree is unresolved')
        for j in ready:
            motion = np.eye(4)
            kind = j.get('type')
            if kind != 'fixed':
                axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
                value = q[active.index(j.get('name'))]
                if kind == 'prismatic':
                    motion[:3, 3] = axis * value
                else:
                    motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
            result[j.find('child').get('link')] = result[j.find('parent').get('link')] @ origin_transform(j.find('origin')) @ motion
            pending.remove(j)
    return result


@lru_cache(maxsize=32)
def finger_geometry(urdf, name, proxy_path=None):
    if proxy_path:
        with np.load(proxy_path, allow_pickle=False) as proxy:
            i = proxy['names'].astype(str).tolist().index(name)
            mesh = trimesh.Trimesh(proxy[f'verts_{i}'], proxy[f'faces_{i}'], process=False)
        source = 'exported IPC collision mesh'
    else:
        mesh = urdf_finger_mesh(urdf, name)
        source = 'URDF collision mesh'
    hull = mesh.convex_hull
    edges = hull.face_adjacency_edges[hull.face_adjacency_angles > np.deg2rad(.5)]
    bounds = mesh.bounds
    box = np.array([[bounds[(i >> 2) & 1, 0], bounds[(i >> 1) & 1, 1], bounds[i & 1, 2]] for i in range(8)])
    return dict(vertices=np.asarray(hull.vertices), faces=np.asarray(hull.faces), edges=edges, box=box,
                source=source + '; visualization convex hull, coplanar triangle edges hidden; not a physics geometry change',
                source_vertices=len(mesh.vertices), source_faces=len(mesh.faces))


def urdf_finger_mesh(urdf, name):
    root = ET.parse(urdf).getroot()
    pieces = []
    for collision in root.find(f"link[@name='{name}']").findall('collision'):
        mesh = collision.find('geometry/mesh')
        if mesh is None:
            raise ValueError('Preview currently requires mesh finger collisions')
        loaded = trimesh.load(str((Path(urdf).parent / mesh.get('filename')).resolve()), force='mesh', process=False)
        loaded.vertices *= np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
        loaded.apply_transform(origin_transform(collision.find('origin')))
        pieces.append(loaded)
    return trimesh.util.concatenate(pieces)


def finger_box(urdf, name):
    """Compatibility helper for bounding-box callers."""
    return finger_geometry(str(urdf), name)['box']


def pose_preview(data, frame, hand):
    index = validate_frame_hand(data, frame, hand)
    tool = f'{hand}_link' + ('16' if hand == 'left' else '26')
    names = [f'{hand}_link{n}' for n in ([17, 18] if hand == 'left' else [27, 28])]
    # Measured replay is intentionally used even when planning-command overrides
    # and a later checkpoint are present in this Workbench's configuration.
    transforms = fk(data.config, data.robot_q[index])
    source = 'measured robot_q URDF FK'
    with np.load(data.replay_path, allow_pickle=False) as replay:
        if 'ipc_link_transforms' in replay and 'ipc_link_names' in replay:
            ipc_names = replay['ipc_link_names'].astype(str).tolist()
            row = replay['ipc_link_transforms'][index]
            if all(n in ipc_names for n in [tool] + names):
                transforms.update({n: row[ipc_names.index(n)] for n in [tool] + names})
                source = 'measured IPC link transforms; orientation projected to SO(3)'
    tf = transforms[tool]
    rotation = Rotation.from_matrix(tf[:3, :3])
    matrix = rotation.as_matrix()
    position = tf[:3, 3] + tf[:3, :3] @ TCP_LOCAL
    quat = rotation.as_quat()[[3, 0, 1, 2]]
    fingers = []
    proxy_path = None
    if data.config.get('action_manifest'):
        candidate = Path(data.config['action_manifest']).parent / 'stage_physics_keyframes/ipc_proxy_meshes.npz'
        if candidate.is_file():
            proxy_path = str(candidate)
    for name in names:
        geometry = finger_geometry(data.config['robot_urdf'], name, proxy_path)
        finger_tf = transforms[name]
        world = geometry['vertices'] @ finger_tf[:3, :3].T + finger_tf[:3, 3]
        box_world = geometry['box'] @ finger_tf[:3, :3].T + finger_tf[:3, 3]
        fingers.append(dict(name=name, vertices_world_m=world.tolist(),
                            vertices_local_m=((world-position) @ matrix).tolist(),
                            edges=geometry['edges'].tolist(), faces=geometry['faces'].tolist(),
                            geometry_source=geometry['source'],
                            source_mesh_counts=dict(vertices=geometry['source_vertices'], faces=geometry['source_faces']),
                            box_vertices_local_m=((box_world-position) @ matrix).tolist(), box_edges=BOX_EDGES))
    vertices = data.cloth_pos[index]
    centers = vertices[data.faces].mean(axis=1)
    chosen = np.flatnonzero(np.linalg.norm(centers-position, axis=1) < .1)
    if len(chosen) > 300:
        chosen = chosen[np.linspace(0, len(chosen)-1, 300).astype(int)]
    ids, inverse = np.unique(data.faces[chosen], return_inverse=True)
    # Verify gripper semantics against URDF, rather than relying on a label.
    root = ET.parse(data.config['robot_urdf']).getroot()
    axes = []
    for name in names:
        joint = next(j for j in root.findall('joint') if j.find('child').get('link') == name)
        axes.append(origin_transform(joint.find('origin'))[:3, :3] @ np.fromstring(joint.find('axis').get('xyz'), sep=' '))
    if not (np.allclose(axes[0], [0, 1, 0]) and np.allclose(axes[1], [0, -1, 0])):
        raise ValueError('Unexpected finger axes; refusing incorrect opening-axis labels')
    return dict(source_frame=int(frame), hand=hand, position_m=position.tolist(),
                quat_wxyz=quat.tolist(), axes_world={a: matrix[:, i].tolist() for i, a in enumerate('xyz')},
                closing_axis_local=[0, 1, 0], closing_axis_world=matrix[:, 1].tolist(),
                tool_axis_local=[1, 0, 0], pose_source=source,
                geometry_source=fingers[0]['geometry_source'],
                axis_semantics={'x': 'tool longitudinal axis', 'y': 'finger opening/closing axis', 'z': 'third orthogonal axis'},
                fingers=fingers, table_z_m=float(data.config.get('table_z_m', .8)),
                cloth_patch=dict(vertices_world_m=vertices[ids].tolist(), faces=inverse.reshape(-1, 3).tolist()),
                preview_only=True, ik_checked=False, collision_checked=False)


def save_pose(data, request):
    frame, hand = request.get('source_frame'), request.get('hand')
    validate_frame_hand(data, frame, hand)
    role = request.get('role')
    if role not in ('grasp', 'placement'):
        raise ValueError('role must be grasp or placement')
    q = np.asarray(request.get('quat_wxyz'), dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or abs(np.linalg.norm(q)-1) > 1e-5:
        raise ValueError('quat_wxyz must contain four finite numbers with unit norm')
    binding = data.config.get('regrasp_edit_binding') or {}
    if 'output_dir' not in binding:
        raise ValueError('regrasp_edit_binding.output_dir is not configured')
    actual = pose_preview(data, int(frame), hand)
    target_matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    record = dict(schema='workbench-gripper-orientation-v1', created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  source_frame=int(frame), hand=hand, role=role, quaternion_frame='world', quaternion_order='wxyz',
                  quat_wxyz=(q / np.linalg.norm(q)).tolist(), source_quat_wxyz=actual['quat_wxyz'],
                  position_m=actual['position_m'], config=str(data.config_path), replay=str(data.replay_path),
                  pose_source=actual['pose_source'], preview_only=True, ik_checked=False, physics_checked=False)
    record['target_axes_world'] = {a: target_matrix[:, i].tolist() for i, a in enumerate('xyz')}
    record['axis_semantics'] = actual['axis_semantics']
    output = Path(binding['output_dir']).expanduser().resolve() / 'orientations'
    output.mkdir(parents=True, exist_ok=True)
    path = output / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8] + '.json')
    with path.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return dict(saved=True, path=str(path), record=record)
