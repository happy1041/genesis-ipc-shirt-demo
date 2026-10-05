"""Freeze a completed seven-stage Scene527 run into a portable data bundle.

Copies assets/states; never starts Genesis, Isaac, or a GPU simulation.
"""
import argparse
import hashlib
import json
import shutil
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

NAMES = ('first_fold', 'first_pull', 'second_fold', 'third_right',
         'third_left', 'sweep', 'press_slide_home')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prefix_digest(q, opening, end):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(q[:end+1], dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(opening[:end+1], dtype=np.float64).tobytes())
    return h.hexdigest()


def portable_args(argv):
    tokens, result, i = argv[2:], [], 0
    path_flags = {'--save-third-fold-checkpoint', '--load-third-fold-checkpoint',
                  '--checkpoint-output-dir', '--save-checkpoint-source-frame'}
    while i < len(tokens):
        t = tokens[i]
        if t in path_flags:
            i += 2
        elif t == '--save-checkpoint-source-frames':
            i += 1
            while i < len(tokens) and not tokens[i].startswith('--'):
                i += 1
        elif t in ('--physics-only', '--render-only', '--viewer'):
            i += 1
        else:
            result.append(t)
            i += 1
    if any('/home/' in t for t in result):
        raise ValueError('Unrelocated path in stage arguments')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--final-manifest', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-hero', type=Path)
    a = p.parse_args()
    workspace, out = a.workspace.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = {}

    def record(path, role, source=None):
        records[str(path.relative_to(out))] = dict(
            bytes=path.stat().st_size, sha256=sha(path), role=role,
            source=str(source.relative_to(workspace)) if source else None)

    def copy(source, rel, role):
        source, target = Path(source).resolve(), out/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        record(target, role, source)
        return target

    chain, seen, path = [], set(), a.final_manifest.resolve()
    while path:
        if path in seen:
            raise ValueError('Cyclic source chain')
        seen.add(path)
        m = json.loads(path.read_text())
        if not m['status'].startswith('COMPLETE'):
            raise ValueError('Incomplete source stage')
        chain.append((path, m))
        path = Path(m['parent_manifest']).resolve() if m.get('parent_manifest') else None
    chain.reverse()
    if len(chain) != 7:
        raise ValueError('Expected seven recorded stages')
    final = chain[-1][1]
    trajectory = copy(final['trajectory'], 'controls/trajectory.npz', 'robot_command')
    with np.load(trajectory, allow_pickle=False) as d:
        q, opening = d['joint_q'], d['openness']
        if q.shape != (4373, 19) or opening.shape != (4373, 2):
            raise ValueError('Expected 4373-frame native command timeline')

    cloth = copy(final['env']['SHIRT_OBJ'], 'assets/cloth/short-shirt-55068f.obj', 'cloth')
    original_urdf = Path(final['env']['ROBOT_URDF']).resolve()
    urdf_rel = original_urdf.relative_to(workspace)
    urdf = copy(original_urdf, urdf_rel, 'robot_urdf')
    for element in ET.parse(original_urdf).iter('mesh'):
        source = (original_urdf.parent/element.attrib['filename']).resolve()
        rel = source.relative_to(workspace)
        if str(rel) not in records:
            copy(source, rel, 'robot_mesh')
        if (urdf.parent/element.attrib['filename']).resolve() != (out/rel).resolve():
            raise ValueError('Robot mesh relocation changed URDF resolution')
    preset = copy(final['env']['DEMO_CONFIG'], 'configs/source_preset.args', 'physics_preset')
    stages, cursor = [], 0
    for i, (source_manifest, m) in enumerate(chain):
        ident = f'{i+1:02d}_{NAMES[i]}'
        first, last = m['frame_range']
        if first != cursor:
            raise ValueError('Noncontiguous source frames')
        cursor = last+1
        checkpoint = Path(m['checkpoint'])
        meta_path = checkpoint.with_suffix('.meta.json')
        meta = json.loads(meta_path.read_text())
        signature = meta['signature']
        if meta['source_frame'] != last or meta['trajectory_index'] != last:
            raise ValueError('Checkpoint frame mismatch')
        if signature['action_prefix_sha256'] != prefix_digest(q, opening, last):
            raise ValueError('Checkpoint does not match cumulative command prefix')
        if signature['robot_urdf_sha256'] != sha(urdf):
            raise ValueError('Checkpoint URDF hash mismatch')
        if signature['shirt_obj_size'] != cloth.stat().st_size:
            raise ValueError('Checkpoint cloth size mismatch')
        with np.load(m['trajectory'], allow_pickle=False) as d:
            if not np.array_equal(d['joint_q'][:last+1], q[:last+1]) or not np.array_equal(d['openness'][:last+1], opening[:last+1]):
                raise ValueError('Source command prefix mismatch')
        members, archive = {}, out/'checkpoints'/f'{ident}.tar.gz'
        archive.parent.mkdir(exist_ok=True)
        with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
            for suffix in ('.pkl', '.meta.json', '.ipc_state.npz'):
                source = checkpoint.with_suffix(suffix)
                members[source.name] = dict(bytes=source.stat().st_size, sha256=sha(source))
                tar.add(source, arcname=source.name, recursive=False)
        record(archive, 'checkpoint_archive')
        replay = copy(m['replay'], f'reference_replays/{ident}.npz', 'saved_replay')
        with np.load(replay, allow_pickle=False) as d:
            np.testing.assert_array_equal(d['source_frames'], np.arange(first, last+1))
        for key, filename in (('metrics', 'metrics.json'), ('tcp_csv', 'tcp.csv'), ('config', 'source_config.json')):
            copy(m[key], f'source_records/{ident}/{filename}', 'source_record')
        copy(source_manifest, f'source_records/{ident}/manifest.json', 'source_record')
        plan = source_manifest.parent.parent/'plan.json'
        if plan.is_file():
            copy(plan, f'source_records/{ident}/plan.json', 'source_record')
        stages.append(dict(id=ident, name=NAMES[i], frame_range=[first,last],
                           previous_checkpoint=stages[-1]['id'] if stages else None,
                           checkpoint_archive=str(archive.relative_to(out)),
                           checkpoint_members=members, checkpoint_meta=meta,
                           replay=str(replay.relative_to(out)), args=portable_args(m['argv']),
                           source_physics=m['physics']))
        print(f'PACKAGED {ident} frames={first}..{last} checkpoint_archive_MiB={archive.stat().st_size/1024**2:.2f}', flush=True)
    if a.reference_hero:
        copy(a.reference_hero, 'reference/hero.mp4', 'reference_video')
    profile = dict(schema='scene527-seven-stage-reproduction-v1', frames=4373, fps=60,
                   cloth=str(cloth.relative_to(out)), robot_urdf=str(urdf_rel),
                   trajectory=str(trajectory.relative_to(out)), preset=str(preset.relative_to(out)),
                   topology=dict(vertices=27811,faces=55068), stages=stages, files=records,
                   checkpoint_native_cache_restored=False,
                   physics_continuous=False,
                   publication_scope='user-requested reproduction data on existing repository branch')
    (out/'bundle.json').write_text(json.dumps(profile, indent=2)+'\n')
    print(f'COMPLETE {out} files={len(records)} total_GiB={sum(x["bytes"] for x in records.values())/1024**3:.3f}', flush=True)


if __name__ == '__main__':
    main()
