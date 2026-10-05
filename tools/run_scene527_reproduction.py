"""Verify, replay, or simulate the bundled 4373-frame Scene527 result."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def digest(path):
    with Path(path).open('rb') as f:
        return digest_stream(f)


def inside(bundle, rel):
    path = (bundle/rel).resolve()
    if not path.is_relative_to(bundle.resolve()):
        raise ValueError('Bundle path escapes its root: '+rel)
    return path


def read_checkpoint_archive(bundle, stage, target=None):
    archive = inside(bundle, stage['checkpoint_archive'])
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        expected = stage['checkpoint_members']
        if {x.name for x in members} != set(expected) or len(members) != len(expected):
            raise ValueError('Checkpoint archive member mismatch')
        for member in members:
            if not member.isfile() or Path(member.name).name != member.name:
                raise ValueError('Unsafe checkpoint archive member')
            if member.size != expected[member.name]['bytes']:
                raise ValueError('Checkpoint member size mismatch')
            with tar.extractfile(member) as stream:
                if digest_stream(stream) != expected[member.name]['sha256']:
                    raise ValueError('Checkpoint member hash mismatch')
        if target:
            target.mkdir(parents=True, exist_ok=True)
            for name, spec in expected.items():
                path = target/name
                if path.exists() and digest(path) != spec['sha256']:
                    raise FileExistsError('Refusing to overwrite modified checkpoint: '+str(path))
            tar.extractall(target, filter='data')
    return target/'checkpoint.pkl' if target else None


def verify(bundle, profile):
    for rel, spec in profile['files'].items():
        path = inside(bundle, rel)
        if not path.is_file() or path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
            raise ValueError(f'Missing/modified file or Git LFS pointer: {rel}; run git lfs pull')
    urdf = inside(bundle, profile['robot_urdf'])
    for mesh in ET.parse(urdf).iter('mesh'):
        path = (urdf.parent/mesh.attrib['filename']).resolve()
        if not path.is_relative_to(bundle.resolve()) or not path.is_file():
            raise ValueError('Missing/nonportable robot mesh: '+str(path))
    with np.load(inside(bundle, profile['trajectory']), allow_pickle=False) as d:
        q, opening = d['joint_q'], d['openness']
        if q.shape != (4373,19) or opening.shape != (4373,2) or not np.isfinite(q).all() or not np.isfinite(opening).all():
            raise ValueError('Invalid command trace')
        cursor = 0
        for stage in profile['stages']:
            first,last = stage['frame_range']
            if first != cursor:
                raise ValueError('Stage frame discontinuity')
            cursor = last+1
            h = hashlib.sha256()
            h.update(np.ascontiguousarray(q[:last+1], dtype=np.float64).tobytes())
            h.update(np.ascontiguousarray(opening[:last+1], dtype=np.float64).tobytes())
            signature = stage['checkpoint_meta']['signature']
            if h.hexdigest() != signature['action_prefix_sha256'] or digest(urdf) != signature['robot_urdf_sha256']:
                raise ValueError('Checkpoint command/URDF mismatch')
            read_checkpoint_archive(bundle, stage)
            with np.load(inside(bundle, stage['replay']), allow_pickle=False) as replay:
                np.testing.assert_array_equal(replay['source_frames'], np.arange(first,last+1))
        if cursor != 4373:
            raise ValueError('Unexpected final frame')
    print('VERIFIED assets, meshes, seven checkpoint triplets, replays and command prefixes', flush=True)


def replace_frames(tokens, count):
    tokens = tokens[:]
    i = tokens.index('--frames')
    tokens[i+1] = str(count)
    return tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT/'reproduction/scene527_55k_73s')
    p.add_argument('--verify', action='store_true', help='CPU-only data verification; no simulation')
    p.add_argument('--mode', choices=('replay','reference-boundaries','chain','continuous'))
    p.add_argument('--stage', default='all', help='all, stage name, or stage ID')
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--physics-only', action='store_true', help='Skip Genesis offline videos after physics')
    p.add_argument('--headless', action='store_true', help='Explicitly disable the physics viewer')
    p.add_argument('--smoke-frames', type=int, help='Short single-stage/continuous diagnostic')
    a = p.parse_args()
    bundle = a.bundle.resolve()
    profile = json.loads((bundle/'bundle.json').read_text())
    if a.verify:
        verify(bundle, profile)
        return
    if not a.mode:
        p.error('Specify --verify or an explicit --mode')
    stages = profile['stages']
    selected = stages if a.stage == 'all' else [s for s in stages if a.stage in (s['id'],s['name'])]
    if not selected:
        p.error('Unknown stage')
    if a.mode == 'continuous' and a.stage != 'all':
        p.error('Continuous mode requires --stage all')
    if a.mode == 'chain' and a.stage != 'all' and selected[0]['previous_checkpoint']:
        p.error('Use --mode reference-boundaries to resume a single later stage')
    if a.smoke_frames and (a.smoke_frames <= 0 or a.mode == 'replay' or (len(selected)>1 and a.mode!='continuous')):
        p.error('--smoke-frames requires positive length and a single physics stage or continuous mode')
    out = (a.output or ROOT/'outputs/scene527_reproduction'/
           (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+a.mode)).resolve()
    if out.exists() and not a.dry_run:
        raise FileExistsError('Use a new output directory')
    if not a.dry_run:
        verify(bundle, profile)
        out.mkdir(parents=True)
    python = os.environ.get('GENESIS_PYTHON', sys.executable)
    preset = [s.strip() for s in inside(bundle,profile['preset']).read_text().splitlines()
              if s.strip() and not s.lstrip().startswith('#')]
    common = ['--sim1-root',str(bundle),'--trajectory',str(inside(bundle,profile['trajectory'])),
              '--shirt-obj',str(inside(bundle,profile['cloth'])),
              '--robot-urdf',str(inside(bundle,profile['robot_urdf'])),*preset]
    commands, previous_output = [], None
    if a.mode == 'continuous':
        selected = [dict(stages[-1], id='continuous', frame_range=[0,4372], previous_checkpoint=None)]
    env = dict(os.environ)
    # Match the standalone launcher's CUDA library discovery without reading .env.
    purelib = subprocess.check_output([python,'-c','import sysconfig;print(sysconfig.get_paths()["purelib"])'],text=True).strip()
    libs = [str(Path(purelib)/'nvidia'/rel) for rel in
            ('cu13/lib','cuda_runtime/lib','cuda_nvrtc/lib','cublas/lib','cusparse/lib','cusolver/lib','nvjitlink/lib')
            if (Path(purelib)/'nvidia'/rel).is_dir()]
    if libs:
        env['LD_LIBRARY_PATH'] = ':'.join(libs+[env.get('LD_LIBRARY_PATH','')])
    if os.environ.get('GENESIS_GPU'):
        env['CUDA_VISIBLE_DEVICES'] = os.environ['GENESIS_GPU']

    def execute(command, stage, kind):
        print(shlex.join(command), flush=True)
        commands.append(dict(stage=stage,kind=kind,argv=command))
        if not a.dry_run:
            (out/'RUN.json').write_text(json.dumps(dict(mode=a.mode,bundle=str(bundle),commands=commands,
                headless=a.headless,smoke_frames=a.smoke_frames),indent=2)+'\n')
            subprocess.run(command, cwd=ROOT, env=env, check=True)

    lock = None
    if not a.dry_run:
        lock = open(f'/tmp/genesis_ipc_shirt_demo_{os.getuid()}.lock','a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for stage in selected:
        stage_out = out/stage['id']
        if not a.dry_run:
            stage_out.mkdir()
        first,last = stage['frame_range']
        actual_last = min(last,first+a.smoke_frames-1) if a.smoke_frames else last
        args = replace_frames(stage['args'], actual_last+1)
        diagnostics = ['--debug-third-fold-dir',str(stage_out/'third_state'),
                       '--debug-second-fold-dir',str(stage_out/'second_state'),
                       '--keyframe-diagnostics-dir',str(stage_out/'keyframes')]
        base = [python,str(ROOT/'src/run_genesis_ipc.py'),*common,*args,*diagnostics,'--no-record']
        output_checkpoint = stage_out/'checkpoint.pkl'
        if a.mode != 'replay':
            physics = [*base,'--output',str(stage_out/'stage_physics.mp4'),
                       '--dump-replay-states',str(stage_out/'stage_physics.replay_states.npz'),
                       '--save-checkpoint-source-frame',str(actual_last),
                       '--save-third-fold-checkpoint',str(output_checkpoint)]
            if stage['previous_checkpoint']:
                if a.mode == 'chain':
                    input_checkpoint = previous_output
                else:
                    prior = next(s for s in stages if s['id']==stage['previous_checkpoint'])
                    input_checkpoint = out/'inputs'/prior['id']/'checkpoint.pkl'
                    if not a.dry_run:
                        read_checkpoint_archive(bundle,prior,input_checkpoint.parent)
                physics += ['--load-third-fold-checkpoint',str(input_checkpoint),'--allow-checkpoint-path-relocation']
            if not a.headless:
                physics += ['--viewer']
            if a.mode == 'continuous' and not a.smoke_frames:
                physics += ['--save-checkpoint-source-frames',*[str(s['frame_range'][1]) for s in stages[:-1]],
                            '--checkpoint-output-dir',str(stage_out/'checkpoints')]
            execute(physics,stage['id'],'physics')
            previous_output = output_checkpoint
        if a.mode == 'replay' or (not a.physics_only and not a.smoke_frames):
            replay = inside(bundle,stage['replay']) if a.mode=='replay' else stage_out/'stage_physics.replay_states.npz'
            render = [*base,'--replay-states',str(replay),'--record-multi-view','--replay-six-view',
                      '--replay-closeup-hand','left','--output',str(stage_out/'stage.mp4')]
            execute(render,stage['id'],'genesis_saved_state_render')


if __name__ == '__main__':
    main()
