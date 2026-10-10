"""Prepare an isolated portable Isaac hero job from bundled saved states.

Preparation is CPU-only. --run explicitly starts the offline GPU renderer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

import numpy as np

from run_scene527_reproduction import ROOT, inside, digest, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT/'reproduction/scene527_55k_73s')
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--isaac-python', type=Path, required=True)
    p.add_argument('--ffmpeg', type=Path)
    p.add_argument('--run', action='store_true')
    p.add_argument('--stage-results', type=Path,
                   help='Root containing stage-ID outputs with new replay and robot_visuals; first fold stays reference.')
    p.add_argument('--replay', type=Path, help='Single new 55k saved replay, including Workbench outputs')
    p.add_argument('--robot-visuals', type=Path, help='Actual robot visuals exported from that same replay')
    p.add_argument('--robot-mapping', type=Path, help='JSON list: USD geometry index to Genesis visual index')
    p.add_argument('--reference-frame', type=int, help='Frozen same-pose frame used only to identify robot geometry order')
    a = p.parse_args()
    if bool(a.replay) != bool(a.robot_visuals):
        p.error('--replay and --robot-visuals must be supplied together')
    if a.replay and a.stage_results:
        p.error('--replay and --stage-results are mutually exclusive')
    bundle, job = a.bundle.resolve(), a.job.resolve()
    profile = json.loads((bundle/'bundle.json').read_text())
    verify(bundle,profile)
    if not a.isaac_python.is_file():
        raise FileNotFoundError('Isaac Python interpreter not found')
    job.mkdir(parents=True,exist_ok=False)
    inputs = profile['isaac']
    archive = inside(bundle,inputs['frozen_scene_archive'])
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            if (not (member.isfile() or member.isdir()) or not (job/member.name).resolve().is_relative_to(job)):
                raise ValueError('Unsafe scene archive member')
        tar.extractall(job,filter='data')

    def relocate_overlay(rel,target,parent):
        text = inside(bundle,rel).read_text()
        text,count = re.subn(r'(subLayers\s*=\s*\[\s*)@[^@]+@',lambda m:m[1]+'@'+parent+'@',text,count=1)
        if count != 1:
            raise ValueError('Expected one overlay dependency')
        target.write_text(text)
    relocate_overlay(inputs['robot_overlay'],job/'robot_scene.usda','scene/scene.usdc')
    relocate_overlay(inputs['cloth_overlay'],job/'scene.usda','robot_scene.usda')
    if not (a.stage_results or a.replay):
        shutil.copy2(inside(bundle,inputs['robot_visuals']),job/'robot.npz')
    shutil.copy2(inside(bundle,inputs['renderer']),job/'renderer.py')
    shutil.copy2(inside(bundle,inputs['controller']),job/'controller.py')
    if a.stage_results or a.replay:
        # Generate the isolated single-attempt controller; preserve the frozen bundle.
        controller_text = (job/'controller.py').read_text()
        if controller_text.count('if failures >= 3:') != 1:
            raise ValueError('Unexpected worker retry policy')
        (job/'controller.py').write_text(controller_text.replace('if failures >= 3:', 'if failures >= 1:'))
    positions = None
    cursor = 0
    scratch = job/'cloth_positions.npy'
    robot_transforms = []
    replay_sources = []
    robot_geometry_mappings = []
    reference_robot = None
    if a.stage_results:
        with np.load(inside(bundle,inputs['robot_visuals']),allow_pickle=False) as robot_data:
            reference_robot = robot_data['transforms']
    for stage in ([] if a.replay else profile['stages']):
        first,last = stage['frame_range']
        stage_dir = a.stage_results.resolve()/stage['id'] if a.stage_results else None
        use_new = bool(stage_dir and first > 0)
        replay_path = stage_dir/'stage_physics.replay_states.npz' if use_new else inside(bundle,stage['replay'])
        replay_sources.append(dict(stage=stage['id'], replay=str(replay_path),
                                   origin='new_checkpoint_resume' if use_new else 'uploaded_reference'))
        if a.stage_results:
            robot_path = stage_dir/'robot_visuals.npz' if use_new else inside(bundle,inputs['robot_visuals'])
            with np.load(robot_path,allow_pickle=False) as robot_data:
                transforms = robot_data['transforms']
                if use_new:
                    np.testing.assert_array_equal(robot_data['source_frames'],np.arange(first,last+1))
                    assert len(transforms) == last-first+1
                    # Fixed-link merging may order Genesis vgeoms differently from
                    # the frozen USD geometry list. Match the first restored pose
                    # to the uploaded same-frame visual transforms, then retain
                    # that one mapping across every frame of this stage.
                    from scipy.optimize import linear_sum_assignment
                    cost = np.linalg.norm(reference_robot[first,:,None]-transforms[0,None],axis=(-2,-1))
                    rows, columns = linear_sum_assignment(cost)
                    np.testing.assert_array_equal(rows,np.arange(reference_robot.shape[1]))
                    if transforms.shape[1] != reference_robot.shape[1] or cost[rows,columns].max() > .05:
                        raise ValueError('New robot visual geometry mapping does not match uploaded USD')
                    robot_geometry_mappings.append(dict(stage=stage['id'],usd_to_genesis=columns.tolist(),
                                                         first_pose_max_matrix_error=float(cost[rows,columns].max())))
                    robot_transforms.append(transforms[:,columns])
                else:
                    robot_transforms.append(transforms[first:last+1])
        with np.load(replay_path,allow_pickle=False) as data:
            frames = data['source_frames']
            selected = (frames>=first)&(frames<=last)
            np.testing.assert_array_equal(frames[selected],np.arange(first,last+1))
            points = data['cloth_pos']
            if positions is None:
                positions = np.lib.format.open_memmap(scratch,mode='w+',dtype=points.dtype,
                    shape=(profile['frames'],profile['topology']['vertices'],3))
            if points.dtype != positions.dtype:
                raise ValueError('Replay dtype differs between stages')
            if cursor != first:
                raise ValueError('Unexpected replay seam')
            positions[first:last+1] = points[selected]
            cursor=last+1
        print('MERGED',stage['id'],flush=True)
    if a.replay:
        with np.load(a.replay, allow_pickle=False) as data, np.load(a.robot_visuals, allow_pickle=False) as robot:
            frames = data['source_frames']
            points = data['cloth_pos']
            transforms = robot['transforms']
            np.testing.assert_array_equal(frames, robot['source_frames'])
            if not len(frames) or points.shape != (len(frames), profile['topology']['vertices'], 3):
                raise ValueError('Expected nonempty saved replay with the bundled 55k topology')
            if not np.isfinite(points).all() or not np.isfinite(transforms).all():
                raise ValueError('Nonfinite saved state')
            with np.load(inside(bundle, inputs['robot_visuals']), allow_pickle=False) as frozen:
                reference = frozen['transforms']
            if transforms.shape != (len(frames), reference.shape[1], 4, 4):
                raise ValueError('Unexpected robot visual transform dimensions')
            if a.robot_mapping:
                columns = np.asarray(json.loads(a.robot_mapping.read_text()), dtype=int)
                if sorted(columns.tolist()) != list(range(reference.shape[1])):
                    raise ValueError('Robot mapping must be a permutation of geometry indices')
                error = None
            else:
                frame = int(frames[0]) if a.reference_frame is None else a.reference_frame
                if not 0 <= frame < len(reference):
                    raise ValueError('Specify a valid --reference-frame or --robot-mapping')
                from scipy.optimize import linear_sum_assignment
                cost = np.linalg.norm(reference[frame, :, None] - transforms[0, None], axis=(-2, -1))
                rows, columns = linear_sum_assignment(cost)
                error = float(cost[rows, columns].max())
                if error > .05:
                    raise ValueError('Initial robot pose differs from reference; supply mapping calibrated on an unchanged pose')
            cursor = len(frames)
            np.savez_compressed(job/'replay.npz', cloth_pos=points, source_frames=np.arange(cursor))
            np.savez_compressed(job/'robot.npz', transforms=transforms[:, columns], source_frames=np.arange(cursor))
            (job/'source_frames.json').write_text(json.dumps(frames.tolist())+'\n')
            (job/'robot_mapping.json').write_text(json.dumps(columns.tolist())+'\n')
            replay_sources.append(dict(replay=str(a.replay.resolve()), robot_visuals=str(a.robot_visuals.resolve()),
                replay_sha256=digest(a.replay), robot_visuals_sha256=digest(a.robot_visuals), origin='new_single_replay'))
            robot_geometry_mappings.append(dict(usd_to_genesis=columns.tolist(), first_pose_max_matrix_error=error))
    else:
        positions.flush()
        np.savez_compressed(job/'replay.npz',cloth_pos=positions,source_frames=np.arange(profile['frames']))
        del positions
        scratch.unlink()  # Newly created temporary memmap; replay contains its data.
    if a.stage_results:
        np.savez_compressed(job/'robot.npz',transforms=np.concatenate(robot_transforms),
                            source_frames=np.arange(profile['frames']))
    cfg = json.loads(inside(bundle,inputs['settings']).read_text())
    cfg['frames'] = cursor
    if a.replay:
        cfg['views'] = ['hero']
    ffmpeg = str(a.ffmpeg.resolve()) if a.ffmpeg else shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cfg.update(scene=str(job/'scene.usda'),replay=str(job/'replay.npz'),
        robot_visuals=str(job/'robot.npz'),renderer=str(job/'renderer.py'),controller=str(job/'controller.py'),
        cpu_python=sys.executable,isaac_python=str(a.isaac_python.absolute()),ffmpeg=ffmpeg,
        deliverables_dir=str(job/'deliverables'),
        gpu_lock=f'/tmp/genesis_ipc_shirt_demo_{os.getuid()}.lock' if (a.stage_results or a.replay) else str(job/'gpu.lock'),
        visual_version='new-single-saved-replay-shadowed-hero' if a.replay else 'uploaded-only-reference-first-fold-six-new-checkpoint-stages' if a.stage_results
                       else 'portable-scene527-4373-frame-shadowed-hero')
    paths=[job/'scene.usda',job/'robot_scene.usda',job/'scene/scene.usdc',job/'robot.npz',job/'replay.npz',job/'renderer.py',job/'controller.py']
    cfg['input_files']=[dict(path=str(path),sha256=digest(path)) for path in paths]
    cfg['visual_fingerprint']=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (job/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    command=[sys.executable,str(job/'controller.py'),'run','--job',str(job)]
    (job/'PREPARATION.json').write_text(json.dumps(dict(bundle=str(bundle),frames=cursor,
        scene_relocated=True,physics_advanced=False,replay_sources=replay_sources,
        robot_geometry_mappings=robot_geometry_mappings,
        worker_failure_limit=1 if (a.stage_results or a.replay) else 3,
        checkpoint_seams=[] if a.replay else [s['frame_range'][0] for s in profile['stages'][1:]],
        render_command=command),indent=2)+'\n')
    print('JOB_READY',job,flush=True)
    print(' '.join(command),flush=True)
    if a.run:
        subprocess.run(command,check=True)


if __name__=='__main__':
    main()
