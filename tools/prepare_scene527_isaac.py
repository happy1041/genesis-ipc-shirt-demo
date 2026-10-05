"""Prepare an isolated portable Isaac hero job from bundled saved states.

Preparation is CPU-only. --run explicitly starts the offline GPU renderer.
"""
import argparse
import hashlib
import json
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
    a = p.parse_args()
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
    shutil.copy2(inside(bundle,inputs['robot_visuals']),job/'robot.npz')
    shutil.copy2(inside(bundle,inputs['renderer']),job/'renderer.py')
    shutil.copy2(inside(bundle,inputs['controller']),job/'controller.py')
    positions = None
    cursor = 0
    scratch = job/'cloth_positions.npy'
    for stage in profile['stages']:
        first,last = stage['frame_range']
        with np.load(inside(bundle,stage['replay']),allow_pickle=False) as data:
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
    positions.flush()
    np.savez_compressed(job/'replay.npz',cloth_pos=positions,source_frames=np.arange(profile['frames']))
    del positions
    scratch.unlink()  # Newly created temporary memmap; replay contains its data.
    cfg = json.loads(inside(bundle,inputs['settings']).read_text())
    ffmpeg = str(a.ffmpeg.resolve()) if a.ffmpeg else shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cfg.update(scene=str(job/'scene.usda'),replay=str(job/'replay.npz'),
        robot_visuals=str(job/'robot.npz'),renderer=str(job/'renderer.py'),controller=str(job/'controller.py'),
        cpu_python=sys.executable,isaac_python=str(a.isaac_python.resolve()),ffmpeg=ffmpeg,
        deliverables_dir=str(job/'deliverables'),gpu_lock=str(job/'gpu.lock'),
        visual_version='portable-scene527-4373-frame-shadowed-hero')
    paths=[job/'scene.usda',job/'robot_scene.usda',job/'scene/scene.usdc',job/'robot.npz',job/'replay.npz',job/'renderer.py']
    cfg['input_files']=[dict(path=str(path),sha256=digest(path)) for path in paths]
    cfg['visual_fingerprint']=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (job/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    command=[sys.executable,str(job/'controller.py'),'run','--job',str(job)]
    (job/'PREPARATION.json').write_text(json.dumps(dict(bundle=str(bundle),frames=cursor,
        scene_relocated=True,physics_advanced=False,render_command=command),indent=2)+'\n')
    print('JOB_READY',job,flush=True)
    print(' '.join(command),flush=True)
    if a.run:
        subprocess.run(command,check=True)


if __name__=='__main__':
    main()
