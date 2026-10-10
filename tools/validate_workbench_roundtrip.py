"""Optional short integration check using a real shipped checkpoint.

CPU prepares a three-frame 0.1mm edit and solves IK; --physics additionally
runs those frames with a viewer and produces Genesis saved-state rendering.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from trajectory_workbench.server import WorkbenchData
from trajectory_workbench.action_service import seed_state, save_plan
from trajectory_workbench.action_plan_runtime import export_workbench_bundle

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--physics',action='store_true');a=p.parse_args()
    out=a.output.resolve()
    subprocess.run([sys.executable,str(ROOT/'tools/prepare_workbench.py'),'--output',str(out)],check=True)
    config=out/'workbench.json';data=WorkbenchData(config);frame=int(data.config['default_frame'])
    _,start=seed_state(data,frame)
    target={h:dict(pos=list(s['pos']),quat=list(s['quat']),opening=s['opening']) for h,s in start.items()}
    target['right']['pos'][2]+=.0001
    plan=dict(start_frame=frame,reference_frame=frame,fps=60,start=start,actions=[dict(id='short_clearance',duration_frames=3,**target)])
    folder,compiled=save_plan(data,dict(plan=plan));assert compiled['accepted']
    subprocess.run([sys.executable,str(ROOT/'tools/trajectory_workbench/action_plan_ik.py'),'--config',str(config),'--compiled',str(folder/'compiled.json'),'--seed',str(folder/'seed.npz'),'--output-dir',str(folder/'ik')],check=True)
    bundle=export_workbench_bundle(out/'run_bundle',config,folder/'ik/robot_q.npz',replay_six_view=True)
    if a.physics:
        env=dict(os.environ,GENESIS_PYTHON=sys.executable,SIM1_ROOT=str(ROOT/'reproduction/scene527_55k_73s'))
        env.update(bundle['env'])
        for cmd,log in [(bundle['argv'],'physics.log'),(bundle['render_argv'],'render.log')]:
            with (out/'run_bundle'/log).open('x') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        metrics=json.loads(Path(bundle['metrics']).read_text());assert metrics['executed_frames']==3
        with np.load(bundle['replay']) as z:assert len(z['source_frames'])==3 and np.isfinite(z['cloth_pos']).all()
    (out/'VALIDATION.json').write_text(json.dumps(dict(ik=True,export=True,physics=a.physics,frame_range=bundle['frame_range'],manifest=str(out/'run_bundle/manifest.json')),indent=2))
    print(out/'VALIDATION.json')
