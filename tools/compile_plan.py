"""Compile a plan using the current Workbench checkpoint, then optionally solve IK."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from trajectory_workbench.server import WorkbenchData
from trajectory_workbench.action_service import save_plan

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--ik',action='store_true')
    a=p.parse_args();data=WorkbenchData(a.config)
    folder,result=save_plan(data,dict(plan=json.loads(a.plan.read_text())))
    print(folder,flush=True)
    if not result['accepted']:raise SystemExit('PATH rejected; inspect compiled.json')
    if a.ik:subprocess.run([sys.executable,str(Path(__file__).parent/'trajectory_workbench/action_plan_ik.py'),'--config',str(a.config.resolve()),'--compiled',str(folder/'compiled.json'),'--seed',str(folder/'seed.npz'),'--output-dir',str(folder/'ik')],check=True)
