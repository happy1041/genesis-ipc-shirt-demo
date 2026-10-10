"""Bind shipped 55k replay/checkpoint to the portable editing workbench."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_scene527_reproduction import read_checkpoint_archive

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--after-stage',default='03_second_fold');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    bundle=ROOT/'reproduction/scene527_55k_73s';profile=json.loads((bundle/'bundle.json').read_text())
    stage=next(s for s in profile['stages'] if s['id']==a.after_stage)
    cp=read_checkpoint_archive(bundle,stage,out/'checkpoint')
    meta=json.loads(cp.with_suffix('.meta.json').read_text());frame=meta['source_frame']
    with np.load(bundle/profile['trajectory']) as z:
        q=z['joint_q'][:frame+1];op=z['openness'][:frame+1]
        np.savez_compressed(out/'prefix.npz',joint_q=q,openness=op,joint_names=z['joint_names'])
    from trajectory_workbench.action_plan_runtime import prefix_hash
    assert prefix_hash(q,op)==meta['signature']['action_prefix_sha256']
    replay=bundle/stage['replay']
    with np.load(replay) as z:begin=int(z['source_frames'][0]);assert int(z['source_frames'][-1])==frame
    cfg=dict(name='55k Workbench after '+a.after_stage,replay=str(replay),cloth_obj=str(bundle/profile['cloth']),
        atlas=str(bundle/'assets/cloth/short-shirt-55068f.garment_atlas.npz'),robot_urdf=str(bundle/profile['robot_urdf']),
        robot_base_pos_m=[0,0,.17],background_mode='replay_cpu',replay_fps=60,
        tcp_csvs=[str(bundle/'source_records'/a.after_stage/'tcp.csv')],patch_output_dir=str(out/'patches'),
        default_frame=frame,display_frame_range=[begin,frame],expected_cloth_vertices=27811,expected_cloth_faces=55068,
        start_checkpoint=str(cp),action_manifest=str(out/'parent_manifest.json'),
        camera=dict(view='overhead',resolution=[800,600],pos_m=[.68,0,2.45],lookat_m=[.68,0,.8],up=[0,1,0],vertical_fov_deg=42),
        regrasp_edit_binding=dict(frame=frame,hands=['left','right'],placement_enabled=True,output_dir=str(out/'edits')))
    cfg['trajectory_editor']=dict(stage='reference',label='Reference replay / next action planning',frame_range=[begin,frame],default_hand='left',material_anchors=[],hands={h:dict(label=h,color=c,nodes=[dict(frame=begin,name='start',draggable=False,support_radius=2),dict(frame=frame,name='checkpoint',draggable=False,support_radius=2)],semantic_path=[dict(frame=begin,zero=True),dict(frame=frame,zero=True)]) for h,c in [('left','#48d58a'),('right','#ff6b78')]})
    cfg['trajectory_editor']['limits']=dict(max_step_mm=20,max_speed_mps=1.2,max_accel_mps2=30)
    (out/'workbench.json').write_text(json.dumps(cfg,indent=2))
    source=json.loads((bundle/'source_records'/a.after_stage/'manifest.json').read_text())
    physics=source.get('physics',dict(action_fps=60,cloth_E=20000,cloth_bending=40,cloth_self_friction=2,initial_shirt_x=.667,initial_shirt_y=.015,initial_shirt_z=.93))
    parent=dict(config=str(out/'workbench.json'),trajectory=str(out/'prefix.npz'),checkpoint=str(cp),physics=physics)
    (out/'parent_manifest.json').write_text(json.dumps(parent,indent=2))
    print(out/'workbench.json')

if __name__=='__main__':main()
