"""Maintainer: extract compact real 8k/13k inspection samples from a local source batch."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source_batch',type=Path)
    args=p.parse_args()
    bundle=ROOT/'reproduction/scene527_55k_73s'
    profile=json.loads((bundle/'bundle.json').read_text())
    for label,n,v in [('8k',8000,4090),('13k',13767,7021)]:
        source=args.source_batch/f'mesh_{n}'
        replay=source/'no_spread_physics.replay_states.npz'
        physical=json.loads((source/'no_spread_physics.json').read_text())
        obj=bundle/f'assets/cloth/short-shirt-{n}f.obj'
        assert sha(Path(physical['shirt_obj']))==sha(obj),'Source mesh bytes differ'
        out=ROOT/'reproduction/workbench_examples'/label
        out.mkdir(parents=True,exist_ok=False)
        with np.load(replay,allow_pickle=False) as z:
            mask=(z['source_frames']>=770)&(z['source_frames']<=829)
            assert mask.sum()==60 and z['cloth_pos'].shape[1]==v
            arrays={k:(z[k] if k=='ipc_link_names' else z[k][mask]) for k in z.files}
            np.savez_compressed(out/'replay.npz',**arrays)
        tcp=source/'no_spread_physics_tcp_trajectory.csv'
        with tcp.open() as src,(out/'tcp.csv').open('w',newline='') as dst:
            reader=csv.DictReader(src);writer=csv.DictWriter(dst,fieldnames=reader.fieldnames);writer.writeheader()
            writer.writerows(row for row in reader if 770<=int(row['source_frame'])<=829)
        subprocess.run([sys.executable,str(ROOT/'diagnostics/build_garment_atlas.py'),str(obj),'--output-prefix',str(out/'atlas')],check=True)
        # Make the atlas metadata portable while retaining its content hash.
        atlas_meta=json.loads((out/'atlas.json').read_text())
        for key,value in list(atlas_meta.items()):
            if isinstance(value,str) and str(ROOT) in value:atlas_meta[key]=value.replace(str(ROOT)+'/', '')
        (out/'atlas.json').write_text(json.dumps(atlas_meta,indent=2)+'\n')
        rel=lambda path:str(path.relative_to(ROOT))
        cfg=dict(name=f'{label} first-fold saved sample / point editing',replay=rel(out/'replay.npz'),cloth_obj=rel(obj),atlas=rel(out/'atlas.npz'),
            robot_urdf=rel(bundle/profile['robot_urdf']),robot_base_pos_m=[0,0,.17],background_mode='replay_cpu',replay_fps=60,
            tcp_csvs=[rel(out/'tcp.csv')],patch_output_dir='GENERATED_BY_PREPARE',default_frame=829,display_frame_range=[770,829],
            expected_cloth_vertices=v,expected_cloth_faces=n,
            camera=dict(view='overhead',resolution=[800,600],pos_m=[.68,0,2.45],lookat_m=[.68,0,.8],up=[0,1,0],vertical_fov_deg=42),
            regrasp_edit_binding=dict(frame=829,hands=['left','right'],placement_enabled=True,output_dir='GENERATED_BY_PREPARE'),
            sample_capabilities=dict(view=True,select_points=True,physical_checkpoint_resume=False))
        (out/'workbench.template.json').write_text(json.dumps(cfg,indent=2)+'\n')
        manifest=dict(source_batch=args.source_batch.name,source_run=f'mesh_{n}',source_replay_sha256=sha(replay),source_tcp_sha256=sha(tcp),
            source_physics_sha256=sha(source/'no_spread_physics.json'),source_mesh_sha256=sha(obj),source_frames=[770,829],
            vertices=v,faces=n,physical_checkpoint_resume=False,stage='first_fold_end',
            extraction='Exact saved world-space frames, robot q/link transforms and matching TCP. No simulation or geometry transformation.',
            files={f.name:sha(f) for f in out.iterdir() if f.is_file()})
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
