"""Apply explicit Workbench selections to a copied action plan via reviewed bindings.

Only endpoint positions are changed. Pose, aperture, timing and arc parameters
remain in the template, and the output must pass PATH/IK and physical checks.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np


def vector(value, label):
    a=np.asarray(value,dtype=float)
    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError(f'{label}: expected finite XYZ')
    return a


def convert(plan, selections, mapping):
    if selections.get('schema')!='workbench-material-regrasp-v1':raise ValueError('Expected Workbench saved points schema')
    frame=int(plan['start_frame'])
    if int(selections['source_frame'])!=frame:raise ValueError('Save points at the plan/checkpoint start frame')
    if int(plan.get('reference_frame',frame))!=frame:raise ValueError('Plan reference/start mismatch')
    actions=plan['actions'];by_id={a['id']:a for a in actions}
    if len(by_id)!=len(actions):raise ValueError('Duplicate action IDs')
    result=copy.deepcopy(plan);changed={a['id']:a for a in result['actions']}
    updated=set();records=[]
    rules=mapping['bindings']
    if not rules:raise ValueError('No point bindings')
    for rule in rules:
        hand=rule['hand'];kind=rule['kind']
        if hand not in ('left','right') or kind not in ('grasp','placement'):raise ValueError('Invalid hand/kind')
        selected=(selections.get('points',{}).get(hand) if kind=='grasp' else selections.get('placement_targets_world_m',{}).get(hand))
        if selected is None:
            records.append(dict(hand=hand,kind=kind,status='unspecified_preserved'));continue
        material=selected['material_world_m'] if kind=='grasp' else selected
        # Offset is explicitly specified in WORLD coordinates; never guess TCP calibration.
        target=vector(material,'selection')+vector(rule['tcp_offset_world_mm'],'offset')*.001
        anchor=by_id[rule['anchor_action']][hand]
        delta=target-vector(anchor['pos'],'anchor')
        for action_id in rule['translate_actions']:
            key=(action_id,hand)
            if key in updated:raise ValueError(f'Overlapping binding: {key}')
            old=vector(by_id[action_id][hand]['pos'],'action position')
            changed[action_id][hand]['pos']=(old+delta).tolist();updated.add(key)
        if rule['anchor_action'] not in rule['translate_actions']:raise ValueError('Anchor must be included in translated actions')
        records.append(dict(hand=hand,kind=kind,status='updated',target_tcp_world_m=target.tolist(),delta_mm=(delta*1000).tolist(),actions=rule['translate_actions']))
    if not updated:raise ValueError('No selected points matched bindings')
    result['point_edit']=dict(source_frame=frame,changes=records,ik_status='not_run',physics_started=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','points','bindings','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    result=convert(json.loads(a.plan.read_text()),json.loads(a.points.read_text()),json.loads(a.bindings.read_text()))
    result['point_edit']['inputs']={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.plan,a.points,a.bindings)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(a.output)
