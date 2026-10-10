"""Workbench action-plan services. Physics is exported, never launched by HTTP."""
import copy
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .action_plan import compile_action_plan, parse_gripper_events
except ImportError:
    from action_plan import compile_action_plan, parse_gripper_events

TOOLS = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get('GENESIS_PYTHON', sys.executable))

def seed_state(data, frame):
    """FK in the same URDF link/TCP convention as the IK and runtime."""
    index=data.source_to_index.get(frame)
    if index is None: raise ValueError('起点帧不存在于当前 replay')
    q=data.robot_q[index].copy()
    if data.config.get('planning_command_trajectory') and not data.config.get('start_checkpoint'):
        # Planning against measured cloth while seeding from the exact command
        # prefix; this is not a substitute for a physical checkpoint.
        with np.load(data.config['planning_command_trajectory']) as source:
            q=np.asarray(source['joint_q'][frame],dtype=float).copy()
    if data.config.get('start_checkpoint'):
        meta=json.loads(Path(data.config['start_checkpoint']).with_suffix('.meta.json').read_text())
        if frame!=int(meta['source_frame']):raise ValueError('续跑计划必须选上一段checkpoint末帧')
        q=np.asarray(meta['previous_q'],dtype=float)
    else:
        q[[9,10,17,18]]=np.clip(q[[9,10,17,18]],0,.044)
    root=ET.parse(data.config['robot_urdf']).getroot()
    joints=root.findall('joint'); active=[j.get('name') for j in joints if j.get('type')!='fixed']
    if len(active)!=19: raise ValueError('仅支持当前19列双X5')
    if not data.config.get('start_checkpoint'):
        # Measured replay can overshoot a hard stop slightly. A fresh initial
        # command must be in bounds; never change a restored checkpoint seed.
        for joint in joints:
            limit=joint.find('limit')
            if joint.get('type')=='fixed' or limit is None:continue
            i=active.index(joint.get('name'))
            bounded=np.clip(q[i],float(limit.get('lower','-inf')),float(limit.get('upper','inf')))
            if abs(bounded-q[i])>1e-3:raise ValueError('参考关节越限超过初态数值修正范围，不能生成合法seed')
            q[i]=bounded
    transforms={'base_link':np.eye(4)}
    transforms['base_link'][:3,3]=data.config['robot_base_pos_m']
    pending=list(joints)
    while pending:
        ready=[j for j in pending if j.find('parent').get('link') in transforms]
        if not ready: raise ValueError('URDF关节树无法解析')
        for joint in ready:
            origin=joint.find('origin'); local=np.eye(4); motion=np.eye(4)
            if origin is not None:
                local[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                local[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            kind=joint.get('type')
            if kind!='fixed':
                axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ');value=q[active.index(joint.get('name'))]
                if kind=='prismatic':motion[:3,3]=axis*value
                else:motion[:3,:3]=Rotation.from_rotvec(axis*value).as_matrix()
            transforms[joint.find('child').get('link')]=transforms[joint.find('parent').get('link')]@local@motion
            pending.remove(joint)
    start={}
    for hand,link,cols in [('left','left_link16',[9,10]),('right','right_link26',[17,18])]:
        tf=transforms[link];quat=Rotation.from_matrix(tf[:3,:3]).as_quat()
        start[hand]=dict(pos=(tf[:3,3]+tf[:3,:3]@np.array([.155,.001786,.014])).tolist(),
                         quat=quat[[3,0,1,2]].tolist(),opening=float(np.mean(q[cols])))
    return q,start

def template(data, request):
    frame=int(request.get('frame',data.config['default_frame']))
    q,start=seed_state(data,frame)
    stage=request.get('stage','A')
    if stage not in ('A','B','C'):raise ValueError('子阶段必须为A/B/C')
    actions=[];last=copy.deepcopy(start)
    def add(name,duration,delta=None,targets=None):
        nonlocal last
        last=copy.deepcopy(last)
        for hand in ('left','right'):
            if targets is not None:last[hand]['pos']=list(targets[hand])
            if delta is not None:last[hand]['pos']=(np.array(last[hand]['pos'])+delta).tolist()
        actions.append(dict(id=name,duration_frames=duration,**copy.deepcopy(last)))
    events=''
    if stage=='A':
        targets={}
        for hand in ('left','right'):
            nodes=data._hand_config(hand)['nodes']
            node=next((n for n in nodes if n.get('semantic_role')=='grasp'),None)
            targets[hand]=data.tcp[int(node['frame'])][hand]['world_m'] if node else start[hand]['pos']
        add('pregrasp',120,targets={h:(np.array(p)+[0,0,.02]).tolist() for h,p in targets.items()})
        add('grasp',60,targets=targets);add('close_hold',30)
        add('lift_50mm',90,delta=[0,0,.05]);add('pull_50mm',90,delta=[-.05,0,0]);add('hold',30)
        events=f't={frame+180} close both'
    elif stage=='B':
        add('move_to_midline_EDIT_TARGET',120);add('lower_EDIT_TARGET',90,delta=[0,0,-.05])
        add('release_hold',30);add('retreat',60,delta=[0,0,.03]);add('settle',30)
        events=f't={frame+210} open both'
    else:
        add('pregrasp_EDIT_TARGET',90);add('grasp_EDIT_TARGET',60);add('close_hold',30)
        add('pull_30mm',90,delta=[-.03,0,0]);add('release_hold',30);add('retreat',60,delta=[0,0,.03])
        events=f't={frame+150} close both\nt={frame+270} open both'
    return dict(start_frame=frame,fps=60,start=start,actions=actions,gripper_text=events,
                stage=stage,reference_frame=frame,reference_config=str(data.config_path),
                note='模板不是已验收轨迹。负X回拉仅为可编辑示例；B/C请先导入上一段实际末态。抓点需核对夹口偏移。')

def compile_request(data, request):
    plan=copy.deepcopy(request['plan'])
    frame=int(plan.get('reference_frame',plan['start_frame']))
    if plan['start_frame']!=frame:raise ValueError('起点帧不能脱离所选初态/checkpoint；通过动作时长调整后续帧')
    q,start=seed_state(data,frame)
    plan['start']=start
    plan['gripper_events']=parse_gripper_events(plan.get('gripper_text',''))
    result=compile_action_plan(plan)
    return plan,result,q

def save_plan(data,request):
    plan,compiled,q=compile_request(data,request)
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=data.patch_output_dir.parent/'action_plans'/stamp;folder.mkdir(parents=True,exist_ok=False)
    for name,value in [('plan.json',plan),('compiled.json',compiled)]:
        (folder/name).write_text(json.dumps(value,indent=2,ensure_ascii=False))
    np.savez_compressed(folder/'seed.npz',robot_q=q)
    return folder,compiled

def handle(data,endpoint,request):
    if endpoint=='/api/action-save-release':
        binding=data.config.get('release_edit_binding')
        if not binding:raise ValueError('当前配置没有放手位置编辑绑定')
        original=json.loads(Path(binding['plan']).read_text())
        plan=copy.deepcopy(original)
        plan['actions']=[a for a in plan['actions'] if not a['id'].startswith('C_')]
        release=next(a for a in plan['actions'] if a['id']=='B_lower')
        targets=request['targets'];changes={}
        for hand in ('left','right'):
            target=np.asarray(targets[hand],dtype=float)
            if target.shape!=(3,) or not np.isfinite(target).all():raise ValueError('放手XYZ必须是有限三维坐标')
            delta=target-np.array(binding['release_tcp_world_m'][hand]);changes[hand]=delta.tolist()
            for action in plan['actions']:
                if not action['id'].startswith('B_'):continue
                pos=np.array(action[hand]['pos'],dtype=float)
                pos[:2]+=delta[:2]
                if action['id']!='B_move_midline':pos[2]+=delta[2]
                action[hand]['pos']=pos.tolist()
        last=plan['start_frame']+sum(a['duration_frames'] for a in plan['actions'])
        events=[e for e in parse_gripper_events(plan.get('gripper_text','')) if e['frame']+e['duration_frames']<=last]
        plan['gripper_text']='\n'.join(f"t={e['frame']} {e['command']} {e['hand']} duration={e['duration_frames']}" for e in events)
        plan['stage']='AB_release_edit';plan['release_edit']={'targets':targets,'command_delta_m':changes,'source_plan':binding['plan'],'reference_frame':request.get('source_frame'),'C_requires_new_B_state':True}
        folder,compiled=save_plan(data,{'plan':plan})
        latest=data.patch_output_dir.parent/'release_latest.json'
        record=dict(saved=True,plan=str(folder/'plan.json'),compiled=str(folder/'compiled.json'),targets=targets,
                    signed_drag_x_mm={h:1000*(targets[h][0]-binding['start_tcp_world_m'][h][0]) for h in targets},
                    path_accepted=compiled['accepted'],ik_status='not_run',physics_started=False,note=binding['note'])
        latest.parent.mkdir(parents=True,exist_ok=True);latest.write_text(json.dumps(record,indent=2,ensure_ascii=False))
        return record
    if endpoint=='/api/action-export':
        try:
            from .action_plan_runtime import export_workbench_bundle
        except ImportError:
            from action_plan_runtime import export_workbench_bundle
        folder=Path(request['path']).resolve()
        allowed=(data.patch_output_dir.parent/'action_plans').resolve()
        if not folder.is_relative_to(allowed):raise ValueError('只能导出当前Workbench生成的动作IK')
        _,compiled,_=compile_request(data,request)
        if compiled!=json.loads((folder/'compiled.json').read_text()):raise ValueError('动作计划已变更，请重新运行IK')
        result=json.loads((folder/'ik/result.json').read_text())
        if not result.get('accepted'):raise ValueError('IK未通过，禁止导出')
        if not data.config.get('action_manifest') and compiled['source_frames'][0]!=0:
            raise ValueError('无checkpoint的第一段必须从初态frame0开始；不能将参考视频中途状态当作物理初态')
        bundle=export_workbench_bundle(folder/'run_bundle',data.config_path,folder/'ik/robot_q.npz')
        return dict(manifest=str(folder/'run_bundle/manifest.json'),command=f"bash {folder/'run_bundle/run.sh'}",started=False,frame_range=bundle['frame_range'])
    if endpoint=='/api/action-import':
        try:
            from .action_plan_runtime import import_completed_stage
        except ImportError:
            from action_plan_runtime import import_completed_stage
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output=data.patch_output_dir.parent/'action_bindings'/f'{stamp}.json'
        import_completed_stage(data.config_path,request['manifest'],output)
        return dict(config=str(output),command=f'{PYTHON} {TOOLS/"server.py"} --config {output} --port 8766 --no-browser',note='生成新绑定，未替换当前会话；确认后打开新服务查看末态并规划B/C')
    if endpoint=='/api/action-template':return template(data,request)
    if endpoint=='/api/action-compile':
        _,compiled,_=compile_request(data,request);return compiled
    if endpoint in ('/api/action-save','/api/action-ik'):
        folder,compiled=save_plan(data,request)
        result=dict(path=str(folder),accepted=compiled['accepted'],compiled=str(folder/'compiled.json'))
        if endpoint=='/api/action-save':return result
        if not compiled['accepted']:raise ValueError('PATH未通过；先延长动作时长或修改位移')
        if not data._ik_lock.acquire(blocking=False):raise ValueError('已有IK任务正在执行')
        try:
            env=dict(os.environ)
            with (folder/'ik.log').open('x') as log:
                process=subprocess.run([str(PYTHON),str(TOOLS/'action_plan_ik.py'),'--config',str(data.config_path),
                    '--compiled',str(folder/'compiled.json'),'--seed',str(folder/'seed.npz'),'--output-dir',str(folder/'ik')],
                    stdout=log,stderr=subprocess.STDOUT,env=env,timeout=600)
            report=folder/'ik/result.json'
            if not report.exists():raise RuntimeError(f'IK未生成报告，请查看 {folder}/ik.log；exit={process.returncode}')
            result.update(ik=json.loads(report.read_text()),report=str(report));return result
        finally:data._ik_lock.release()
    raise ValueError('未知动作接口')
