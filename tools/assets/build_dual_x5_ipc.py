#!/usr/bin/env python3
"""Build a dual X5-2025 IPC candidate with an articulated compatible gripper.

Official X5-2025 CAD supplies bases/arms. Its closed, merged gripper is replaced
by the geometry-matched Acone housing + moving jaw decomposition. Existing
19-column aliases are retained as an interface contract, including three
non-geometric legacy padding joints; the working robot has 12 arm + 4 jaw DOFs.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve(); output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    (output/'meshes').mkdir()
    source=root/'assets/shirt-demo-assets/runtime/acone_ipc/acone_ipc.urdf'
    official=root/'transport/arx_x5_official_20260920/extracted_2025/X5/urdf/X5.urdf'
    at=ET.parse(source); ar=at.getroot(); xr=ET.parse(official).getroot()
    ar.set('name','dual_x5_2025_ipc_articulated_gripper')
    for mesh in ar.iter('mesh'):
        mesh.set('filename',str((source.parent/mesh.get('filename')).resolve()))
    for name in ['base_link','body4','wheel1','wheel2','wheel3']:
        link=ar.find(f"link[@name='{name}']")
        for tag in ['visual','collision']:
            for child in list(link.findall(tag)):
                link.remove(child)
    provenance={}; topology=[]
    for side,prefix,y in [('left','1',.25),('right','2',-.25)]:
        # Prove the shared chain before retaining the existing controller aliases.
        for i in range(1,7):
            a=ar.find(f"joint[@name='{side}_joint{prefix}{i}']")
            x=xr.find(f"joint[@name='joint{i}']")
            if a.find('axis').get('xyz')!=x.find('axis').get('xyz'):
                raise ValueError('Joint axes differ')
            if i>1:
                for attr in ['xyz','rpy']:
                    av=np.fromstring(a.find('origin').get(attr),sep=' ')
                    xv=np.fromstring(x.find('origin').get(attr),sep=' ')
                    if not np.array_equal(av,xv):
                        raise ValueError('Arm origins differ')
            a.find('limit').attrib.update(x.find('limit').attrib)
            topology.append(dict(alias=a.get('name'),official_joint=x.get('name')))
        for i in range(1,6):
            name=f'{side}_link{prefix}{i}'
            a=ar.find(f"link[@name='{name}']")
            x=xr.find(f"link[@name='link{i}']")
            for child in list(a):
                a.remove(child)
            for element in x:
                a.append(copy.deepcopy(element))
            for mesh in a.iter('mesh'):
                asset=official.parent.parent/'meshes'/Path(mesh.get('filename')).name
                mesh.set('filename',str(asset))
            provenance[name]='official X5-2025 CAD'
        provenance[f'{side}_link{prefix}6']='Acone common X5 gripper housing, static fingers removed'
        for finger in (7,8):
            provenance[f'{side}_link{prefix}{finger}']='Acone common X5 articulated jaw; source origin/axis/0..44mm retained'

        base=copy.deepcopy(xr.find("link[@name='base_link']"))
        base.set('name',f'{side}_x5_base')
        for mesh in base.iter('mesh'):
            mesh.set('filename',str(official.parent.parent/'meshes'/Path(mesh.get('filename')).name))
        ar.append(base)
        mount=ET.SubElement(ar,'joint',name=f'{side}_x5_base_mount',type='fixed')
        ET.SubElement(mount,'parent',link='body4')
        ET.SubElement(mount,'child',link=f'{side}_x5_base')
        ET.SubElement(mount,'origin',xyz=f'.059949 {y} -.036',rpy='0 0 0')
        # Explicit simulation mounting pedestal, spanning tabletop .8 to base .904.
        support=ET.SubElement(ar,'link',name=f'{side}_x5_pedestal')
        for tag in ['visual','collision']:
            part=ET.SubElement(support,tag)
            ET.SubElement(part,'origin',xyz='0 0 0',rpy='0 0 0')
            geom=ET.SubElement(part,'geometry')
            ET.SubElement(geom,'cylinder',radius='.045',length='.104')
        mount=ET.SubElement(ar,'joint',name=f'{side}_x5_pedestal_mount',type='fixed')
        ET.SubElement(mount,'parent',link='body4')
        ET.SubElement(mount,'child',link=f'{side}_x5_pedestal')
        ET.SubElement(mount,'origin',xyz=f'.059949 {y} -.088',rpy='0 0 0')

    # Rebuild conservative wrist hulls from the actual candidate visual geometry.
    hulls={}
    for name in ['left_link15','left_link16','right_link25','right_link26']:
        link=ar.find(f"link[@name='{name}']")
        visual=Path(link.find('visual/geometry/mesh').get('filename'))
        m=trimesh.load_mesh(visual,process=True); h=m.convex_hull
        hull_path=output/'meshes'/(name+'_ipc_hull.obj'); h.export(hull_path)
        link.find('collision/geometry/mesh').set('filename',str(hull_path))
        hulls[name]=dict(source=str(visual),source_sha256=sha(visual),hull=str(hull_path),
                         hull_sha256=sha(hull_path),watertight=bool(h.is_watertight),faces=len(h.faces))
        assert h.is_watertight
    used={}
    for mesh in ar.iter('mesh'):
        path=Path(mesh.get('filename')); assert path.is_file(),path
        used[str(path)]=sha(path)
        mesh.set('filename',os.path.relpath(path,output))
    # Keep the three padding joint names (SIM1 ABI) but remove misleading wheel names.
    for i in range(1,4):
        ar.find(f"link[@name='wheel{i}']").set('name',f'legacy_padding_{i}')
        ar.find(f"joint[@name='joint{i}']/child").set('link',f'legacy_padding_{i}')
    ET.indent(ar)
    target=output/'dual_x5_2025_ipc.urdf'; at.write(target,encoding='utf-8',xml_declaration=True)
    manifest=dict(schema_version=1,kind='dual_x5_2025_ipc_adapter',urdf=str(target),urdf_sha256=sha(target),
                  official_source=str(official),official_source_sha256=sha(official),
                  gripper_source=str(source),gripper_source_sha256=sha(source),
                  base_world_m={'left':[.154067,.2510067,.904],'right':[.154067,-.2489933,.904]},
                  runner_root_translation_m=[0,0,.17],tcp_local_m=[.155,.001786,.014],
                  arm_joint_aliases=topology,physical_arm_dofs=12,physical_jaw_dofs=4,
                  legacy_padding_dofs=3,provenance=provenance,mesh_hashes=used,ipc_hulls=hulls,
                  jaw_control='Two prismatic simulated jaws per hand, preserving baseline staged-close/PD; real actuator synchronization not calibrated.',
                  body_geometry='Two independent X5 bases and simulation pedestals; no Acone torso/chassis/wheels.',
                  source_checkpoint_reuse=False,script_sha256=sha(__file__),
                  scope='Physics candidate; require CPU FK/geometry and bounded IPC validation before full run.')
    (output/'adapter_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'urdf':str(target),'sha256':manifest['urdf_sha256'],'physical_dofs':16,'padding_dofs':3,'ipc_hulls':hulls},indent=2))


if __name__=='__main__':
    main()
