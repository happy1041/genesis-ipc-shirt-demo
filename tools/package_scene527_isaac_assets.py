"""Add the verified three-layer static hero scene to a reproduction bundle."""
import argparse
import json
import re
import shutil
import tarfile
from pathlib import Path

from package_scene527_reproduction import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--frozen-scene', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    a = p.parse_args()
    bundle, workspace = a.bundle.resolve(), a.workspace.resolve()
    profile = json.loads((bundle/'bundle.json').read_text())
    cfg = json.loads((a.job/'config.json').read_text())
    directory = bundle/'isaac'
    directory.mkdir(exist_ok=False)

    def record(path, role, source=None):
        profile['files'][str(path.relative_to(bundle))] = dict(
            bytes=path.stat().st_size,sha256=sha(path),role=role,
            source=str(source.resolve().relative_to(workspace)) if source else None)

    def copy(source, name):
        source, target = Path(source), directory/name
        shutil.copy2(source,target)
        record(target,'isaac_input',source)
        return target

    overlay = Path(cfg['scene'])
    robot_overlay = Path(re.search(r'subLayers\s*=\s*\[\s*@([^@]+)@',overlay.read_text()).group(1))
    base = Path(re.search(r'subLayers\s*=\s*\[\s*@([^@]+)@',robot_overlay.read_text()).group(1))
    if base.resolve() != (a.frozen_scene/'scene.usdc').resolve():
        raise ValueError('Unexpected scene dependency')
    files = dict(cloth_overlay=str(copy(overlay,'cloth_overlay.usda').relative_to(bundle)),
                 robot_overlay=str(copy(robot_overlay,'robot_overlay.usda').relative_to(bundle)),
                 robot_visuals=str(copy(cfg['robot_visuals'],'robot_visuals.npz').relative_to(bundle)),
                 renderer=str(copy(cfg['renderer'],'renderer.py').relative_to(bundle)),
                 controller=str(copy(cfg['controller'],'controller.py').relative_to(bundle)))
    archive = directory/'frozen_scene.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        tar.add(a.frozen_scene,arcname='scene')
    record(archive,'isaac_scene_archive')
    files['frozen_scene_archive'] = str(archive.relative_to(bundle))
    settings = {k:cfg[k] for k in ('frames','fps','resolution','chunk_size','views',
        'camera_config','demo_translation_m','rt_subframes','expected_cloth_vertices','expected_cloth_faces')}
    settings_path = directory/'settings.json'
    settings_path.write_text(json.dumps(settings,indent=2)+'\n')
    record(settings_path,'isaac_settings')
    files['settings'] = str(settings_path.relative_to(bundle))
    profile['isaac'] = files
    profile['isaac_dependency_audit'] = dict(layers=3,resolved_scene_assets=442,
        required_builtin_mdl='OmniPBR.mdl supplied by Isaac runtime',cpu_audited=True,gpu_render_rerun=False)
    (bundle/'bundle.json').write_text(json.dumps(profile,indent=2)+'\n')
    print('ISAAC_PACKAGED',archive.stat().st_size,'archive bytes',flush=True)


if __name__ == '__main__':
    main()
