"""One camera per process; persistent RTX scene, atomic verified video chunks."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, record):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


parser = argparse.ArgumentParser()
parser.add_argument('--job', type=Path, required=True)
parser.add_argument('--view', required=True)
parser.add_argument('--start-frame', type=int, required=True)
args = parser.parse_args()
job = args.job.resolve()
config = json.loads((job / 'config.json').read_text())
identity = {key: value for key, value in config.items() if key != 'visual_fingerprint'}
assert hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == config['visual_fingerprint'], 'Config identity changed'
assert args.view in config['views']
assert 0 <= args.start_frame <= config['frames']
for entry in config['input_files']:
    if digest(entry['path']) != entry['sha256']:
        raise RuntimeError('Frozen input changed: ' + entry['path'])
gpu_lock = Path(config['gpu_lock']).open('a')
fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
# No existing compute workload may be displaced by this renderer. Desktop capture
# can appear in the compute list on this workstation and is safe to coexist with.
gpu_processes = subprocess.run(
    ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'],
    capture_output=True, text=True, check=True)
desktop_helpers = {'gnome-remote-desktop-daemon'}
busy = []
for line in gpu_processes.stdout.splitlines():
    fields = [field.strip() for field in line.split(',', 1)]
    if len(fields) == 2 and fields[0].isdigit() and Path(fields[1]).name not in desktop_helpers:
        busy.append(fields[0] + ':' + fields[1])
if busy:
    raise RuntimeError('GPU already has compute processes: ' + ','.join(busy))
view_root = job / 'views' / args.view
(view_root / 'segments').mkdir(parents=True, exist_ok=True)
attempt = view_root / 'attempts' / str(time.time_ns())
attempt.mkdir(parents=True)
progress_path = view_root / 'progress.json'
stop_requested = False


def stop(signum, frame):
    global stop_requested
    stop_requested = True


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
t0 = time.perf_counter()
last_verified = args.start_frame - 1
last_rendered = args.start_frame - 1


def progress(phase, **extra):
    atomic_json(progress_path, dict(view=args.view, phase=phase,
        worker_pid=os.getpid(), last_rendered_frame=last_rendered,
        last_verified_frame=last_verified, next_frame=last_rendered + 1,
        updated_at_unix=time.time(), elapsed_seconds=time.perf_counter() - t0,
        visual_fingerprint=config['visual_fingerprint'], **extra))


progress('initializing')
timings = (attempt / 'timings.jsonl').open('w')
gpu_log = (attempt / 'gpu.csv').open('w')
telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.current.graphics', '--format=csv', '--loop=5'], stdout=gpu_log, stderr=subprocess.STDOUT)
writer = None
success = False
chunk_frames = []


def verify_video(path, count):
    # PyAV lives in the CPU environment, not the Isaac Python environment.
    program = """import av,sys,json
with av.open(sys.argv[1]) as c:
 s=c.streams.video[0]; info={'width':s.width,'height':s.height,'fps':float(s.average_rate),'frames':sum(1 for _ in c.decode(s))}
assert [info['width'],info['height'],info['fps'],info['frames']]==list(map(int,sys.argv[2:])),info
print(json.dumps(info))
"""
    # Isaac injects its Python 3.11 stdlib and native libraries into the environment.
    # The CPU decoder uses Python 3.12 and must not inherit those search paths.
    environment = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'LD_PRELOAD'):
        environment.pop(key, None)
    result = subprocess.run([config['cpu_python'], '-I', '-c', program, str(path),
        str(config['resolution'][0]), str(config['resolution'][1]), str(config['fps']), str(count)],
        env=environment, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Video validation failed: ' + result.stderr[-3000:] + result.stdout[-500:])


def commit_chunk():
    global writer, last_verified
    if writer is None:
        return
    writer.stdin.close()
    if writer.wait(timeout=120) != 0:
        raise RuntimeError('Video encoder failed')
    writer = None
    verify_video(chunk_folder / 'video.mp4', len(chunk_frames))
    record = dict(view=args.view, start_frame=chunk_frames[0], end_frame=chunk_frames[-1] + 1,
        source_frames=chunk_frames, frames=len(chunk_frames), fps=config['fps'],
        resolution=config['resolution'], visual_fingerprint=config['visual_fingerprint'],
        visual_version=config['visual_version'], video_finalized=True,
        video_sha256=digest(chunk_folder / 'video.mp4'))
    atomic_json(chunk_folder / 'manifest.json', record)
    last_verified = chunk_frames[-1]
    progress('chunk_verified')
    print('CHUNK_VERIFIED ' + json.dumps(record), flush=True)


try:
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, renderer='RaytracedLighting',
        disable_viewport_updates=True, width=config['resolution'][0], height=config['resolution'][1],
        multi_gpu=False, sync_loads=True, limit_cpu_threads=8))
    import carb
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    import numpy as np
    from PIL import Image
    from pxr import UsdGeom, Gf, Vt

    ctx = omni.usd.get_context()
    ctx.open_stage(config['scene'])
    stage = ctx.get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    for path in ['/Render/HeroProduct', '/Render/HeroSettings']:
        if stage.GetPrimAtPath(path):
            stage.GetPrimAtPath(path).SetActive(False)
    cloth = UsdGeom.Mesh(stage.GetPrimAtPath('/Demo/Cloth'))
    faces = np.asarray(cloth.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    assert set(cloth.GetFaceVertexCountsAttr().Get()) == {3}
    assert len(cloth.GetPointsAttr().Get()) == config['expected_cloth_vertices']
    assert faces.shape == (config['expected_cloth_faces'], 3)
    replay = np.load(config['replay'])
    cloth_states = replay['cloth_pos']
    assert len(cloth_states) == config['frames'] and np.isfinite(cloth_states).all()
    assert np.array_equal(replay['source_frames'], np.arange(config['frames']))
    robot = np.load(config['robot_visuals'])
    transforms = robot['transforms']
    assert len(transforms) == config['frames'] and np.isfinite(transforms).all()
    robot_ops = [UsdGeom.Xformable(stage.GetPrimAtPath(f'/Demo/CurrentRobot/g{i}')).GetOrderedXformOps()[0]
                 for i in range(transforms.shape[1])]
    parent_transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/Demo'))).T
    assert np.allclose(parent_transform[:3, :3], np.eye(3))
    assert np.allclose(parent_transform[:3, 3], config['demo_translation_m'])
    # Mesh points/transforms remain in source coordinates: the parent provides the relocation.
    camera = UsdGeom.Camera(stage.GetPrimAtPath('/HeroCamera'))
    camera_op = UsdGeom.Xformable(camera).GetOrderedXformOps()[0]
    camera.CreateHorizontalApertureAttr(36.)
    camera.CreateVerticalApertureAttr(36. * config['resolution'][1] / config['resolution'][0])
    cam = config['camera_config'][args.view]
    camera.CreateClippingRangeAttr(Gf.Vec2f(float(cam.get('near_m', .03)), float(cam.get('far_m', 100.))))
    camera.CreateFocalLengthAttr(float(18 / np.tan(np.deg2rad(cam['fov_deg'] / 2))))
    if cam['mode'] == 'right_tcp_follow':
        link_index = list(replay['ipc_link_names'].astype(str)).index(cam['driver_link'])
        tcp_transforms = replay['ipc_link_transforms'][:, link_index]
    settings = carb.settings.get_settings()
    settings.set('/rtx/rendermode', 'RaytracedLighting')
    settings.set('/omni/replicator/captureMotionBlur', False)
    rep.orchestrator.set_capture_on_play(False)
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    timeline.set_auto_update(False)
    product = rep.create.render_product('/HeroCamera', tuple(config['resolution']))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach([product])
    progress('ready')
    for frame in range(args.start_frame, config['frames']):
        if stop_requested or (job / 'STOP').exists() or (view_root / 'STOP').exists():
            break
        frame_t0 = time.perf_counter()
        if writer is None:
            end = min(frame + config['chunk_size'], config['frames'])
            chunk_folder = view_root / 'segments' / f'{frame:04d}_{end:04d}'
            if chunk_folder.exists():
                chunk_folder = chunk_folder.with_name(chunk_folder.name + '_retry_' + str(time.time_ns()))
            chunk_folder.mkdir()
            chunk_frames = []
            writer = subprocess.Popen([config['ffmpeg'], '-v', 'error', '-n', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', f"{config['resolution'][0]}x{config['resolution'][1]}",
                '-r', str(config['fps']), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '4',
                '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                str(chunk_folder / 'video.mp4')], stdin=subprocess.PIPE)
        vertices = cloth_states[frame]
        cloth.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices))
        normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        total = np.zeros_like(vertices)
        for corner in range(3):
            np.add.at(total, faces[:, corner], normals)
        total /= np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-20)
        cloth.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(total))
        cloth.SetNormalsInterpolation('vertex')
        for i, op in enumerate(robot_ops):
            op.Set(Gf.Matrix4d(transforms[frame, i].T.tolist()))
        if cam['mode'] == 'right_tcp_follow':
            tf = tcp_transforms[frame]
            target = tf[:3, 3] + tf[:3, :3] @ np.array([.155, .001786, .014]) + np.array(config['demo_translation_m'])
            position = target + np.array(cam['offset_world_m'])
        else:
            position, target = np.array(cam['pos_m']), np.array(cam['lookat_m'])
        camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(*target), Gf.Vec3d(*cam['up'])).GetInverse())
        before = timeline.get_current_time()
        tick = time.perf_counter()
        rep.orchestrator.step(rt_subframes=config['rt_subframes'], pause_timeline=True, delta_time=0.0)
        render_seconds = time.perf_counter() - tick
        after = timeline.get_current_time()
        if abs(after - before) > 1e-9:
            raise RuntimeError('Unexpected timeline advance')
        pixels = np.ascontiguousarray(np.asarray(rgb.get_data())[:, :, :3], dtype=np.uint8)
        assert pixels.shape == (config['resolution'][1], config['resolution'][0], 3)
        writer.stdin.write(pixels.tobytes())
        if frame % 100 == 0 or frame in (339, 619, 900, 979):
            Image.fromarray(pixels).save(view_root / f'frame_{frame:04d}.png')
        chunk_frames.append(frame)
        last_rendered = frame
        elapsed = time.perf_counter() - frame_t0
        timings.write(json.dumps(dict(frame=frame, render_seconds=render_seconds, total_seconds=elapsed,
            timeline_before=before, timeline_after=after)) + '\n')
        timings.flush()
        progress('rendering', frame_seconds=elapsed, render_seconds=render_seconds)
        if len(chunk_frames) >= config['chunk_size']:
            commit_chunk()
    commit_chunk()
    progress('completed' if last_verified == config['frames'] - 1 else 'stopped')
    success = True
except Exception:
    import traceback
    traceback.print_exc()
    progress('failed')
finally:
    try:
        if writer is not None:
            try:
                writer.stdin.close()
                writer.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                if writer.poll() is None:
                    writer.kill()
    finally:
        telemetry.terminate()
        try:
            telemetry.wait(timeout=5)
        except subprocess.TimeoutExpired:
            telemetry.kill()
        gpu_log.close()
        timings.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if success else 1)
