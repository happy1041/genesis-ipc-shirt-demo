#!/usr/bin/env python3
"""Serial, resumable view rendering. Never removes STOP or invalid artifacts."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

VIEWS = ['hero']

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()

def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {} if default is None else default

def write(path, data):
    p = Path(path)
    temp = p.with_name(p.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temp, p)

def identity(pid):
    try:
        # comm may include spaces; fields after ')' begin at state (field 3).
        tail = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if tail[0] == 'Z' else tail[19]
    except (OSError, IndexError):
        return None

def alive(state):
    return bool(state.get('pid') and state.get('starttime') and identity(state['pid']) == state['starttime'])

def has_stop(job):
    return (job / 'STOP').exists() or any((job / 'views' / view / 'STOP').exists() for view in VIEWS)

def gpu_busy(cfg):
    if not cfg.get('gpu_lock'):
        return False
    with open(cfg['gpu_lock'], 'a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False

def preserve_previous(output):
    # Called only after a replacement video passed full verification.
    token = '.preserved.' + str(time.time_ns()) + '.' + uuid.uuid4().hex
    manifest = Path(str(output) + '.manifest.json')
    if output.exists():
        os.replace(output, output.with_name(output.name + token))
    if manifest.exists():
        os.replace(manifest, manifest.with_name(manifest.name + token))

def workers(job, cfg):
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = (p / 'cmdline').read_bytes().decode().split('\0')
            if cfg['renderer'] in args and '--job' in args and str(job) in args and identity(int(p.name)):
                found.append(int(p.name))
        except (OSError, UnicodeError):
            pass
    return found

def validate_config(cfg):
    canonical = {k: v for k, v in cfg.items() if k != "visual_fingerprint"}
    fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if cfg.get("visual_fingerprint") != fingerprint:
        raise ValueError("Frozen config fingerprint mismatch")

def verify_inputs(cfg):
    for item in cfg['input_files']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen input changed: ' + item['path'])

def decode(path, frames, resolution, fps):
    import av
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Expected one video stream: ' + str(path))
        stream = container.streams.video[0]
        if [stream.width, stream.height] != list(resolution) or abs(float(stream.average_rate or 0) - fps) > .001:
            raise ValueError('Video dimensions or fps mismatch: ' + str(path))
        count = sum(1 for _ in container.decode(stream))
        if count != frames:
            raise ValueError(f'Frame count {count} != {frames}: {path}')

def segments(job, cfg, view):
    valid, invalid = [], []
    for manifest in sorted((job / 'views' / view / 'segments').glob('*/manifest.json')):
        try:
            m = read(manifest)
            start, end = m['start_frame'], m['end_frame']
            if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= cfg['frames']):
                raise ValueError('Invalid bounds')
            if m['view'] != view or m['visual_fingerprint'] != cfg['visual_fingerprint'] or m['video_finalized'] is not True:
                raise ValueError('Version/view/finalization mismatch')
            if m['source_frames'] != list(range(start, end)) or m['frames'] != end-start or m['fps'] != cfg['fps'] or m['resolution'] != cfg['resolution']:
                raise ValueError('Manifest frame contract mismatch')
            video = manifest.parent / 'video.mp4'
            if sha(video) != m['video_sha256']:
                raise ValueError('Video hash mismatch')
            decode(video, end-start, cfg['resolution'], cfg['fps'])
            valid.append((start, end, video))
        except Exception as e:
            invalid.append({'manifest': str(manifest), 'error': str(e)})
    # Retries may duplicate a complete segment. Choose a continuous, non-overlapping chain.
    chain, cursor = [], 0
    while True:
        candidates = [s for s in valid if s[0] == cursor]
        if not candidates:
            break
        s = max(candidates, key=lambda s: (s[1], str(s[2])))
        chain.append(s)
        cursor = s[1]
    return chain, cursor, invalid

def publish(video, cfg, resolution):
    decode(video, cfg['frames'], resolution, cfg['fps'])
    return {'path': str(video), 'sha256': sha(video), 'visual_fingerprint': cfg['visual_fingerprint'], 'frames': cfg['frames'], 'resolution': resolution, 'fps': cfg['fps']}

def existing(output, cfg, resolution):
    m = read(str(output) + '.manifest.json')
    if not output.exists() or m.get('visual_fingerprint') != cfg['visual_fingerprint'] or m.get('sha256') != sha(output):
        return None
    try:
        return publish(output, cfg, resolution)
    except Exception:
        return None

def merge(job, cfg, view, chain):
    dest = Path(cfg['deliverables_dir'])
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / (view + '.mp4')
    old = existing(output, cfg, cfg['resolution'])
    if old:
        return old
    uid = uuid.uuid4().hex
    listing = job / ('concat-' + view + '-' + uid + '.txt')
    listing.write_text(''.join("file '" + str(s[2]).replace("'", "'\\''") + "'\n" for s in chain))
    temp = dest / (view + '.' + uid + '.tmp.mp4')
    subprocess.run([cfg['ffmpeg'], '-nostdin', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(temp)], check=True)
    result = publish(temp, cfg, cfg['resolution'])
    preserve_previous(output)
    os.replace(temp, output)
    result['path'] = str(output)
    write(str(output) + '.manifest.json', result)
    return result

def combine(cfg):
    dest = Path(cfg['deliverables_dir'])
    output = dest / 'four_in_one.mp4'
    old = existing(output, cfg, [2560, 1440])
    if old:
        return old
    temp = dest / ('four_in_one.' + uuid.uuid4().hex + '.tmp.mp4')
    cmd = [cfg['ffmpeg'], '-nostdin', '-v', 'error']
    for view in cfg['views']:
        cmd += ['-i', str(dest / (view + '.mp4'))]
    graph = ';'.join(f'[{i}:v]scale=1280:720,setsar=1[v{i}]' for i in range(4)) + ';[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|1280_0|0_720|1280_720[out]'
    cmd += ['-filter_complex_threads', '1', '-filter_complex', graph, '-map', '[out]', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temp)]
    subprocess.run(cmd, check=True)
    result = publish(temp, cfg, [2560, 1440])
    preserve_previous(output)
    os.replace(temp, output)
    result['path'] = str(output)
    write(str(output) + '.manifest.json', result)
    return result

@contextlib.contextmanager
def lock(job):
    with open(job / '.pipeline.lock', 'a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Controller already owns pipeline lock')
        yield

def run(job, cfg):
    with lock(job):
        state = read(job / 'state.json')
        if state.get('status') == 'blocked':
            return
        state.update(pid=os.getpid(), starttime=identity(os.getpid()), status='running', updated_at=time.time())
        write(job / 'state.json', state)
        try:
            verify_inputs(cfg)
            if workers(job, cfg):
                raise RuntimeError('Orphan renderer is still running; refusing duplicate')
            for view in cfg['views']:
                state['current_view'] = view
                while True:
                    chain, count, invalid = segments(job, cfg, view)
                    state.setdefault('verified_frames', {})[view] = count
                    state['invalid_segments'] = invalid
                    state['updated_at'] = time.time()
                    write(job / 'state.json', state)
                    if has_stop(job):
                        state['status'] = 'stopped'
                        return
                    if count == cfg['frames']:
                        state.setdefault('outputs', {})[view] = merge(job, cfg, view, chain)
                        state['consecutive_failures'] = 0
                        break
                    verify_inputs(cfg)
                    with open(job / (view + '.worker.log'), 'ab', buffering=0) as log:
                        worker_env = dict(os.environ)
                        worker_env['OMNI_KIT_ACCEPT_EULA'] = 'YES'
                        child = subprocess.Popen([cfg['isaac_python'], cfg['renderer'], '--job', str(job), '--view', view, '--start-frame', str(count)],
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            env=worker_env)
                        state['worker_pid'] = child.pid
                        state['worker_starttime'] = identity(child.pid)
                        write(job / 'state.json', state)
                        code = child.wait()
                    state.pop('worker_pid', None)
                    state.pop('worker_starttime', None)
                    if has_stop(job):
                        state['status'] = 'stopped'
                        return
                    _, after, _ = segments(job, cfg, view)
                    if code == 0 and after == cfg['frames']:
                        continue
                    failures = state.get('consecutive_failures', 0) + 1
                    state['consecutive_failures'] = failures
                    state['error'] = f'{view} worker exit={code}, verified={after}/{cfg["frames"]}'
                    if failures >= 3:
                        state['status'] = 'blocked'
                        return
            if has_stop(job):
                state['status'] = 'stopped'
                return
            if len(cfg['views']) > 1:
                state.setdefault('outputs', {})['four_in_one'] = combine(cfg)
            state['status'] = 'complete'
        except Exception as e:
            state['error'] = repr(e)
            state['consecutive_failures'] = state.get('consecutive_failures', 0) + 1
            state['status'] = 'blocked' if state['consecutive_failures'] >= 3 else 'failed'
        finally:
            state['updated_at'] = time.time()
            write(job / 'state.json', state)

def status(job, cfg):
    state = read(job / 'state.json')
    result = dict(state)
    result['controller_alive'] = alive(state)
    result['worker_pids'] = workers(job, cfg)
    result['stop'] = has_stop(job)
    result['verified_frames'] = {}
    latest = 0
    for view in cfg['views']:
        chain, count, invalid = segments(job, cfg, view)
        result['verified_frames'][view] = count
        progress = read(job / 'views' / view / 'progress.json')
        latest = max(latest, progress.get('updated_at_unix', 0))
        result.setdefault('view_progress', {})[view] = progress
        for _, _, video in chain:
            latest = max(latest, video.stat().st_mtime)
    result['latest_verified_progress_at'] = latest or state.get('updated_at', time.time())
    result['reported_stalled'] = bool((result['controller_alive'] or result['worker_pids']) and time.time() - result['latest_verified_progress_at'] > 3600)
    return result

def start(job, cfg):
    with lock(job):
        state = read(job / 'state.json')
        if has_stop(job) or state.get('status') in ('blocked', 'complete') or alive(state) or workers(job, cfg) or gpu_busy(cfg):
            return {'started': False, 'reason': 'STOP, blocked, complete, existing process, or GPU lock held'}
        if state.get('status') in ('starting', 'running'):
            state['consecutive_failures'] = state.get('consecutive_failures', 0) + 1
            if state['consecutive_failures'] >= 3:
                state.update(status='blocked', error='Repeated unexpected controller exits')
                write(job / 'state.json', state)
                return {'started': False, 'reason': 'Repeated unexpected exits; blocked'}
        verify_inputs(cfg)
        with open(job / 'controller.launch.log', 'ab', buffering=0) as log:
            child = subprocess.Popen([cfg['cpu_python'], str(Path(__file__).resolve()), 'run', '--job', str(job)], start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
        # Child blocks very briefly until parent releases lock (run retries acquisition).
        state.update(pid=child.pid, starttime=identity(child.pid), status='starting', updated_at=time.time())
        write(job / 'state.json', state)
        return {'started': True, 'pid': child.pid}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'run', 'status', 'monitor'])
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = Path(args.job).resolve()
    cfg = read(job / 'config.json')
    validate_config(cfg)
    if cfg['views'] != VIEWS:
        raise ValueError('Unexpected view order')
    if args.action == 'run':
        # Detached launch parent is holding lock while recording child identity.
        for attempt in range(30):
            try:
                run(job, cfg)
                return
            except RuntimeError as e:
                if 'pipeline lock' not in str(e) or attempt == 29:
                    raise
                time.sleep(.1)
    elif args.action == 'start':
        print(json.dumps(start(job, cfg), indent=2))
    else:
        result = status(job, cfg)
        if args.action == 'monitor' and not result['controller_alive'] and not result['worker_pids'] and not result['stop'] and result.get('status') not in ('complete', 'blocked'):
            result['repair'] = start(job, cfg)
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
