#!/usr/bin/env python3
"""Fetch pinned official ARX model assets with direct connections and blob checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import build_opener, ProxyHandler, Request

MODEL_COMMIT = '74386441845f12af4da203843b6400951d6c5530'
SDK_COMMIT = 'c78328785ea23a81d908e1dcc551eca992f2f1e9'


def get(url):
    request = Request(url, headers={'User-Agent': 'ShirtDemo-model-audit'})
    with build_opener(ProxyHandler({})).open(request, timeout=45) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = []
    repositories = [('ARX_Model', MODEL_COMMIT), ('ARX_X5', SDK_COMMIT)]
    for repo, commit in repositories:
        url = f'https://api.github.com/repos/ARXroboticsX/{repo}/git/trees/{commit}?recursive=1'
        tree = json.loads(get(url))
        if tree.get('truncated') or tree['sha'] != commit:
            raise RuntimeError('Incomplete or mismatched source tree')
        for entry in tree['tree']:
            path = entry['path']
            wanted = (
                repo == 'ARX_Model' and (
                    path == 'X5/X5-2025.7z' or
                    (path.startswith('X5/X5A/') and path != 'X5/X5A/export.log')
                )
            ) or (
                repo == 'ARX_X5' and (
                    path == 'LICENSE' or
                    (path.startswith('ROS2/X5_ws/src/arx_x5_ros2/arx_x5_controller/')
                     and path.endswith(('/x5.urdf', '/x5_2025.urdf')))
                )
            )
            if entry['type'] == 'blob' and wanted:
                tasks.append((repo, commit, entry))

    def download(task):
        repo, commit, entry = task
        relative = Path(repo) / entry['path']
        dest = args.output / relative
        url = f'https://raw.githubusercontent.com/ARXroboticsX/{repo}/{commit}/' + quote(entry['path'])
        content = dest.read_bytes() if dest.exists() else get(url)
        blob = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        if blob != entry['sha'] or len(content) != entry['size']:
            raise RuntimeError(f'Git blob verification failed: {relative}')
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as stream:
                stream.write(content)
        print(f'verified {relative} ({len(content)} bytes)', flush=True)
        return dict(path=str(relative), url=url, commit=commit, bytes=len(content),
                    git_blob_sha1=blob, sha256=hashlib.sha256(content).hexdigest())

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(download, tasks))
    manifest = dict(repositories=dict(repositories), network='direct; proxies disabled',
                    scope='official model files only; no hardware SDK/install scripts executed',
                    model_repository_license='No repository-level license file found in pinned ARX_Model tree; SDK license is separate.',
                    files=files, total_bytes=sum(item['bytes'] for item in files))
    target = args.output / 'download_manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise FileExistsError('Refusing to overwrite different download manifest')
    if not target.exists():
        target.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'files':len(files), 'bytes':manifest['total_bytes'], 'manifest':str(target)}))


if __name__ == '__main__':
    main()
