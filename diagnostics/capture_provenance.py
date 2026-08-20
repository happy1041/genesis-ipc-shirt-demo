#!/usr/bin/env python3
"""Capture input hashes, repository states, Python packages and GPU identity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENESIS_ROOT = Path("/home/happy1041/work/genesis-world")
CURRENT_SIM1_ROOT = Path(os.environ.get("SIM1_ROOT", "/home/happy1041/Workspace/SIM1"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_json", type=Path)
    parser.add_argument("--genesis-root", type=Path, default=DEFAULT_GENESIS_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    path = path.expanduser().resolve()
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
    return record


def relocate_sim1_path(path: Path | None) -> tuple[Path | None, str | None]:
    """Resolve archived absolute SIM1 paths after the repository was moved."""
    if path is None:
        return path, None
    expanded = path.expanduser()
    parts = expanded.parts
    if "SIM1" in parts and CURRENT_SIM1_ROOT.exists():
        marker = max(index for index, part in enumerate(parts) if part == "SIM1")
        candidate = CURRENT_SIM1_ROOT.joinpath(*parts[marker + 1 :])
        # Prefer the user-designated canonical root even while the archived
        # mount remains readable.  Otherwise provenance changes depending on
        # whether the old disk happens to be mounted during diagnosis.
        if candidate.exists():
            if candidate.resolve() == expanded.resolve():
                return candidate, None
            return candidate, str(expanded)
    return path, None


def run(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def command_sha256(command: list[str], cwd: Path | None = None) -> str | None:
    """Hash command output incrementally so a large dirty diff is never buffered."""
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    digest = hashlib.sha256()
    assert process.stdout is not None
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    if process.wait() != 0:
        return None
    return digest.hexdigest()


def git_record(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    top = run(["git", "rev-parse", "--show-toplevel"], root)
    if top is None:
        return {"path": str(root), "is_git_repository": False}
    top_path = Path(top)
    status = run(["git", "status", "--short"], top_path) or ""
    status_lines = status.splitlines()
    return {
        "path": str(top_path),
        "is_git_repository": True,
        "head": run(["git", "rev-parse", "HEAD"], top_path),
        "branch": run(["git", "branch", "--show-current"], top_path),
        "dirty": bool(status),
        "status_entry_count": len(status_lines),
        "status_short_first_200": status_lines[:200],
        "working_tree_diff_sha256": command_sha256(
            ["git", "diff", "--binary", "HEAD"], top_path
        ),
    }


def gpu_records() -> list[dict[str, str]]:
    output = run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    records = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 4:
            records.append(
                dict(zip(("name", "uuid", "driver_version", "memory_total_mib"), fields))
            )
    return records


def main() -> None:
    args = parse_args()
    run_json = args.run_json.expanduser().resolve()
    metrics = json.loads(run_json.read_text(encoding="utf-8"))
    recorded_sim1_root = Path(metrics["sim1_root"]) if metrics.get("sim1_root") else None
    if recorded_sim1_root is None and metrics.get("urdf"):
        recorded_urdf_for_root = Path(metrics["urdf"]).expanduser()
        if "SIM1" in recorded_urdf_for_root.parts:
            marker = max(
                index
                for index, part in enumerate(recorded_urdf_for_root.parts)
                if part == "SIM1"
            )
            recorded_sim1_root = Path(
                *recorded_urdf_for_root.parts[: marker + 1]
            )
    sim1_root, sim1_root_relocated_from = relocate_sim1_path(recorded_sim1_root)
    recorded_urdf = Path(metrics["urdf"]) if metrics.get("urdf") else None
    urdf, urdf_relocated_from = relocate_sim1_path(recorded_urdf)
    input_paths = {
        "run_json": run_json,
        "trajectory": Path(metrics["trajectory"]) if metrics.get("trajectory") else None,
        "shirt_obj": Path(metrics["shirt_obj"]) if metrics.get("shirt_obj") else None,
        "urdf": urdf,
        "runner": PROJECT_ROOT / "run_genesis_ipc.py",
        "genesis_ipc_coupler": args.genesis_root
        / "genesis/engine/couplers/ipc_coupler/coupler.py",
    }
    packages = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    manifest = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "captured_after_run": True,
        "warning": (
            "Runner/repository hashes describe capture time. Future runs should capture "
            "this manifest immediately after completion."
        ),
        "path_relocations": {
            "sim1_root": {
                "recorded": sim1_root_relocated_from,
                "resolved": str(sim1_root) if sim1_root_relocated_from else None,
            },
            "urdf": {
                "recorded": urdf_relocated_from,
                "resolved": str(urdf) if urdf_relocated_from else None,
            },
        },
        "run": {
            "backend": metrics.get("backend"),
            "frames": metrics.get("frames"),
            "executed_frames": metrics.get("executed_frames"),
            "seed": metrics.get("seed"),
        },
        "files": {name: file_record(path) for name, path in input_paths.items()},
        "repositories": {
            "project": git_record(PROJECT_ROOT),
            "genesis": git_record(args.genesis_root),
            "sim1": git_record(sim1_root) if sim1_root is not None else None,
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "gpus": gpu_records(),
        },
    }
    output = (
        args.output.expanduser().resolve()
        if args.output
        else run_json.with_suffix(".provenance.json")
    )
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest={output}")


if __name__ == "__main__":
    main()
