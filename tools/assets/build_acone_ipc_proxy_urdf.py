#!/usr/bin/env python3
"""Build an Acone URDF with usable IPC collision geometry for wrist links.

The public SIM1 URDF gives links 15/16/25/26 a 1 mm cube collision mesh even
though their visual meshes are 13--18 cm across.  That placeholder makes IPC
incapable of preventing the visible wrist and gripper housing from entering the
table or cloth.  This script creates conservative, watertight convex hulls and
rewrites a project-local URDF without modifying the SIM1 checkout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import trimesh


PROXY_LINKS = ("left_link15", "left_link16", "right_link25", "right_link26")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    source_root = source.parent
    output_dir = args.output_dir.expanduser().resolve()
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source)
    root = tree.getroot()

    # A project-local URDF cannot resolve the original relative visual paths.
    # Make every source mesh absolute, then replace only the four bad collision
    # meshes with the generated local hulls.
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename and not Path(filename).is_absolute():
            mesh.set("filename", str((source_root / filename).resolve()))

    for link_name in PROXY_LINKS:
        link = root.find(f"./link[@name='{link_name}']")
        if link is None:
            raise RuntimeError(f"Missing link {link_name} in {source}")
        visual_mesh = link.find("./visual/geometry/mesh")
        collision_mesh = link.find("./collision/geometry/mesh")
        if visual_mesh is None or collision_mesh is None:
            raise RuntimeError(f"Missing visual/collision mesh for {link_name}")

        visual_path = Path(visual_mesh.get("filename", ""))
        source_mesh = trimesh.load_mesh(visual_path, process=True)
        hull = source_mesh.convex_hull
        proxy_path = mesh_dir / f"{link_name}_ipc_hull.obj"
        hull.export(proxy_path)
        collision_mesh.set("filename", str(proxy_path))
        print(
            f"{link_name}: visual_faces={len(source_mesh.faces)} "
            f"proxy_faces={len(hull.faces)} watertight={hull.is_watertight} "
            f"extents_m={hull.extents.tolist()}"
        )

    output_urdf = output_dir / "acone_ipc.urdf"
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {output_urdf}")


if __name__ == "__main__":
    main()
