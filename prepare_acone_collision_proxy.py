#!/usr/bin/env python3
"""Build an Acone URDF whose finger collision meshes are conservatively reduced.

Visual meshes remain the original SIM1 assets.  Only the collision elements of
the four finger links are redirected to generated proxies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import trimesh


FINGER_LINKS = {"left_link17", "left_link18", "right_link27", "right_link28"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_urdf", type=Path)
    parser.add_argument("output_urdf", type=Path)
    parser.add_argument("--faces", type=int, default=4000)
    args = parser.parse_args()

    source_urdf = args.source_urdf.expanduser().resolve()
    output_urdf = args.output_urdf.expanduser().resolve()
    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    proxy_dir = output_urdf.parent / "collision_meshes"
    proxy_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source_urdf)
    root = tree.getroot()
    source_root = source_urdf.parent

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        for mesh in link.findall("./visual/geometry/mesh"):
            mesh.set("filename", str((source_root / mesh.attrib["filename"]).resolve()))
        for mesh in link.findall("./collision/geometry/mesh"):
            source_mesh = (source_root / mesh.attrib["filename"]).resolve()
            if link_name not in FINGER_LINKS:
                mesh.set("filename", str(source_mesh))
                continue

            proxy_mesh = proxy_dir / f"{link_name}_{args.faces}f.stl"
            original = trimesh.load_mesh(source_mesh, process=True)
            simplified = original.simplify_quadric_decimation(face_count=args.faces)
            if len(simplified.faces) > args.faces:
                raise RuntimeError(f"Failed to reduce {link_name}: {len(simplified.faces)} faces")
            if not simplified.is_watertight:
                raise RuntimeError(f"Collision proxy for {link_name} is not watertight")
            max_bound_error = float(abs(simplified.bounds - original.bounds).max())
            if max_bound_error > 5.0e-4:
                raise RuntimeError(
                    f"Collision proxy for {link_name} changes its bounds by {max_bound_error:.6f} m"
                )
            simplified.export(proxy_mesh)
            mesh.set("filename", str(proxy_mesh))
            print(
                f"{link_name}: {len(original.faces)} -> {len(simplified.faces)} faces, "
                f"max_bound_error={max_bound_error * 1000.0:.3f} mm"
            )

    ET.indent(tree, space="  ")
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {output_urdf}")


if __name__ == "__main__":
    main()
