import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


SIM1_ROOT = Path(os.environ.get("SIM1_ROOT", "/home/happy1041/Workspace/SIM1"))

snapshot_sets = {
    "no_lateral": "outputs/episode_000000/third_fold_debug_recover_20260813_170645",
    "lateral12": "outputs/episode_000000/third_fold_debug_lateral12_valid_20260813_172824",
}

for frame in (680, 688, 690, 700, 720):
    print(f"\nFRAME {frame}")
    for label, base in snapshot_sets.items():
        data = np.load(f"{base}/frame_{frame:04d}.npz")
        cloth_pos = data["cloth_pos"]
        w, x, y, z = data["tcp_quat_wxyz"]
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        local_pos = (cloth_pos - data["right_link26_pos"]) @ rotation
        selected = (
            (local_pos[:, 0] > 0.10)
            & (local_pos[:, 0] < 0.18)
            & (local_pos[:, 2] > -0.07)
            & (local_pos[:, 2] < 0.07)
            & (np.abs(local_pos[:, 1]) < 0.12)
        )
        cloth_y = local_pos[selected, 1]
        q27, q28 = data["robot_q"][-2:]
        # Approximate distal inner surfaces from the exact finger meshes.
        inner_positive_y = 0.02721 + q27 - 0.0254
        inner_negative_y = -0.023638 - q28 + 0.0254
        in_gap = ((cloth_y >= inner_negative_y) & (cloth_y <= inner_positive_y)).sum()
        print(
            label,
            "q=", np.round([q27, q28], 4),
            "axis_world=", np.round(rotation[:, 1], 4),
            "gap_y=", np.round([inner_negative_y, inner_positive_y], 4),
            "cloth_y_quantiles=", np.round(np.quantile(cloth_y, [0, 0.1, 0.5, 0.9, 1]), 4),
            "in_gap=", f"{in_gap}/{len(cloth_y)}",
        )

        # Use dense vertices of the exact collision STL as a conservative
        # approximation of distance to each finger surface. The meshes contain
        # ~23k vertices each, so this is substantially more informative than
        # TCP or link-origin distance for the thin tapered fingertips.
        near_each_finger = []
        for link_name in ("right_link27", "right_link28"):
            mesh = trimesh.load(
                SIM1_ROOT / "assets" / "acone" / "meshes" / f"{link_name}.STL",
                process=False,
            )
            lw, lx, ly, lz = data[f"{link_name}_quat_wxyz"]
            link_rotation = np.array(
                [
                    [1 - 2 * (ly * ly + lz * lz), 2 * (lx * ly - lz * lw), 2 * (lx * lz + ly * lw)],
                    [2 * (lx * ly + lz * lw), 1 - 2 * (lx * lx + lz * lz), 2 * (ly * lz - lx * lw)],
                    [2 * (lx * lz - ly * lw), 2 * (ly * lz + lx * lw), 1 - 2 * (lx * lx + ly * ly)],
                ]
            )
            finger_world = np.asarray(mesh.vertices) @ link_rotation.T + data[f"{link_name}_pos"]
            distances, _ = cKDTree(finger_world).query(cloth_pos, k=1)
            near_each_finger.append(distances)

        d27, d28 = near_each_finger
        local_region = (
            (local_pos[:, 0] > 0.06)
            & (local_pos[:, 0] < 0.20)
            & (np.abs(local_pos[:, 1]) < 0.15)
            & (np.abs(local_pos[:, 2]) < 0.12)
        )
        print(
            "surface_distance_counts",
            {
                threshold_mm: {
                    "finger27": int(np.sum(local_region & (d27 < threshold_mm / 1000))),
                    "finger28": int(np.sum(local_region & (d28 < threshold_mm / 1000))),
                    "both": int(np.sum(local_region & (d27 < threshold_mm / 1000) & (d28 < threshold_mm / 1000))),
                }
                for threshold_mm in (2, 5, 8, 12)
            },
        )
