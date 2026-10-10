import unittest

import numpy as np

from tools.trajectory_workbench.trajectory_compiler import (
    PositionEdit,
    PositionLimits,
    compile_position_edits,
)


class TrajectoryCompilerTest(unittest.TestCase):
    def setUp(self):
        self.frames = np.arange(100, 201)
        self.base = np.column_stack((self.frames * 0.001, np.zeros(101), np.zeros(101)))
        self.limits = PositionLimits(max_step_mm=5.0, max_speed_mps=1.0, max_accel_mps2=30.0)

    def compile(self, edits):
        return compile_position_edits(
            self.frames,
            self.base,
            edits,
            stage_range=(100, 200),
            fps=60.0,
            limits=self.limits,
        )

    def test_no_edit_identity(self):
        result = self.compile([])
        np.testing.assert_array_equal(result["resolved_position_m"], self.base)

    def test_local_node_exact_and_compact(self):
        delta = np.array((0.0, 0.02, 0.01))
        result = self.compile([PositionEdit(150, delta, 20)])
        index = np.flatnonzero(self.frames == 150)[0]
        np.testing.assert_allclose(result["correction_m"][index], delta, atol=1.0e-12)
        np.testing.assert_array_equal(result["correction_m"][self.frames <= 130], 0.0)
        np.testing.assert_array_equal(result["correction_m"][self.frames >= 170], 0.0)

    def test_two_nodes_are_exact(self):
        edits = [
            PositionEdit(140, np.array((0.01, 0.0, 0.0)), 25),
            PositionEdit(160, np.array((0.0, 0.02, 0.0)), 25),
        ]
        result = self.compile(edits)
        for edit in edits:
            index = np.flatnonzero(self.frames == edit.frame)[0]
            np.testing.assert_allclose(result["correction_m"][index], edit.delta_m, atol=1.0e-12)

    def test_hold_after_ramps_and_holds(self):
        delta = np.array((0.0, 0.0, 0.01))
        result = self.compile([PositionEdit(170, delta, 20, "hold_after")])
        np.testing.assert_array_equal(result["correction_m"][self.frames < 150], 0.0)
        held = result["correction_m"][self.frames >= 170]
        np.testing.assert_allclose(held, np.broadcast_to(delta, held.shape), atol=1.0e-12)

    def test_semantic_keyframes_hold_and_blend(self):
        frames = np.arange(0, 101, dtype=np.int64)
        base = np.zeros((len(frames), 3), dtype=np.float64)
        target = np.array((0.08, -0.03, 0.0))
        apex = np.array((0.0, 0.0, 0.04))
        result = compile_position_edits(
            frames,
            base,
            [],
            stage_range=(0, 100),
            fps=60.0,
            limits=PositionLimits(
                max_step_mm=20.0,
                max_speed_mps=2.0,
                max_accel_mps2=100.0,
            ),
            semantic_keyframes=[
                (10, np.zeros(3)),
                (30, target),
                (40, target),
                (70, apex),
                (90, np.zeros(3)),
                (100, np.zeros(3)),
            ],
        )
        correction = result["correction_m"]
        np.testing.assert_allclose(correction[0], 0.0)
        np.testing.assert_allclose(correction[30], target)
        np.testing.assert_allclose(correction[35], target)
        np.testing.assert_allclose(correction[40], target)
        np.testing.assert_allclose(correction[70], apex)
        np.testing.assert_allclose(correction[90:], 0.0)

    def test_speed_gate_rejects_aggressive_edit(self):
        result = self.compile([PositionEdit(150, np.array((0.0, 0.30, 0.0)), 4)])
        self.assertFalse(result["accepted"])
        self.assertGreater(result["required_duration_scale"], 1.0)

    def test_first_fold_grasp_lift_release_semantics(self):
        frames = np.arange(0, 340, dtype=np.int64)
        base = np.zeros((len(frames), 3), dtype=np.float64)
        grasp = np.array((0.01, -0.02, 0.003))
        lift = np.array((0.01, -0.02, 0.050))
        release = np.array((0.04, 0.01, 0.008))
        result = compile_position_edits(
            frames,
            base,
            [],
            stage_range=(0, 339),
            fps=60.0,
            limits=PositionLimits(100.0, 10.0, 1000.0),
            semantic_keyframes=[
                (0, np.zeros(3)),
                (54, grasp),
                (105, grasp),
                (150, lift),
                (315, release),
                (332, release),
                (339, np.zeros(3)),
            ],
        )
        correction = result["correction_m"]
        np.testing.assert_allclose(correction[54], grasp)
        np.testing.assert_allclose(correction[105], grasp)
        np.testing.assert_allclose(correction[150], lift)
        np.testing.assert_allclose(correction[315], release)
        np.testing.assert_allclose(correction[332], release)
        np.testing.assert_allclose(correction[339], 0.0)


if __name__ == "__main__":
    unittest.main()
