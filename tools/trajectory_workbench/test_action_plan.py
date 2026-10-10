import copy
import json
import unittest

import numpy as np

from tools.trajectory_workbench.action_plan import compile_action_plan, default_plan, parse_gripper_events


class ActionPlanTest(unittest.TestCase):
    def setUp(self):
        self.start = {hand: {"pos": [0., 0., .1], "quat": [1., 0., 0., 0.], "opening": .044}
                      for hand in ("left", "right")}
        self.plan = default_plan(self.start, start_frame=50)

    def test_duration_intervals_endpoints_and_no_return(self):
        result = compile_action_plan(self.plan)
        self.assertEqual(result["source_frames"], list(range(50, 201)))
        self.assertEqual(result["checkpoint_frames"], [200])
        np.testing.assert_allclose(result["left_pos"][60], [0, 0, .15])
        np.testing.assert_allclose(result["left_pos"][-1], [-.05, 0, .15])
        self.assertTrue(result["accepted"])
        json.dumps(result, allow_nan=False)

    def test_independent_hands_and_arbitrary_timing(self):
        self.plan["actions"][0]["duration_frames"] = 97
        self.plan["actions"][0]["right"]["pos"] = [.01, .02, .11]
        result = compile_action_plan(self.plan)
        self.assertEqual(result["events"][0]["frame"], 147)
        np.testing.assert_allclose(result["right_pos"][97], [.01, .02, .11])
        np.testing.assert_allclose(result["left_pos"][97], [0, 0, .15])

    def test_quaternion_shortest_arc_and_inheritance(self):
        self.plan["actions"][0]["left"]["quat"] = [-2, 0, 0, 0]
        del self.plan["actions"][1]["left"]["quat"]
        result = compile_action_plan(self.plan)
        np.testing.assert_allclose(np.linalg.norm(result["left_quat"], axis=1), 1.)
        np.testing.assert_allclose(result["left_quat"], np.tile([1, 0, 0, 0], (151, 1)))

    def test_gripper_event_hold_and_smooth_transition(self):
        self.plan["gripper_events"] = "t=60 close both\nt=120 open right duration=20"
        result = compile_action_plan(self.plan)
        self.assertEqual(result["left_opening"][10], .044)
        self.assertGreater(result["left_opening"][19], 0)
        self.assertLess(result["left_opening"][19], .044)
        self.assertEqual(result["left_opening"][28], 0)
        self.assertEqual(result["left_opening"][-1], 0)
        self.assertEqual(result["right_opening"][-1], .044)

    def test_overlap_and_range_rejected(self):
        for events in ("t=60 close both\nt=65 open right", "t=195 close both", "t=49 close left"):
            with self.subTest(events=events), self.assertRaises(ValueError):
                compile_action_plan({**self.plan, "gripper_events": events})

    def test_simultaneous_different_hands_allowed(self):
        compile_action_plan({**self.plan, "gripper_events": "t=60 close left\nt=60 open right"})

    def test_event_parser(self):
        self.assertEqual(parse_gripper_events("# comment\nt=120 close both # hi")[0]["duration_frames"], 18)
        with self.assertRaises(ValueError):
            parse_gripper_events("close at 120")

    def test_too_fast_rejected_including_one_interval(self):
        self.plan["actions"][0]["duration_frames"] = 1
        result = compile_action_plan(self.plan)
        self.assertFalse(result["accepted"])
        self.assertGreater(result["metrics"]["left"]["max_accel_mps2"], 20)

    def test_input_not_mutated(self):
        original = copy.deepcopy(self.plan)
        compile_action_plan(self.plan)
        self.assertEqual(original, self.plan)

    def test_validation(self):
        changes = [lambda p: p.update(start_frame=1.5),
                   lambda p: p.update(fps=float("nan")),
                   lambda p: p.update(boundaries=["unknown"]),
                   lambda p: p.update(limits={"max_speed_mps": -1}),
                   lambda p: p["actions"][0].update(duration_frames=True),
                   lambda p: p["actions"][0].update(duration_frames=10001),
                   lambda p: p["actions"][0].update(id="A_hold"),
                   lambda p: p["start"]["left"].update(quat=[0, 0, 0, 0]),
                   lambda p: p["start"]["left"].update(pos=[0, 0]),
                   lambda p: p["start"]["left"].update(opening=.045),
                   lambda p: p["start"]["left"].update(pos=[0, float("inf"), 0])]
        for i, change in enumerate(changes):
            plan = copy.deepcopy(self.plan)
            change(plan)
            with self.subTest(case=i), self.assertRaises(ValueError):
                compile_action_plan(plan)


if __name__ == "__main__":
    unittest.main()
