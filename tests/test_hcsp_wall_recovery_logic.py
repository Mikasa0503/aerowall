"""Contract tests for rally and physical-perturbation scoring."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hcsp_wall_recovery_logic import center_band_wall_reward, recenter_prediction, score_rally_events, score_recovery


def rally_events(rounds, wall_y=0.0):
    events = [{"kind": "cap", "y": 0.0, "step": 0, "substep": 0}]
    for ordinal in range(1, rounds + 1):
        events.extend((
            {"kind": "wall", "y": wall_y, "step": ordinal * 2, "substep": 0},
            {"kind": "cap", "y": 0.0, "step": ordinal * 2 + 1, "substep": 0},
        ))
    return events


class RecoveryLogicTest(unittest.TestCase):
    def test_center_band_wall_reward_is_symmetric_continuous_and_bounded(self):
        expected = [(0.0, 10.0), (0.25, 9.375), (0.5, 7.5), (0.75, 4.375),
                    (1.0, 0.0), (1.25, -5.0), (1.5, -10.0), (3.0, -10.0)]
        for y, reward in expected:
            with self.subTest(y=y):
                self.assertAlmostEqual(center_band_wall_reward(y), reward)
                self.assertAlmostEqual(center_band_wall_reward(-y), reward)

    def test_lateral_mirror_and_center(self):
        right = recenter_prediction(0.8, 3.0, -2.3, 0.0, 6.5)
        left = recenter_prediction(-0.8, 3.0, -2.3, 0.0, 6.5)
        center = recenter_prediction(0.0, 3.0, -2.3, 0.0, 6.5)
        self.assertIsNotNone(right)
        self.assertAlmostEqual(right["desired_vy"], -left["desired_vy"])
        self.assertEqual(center["desired_vy"], 0.0)

    def test_invalid_trajectory(self):
        self.assertIsNone(recenter_prediction(0, 1.0, -2.3, 0, 0))
        self.assertIsNone(recenter_prediction(0, 3.0, 2.3, 0, 6.5))
        self.assertIsNone(recenter_prediction(0, 3.0, -0.001, 0, 6.5))

    def test_ten_rounds_need_eleven_caps_and_centered_wall_hits(self):
        self.assertFalse(score_rally_events(rally_events(9))["safe10"])
        self.assertTrue(score_rally_events(rally_events(10))["safe10"])
        events = rally_events(10)
        events[5]["y"] = 1.1
        self.assertFalse(score_rally_events(events)["safe10"])

    def test_contact_reports_in_same_phase_do_not_add_rallies(self):
        events = rally_events(3)
        events.insert(1, dict(events[0]))
        events.insert(3, dict(events[2]))
        score = score_rally_events(events)
        self.assertEqual((score['caps'], score['walls'], score['rallies']), (4, 3, 3))

    def test_recovery_three_wall_window_and_two_extra_rounds(self):
        for centered_wall in (3, 4, 5):
            events = rally_events(centered_wall + 2)
            for event in events:
                if event["kind"] == "wall":
                    event["y"] = 1.5
            events[2 * centered_wall - 1]["y"] = 0.0
            score = score_recovery(events, impulse_end=(4, 1), failed=False)
            self.assertEqual(score["recovery_wall_ordinal"], centered_wall)
            self.assertTrue(score["recovery_success"])
            self.assertFalse(score_recovery(events[:-2], (4, 1), failed=True)["recovery_success"])
        events = rally_events(7, wall_y=1.5)
        events[11]["y"] = 0.0  # Sixth wall is outside the allowed window.
        self.assertFalse(score_recovery(events, (4, 1), failed=False)["recovery_success"])


if __name__ == "__main__":
    unittest.main()
