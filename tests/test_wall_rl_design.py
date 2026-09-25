"""Simulator-free contracts for the dynamic wall-volley curriculum."""

import unittest
import random
from pathlib import Path

from aerowall.wall_rl.curriculum import (
    case_is_feasible,
    make_cases,
    make_hit_start,
    validate_handoff_case,
)
from aerowall.wall_rl.skill_fsm import Skill, select_skill
from aerowall.wall_rl.trajectory import predict_intercept
from aerowall.wall_rl.upgrade_config import load_upgrade_config, verify_hashed_file


class WallRLDesignTest(unittest.TestCase):
    def test_upgrade_config_is_versioned_and_frozen_banks_match_hashes(self):
        root = Path(__file__).resolve().parents[1]
        config = load_upgrade_config(root / 'configs/wall_skill_upgrade_v3.json', root)
        self.assertEqual(config['experiment'], 'aerowall_skill_upgrade_v3')
        self.assertEqual(config['observation_version'], 'aerowall_goal_v1')
        self.assertEqual(config['skills']['launch']['observation_version'], 'legacy')
        self.assertEqual(config['skills']['recover']['observation_version'], 'legacy')
        self.assertEqual(config['skills']['hit']['observation_version'], 'aerowall_goal_v1')
        self.assertEqual(config['reset']['artificial_ratio'], 0.7)
        self.assertEqual(config['reset']['real_policy_prefix_ratio'], 0.3)
        self.assertTrue(config['training']['fixed_incoming'])

    def test_hashed_file_verification_rejects_a_changed_asset(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'asset.json'
            path.write_text('{"version": 1}\n')
            original = verify_hashed_file(path, __import__('hashlib').sha256(path.read_bytes()).hexdigest(), 'test asset')
            path.write_text('{"version": 2}\n')
            self.assertNotEqual(original, __import__('hashlib').sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                verify_hashed_file(path, original, 'test asset')

    def test_banks_are_deterministic_feasible_and_out_of_distribution(self):
        train = make_cases('train', 8121, 128)
        heldout = make_cases('heldout', 9121, 128)
        self.assertEqual(train, make_cases('train', 8121, 128))
        self.assertTrue(all(case_is_feasible(case) for case in train + heldout))
        self.assertTrue(all(abs(case['ball_position'][1]) <= 0.4 for case in train))
        self.assertTrue(all(abs(case['ball_position'][1]) >= 0.45 for case in heldout))
        self.assertTrue(all(abs(case['ball_velocity'][1]) <= 0.3 for case in train))
        self.assertTrue(all(abs(case['ball_velocity'][1]) >= 0.35 for case in heldout))
        impossible = dict(heldout[0], ball_position=[1.5, 4.0, 4.8])
        self.assertFalse(case_is_feasible(impossible))

    def test_intercept_includes_wall_reflection(self):
        outbound = predict_intercept((1.5, 0.0, 4.8), (-2.0, 0.1, 0.0))
        inbound = predict_intercept((1.5, 0.0, 4.8), (2.0, 0.1, 0.0))
        self.assertIsNotNone(outbound)
        self.assertIsNotNone(outbound['wall_time'])
        self.assertGreater(outbound['position'][0], 0.2)
        self.assertIsNone(inbound['wall_time'])
        self.assertGreater(inbound['position'][0], outbound['position'][0])

    def test_finite_wall_side_miss_does_not_reflect(self):
        prediction = predict_intercept((1.5, 2.9, 4.8), (-2.0, 0.0, 0.0))
        self.assertIsNotNone(prediction)
        self.assertIsNone(prediction['wall_time'])
        self.assertLess(prediction['position'][0], 0.2)

    def test_hit_start_sampler_stays_inside_legal_racket_window(self):
        rng = random.Random(771)
        cases = [make_hit_start(rng) for _ in range(256)]
        self.assertTrue(all(0.35 <= case['time_to_contact'] <= 0.45 for case in cases))
        self.assertTrue(all((case['contact_position'][0] - case['drone_position'][0]) ** 2
                            + (case['contact_position'][1] - case['drone_position'][1]) ** 2 <= 0.2 ** 2
                            for case in cases))
        self.assertTrue(all(2.8 <= case['ball_position'][2] <= 3.8 for case in cases))

    def test_handoff_bank_requires_full_state_and_physx_audited_prefix(self):
        case = {
            'case_id': 'handoff-test', 'handoff_state_version': 1,
            'drone_position': [1.5, 0.0, 2.0],
            'drone_orientation': [1.0, 0.0, 0.0, 0.0],
            'drone_velocity': [0.0] * 6, 'motor_throttle': [0.4] * 4,
            'prev_action': [0.0] * 4, 'action_before': [0.0] * 4,
            'ball_position': [1.2, 0.0, 3.0],
            'ball_orientation': [1.0, 0.0, 0.0, 0.0],
            'ball_velocity': [1.0, 0.0, -1.0], 'ball_angular_velocity': [0.0] * 3,
            'prev_ball_velocity': [1.0, 0.0, -1.0], 'wall_target': [0.0, 4.0],
            'phase': 2, 'caps': 1, 'walls': 1, 'rallies': 0,
            'streak': 0, 'max_streak': 0, 'skill_id': 1, 'skill_held_steps': 0,
            'source_event_prefix': [
                {'kind': 'cap', 'y': 0.0, 'step': 20, 'substep': 2},
                {'kind': 'wall', 'y': 0.0, 'step': 50, 'substep': 4},
            ],
        }
        validate_handoff_case(case)
        invalid = dict(case, motor_throttle=[-0.1, 0.4, 0.4, 0.4])
        with self.assertRaisesRegex(ValueError, 'motor_throttle'):
            validate_handoff_case(invalid)
        invalid = dict(case, source_event_prefix=[{'kind': 'wall', 'y': 0.0, 'step': 20, 'substep': 2}])
        with self.assertRaisesRegex(ValueError, 'event prefix'):
            validate_handoff_case(invalid)

    def test_fsm_switches_only_at_control_boundaries_after_dwell(self):
        self.assertEqual(select_skill(Skill.INTERCEPT, caps=0, phase=0,
                                      contact_time=0.1, feasible=True, held_steps=1), Skill.INTERCEPT)
        self.assertEqual(select_skill(Skill.INTERCEPT, caps=0, phase=0,
                                      contact_time=0.1, feasible=True, held_steps=2), Skill.HIT)
        self.assertEqual(select_skill(Skill.INTERCEPT, caps=0, phase=0,
                                      contact_time=0.3, feasible=True, held_steps=2,
                                      hit_window=0.42), Skill.HIT)
        self.assertEqual(select_skill(Skill.HIT, caps=1, phase=1,
                                      contact_time=0.1, feasible=True, held_steps=2), Skill.RECOVER)

    def test_fsm_physical_event_precedes_dwell_and_synthetic_hit_keeps_ownership(self):
        self.assertEqual(select_skill(
            Skill.INTERCEPT, caps=1, phase=1, contact_time=0.0,
            feasible=False, held_steps=0,
        ), Skill.RECOVER)
        self.assertEqual(select_skill(
            Skill.HIT, caps=0, phase=0, contact_time=0.4,
            feasible=True, held_steps=3, artificial_hit=True,
        ), Skill.HIT)

    def test_fsm_uses_entry_exit_hysteresis(self):
        self.assertEqual(select_skill(
            Skill.INTERCEPT, caps=0, phase=0, contact_time=0.17,
            feasible=True, held_steps=2, hit_enter_seconds=0.18,
            hit_exit_seconds=0.22,
        ), Skill.HIT)
        self.assertEqual(select_skill(
            Skill.HIT, caps=0, phase=0, contact_time=0.20,
            feasible=True, held_steps=2, hit_enter_seconds=0.18,
            hit_exit_seconds=0.22,
        ), Skill.HIT)


if __name__ == '__main__':
    unittest.main()
