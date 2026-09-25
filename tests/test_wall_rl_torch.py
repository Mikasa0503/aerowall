"""Tensor contracts, runnable in the isolated Isaac Sim Python environment."""

import importlib.util
import sys
import unittest
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/HCSP'))
sys.path.insert(0, str(ROOT / 'scripts'))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch requires the Isaac Sim runtime')
class WallRLTensorTest(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def test_relative_observation_width_and_coordinates(self):
        from aerowall.wall_rl.observations import build_relative_observation
        torch = self.torch
        root = torch.zeros(2, 1, 23)
        root[..., 3] = 1
        root[..., 2] = 2
        prefix = torch.zeros(2, 1, 26)
        ball = torch.tensor([[[1., 0., 3.]], [[0., -1., 3.]]])
        velocity = torch.zeros(2, 1, 3)
        intercept = ball.clone()
        obs = build_relative_observation(
            prefix, root, ball, velocity, intercept,
            torch.tensor([.5, .5]), torch.tensor([True, True]),
            torch.tensor([0, 2]), torch.tensor([[0., 4.], [.5, 4.]]),
            torch.zeros(2, 1, 4),
        )
        self.assertEqual(tuple(obs.shape), (2, 1, 46))
        self.assertTrue(torch.allclose(obs[0, 0, 26:29], torch.tensor([1., 0., 1.])))
        self.assertTrue(torch.allclose(obs[1, 0, 26:29], torch.tensor([0., -1., 1.])))

    def test_aerowall_goal_view_appends_normalized_command_without_changing_legacy_features(self):
        from aerowall.wall_rl.observations import build_aerowall_goal_observation
        torch = self.torch
        legacy = torch.arange(46, dtype=torch.float32).reshape(1, 1, 46)
        target = torch.tensor([[0.15, 4.1]])
        view = build_aerowall_goal_observation(
            legacy, target, y_bounds=(-0.3, 0.3), z_bounds=(3.8, 4.2),
        )
        self.assertEqual(tuple(view.shape), (1, 1, 48))
        self.assertTrue(torch.equal(view[..., :46], legacy))
        self.assertTrue(torch.allclose(view[0, 0, 46:], torch.tensor([0.5, 0.5])))

    def test_relative_velocity_rotates_world_velocity_difference(self):
        from aerowall.wall_rl.observations import build_relative_observation
        torch = self.torch
        root = torch.zeros(2, 1, 23)
        root[..., 2] = 2.0
        root[0, 0, 3:7] = torch.tensor([2 ** -0.5, 0., 0., 2 ** -0.5])
        root[1, 0, 3:7] = torch.tensor([2 ** -0.5, 0., 2 ** -0.5, 0.])
        root[0, 0, 7:10] = torch.tensor([1., 0., 0.])
        root[1, 0, 7:10] = torch.tensor([1., 0., 1.])
        ball = torch.tensor([[[1., 0., 3.]], [[0., 0., 3.]]])
        ball_velocity = torch.tensor([[[2., 0., 0.]], [[1., 0., 2.]]])
        prefix = torch.zeros(2, 1, 26)
        obs = build_relative_observation(
            prefix, root, ball, ball_velocity, ball,
            torch.ones(2), torch.ones(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.long), torch.zeros(2, 2),
            torch.zeros(2, 1, 4),
        )
        expected = torch.tensor([[0., -1., 0.], [-1., 0., 0.]])
        self.assertTrue(torch.allclose(obs[:, 0, 29:32], expected, atol=1e-6))

    def test_relative_v2_keeps_its_historical_velocity_semantics(self):
        from aerowall.wall_rl.observations import build_relative_observation_v2
        torch = self.torch
        root = torch.zeros(1, 1, 23)
        root[..., 3] = 2 ** -0.5
        root[..., 6] = 2 ** -0.5
        root[..., 2] = 2.0
        root[..., 7] = 1.0
        velocity = torch.tensor([[[2., 0., 0.]]])
        obs = build_relative_observation_v2(
            torch.zeros(1, 1, 26), root, torch.tensor([[[1., 0., 3.]]]),
            velocity, torch.tensor([[[1., 0., 3.]]]), torch.ones(1),
            torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.long),
            torch.zeros(1, 2), torch.zeros(1, 1, 4),
        )
        self.assertTrue(torch.allclose(obs[0, 0, 29:32], torch.tensor([-1., -2., 0.]), atol=1e-6))

    def test_reward_is_bounded_by_one_shot_events(self):
        from aerowall.wall_rl.rewards import compute_skill_reward
        torch = self.torch
        zero = torch.zeros(2)
        false = torch.zeros(2, dtype=torch.bool)
        reward = compute_skill_reward(
            potential_before=zero, potential_after=zero, approach_mask=false,
            legal_contact=torch.tensor([True, False]), next_contact=false,
            legal_wall=false, outbound_quality=torch.tensor([.5, 0.]),
            illegal=false, crash=false, out=false,
            action=torch.zeros(2, 4), previous_action=torch.zeros(2, 4),
        )
        self.assertTrue(torch.allclose(reward, torch.tensor([3., 0.])))

    def test_causal_v2_scores_high_recoverable_wall_targets(self):
        from aerowall.wall_rl.trajectory import outbound_quality_v2
        torch = self.torch
        pos = torch.tensor([[1.5, 0.0, 2.18]])
        vel = torch.tensor([[-2.0, 0.0, 6.2]])
        drone = torch.tensor([[1.5, 0.0, 2.0]])
        high_target = torch.tensor([[0.0, 4.1]])
        low_target = torch.tensor([[0.0, 2.2]])
        high = outbound_quality_v2(pos, vel, high_target, drone)
        low = outbound_quality_v2(pos, vel, low_target, drone)
        self.assertTrue(torch.isfinite(high).all() and torch.isfinite(low).all())
        self.assertGreater(float(high[0]), float(low[0]))

    def test_wall_quality_is_one_shot_and_monotonic(self):
        from aerowall.wall_rl.rewards import compute_skill_reward
        torch = self.torch
        zero = torch.zeros(2)
        false = torch.zeros(2, dtype=torch.bool)
        wall = torch.ones(2, dtype=torch.bool)
        reward = compute_skill_reward(
            potential_before=zero, potential_after=zero, approach_mask=false,
            legal_contact=false, next_contact=false, legal_wall=wall,
            outbound_quality=zero, wall_quality=torch.tensor([0.0, 1.0]),
            illegal=false, crash=false, out=false,
            action=torch.zeros(2, 4), previous_action=torch.zeros(2, 4),
        )
        self.assertTrue(torch.allclose(reward, torch.tensor([2.0, 10.0])))

    def test_batch_prediction_matches_scalar(self):
        from aerowall.wall_rl.trajectory import predict_intercept, predict_intercept_batch
        torch = self.torch
        pos = torch.tensor([[1.5, 0., 4.8]])
        vel = torch.tensor([[-3., .1, 0.]])
        batched, time, feasible = predict_intercept_batch(pos, vel, torch.tensor([[1.5, 0., 2.]]))
        scalar = predict_intercept(pos[0].tolist(), vel[0].tolist())
        self.assertTrue(torch.allclose(batched[0], torch.tensor(scalar['position']), atol=1e-5))
        self.assertAlmostEqual(float(time[0]), scalar['time'], places=5)
        self.assertTrue(bool(feasible[0]))

    def test_finite_wall_side_miss_matches_scalar_without_reflection(self):
        from aerowall.wall_rl.trajectory import predict_intercept, predict_intercept_batch
        torch = self.torch
        pos = torch.tensor([[1.5, 2.9, 4.8]])
        vel = torch.tensor([[-2., 0., 0.]])
        scalar = predict_intercept(pos[0].tolist(), vel[0].tolist())
        batch, time, feasible = predict_intercept_batch(pos, vel, torch.tensor([[1.5, 2.9, 2.]]))
        self.assertIsNone(scalar['wall_time'])
        self.assertLess(scalar['position'][0], 0.2)
        self.assertTrue(torch.allclose(batch[0], torch.tensor(scalar['position']), atol=1e-5))
        self.assertAlmostEqual(float(time[0]), scalar['time'], places=5)
        self.assertFalse(bool(feasible[0]))

    def test_causal_v3_reward_terms_suppress_invalid_and_terminal_potential_jumps(self):
        from aerowall.wall_rl.rewards import compute_aerowall_causal_v3_reward_terms
        torch = self.torch
        zero = torch.zeros(2)
        false = torch.zeros(2, dtype=torch.bool)
        terms = compute_aerowall_causal_v3_reward_terms(
            potential_before=torch.tensor([-0.8, -0.8]),
            potential_after=torch.tensor([0.0, -0.7]),
            potential_before_valid=torch.tensor([True, True]),
            potential_after_valid=torch.tensor([False, True]),
            approach_mask=torch.tensor([True, True]), terminal=false,
            legal_contact=false, next_contact=false, legal_wall=false,
            outbound_quality=zero, illegal=false, crash=false, out=false,
            action=torch.zeros(2, 4), previous_action=torch.zeros(2, 4),
        )
        self.assertTrue(torch.allclose(terms['approach_shaping'], torch.tensor([0., 0.05175]), atol=1e-6))
        self.assertTrue(torch.allclose(terms['total'], sum(
            value for name, value in terms.items() if name != 'total'
        )))

    def test_hit_reset_sampler_uses_the_tested_legal_contact_window(self):
        from aerowall.wall_rl.curriculum import sample_hit_starts
        torch = self.torch
        drone = torch.tensor([[1.5, 0.0, 2.0]]).expand(512, 3).clone()
        starts = sample_hit_starts(drone)
        radial = (starts['contact_position'][:, :2] - drone[:, :2]).norm(dim=-1)
        self.assertTrue(bool(((starts['time_to_contact'] >= .35) &
                              (starts['time_to_contact'] <= .45)).all()))
        self.assertTrue(bool((radial <= .2).all()))
        reconstructed = (starts['ball_position'][:, 2]
                         + starts['ball_velocity'][:, 2] * starts['time_to_contact']
                         - 4.905 * starts['time_to_contact'].square())
        self.assertTrue(torch.allclose(reconstructed, torch.full_like(reconstructed, 2.18), atol=1e-5))

    def test_batch_fsm_matches_scalar_for_hysteresis_and_physical_priority(self):
        from aerowall.wall_rl.skill_fsm import Skill, select_skill, select_skill_batch
        torch = self.torch
        previous = torch.tensor([0, 1, 1, 1, 2])
        caps = torch.tensor([0, 0, 0, 1, 0])
        phase = torch.tensor([0, 0, 0, 1, 2])
        contact = torch.tensor([.17, .20, .40, 0., .23])
        feasible = torch.tensor([True, True, True, False, True])
        held = torch.tensor([2, 3, 4, 0, 5])
        artificial = torch.tensor([False, False, True, False, False])
        batch = select_skill_batch(
            previous, caps, phase, contact, feasible, held,
            hit_enter_seconds=.18, hit_exit_seconds=.22, min_dwell_steps=2,
            artificial_hit=artificial,
        )
        scalar = torch.tensor([
            int(select_skill(Skill(int(previous[i])), caps=int(caps[i]), phase=int(phase[i]),
                             contact_time=float(contact[i]), feasible=bool(feasible[i]),
                             held_steps=int(held[i]), hit_enter_seconds=.18,
                             hit_exit_seconds=.22, min_dwell_steps=2,
                             artificial_hit=bool(artificial[i])))
            for i in range(len(previous))
        ])
        self.assertTrue(torch.equal(batch, scalar))
        self.assertEqual(int(batch[3]), int(Skill.RECOVER))

    def test_real_hcsp_skill_chain_keeps_sample_and_update_actor_inputs_identical(self):
        torch = self.torch
        from omegaconf import OmegaConf
        from tensordict import TensorDict
        from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
        from hcsp.utils.torchrl.env import AgentSpec, DummyEnv
        from aerowall_policy_encoder import configure_policy_encoder

        configure_policy_encoder()
        from hcsp.learning import MAPPOPolicy
        from aerowall_skill_policies import AeroWallSkillChainPolicy

        observation_spec = CompositeSpec({
            'agents': CompositeSpec({
                'observation': UnboundedContinuousTensorSpec((1, 46)),
            }),
        })
        action_spec = CompositeSpec({
            'agents': CompositeSpec({
                'action': BoundedTensorSpec(-1.0, 1.0, (1, 4)),
            }),
        })
        reward_spec = CompositeSpec({
            'agents': CompositeSpec({
                'reward': UnboundedContinuousTensorSpec((1, 1)),
            }),
        })
        spec_env = DummyEnv(observation_spec, action_spec, reward_spec)
        agent_spec = AgentSpec(
            'drone', 1, observation_key=('agents', 'observation'),
            action_key=('agents', 'action'), reward_key=('agents', 'reward'),
            _env=spec_env,
        )
        cfg = OmegaConf.load(str(ROOT / 'third_party/HCSP/cfg/algo/mappo.yaml'))
        torch.manual_seed(2901)
        source = MAPPOPolicy(cfg, agent_spec, device='cpu')
        with tempfile.TemporaryDirectory() as temp:
            launch_path = Path(temp) / 'launch.pt'
            recovery_path = Path(temp) / 'recovery.pt'
            hit_path = Path(temp) / 'hit.pt'
            torch.save(source.state_dict(), launch_path)
            torch.save(source.state_dict(), recovery_path)
            torch.manual_seed(2902)
            hit_source = MAPPOPolicy(cfg, agent_spec, device='cpu')
            torch.save(hit_source.state_dict(), hit_path)
            policy = AeroWallSkillChainPolicy(
                cfg, agent_spec, 'cpu', train_skill='hit',
                launch_checkpoint=launch_path, recovery_checkpoint=recovery_path,
                launch_observation_version='relative_v2',
                skill_observation_version='legacy',
            )

            td = TensorDict({
                'agents': {
                    'observation': torch.randn(2, 1, 46),
                    'legacy_observation': torch.randn(2, 1, 46),
                    'relative_v2_observation': torch.randn(2, 1, 46),
                    'relative_v3_observation': torch.randn(2, 1, 46),
                },
                'stats': {'caps': torch.zeros(2, 1, dtype=torch.long)},
                'info': {'skill_id': torch.tensor([[1.], [0.]])},
            }, batch_size=[2])
            output = policy(td)
            self.assertTrue(torch.equal(
                output['agents', 'actor_observation'][0],
                output['agents', 'legacy_observation'][0],
            ))
            self.assertTrue(torch.equal(output['executed_skill_id'], torch.tensor([1, 0])))
            new_log_prob = policy.recompute_action_log_prob(output)
            sampled_log_prob = output[policy.act_logps_name]
            ratio = torch.exp(new_log_prob[0] - sampled_log_prob[0])
            self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6))

            launch_input = output.clone()
            launch_input['agents', 'observation'] = output['agents', 'relative_v2_observation']
            expected_launch = policy.frozen['launch'](launch_input, deterministic=True)
            self.assertTrue(torch.allclose(
                output['agents', 'action'][1], expected_launch['agents', 'action'][1], atol=1e-6,
            ))

            before_launch = [value.detach().clone() for value in policy.frozen['launch'].actor_params.parameters()]
            before_recovery = [value.detach().clone() for value in policy.frozen['recovery'].actor_params.parameters()]
            empty = output.clone()
            empty['executed_skill_id'] = torch.zeros(2, dtype=torch.long)
            self.assertEqual(policy.update_actor(empty)['actor_grad_norm'], 0.0)
            for before, after in zip(before_launch, policy.frozen['launch'].actor_params.parameters()):
                self.assertTrue(torch.equal(before, after))
            for before, after in zip(before_recovery, policy.frozen['recovery'].actor_params.parameters()):
                self.assertTrue(torch.equal(before, after))

            # The AeroWall target-command extension adds two actor inputs but
            # keeps the C350 legacy action exactly before those weights train.
            goal_observation_spec = CompositeSpec({
                'agents': CompositeSpec({
                    'observation': UnboundedContinuousTensorSpec((1, 48)),
                }),
            })
            goal_env = DummyEnv(goal_observation_spec, action_spec, reward_spec)
            goal_agent_spec = AgentSpec(
                'drone', 1, observation_key=('agents', 'observation'),
                action_key=('agents', 'action'), reward_key=('agents', 'reward'),
                _env=goal_env,
            )
            goal_policy = AeroWallSkillChainPolicy(
                cfg, goal_agent_spec, 'cpu', train_skill='hit',
                launch_checkpoint=launch_path, recovery_checkpoint=recovery_path,
                launch_observation_version='legacy',
                skill_observation_version='aerowall_goal_v1',
                hit_observation_version='aerowall_goal_v1',
            )
            from train_aerowall_wall_rl import warmstart_actor
            warmstart = warmstart_actor(goal_policy, recovery_path)
            self.assertEqual(warmstart['padded_input_tensors'] != [], True)
            self.assertLessEqual(warmstart['action_preservation_max_error'], 1e-5)

            frozen_route = AeroWallSkillChainPolicy(
                cfg, agent_spec, 'cpu', train_skill=None,
                launch_checkpoint=launch_path, recovery_checkpoint=recovery_path,
                hit_checkpoint=hit_path, launch_observation_version='relative_v2',
                hit_observation_version='relative_v3', recovery_observation_version='legacy',
                skill_observation_version='aerowall_goal_v1',
            )
            route_td = TensorDict({
                'agents': {
                    'observation': torch.randn(5, 1, 46),
                    'legacy_observation': torch.randn(5, 1, 46),
                    'relative_v2_observation': torch.randn(5, 1, 46),
                    'relative_v3_observation': torch.randn(5, 1, 46),
                    'aerowall_goal_v1_observation': torch.randn(5, 1, 48),
                },
                'stats': {'caps': torch.tensor([[0], [0], [1], [1], [1]])},
                'info': {'skill_id': torch.tensor([[0.], [1.], [0.], [1.], [2.]])},
            }, batch_size=[5])
            routed = frozen_route(route_td)
            for row, name in enumerate(('launch', 'launch', 'recovery', 'hit', 'recovery')):
                expected_td = route_td.clone()
                version = frozen_route.observation_versions[name]
                expected_td[frozen_route.obs_name] = route_td[frozen_route.OBSERVATION_KEYS[version]]
                expected = frozen_route.frozen[name](expected_td, deterministic=True)
                self.assertTrue(torch.allclose(
                    routed[frozen_route.act_name][row], expected[frozen_route.act_name][row], atol=1e-6,
                ))
            self.assertEqual(frozen_route.update_actor(routed)['actor_grad_norm'], 0.0)

            all_hit = output.clone()
            all_hit['executed_skill_id'] = torch.ones(2, dtype=torch.long)
            all_hit['train_skill_mask'] = torch.ones(2, dtype=torch.bool)
            all_hit['advantages'] = torch.tensor([[[-1.]], [[1.]]])
            before_target = [value.detach().clone() for value in policy.actor_params.parameters()]
            policy.update_actor(all_hit)
            self.assertTrue(any(not torch.equal(before, after)
                                for before, after in zip(before_target, policy.actor_params.parameters())))
            for before, after in zip(before_launch, policy.frozen['launch'].actor_params.parameters()):
                self.assertTrue(torch.equal(before, after))
            for before, after in zip(before_recovery, policy.frozen['recovery'].actor_params.parameters()):
                self.assertTrue(torch.equal(before, after))


if __name__ == '__main__':
    unittest.main()
