"""Development curriculum: physical cap juggling with all body collisions enabled.

Keeps upstream observations and CTBR control. This is not WallRally. Rewards
use actual lifecycle-qualified contacts; no velocity proxy or hit cooldown.
"""
import torch
from tensordict import TensorDict
from aerowall.envs.wall_scene import WallContactScene
from aerowall.contact_router import WallContactRouter
from aerowall.rally_events import Kind, ILLEGAL


class AlignedJuggle(WallContactScene):
    def __init__(self, cfg, headless=True):
        self.router = None
        super().__init__(cfg, headless=headless)
        self.physics_dt = self.dt
        self.policy_dt = self.dt * self.substeps
        assert abs(self.policy_dt - .02) < 1e-9, 'Preserve 50 Hz policy timing'
        self.record_contact_kinematics = cfg.get('record_contact_kinematics', False)
        assert not self.hit_random_vel, 'No mid-flight state rewriting in this curriculum'
        assert self.bat_overlay['enabled']
        self.router = WallContactRouter(self, eager_gpu=False)
        self.completed_episodes = []
        self.contact_totals = {'legal_cap': 0, 'illegal': 0, 'resets': 0, 'gpu_queries': 0, 'physics_steps': 0, 'policy_steps': 0}

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self.router is not None:
            self.router.reset(env_ids.cpu().tolist())
            self.contact_totals['resets'] += len(env_ids)

    def _compute_state_and_obs(self):
        # Upstream finite-difference diagnostics observe policy-rate samples.
        physics_dt = self.dt
        self.dt = getattr(self, 'policy_dt', self.dt * self.substeps)
        try:
            return super()._compute_state_and_obs()
        finally:
            self.dt = physics_dt

    def _step(self, tensordict):
        self.last_impacts = [[] for _ in range(self.num_envs)]
        self.last_events = []
        self.action_cap_counts = torch.zeros(self.num_envs, 1, device=self.device)
        self.action_illegal = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self.action_gpu_queries = 0
        self.action_failure_reason = [None] * self.num_envs
        for substep in range(self.substeps):
            if substep == 0:
                self._pre_sim_step(tensordict)
            else:
                # Motor commands are held; refresh physical state for rotor and
                # aerodynamic dynamics without running the PID or policy again.
                self.drone.get_state()
                self.effort = self.drone.apply_action(tensordict['agents', 'action'])
            before = self.ball.get_velocities().clone() if self.record_contact_kinematics else None
            self.sim.step(self._should_render(substep))
            impacts, events = self.router.read()
            self.action_gpu_queries += self.router.gpu_queries_this_step
            bad = [any(x.kind in ILLEGAL or x.kind == Kind.WALL for x in row) for row in impacts]
            bad_t = torch.tensor(bad, device=self.device).unsqueeze(-1)
            cap = torch.tensor([any(x.kind == Kind.CAP for x in row) for row in impacts], device=self.device).unsqueeze(-1)
            legal = cap & ~bad_t & ~self.action_illegal
            self.action_cap_counts += legal.float()
            for i, failed in enumerate(bad):
                if failed and self.action_failure_reason[i] is None:
                    self.action_failure_reason[i] = sorted({x.kind.value for x in impacts[i] if x.kind in ILLEGAL or x.kind == Kind.WALL})[0]
            self.action_illegal |= bad_t
            for i, row in enumerate(impacts):
                self.last_impacts[i].extend(row)
            if self.record_contact_kinematics and events:
                from omni_drones.utils.torch import quat_rotate
                after = self.ball.get_velocities()
                bp,bq = self.router.bats.get_world_poses()
                bv = self.router.bats.get_velocities()
                order = self.router.bat_order
                bp,bq,bv = bp[order],bq[order],bv[order]
                com = bp + quat_rotate(bq,self.router.bats.get_coms()[0].reshape(self.num_envs,3)[order])
            for event in events:
                i = event['env_id']
                event.update(physics_step_illegal_priority=bad[i], physics_substep=substep, physics_time_offset=(substep+1)*self.physics_dt,
                             curriculum_cap_credit=bool(legal[i]) and event['kind']=='cap' and event['credited'])
                if self.record_contact_kinematics and event['points']:
                    point = torch.tensor(event['points'][0]['point'],device=self.device)
                    velocity = bv[i,:3] + torch.cross(bv[i,3:],point-com[i],dim=0)
                    normal = quat_rotate(bq[i:i+1],torch.tensor([[0.,0.,1.]],device=self.device))[0]
                    event.update(ball_velocity_before=before[i,0].cpu().tolist(),ball_velocity_after=after[i,0].cpu().tolist(),
                                 bat_normal=normal.cpu().tolist(),bat_contact_point_velocity=velocity.cpu().tolist())
            self.last_events.extend(events)
        self.contact_totals['physics_steps'] += self.substeps
        self.contact_totals['policy_steps'] += 1
        self._post_sim_step(tensordict)
        self.progress_buf += 1
        result = self._compute_state_and_obs()
        result.update(self._compute_reward_and_done())
        return result

    def _compute_reward_and_done(self):
        illegal, cap = self.action_illegal, self.action_cap_counts
        # Dense interception shaping plus discrete legal cap reward. The apex
        # height factor prefers a repeatable juggling flight, without editing it.
        distance = (self.ball_pos[..., :2] - self.drone.pos[..., :2]).norm(dim=-1)
        height_error = (self.drone.pos[..., 2] - 1.).abs()
        apex = self.ball_pos[..., 2] + torch.clamp(self.ball_linear_vel[..., 2], min=0).square() / 19.62
        reward = self.policy_dt * (torch.exp(-4 * distance) + torch.exp(-4 * height_error))
        reward += cap.float() * (5. + 5. * torch.exp(-4 * (apex - 1.7).square()))
        reward -= 10. * illegal.float()
        terminated = illegal | (self.drone.pos[..., 2] < .3) | (self.drone.pos[..., 2] > 3.5)
        terminated |= (self.ball_pos[..., 2] < .2) | (self.ball_pos[..., 2] > 4.)
        terminated |= (self.ball_pos[..., :2].abs() > 2.).any(-1)
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) & ~terminated
        done = terminated | truncated
        self.ball_last_2_vel = self.ball_last_vel.clone()
        self.ball_last_vel = self.ball_linear_vel.clone()
        self.hited_mark.zero_()
        self.stats['return'].add_(reward)
        self.stats['episode_len'][:] = self.progress_buf.unsqueeze(-1)
        self.stats['num_true_hits'].add_(cap.float())
        self.stats['wrong_hit'].add_(illegal.float())
        self.stats['ball_too_low'].add_((self.ball_pos[..., 2] < .2).float())
        self.stats['drone_too_low'].add_((self.drone.pos[..., 2] < .3).float())
        self.stats['done'].add_(done.float())
        self.stats['truncated'].add_(truncated.float())
        for i in done.squeeze(-1).nonzero().flatten().cpu().tolist():
            self.completed_episodes.append({'env_id': i, 'return': float(self.stats['return'][i]),
                'steps': int(self.progress_buf[i]), 'legal_caps': int(self.stats['num_true_hits'][i]),
                'contact_reason': self.action_failure_reason[i],
                'reason': 'illegal_contact' if bool(illegal[i]) else 'boundary' if bool(terminated[i]) else 'time_limit'})
        self.contact_totals['legal_cap'] += int(cap.sum())
        self.contact_totals['illegal'] += int(illegal.sum())
        self.contact_totals['reset_reentries'] = self.router.reset_reentries
        self.contact_totals['gpu_queries'] += self.action_gpu_queries
        return TensorDict({'stats': self.stats.clone(), 'agents': {'reward': reward.unsqueeze(-1)}, 'done': done,
                           'terminated': terminated, 'truncated': truncated}, self.num_envs)
