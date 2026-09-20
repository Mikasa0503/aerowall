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
        assert self.substeps == 1, 'Every physics step must be observed by the contact ledger'
        assert not self.hit_random_vel, 'No mid-flight state rewriting in this curriculum'
        assert self.bat_overlay['enabled']
        self.router = WallContactRouter(self, eager_gpu=False)
        self.completed_episodes = []
        self.contact_totals = {'legal_cap': 0, 'illegal': 0, 'resets': 0, 'gpu_queries': 0}

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self.router is not None:
            self.router.reset(env_ids.cpu().tolist())
            self.contact_totals['resets'] += len(env_ids)

    def _compute_reward_and_done(self):
        impacts, _ = self.router.read()
        illegal = torch.tensor([any(v.kind in ILLEGAL or v.kind == Kind.WALL for v in row)
                                for row in impacts], device=self.device).unsqueeze(-1)
        cap = torch.tensor([any(v.kind == Kind.CAP for v in row) for row in impacts],
                           device=self.device).unsqueeze(-1) & ~illegal
        # Dense interception shaping plus discrete legal cap reward. The apex
        # height factor prefers a repeatable juggling flight, without editing it.
        distance = (self.ball_pos[..., :2] - self.drone.pos[..., :2]).norm(dim=-1)
        height_error = (self.drone.pos[..., 2] - 1.).abs()
        apex = self.ball_pos[..., 2] + torch.clamp(self.ball_linear_vel[..., 2], min=0).square() / 19.62
        reward = self.dt * (torch.exp(-4 * distance) + torch.exp(-4 * height_error))
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
                'reason': 'illegal_contact' if bool(illegal[i]) else 'boundary' if bool(terminated[i]) else 'time_limit'})
        self.contact_totals['legal_cap'] += int(cap.sum())
        self.contact_totals['illegal'] += int(illegal.sum())
        self.contact_totals['gpu_queries'] += self.router.gpu_queries_this_step
        return TensorDict({'stats': self.stats.clone(), 'agents': {'reward': reward.unsqueeze(-1)}, 'done': done,
                           'terminated': terminated, 'truncated': truncated}, self.num_envs)
