"""AeroWall skill policies composed from the upstream HCSP MAPPO actor."""
import copy
import torch
from hcsp.learning import MAPPOPolicy
from aerowall.wall_rl.skill_fsm import Skill
from aerowall.wall_rl.policy_specs import agent_spec_for_observation_dim


class AeroWallLaunchRecoveryPolicy(MAPPOPolicy):
    def __init__(self, cfg, agent_spec, device='cuda', launch_checkpoint=None,
                 recovery_checkpoint=None):
        super().__init__(cfg, agent_spec, device)
        assert launch_checkpoint is not None
        if recovery_checkpoint is not None:
            self.load_state_dict(torch.load(recovery_checkpoint, map_location=device))
        self.launch = MAPPOPolicy(cfg, agent_spec, device)
        self.launch.load_state_dict(torch.load(launch_checkpoint, map_location=device))
        self.launch.actor_params.requires_grad_(False)
        self.train_in_keys.append('recovery_mask')

    def __call__(self, td, deterministic=False):
        # Start preparing immediately after launch, while the ball is airborne
        # toward the wall. Waiting for wall contact loses ~0.56s of travel time.
        recovery = td['stats', 'caps'].squeeze(-1) > 0
        launch_output = self.launch(td.clone(), deterministic=True)
        td = super().__call__(td, deterministic=deterministic)
        for key in (self.act_name, self.act_logps_name, f'{self.agent_spec.name}.action_entropy'):
            mask = recovery.reshape(*recovery.shape, *([1] * (td[key].ndim - recovery.ndim)))
            td[key] = torch.where(mask, td[key], launch_output[key])
        td['recovery_mask'] = recovery
        return td

    def update_actor(self, batch):
        mask = batch['recovery_mask'].reshape(batch.batch_size[0], -1).all(-1)
        if not bool(mask.any()):
            return {'policy_loss': 0., 'actor_grad_norm': 0., 'entropy': 0., 'ESS': 1.}
        # Prefix actions came from a different fixed policy. Exclude them from
        # both PPO likelihood loss and entropy loss; only recovery is on-policy.
        return super().update_actor(batch[mask])


class AeroWallSkillChainPolicy(MAPPOPolicy):
    """Train one PRT skill while frozen policies execute every other segment."""

    OBSERVATION_VERSIONS = (
        'legacy', 'relative_v2', 'relative_v3', 'aerowall_goal_v1',
        'aerowall_intercept_target_retention_v1',
    )
    OBSERVATION_KEYS = {
        'legacy': ('agents', 'legacy_observation'),
        'relative_v2': ('agents', 'relative_v2_observation'),
        'relative_v3': ('agents', 'relative_v3_observation'),
        'aerowall_goal_v1': ('agents', 'aerowall_goal_v1_observation'),
        'aerowall_intercept_target_retention_v1': (
            'agents', 'aerowall_intercept_target_retention_v1_observation',
        ),
    }

    def __init__(self, cfg, agent_spec, device='cuda', *, train_skill,
                 launch_checkpoint, recovery_checkpoint=None, hit_checkpoint=None,
                 launch_observation_version='legacy',
                 recovery_observation_version='legacy',
                 hit_observation_version='legacy',
                 skill_observation_version='relative_v3',
                 launch_action_distribution='default',
                 hit_action_distribution='default',
                 recovery_action_distribution='default'):
        super().__init__(cfg, agent_spec, device)
        if train_skill not in (None, 'intercept', 'hit', 'recover'):
            raise ValueError('train_skill must be intercept, hit, recover, or None for frozen evaluation')
        if train_skill is None and (hit_checkpoint is None or recovery_checkpoint is None):
            raise ValueError('frozen skill-chain evaluation requires Hit and Recover checkpoints')
        if train_skill == 'intercept' and (hit_checkpoint is None or recovery_checkpoint is None):
            raise ValueError('intercept training requires frozen Hit and Recover checkpoints')
        if train_skill == 'hit' and recovery_checkpoint is None:
            raise ValueError('hit training requires a frozen recovery checkpoint')
        if train_skill == 'recover' and hit_checkpoint is None:
            raise ValueError('recovery training requires a frozen hit checkpoint')
        versions = {
            'launch': launch_observation_version,
            'recovery': recovery_observation_version,
            'hit': hit_observation_version,
            'trainable': skill_observation_version,
        }
        for name, version in versions.items():
            if version not in self.OBSERVATION_VERSIONS:
                raise ValueError(f'{name} observation version must be one of {self.OBSERVATION_VERSIONS}')
        self.train_skill = train_skill
        if launch_action_distribution not in ('default', 'tanh'):
            raise ValueError('launch_action_distribution must be default or tanh')
        if hit_action_distribution not in ('default', 'tanh'):
            raise ValueError('hit_action_distribution must be default or tanh')
        if recovery_action_distribution not in ('default', 'tanh'):
            raise ValueError('recovery_action_distribution must be default or tanh')
        self.launch_action_distribution = launch_action_distribution
        self.hit_action_distribution = hit_action_distribution
        self.recovery_action_distribution = recovery_action_distribution
        self.observation_versions = versions
        # The evaluator may briefly enable this to capture the exact inputs
        # routed to each skill actor at selected diagnostic steps.
        self.record_reproducibility = False
        self.last_actor_observation_views = None
        self.actor_observation_key = (*self.agent_spec.observation_key[:1], 'actor_observation')
        self.frozen = {}
        observation_dimensions = {
            'legacy': 46, 'relative_v2': 46, 'relative_v3': 46,
            'aerowall_goal_v1': 48, 'aerowall_intercept_target_retention_v1': 46,
        }
        for name, checkpoint in [('launch', launch_checkpoint),
                                 ('recovery', recovery_checkpoint), ('hit', hit_checkpoint)]:
            if checkpoint is None:
                continue
            version = versions[name]
            source_spec = agent_spec_for_observation_dim(
                agent_spec, observation_dimensions[version],
            )
            target_distribution = (
                launch_action_distribution if name == 'launch' else
                hit_action_distribution if name == 'hit' else
                recovery_action_distribution if name == 'recovery' else 'default'
            )
            configured_distribution = cfg.actor.get('create_dist_func', 'default')
            policy_cfg = cfg
            if configured_distribution != target_distribution:
                # Keep frozen actors on their declared HCSP distribution even
                # when the trainable actor uses a different mapping. AeroWall
                # selects the existing HCSP TanhNormal only for the requested
                # role; it does not change the upstream distribution code.
                policy_cfg = copy.deepcopy(cfg)
                policy_cfg.actor.create_dist_func = target_distribution
            policy = MAPPOPolicy(policy_cfg, source_spec, device)
            policy.load_state_dict(torch.load(checkpoint, map_location=device))
            policy.actor_params.requires_grad_(False)
            self.frozen[name] = policy
        self.train_in_keys.append('train_skill_mask')
        self.train_in_keys.append('executed_skill_id')
        self.train_in_keys.append(self.actor_observation_key)

    def _observation_view(self, td, version):
        key = self.OBSERVATION_KEYS[version]
        if key in td.keys(include_nested=True, leaves_only=True):
            return td[key]
        if version == 'legacy' and self.obs_name in td.keys(include_nested=True, leaves_only=True):
            return td[self.obs_name]
        raise KeyError(f'environment did not provide the explicit {version} observation view')

    def _frozen_output(self, name, td):
        source = td.clone()
        version = self.observation_versions[name]
        source[self.obs_name] = self._observation_view(td, version)
        return self.frozen[name](source, deterministic=True)

    def _actor_eval_output(self, batch):
        """Recompute likelihood from the exact actor view saved at sampling."""
        actor_input = batch.select(*self.actor_in_keys, strict=False).clone()
        if self.actor_observation_key in batch.keys(include_nested=True, leaves_only=True):
            actor_input[self.obs_name] = batch[self.actor_observation_key]
        if 'is_init' in actor_input.keys():
            from tensordict.utils import expand_right
            actor_input['is_init'] = expand_right(
                actor_input['is_init'], (*actor_input.batch_size, self.agent_spec.n)
            )
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]
        if self.cfg.share_actor:
            return self.actor(actor_input, self.actor_params, eval_action=True)
        return torch.vmap(
            self.actor, in_dims=(1, 0), out_dims=1, randomness='different',
        )(actor_input, self.actor_params, eval_action=True)

    def recompute_action_log_prob(self, batch):
        """Expose the stored-action likelihood for rollout/update contract tests."""
        return self._actor_eval_output(batch)[self.act_logps_name]

    def __call__(self, td, deterministic=False):
        skill = td['info', 'skill_id'].squeeze(-1).long()
        caps = td['stats', 'caps'].squeeze(-1).long()
        if self.train_skill is None:
            train_mask = torch.zeros_like(skill, dtype=torch.bool)
            actor_observation = self._observation_view(
                td, self.observation_versions['trainable'],
            ).clone()
            # Frozen evaluation has no trainable actor. Seed the output with
            # the launch policy so a configured training-only observation
            # (for example 48-D aerowall_goal_v1) is never passed to the
            # 46-D base actor.
            launch = self._frozen_output('launch', td)
            actor_td = launch.clone()
        else:
            target = int({
                'intercept': Skill.INTERCEPT,
                'hit': Skill.HIT,
                'recover': Skill.RECOVER,
            }[self.train_skill])
            train_mask = skill == target
            actor_observation = self._observation_view(
                td, self.observation_versions['trainable'],
            ).clone()
            actor_td = td.clone()
            actor_td[self.obs_name] = actor_observation
            actor_td = super().__call__(actor_td, deterministic=deterministic)
            launch = None
        if self.record_reproducibility:
            views = {
                'trainable': actor_observation,
            }
            for name in ('launch', 'hit', 'recovery'):
                if name in self.frozen:
                    views[name] = self._observation_view(
                        td, self.observation_versions[name],
                    )
            self.last_actor_observation_views = {
                name: value.detach().clone() for name, value in views.items()
            }
        else:
            self.last_actor_observation_views = None
        public_observation = td[self.obs_name].clone()
        # Keep the critic on the public state view. Only the trainable actor is
        # routed through its declared skill-specific observation.
        actor_td.update(self.value_op(td.clone()))
        actor_td[self.obs_name] = public_observation
        actor_td[self.actor_observation_key] = actor_observation
        actor_td['executed_skill_id'] = skill
        if launch is None:
            launch = self._frozen_output('launch', td)
        recovery = self._frozen_output('recovery', td) if 'recovery' in self.frozen else None
        hit = self._frozen_output('hit', td) if 'hit' in self.frozen else None
        for key in (self.act_name, self.act_logps_name,
                    f'{self.agent_spec.name}.action_entropy'):
            value = actor_td[key]
            def use(mask, source):
                nonlocal value
                if source is None:
                    return
                expanded = mask.reshape(*mask.shape, *([1] * (value.ndim - mask.ndim)))
                value = torch.where(expanded, source[key], value)
            # Preserve the complete launch prefix through its first physical
            # cap. After that, route every non-Hit state to recovery so an
            # intermediate FSM state can never fall through to the random
            # trainable actor during frozen evaluation.
            use((caps == 0) & ~train_mask, launch)
            if recovery is not None:
                use((caps > 0) & (skill != int(Skill.HIT)) & ~train_mask, recovery)
            if hit is not None:
                use((caps > 0) & (skill == int(Skill.HIT)) & ~train_mask, hit)
            actor_td[key] = value
        actor_td['train_skill_mask'] = train_mask
        return actor_td

    def update_actor(self, batch):
        if self.train_skill is None:
            return {'policy_loss': 0., 'actor_grad_norm': 0., 'entropy': 0., 'ESS': 1.}
        target = int({
            'intercept': Skill.INTERCEPT,
            'hit': Skill.HIT,
            'recover': Skill.RECOVER,
        }[self.train_skill])
        executed = batch['executed_skill_id']
        mask = (executed == target).reshape(batch.batch_size[0], -1).all(-1)
        if not bool(mask.any()):
            return {'policy_loss': 0., 'actor_grad_norm': 0., 'entropy': 0., 'ESS': 1.}
        selected = batch[mask].clone()
        selected[self.obs_name] = selected[self.actor_observation_key]
        # Upstream PPO normalized over all chain segments before this mask.
        # Renormalize the valid on-policy segment to avoid source-only rollout
        # rewards setting the target actor's advantage scale.
        advantages = selected['advantages']
        selected['advantages'] = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        # HCSP's TanhNormalWithEntropy estimates transformed entropy by drawing
        # a sample in Actor.forward(). Its MAPPO update vmaps that call without
        # a randomness policy, which PyTorch rejects. Use the same HCSP PPO
        # objective with independent randomness across mapped agents.
        actor_output = self._actor_eval_output(selected)
        log_probs_new = actor_output[self.act_logps_name]
        log_probs_old = selected[self.act_logps_name]
        dist_entropy = actor_output[f'{self.agent_spec.name}.action_entropy']
        advantages = selected['advantages']
        assert advantages.shape == log_probs_new.shape == dist_entropy.shape

        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param,
        ) * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2) * self.act_dim)
        entropy_loss = -torch.mean(dist_entropy)

        self.actor_opt.zero_grad()
        (policy_loss + entropy_loss * self.cfg.entropy_coef).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor_opt.param_groups[0]['params'], self.cfg.max_grad_norm,
        )
        self.actor_opt.step()

        ess = (2 * ratio.logsumexp(0) - (2 * ratio).logsumexp(0)).exp().mean() / ratio.shape[0]
        return {
            'policy_loss': policy_loss.item(),
            'actor_grad_norm': grad_norm.item(),
            'entropy': -entropy_loss.item(),
            'ESS': ess.item(),
        }
