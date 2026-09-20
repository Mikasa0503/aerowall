"""Short real PPO update/save/reload gate; no task performance claim."""
import math
from pathlib import Path
import torch


def check_ppo(env, base, cfg, checkpoint, record):
    from omni_drones.learning import MAPPOPolicy
    from omni_drones.utils.torchrl import SyncDataCollector
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        raise FileExistsError(f'Preserving checkpoint: {checkpoint}')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    policy = MAPPOPolicy(cfg.algo, agent_spec=env.agent_spec['drone'], device=base.device)
    before = {key: value.detach().clone() for key, value in policy.actor_params.items(True, True)}
    frames_per_batch = base.num_envs * int(cfg.algo.train_every)
    collector = SyncDataCollector(env, policy=policy, frames_per_batch=frames_per_batch,
                                  total_frames=4 * frames_per_batch, device=base.device,
                                  return_same_td=True)
    metrics = []
    for batch_index, data in enumerate(collector):
        leaking = [str(key) for key, value in data.items(True, True)
                   if isinstance(value, torch.Tensor) and value.requires_grad]
        assert not leaking, f'Collected environment data retains autograd state: {leaking}'
        result = policy.train_op(data.to_tensordict())
        numeric = {key: float(value) for key, value in result.items()}
        assert all(math.isfinite(value) for value in numeric.values()), 'Non-finite PPO metric'
        metrics.append(numeric)
        record(ppo_progress={'updates': batch_index + 1, 'frames': collector._frames,
                             'metrics': numeric})
        # Upstream closes the env when total_frames is reached. Stop one batch
        # earlier so checkpoint reload evaluation can use this same live scene.
        if batch_index == 2:
            break
    delta = max(float((value.detach() - before[key]).abs().max().item())
                for key, value in policy.actor_params.items(True, True))
    assert delta > 1e-10, 'PPO did not change actor parameters'
    payload = {'policy': policy.state_dict(), 'actor_optimizer': policy.actor_opt.state_dict(),
               'critic_optimizer': policy.critic_opt.state_dict(),
               'n_updates': policy.n_updates, 'environment_frames': collector._frames,
               'torch_rng_state': torch.get_rng_state(),
               'cuda_rng_state': torch.cuda.get_rng_state_all(),
               'scope': 'deployment gate only; not a trained juggling success result'}
    torch.save(payload, checkpoint)
    loaded = torch.load(checkpoint, map_location=base.device)
    restored = MAPPOPolicy(cfg.algo, agent_spec=env.agent_spec['drone'], device=base.device)
    restored.load_state_dict(loaded['policy'])
    policy.eval()
    restored.eval()
    td = env.reset()
    with torch.no_grad():
        expected = policy(td.clone(), deterministic=True)['agents', 'action']
        actual = restored(td.clone(), deterministic=True)['agents', 'action']
        reload_error = float((expected - actual).abs().max().item())
        assert reload_error < 1e-6, 'Checkpoint reload changed deterministic actions'
        terminated = 0
        reward_total = 0.0
        for _ in range(128):
            action_td = restored(td, deterministic=True)
            nxt = env.step(action_td)['next']
            reward = nxt['agents', 'reward']
            assert torch.isfinite(reward).all() and torch.isfinite(nxt['agents', 'observation']).all()
            reward_total += float(reward.sum().item())
            done = nxt['done']
            if done.any():
                terminated += int(done.sum().item())
                nxt.set('_reset', done)
                td = env.reset(nxt)
            else:
                td = nxt
    return {'passed': True, 'updates': len(metrics), 'environment_frames': collector._frames,
            'actor_parameter_max_change': delta, 'checkpoint': str(checkpoint),
            'reload_action_max_error': reload_error, 'evaluation_steps': 128,
            'evaluation_episode_ends': terminated, 'evaluation_reward_sum': reward_total,
            'metrics': metrics, 'performance_claim': False}
