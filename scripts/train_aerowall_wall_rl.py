"""Train and audit AeroWall's single-Iris wall task with upstream HCSP MAPPO/PPO."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HCSP = ROOT / "third_party/HCSP"
sys.path.insert(0, str(ROOT))


def state_snapshot(state, prefix=""):
    """Flatten a nested MAPPO state into cloned CPU tensors."""
    import torch
    if isinstance(state, torch.Tensor):
        return {prefix: state.detach().cpu().contiguous().clone()}
    if hasattr(state, "items"):
        flat = {}
        for name, value in sorted(state.items(), key=lambda item: str(item[0])):
            child = f"{prefix}/{name}" if prefix else str(name)
            flat.update(state_snapshot(value, child))
        return flat
    raise TypeError(f"Unsupported state value at {prefix}: {type(state)!r}")


def state_digest(snapshot):
    """Stable digest over parameter names, metadata, and raw tensor values."""
    digest = hashlib.sha256()
    for name, value in sorted(snapshot.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_equal(left, right):
    import torch
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def scalarize(values):
    import torch
    out = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1: out[key] = float(value)
        elif isinstance(value, (float, int)): out[key] = float(value)
    return out


def warmstart_actor(policy, checkpoint_path):
    """Copy an HCSP actor into the task actor, padding new observation inputs with zero."""
    import torch
    payload = torch.load(checkpoint_path, map_location=policy.device)
    source = dict(payload["actor_params"].items(True, True))
    target = dict(policy.actor_params.items(True, True))
    exact = []
    padded = []
    skipped = []
    with torch.no_grad():
        for key, target_value in target.items():
            source_value = source.get(key)
            if source_value is None:
                skipped.append(str(key))
            elif target_value.shape == source_value.shape:
                target_value.copy_(source_value)
                exact.append(str(key))
            elif (target_value.ndim == source_value.ndim
                  and target_value.shape[:-1] == source_value.shape[:-1]
                  and target_value.shape[-1] > source_value.shape[-1]):
                target_value.zero_()
                if tuple(key) == ('module', 'encoder', '0', 'weight'):
                    target_value.fill_(1.)
                target_value[..., :source_value.shape[-1]].copy_(source_value)
                padded.append({"key": str(key), "source": list(source_value.shape), "target": list(target_value.shape)})
            else:
                skipped.append(str(key))
    assert not skipped, skipped
    preservation_error = None
    if padded or exact:
        from hcsp.learning import MAPPOPolicy
        from tensordict import TensorDict
        from torchrl.data import UnboundedContinuousTensorSpec
        input_weight = source.get(('module', 'encoder', '1', 'layers', '0', 'weight'))
        if input_weight is None:
            raise KeyError('cannot infer checkpoint observation width from the first actor layer')
        source_dim = int(input_weight.shape[-1])
        target_dim = int(policy.agent_spec.observation_spec.shape[-1])
        from aerowall.wall_rl.policy_specs import agent_spec_for_observation_dim
        spec = agent_spec_for_observation_dim(policy.agent_spec, source_dim)
        reference = MAPPOPolicy(policy.cfg, spec, device=policy.device)
        reference.load_state_dict(payload)
        observation = torch.randn(64, 1, target_dim, device=policy.device)
        with torch.no_grad():
            left = reference(TensorDict({'agents': {'observation': observation[..., :source_dim]}}, [64], device=policy.device), deterministic=True)['agents', 'action']
            right = MAPPOPolicy.__call__(policy, TensorDict({'agents': {'observation': observation}}, [64], device=policy.device), deterministic=True)['agents', 'action']
            preservation_error = float((left-right).abs().max())
            assert torch.allclose(left, right, atol=1e-5, rtol=1e-5), preservation_error
    return {"checkpoint": str(checkpoint_path), "exact_tensors": len(exact),
            "padded_input_tensors": padded, "skipped_tensors": skipped,
            "action_preservation_max_error": preservation_error}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument('--candidate-name', type=str,
                   help='AeroWall-owned identity recorded in training reports and actor metadata')
    p.add_argument("--updates", type=int)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--stage", choices=["A0", "A1", "A2", "INTERCEPT", "WALL", "RETURN", "RALLY", "A", "B", "C"])
    p.add_argument("--resume", type=Path)
    p.add_argument("--actor-warmstart", type=Path)
    p.add_argument('--actor-lr', type=float, default=1e-4)
    p.add_argument('--actor-action-distribution', choices=['default', 'tanh'], default='default',
                   help='HCSP distribution selected for the trainable skill actor')
    p.add_argument('--prerequisite', type=Path)
    p.add_argument('--launch-checkpoint', type=Path)
    p.add_argument('--recovery-checkpoint', type=Path)
    p.add_argument('--hit-checkpoint', type=Path)
    p.add_argument('--launch-observation-version', choices=['legacy', 'relative_v2', 'relative_v3', 'aerowall_goal_v1'])
    p.add_argument('--recovery-observation-version', choices=['legacy', 'relative_v2', 'relative_v3', 'aerowall_goal_v1'])
    p.add_argument('--hit-observation-version', choices=['legacy', 'relative_v2', 'relative_v3', 'aerowall_goal_v1'])
    p.add_argument('--train-skill', choices=['intercept', 'hit', 'recover'])
    p.add_argument('--launch-distribution-config', type=Path,
                   help='AeroWall-owned initial-state distribution used only for Intercept training')
    p.add_argument('--observation-version', choices=['legacy', 'relative_v2', 'relative_v3', 'aerowall_goal_v1'])
    p.add_argument('--skill-observation-version', choices=['legacy', 'relative_v2', 'relative_v3', 'aerowall_goal_v1'],
                   help='Observation view used by the trainable skill actor in a chained run')
    p.add_argument('--reward-design', choices=['legacy', 'causal_v1', 'causal_v2', 'aerowall_causal_v3', 'aerowall_causal_v4', 'aerowall_causal_v5', 'aerowall_causal_v6'])
    p.add_argument('--case-mode', choices=['fixed', 'train', 'bank'])
    p.add_argument('--hit-artificial-ratio', type=float,
                   help='Fraction of Hit-training resets initialized near a feasible contact')
    p.add_argument('--hit-window', type=float,
                   help='Seconds before contact when the event FSM transfers control to Hit')
    p.add_argument('--hit-exit-window', type=float)
    p.add_argument('--min-dwell-steps', type=int)
    p.add_argument('--experiment-config', type=Path)
    p.add_argument('--upgrade-config', type=Path,
                   help='Load and hash-check configs/wall_skill_upgrade_v3.json; explicit CLI values override it')
    p.add_argument('--group', choices=['B', 'C'])
    p.add_argument('--training-state', type=Path, help='Full optimizer/RNG resume, unlike --resume')
    p.add_argument('--branch-from-training-state', type=Path,
                   help='Start C at update 51 from the completed B update-50 training state')
    p.add_argument('--save-every', type=int)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--max-gpu-seconds', type=float)
    a = p.parse_args()
    if a.candidate_name and not a.candidate_name.startswith('AeroWall'):
        p.error('--candidate-name must use an AeroWall-owned name')
    upgrade = None
    upgrade_sha = None
    if a.upgrade_config:
        from aerowall.wall_rl.upgrade_config import load_upgrade_config, configured_checkpoint
        if a.experiment_config:
            p.error('--upgrade-config and --experiment-config are separate protocols')
        upgrade = load_upgrade_config(a.upgrade_config, ROOT)
        upgrade_sha = hashlib.sha256(a.upgrade_config.read_bytes()).hexdigest()

        def configured(cli_value, value, fallback):
            return cli_value if cli_value is not None else value if value is not None else fallback

        training = upgrade['training']
        versions = upgrade['skills']
        a.num_envs = configured(a.num_envs, training.get('num_envs'), 64)
        a.seed = configured(a.seed, (training.get('pilot_seeds') or [0])[0], 0)
        a.stage = configured(a.stage, 'RALLY', 'A0')
        a.updates = configured(a.updates, None, max(1, training['max_frames'] // (a.num_envs * 64)))
        a.train_skill = configured(a.train_skill, training.get('train_skill'), None)
        a.observation_version = configured(a.observation_version, upgrade.get('observation_version'), 'legacy')
        a.reward_design = configured(a.reward_design, upgrade.get('reward_version'), 'legacy')
        a.case_mode = configured(a.case_mode, training.get('case_mode'), 'fixed')
        a.launch_observation_version = configured(a.launch_observation_version, versions['launch']['observation_version'], 'legacy')
        a.recovery_observation_version = configured(a.recovery_observation_version, versions['recover']['observation_version'], 'legacy')
        a.hit_observation_version = configured(a.hit_observation_version, versions['hit']['observation_version'], 'legacy')
        a.skill_observation_version = configured(a.skill_observation_version, training.get('skill_observation_version'), 'relative_v2')
        a.hit_artificial_ratio = configured(a.hit_artificial_ratio, upgrade['reset']['artificial_ratio'], 0.7)
        a.hit_window = configured(a.hit_window, upgrade['fsm']['hit_enter_seconds'], 0.18)
        a.hit_exit_window = configured(a.hit_exit_window, upgrade['fsm']['hit_exit_seconds'], a.hit_window)
        a.min_dwell_steps = configured(a.min_dwell_steps, upgrade['fsm']['min_dwell_steps'], 2)
        a.save_every = configured(a.save_every, training.get('save_every_updates'), None)
        a.max_frames = configured(a.max_frames, training.get('max_frames'), None)
        for role, arg_name in (('launch', 'launch_checkpoint'), ('c350_recovery', 'recovery_checkpoint')):
            expected = configured_checkpoint(upgrade, role, ROOT)
            supplied = getattr(a, arg_name)
            if supplied is not None and supplied.resolve() != expected:
                p.error(f'--{arg_name.replace("_", "-")} conflicts with the checkpoint pinned by --upgrade-config')
            setattr(a, arg_name, expected)
        warmstart_ref = training.get('hit_actor_warmstart_ref')
        if warmstart_ref:
            expected = configured_checkpoint(upgrade, warmstart_ref, ROOT)
            if a.actor_warmstart is not None and a.actor_warmstart.resolve() != expected:
                p.error('--actor-warmstart conflicts with the source pinned by --upgrade-config')
            a.actor_warmstart = expected
    else:
        a.num_envs = a.num_envs if a.num_envs is not None else 64
        a.seed = a.seed if a.seed is not None else 0
        a.stage = a.stage if a.stage is not None else 'A0'
        a.updates = a.updates if a.updates is not None else 5
        a.launch_observation_version = a.launch_observation_version or 'legacy'
        a.recovery_observation_version = a.recovery_observation_version or 'legacy'
        a.hit_observation_version = a.hit_observation_version or 'legacy'
        a.observation_version = a.observation_version or 'legacy'
        a.skill_observation_version = a.skill_observation_version or (
            'relative_v3' if a.train_skill == 'intercept' else 'relative_v2')
        a.reward_design = a.reward_design or 'legacy'
        a.case_mode = a.case_mode or 'fixed'
        a.hit_artificial_ratio = a.hit_artificial_ratio if a.hit_artificial_ratio is not None else 0.7
        a.hit_window = a.hit_window if a.hit_window is not None else 0.18
        a.hit_exit_window = a.hit_exit_window if a.hit_exit_window is not None else a.hit_window
        a.min_dwell_steps = a.min_dwell_steps if a.min_dwell_steps is not None else 2
    a.output = a.output.resolve(); a.output.parent.mkdir(parents=True, exist_ok=True)
    experiment = json.loads(a.experiment_config.read_text()) if a.experiment_config else None
    launch_distribution = None
    launch_distribution_sha = None
    if a.launch_distribution_config:
        if a.train_skill != 'intercept' or a.upgrade_config or experiment:
            p.error('--launch-distribution-config requires standalone --train-skill intercept training')
        launch_distribution_path = a.launch_distribution_config.resolve()
        try:
            launch_distribution = json.loads(launch_distribution_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            p.error(f'cannot read --launch-distribution-config: {exc}')
        if (launch_distribution.get('schema_version') != 1
                or launch_distribution.get('distribution') != 'AeroWallLateralInterceptV1'
                or launch_distribution.get('training_skill') != 'intercept'
                or launch_distribution.get('observation_version') != 'relative_v3'
                or launch_distribution.get('sampling') != 'independent_uniform_per_reset'):
            p.error('unsupported AeroWall launch distribution config')
        ranges = launch_distribution.get('ranges', {})
        try:
            launch_y_bounds = tuple(float(x) for x in ranges['ball_position_y_m'])
            launch_vy_bounds = tuple(float(x) for x in ranges['ball_velocity_y_mps'])
        except (KeyError, TypeError, ValueError):
            p.error('launch distribution config requires two-value lateral position and velocity ranges')
        if (len(launch_y_bounds) != 2 or len(launch_vy_bounds) != 2
                or launch_y_bounds[0] >= launch_y_bounds[1]
                or launch_vy_bounds[0] >= launch_vy_bounds[1]):
            p.error('launch distribution ranges must be increasing two-value intervals')
        launch_distribution_sha = hashlib.sha256(launch_distribution_path.read_bytes()).hexdigest()
    else:
        launch_distribution_path = None
        launch_y_bounds = (-0.4, 0.4)
        launch_vy_bounds = (-0.3, 0.3)
    if a.train_skill == 'intercept':
        if not a.launch_distribution_config:
            p.error('--train-skill intercept requires an explicit AeroWall launch distribution config')
        if not a.actor_warmstart:
            p.error('--train-skill intercept requires --actor-warmstart from a compatible 46-D actor')
        if not a.hit_checkpoint or not a.recovery_checkpoint:
            p.error('--train-skill intercept requires frozen --hit-checkpoint and --recovery-checkpoint')
        if (a.stage != 'RALLY' or a.case_mode != 'train'
                or a.observation_version != 'relative_v3'
                or a.skill_observation_version != 'relative_v3'):
            p.error('AeroWall Intercept training requires RALLY/train with relative_v3 actor and environment views')
        if a.launch_observation_version not in ('relative_v2', 'relative_v3'):
            p.error('the frozen launch source checkpoint requires its declared relative observation view')
    if a.actor_action_distribution != 'default' and a.train_skill not in ('intercept', 'hit'):
        p.error('--actor-action-distribution tanh currently requires --train-skill intercept or hit')
    if a.train_skill != 'intercept' and a.launch_distribution_config:
        p.error('--launch-distribution-config is only valid with --train-skill intercept')
    launch_candidate_name = None
    candidate_name = None
    if a.train_skill == 'intercept':
        reward_label = {
            'aerowall_causal_v3': 'CausalV3',
            'aerowall_causal_v4': 'CausalV4',
            'aerowall_causal_v5': 'CausalV5',
            'aerowall_causal_v6': 'CausalV6',
        }.get(a.reward_design, 'Legacy')
        lr_label = '' if abs(a.actor_lr - 1e-4) < 1e-12 else f'-ActorLr{a.actor_lr:g}'
        if a.actor_action_distribution == 'tanh':
            launch_candidate_name = f'AeroWallBoundedInterceptV1-{reward_label}{lr_label}'
        else:
            launch_candidate_name = f'AeroWallLateralInterceptV1-{reward_label}{lr_label}'
        candidate_name = launch_candidate_name
    elif a.train_skill == 'hit' and a.actor_action_distribution == 'tanh':
        candidate_name = f'AeroWallBoundedGoalHitV1-Tanh-S{a.seed}'
    if a.candidate_name:
        if not a.train_skill:
            p.error('--candidate-name requires --train-skill')
        candidate_name = a.candidate_name
        if a.train_skill == 'intercept':
            launch_candidate_name = a.candidate_name
    if a.max_frames is not None:
        frame_capacity = a.max_frames // (a.num_envs * 64)
        if frame_capacity < 1:
            p.error('--max-frames must cover at least one 64-step rollout')
        if a.updates > frame_capacity:
            p.error('--updates exceeds the configured --max-frames budget')
    if a.train_skill:
        assert a.stage == 'RALLY' and a.observation_version in ('relative_v2', 'relative_v3', 'aerowall_goal_v1')
        assert a.launch_checkpoint and not experiment
        assert 0.0 <= a.hit_artificial_ratio <= 1.0
        assert 0.05 <= a.hit_window <= 1.0
        if a.train_skill == 'hit':
            assert a.recovery_checkpoint and a.case_mode == 'train'
        elif a.train_skill == 'recover':
            assert a.hit_checkpoint
        else:
            assert a.hit_checkpoint and a.recovery_checkpoint and a.case_mode == 'train'
    elif a.hit_artificial_ratio != 0.7:
        p.error('--hit-artificial-ratio is only meaningful with --train-skill hit')
    if experiment:
        assert (a.observation_version, a.reward_design, a.case_mode) == ('legacy', 'legacy', 'fixed')
    experiment_sha = hashlib.sha256(a.experiment_config.read_bytes()).hexdigest() if a.experiment_config else None
    if experiment:
        assert a.group in ('B', 'C') and a.launch_checkpoint
        assert sum(x is not None for x in (a.resume, a.actor_warmstart, a.training_state,
                                           a.branch_from_training_state)) == 1
        assert a.stage == 'RALLY'
        assert a.num_envs == experiment['training']['num_envs']
        assert a.seed == experiment['training']['seed']
        assert not a.branch_from_training_state or a.group == 'C'
        assert a.updates <= experiment['training']['max_updates_per_group']
        assert a.save_every and a.save_every > 0
        assert a.max_gpu_seconds and a.max_gpu_seconds > 0
    if a.stage == 'WALL':
        assert a.prerequisite, 'WALL training requires an audited A2 gate report'
        prerequisite = json.loads(a.prerequisite.read_text())
        assert prerequisite.get('stage') == 'A2' and prerequisite.get('gate', {}).get('passed')
        audit = prerequisite.get('contact_audit', {})
        assert audit.get('corroborated_events', 0) == audit.get('legal_events') and not audit.get('callback_errors')
    if a.stage == 'RETURN':
        assert a.prerequisite and a.launch_checkpoint
        prerequisite = json.loads(a.prerequisite.read_text())
        assert prerequisite.get('stage') == 'WALL' and prerequisite.get('gate', {}).get('passed')
        assert hashlib.sha256(a.launch_checkpoint.read_bytes()).hexdigest() == prerequisite['checkpoint_sha256']
    if a.stage == 'RALLY' and not experiment and not a.train_skill:
        assert a.prerequisite and a.launch_checkpoint and a.resume
        prerequisite = json.loads(a.prerequisite.read_text())
        assert prerequisite.get('stage') in ('RETURN', 'RALLY') and prerequisite.get('gate', {}).get('passed')
        assert prerequisite.get('contact_audit', {}).get('corroborated_events', 0) > 0
    report = {"status": "initializing", "pid": os.getpid(), "seed": a.seed, "stage": a.stage,
              "group": a.group, "updates_requested": a.updates, "experiment_sha256": experiment_sha,
              "candidate_name": candidate_name,
              "upgrade_config": str(a.upgrade_config.resolve()) if a.upgrade_config else None,
              "upgrade_config_sha256": upgrade_sha,
              "launch_distribution_config": str(launch_distribution_path) if launch_distribution_path else None,
              "launch_distribution_config_sha256": launch_distribution_sha,
              "launch_distribution": launch_distribution.get('distribution', 'standard') if launch_distribution else 'standard',
              "actor_action_distribution": a.actor_action_distribution,
              "actor_learning_rate": a.actor_lr}
    def record(**values):
        report.update(values); tmp = a.output.with_suffix(".tmp"); tmp.write_text(json.dumps(report, indent=2) + "\n"); tmp.replace(a.output); print(json.dumps(values), flush=True)
    app = None; record()
    try:
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        sys.path.insert(0, str(HCSP))
        # The remote experiment checkout can contain pre-rename root-level
        # modules. Prefer this repository's AeroWall-owned script modules.
        sys.path.insert(0, str(ROOT / "scripts"))
        from hcsp import init_simulation_app
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        with initialize_config_dir(version_base=None, config_dir=str(HCSP / "cfg")):
            cfg = compose(config_name="train", overrides=["task=Serve", "headless=true", "wandb.mode=disabled", f"task.env.num_envs={a.num_envs}", f"seed={a.seed}"])
        OmegaConf.resolve(cfg); OmegaConf.set_struct(cfg, False)
        cfg.sim.dt = 0.0025; cfg.sim.substeps = 8
        cfg.env.max_episode_length = cfg.task.env.max_episode_length = (
            experiment['horizon_steps'] if experiment else
            upgrade['evaluation']['horizon_steps'] if upgrade else 600
        )
        cfg.task.drone_model = "IrisTest"; cfg.task.action_transform = None; cfg.task.wall_curriculum_stage = a.stage
        cfg.task.wall_observation_version = a.observation_version
        cfg.task.wall_reward_design = a.reward_design
        cfg.task.wall_case_mode = a.case_mode
        cfg.task.wall_train_skill = a.train_skill or ''
        cfg.task.wall_launch_distribution = launch_distribution.get('distribution', 'standard') if launch_distribution else 'standard'
        cfg.task.wall_launch_y_bounds_m = list(launch_y_bounds)
        cfg.task.wall_launch_vy_bounds_mps = list(launch_vy_bounds)
        cfg.task.wall_hit_artificial_ratio = a.hit_artificial_ratio if a.train_skill == 'hit' else 0.0
        cfg.task.wall_hit_window = a.hit_window
        cfg.task.wall_hit_enter_seconds = a.hit_window
        cfg.task.wall_hit_exit_seconds = a.hit_exit_window
        cfg.task.wall_min_dwell_steps = a.min_dwell_steps
        if upgrade:
            trajectory = upgrade['trajectory']
            cfg.task.wall_restitution = trajectory['restitution_x']
            cfg.task.wall_y_bounds = trajectory['wall_y_bounds_m']
            cfg.task.wall_z_bounds = trajectory['wall_z_bounds_m']
            cfg.task.wall_contact_height = trajectory['contact_height_m']
            cfg.task.wall_hit_fixed_incoming = bool(upgrade['training'].get('fixed_incoming', False))
            cfg.task.wall_goal_y_bounds = upgrade['training'].get('wall_goal_y_bounds_m', [-0.3, 0.3])
            cfg.task.wall_goal_z_bounds = upgrade['training'].get('wall_goal_z_bounds_m', [3.8, 4.2])
        cfg.task.wall_recenter_reward = bool(experiment and experiment['reward_variant'] == 'recenter')
        cfg.task.wall_reward_variant = experiment.get('wall_reward_variant', 'constant') if experiment else 'constant'
        cfg.task.wall_perturb_mode = ('curriculum' if a.group == 'C' else 'none') if experiment else 'none'
        cfg.task.wall_record_events = not bool(experiment or a.train_skill or a.stage == 'INTERCEPT')
        cfg.task.wall_perturb_light = float(experiment['perturbation']['light_delta_vy'] or 0.0) if experiment else 0.0
        cfg.task.wall_perturb_medium = float(experiment['perturbation']['medium_delta_vy'] or 0.0) if experiment else 0.0
        cfg.algo.actor.lr = experiment['training']['actor_lr'] if experiment else a.actor_lr
        cfg.algo.actor.create_dist_func = a.actor_action_distribution
        cfg.algo.critic.lr = experiment['training']['critic_lr'] if experiment else 5e-4
        cfg.algo.train_every = 64; cfg.algo.num_minibatches = 16; cfg.algo.ppo_epochs = 4
        OmegaConf.save(cfg, a.output.with_suffix(".yaml"))
        random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)
        sys.argv = [sys.argv[0], "--portable", "--portable-root", str(ROOT / ".cache/kit")]
        app = init_simulation_app(cfg)
        from hcsp_offline_assets import configure_local_asset_root
        configure_local_asset_root()
        from hcsp.learning import MAPPOPolicy
        from aerowall_policy_encoder import configure_policy_encoder
        configure_policy_encoder()
        from hcsp.utils.torchrl import SyncDataCollector
        from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
        from aerowall_wall_rally_env import AeroWallSingleWallRallyEnv
        base = AeroWallSingleWallRallyEnv(cfg, headless=True)
        env = TransformedEnv(base, Compose(InitTracker())).train(); env.set_seed(a.seed)
        if a.train_skill:
            from aerowall_skill_policies import AeroWallSkillChainPolicy
            policy = AeroWallSkillChainPolicy(
                cfg.algo, env.agent_spec['drone'], base.device,
                train_skill=a.train_skill, launch_checkpoint=a.launch_checkpoint,
                launch_observation_version=a.launch_observation_version,
                skill_observation_version=a.skill_observation_version,
                recovery_checkpoint=a.recovery_checkpoint,
                hit_checkpoint=a.hit_checkpoint,
                recovery_observation_version=a.recovery_observation_version,
                hit_observation_version=a.hit_observation_version,
            )
        elif a.launch_checkpoint:
            from aerowall_skill_policies import AeroWallLaunchRecoveryPolicy
            policy = AeroWallLaunchRecoveryPolicy(
                cfg.algo, env.agent_spec['drone'], base.device, a.launch_checkpoint,
                recovery_checkpoint=a.recovery_checkpoint,
            )
        else:
            policy = MAPPOPolicy(cfg.algo, agent_spec=env.agent_spec["drone"], device=base.device)
        warmstart = None
        if a.resume:
            policy.load_state_dict(torch.load(a.resume, map_location=base.device))
        elif a.actor_warmstart:
            warmstart = warmstart_actor(policy, a.actor_warmstart.resolve())
            if experiment:
                assert not warmstart['padded_input_tensors'] and not warmstart['skipped_tensors']
            if a.train_skill == 'intercept':
                assert not warmstart['padded_input_tensors'] and not warmstart['skipped_tensors']
        start_update = 0
        prior_gpu_seconds = 0.0
        launch_sha = hashlib.sha256(a.launch_checkpoint.read_bytes()).hexdigest() if a.launch_checkpoint else None
        resume_path = a.training_state or a.branch_from_training_state
        if resume_path:
            payload = torch.load(resume_path, map_location=base.device)
            assert payload['experiment_sha256'] == experiment_sha
            assert payload.get('upgrade_config_sha256') == upgrade_sha
            assert payload['group'] == ('B' if a.branch_from_training_state else a.group)
            if a.branch_from_training_state:
                assert int(payload['updates_completed']) == 50
            assert payload['launch_sha256'] == launch_sha and payload['seed'] == a.seed
            assert payload.get('actor_action_distribution', 'default') == a.actor_action_distribution
            policy.load_state_dict(payload['policy'])
            policy.actor_opt.load_state_dict(payload['actor_optimizer'])
            policy.critic_opt.load_state_dict(payload['critic_optimizer'])
            if 'critic_scheduler' in payload:
                policy.critic_opt_scheduler.load_state_dict(payload['critic_scheduler'])
            policy.n_updates = payload['policy_n_updates']
            random.setstate(payload['python_rng'])
            np.random.set_state(payload['numpy_rng'])
            torch.set_rng_state(payload['torch_rng'].cpu())
            if torch.cuda.is_available() and payload.get('cuda_rng') is not None:
                torch.cuda.set_rng_state_all([item.cpu() for item in payload['cuda_rng']])
            start_update = int(payload['updates_completed'])
            prior_gpu_seconds = float(payload['cumulative_gpu_seconds'])
        assert a.updates > start_update
        if experiment and a.group == 'C':
            base.set_curriculum_update(start_update + 1)
        initial_state = state_snapshot(policy.state_dict())
        launch_initial = state_snapshot(policy.launch.state_dict()) if hasattr(policy, 'launch') else None
        frozen_initial = ({name: state_snapshot(source.state_dict())
                           for name, source in policy.frozen.items()}
                          if hasattr(policy, 'frozen') else {})
        initial_digest = state_digest(initial_state)
        frames_per_batch = a.num_envs * int(cfg.algo.train_every)
        collector = SyncDataCollector(env, policy=policy, frames_per_batch=frames_per_batch,
                                      total_frames=(a.updates - start_update) * frames_per_batch, device=cfg.sim.device, return_same_td=True)
        history = []
        started = time.monotonic()
        def save_checkpoint(completed):
            checkpoint = a.output.with_suffix('.pt')
            temp_checkpoint = checkpoint.with_suffix('.pt.tmp')
            torch.save(policy.state_dict(), temp_checkpoint)
            temp_checkpoint.replace(checkpoint)
            state_path = a.output.with_suffix('.state.pt')
            temp_state = state_path.with_suffix('.state.pt.tmp')
            state = {
                'schema_version': 2,
                'policy': policy.state_dict(),
                'actor_optimizer': policy.actor_opt.state_dict(),
                'critic_optimizer': policy.critic_opt.state_dict(),
                'policy_n_updates': policy.n_updates,
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'updates_completed': completed,
                'frames_completed': completed * frames_per_batch,
                'cumulative_gpu_seconds': prior_gpu_seconds + time.monotonic() - started,
                'experiment_sha256': experiment_sha, 'upgrade_config_sha256': upgrade_sha,
                'group': a.group,
                'launch_sha256': launch_sha, 'seed': a.seed,
                'actor_action_distribution': a.actor_action_distribution,
            }
            if hasattr(policy, 'critic_opt_scheduler'):
                state['critic_scheduler'] = policy.critic_opt_scheduler.state_dict()
            torch.save(state, temp_state)
            temp_state.replace(state_path)
            source_hashes = {'launch': launch_sha}
            for name, source_path in (('recovery', a.recovery_checkpoint), ('hit', a.hit_checkpoint)):
                if source_path:
                    source_hashes[name] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            metadata = {
                'schema_version': 1,
                'framework': 'AeroWall',
                'policy_class': policy.__class__.__name__,
                'environment_class': 'AeroWallSingleWallRallyEnv',
                'experiment': upgrade['experiment'] if upgrade else None,
                'config_sha256': upgrade_sha or experiment_sha,
                'seed': a.seed,
                'updates_completed': completed,
                'frames_completed': completed * frames_per_batch,
                'observation_version': a.observation_version,
                'reward_version': a.reward_design,
                'candidate_name': candidate_name,
                'actor_action_distribution': a.actor_action_distribution,
                'actor_distribution_implementation': (
                    'HCSP.TanhNormalWithEntropy(tanh_loc=True)'
                    if a.actor_action_distribution == 'tanh'
                    else 'HCSP default (IndependentNormal)'
                ),
                'skill_observation_versions': {
                    'launch': a.launch_observation_version,
                    'intercept': a.skill_observation_version if a.train_skill == 'intercept' else None,
                    'hit': a.skill_observation_version if a.train_skill == 'hit' else a.hit_observation_version,
                    'recover': a.skill_observation_version if a.train_skill == 'recover' else a.recovery_observation_version,
                },
                'launch_distribution': cfg.task.wall_launch_distribution,
                'launch_distribution_config_sha256': launch_distribution_sha,
                'source_checkpoint_sha256': source_hashes,
                'fsm': {'hit_enter_seconds': a.hit_window,
                        'hit_exit_seconds': a.hit_exit_window,
                        'min_dwell_steps': a.min_dwell_steps},
            }
            metadata['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            checkpoint.with_suffix('.meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
            return checkpoint, state_path
        wall_reward_files = [ROOT / "aerowall/wall_rl/rewards.py",
                             ROOT / "aerowall/wall_rl/trajectory.py",
                             ROOT / "scripts/aerowall_wall_reward_logic.py"]
        reward_module_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in wall_reward_files}
        record(status="running", hcsp_commit=subprocess.check_output(["git", "-C", str(HCSP), "rev-parse", "HEAD"], text=True).strip(),
               initial_policy_sha256=initial_digest, frames_per_batch=frames_per_batch,
               framework='AeroWall', policy_class=policy.__class__.__name__,
               env_sha256=hashlib.sha256((ROOT / "scripts/aerowall_wall_rally_env.py").read_bytes()).hexdigest(),
               environment_class='AeroWallSingleWallRallyEnv',
               reward_logic_sha256=hashlib.sha256((ROOT / "scripts/aerowall_wall_reward_logic.py").read_bytes()).hexdigest(),
               wall_reward_module_sha256=reward_module_hashes,
               actor_warmstart=warmstart, launch_checkpoint=str(a.launch_checkpoint) if a.launch_checkpoint else None,
               train_skill=a.train_skill, observation_version=a.observation_version,
               skill_observation_version=a.skill_observation_version,
               launch_distribution=cfg.task.wall_launch_distribution,
               launch_distribution_config_sha256=launch_distribution_sha,
               launch_distribution_ranges={'ball_position_y_m': list(launch_y_bounds),
                                           'ball_velocity_y_mps': list(launch_vy_bounds)},
               reward_design=a.reward_design, actor_learning_rate=cfg.algo.actor.lr,
               actor_action_distribution=a.actor_action_distribution,
               actor_distribution_implementation=(
                   'HCSP.TanhNormalWithEntropy(tanh_loc=True)'
                   if a.actor_action_distribution == 'tanh'
                   else 'HCSP default (IndependentNormal)'
               ),
               frozen_actor_distribution_by_skill={
                   'launch': 'HCSP default (IndependentNormal)',
                   'hit': 'HCSP default (IndependentNormal)',
                   'recovery': 'HCSP default (IndependentNormal)',
               },
               case_mode=a.case_mode,
               hit_artificial_ratio=cfg.task.wall_hit_artificial_ratio,
               hit_window=cfg.task.wall_hit_window,
               source_checkpoint_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for name, path in [('recovery', a.recovery_checkpoint), ('hit', a.hit_checkpoint)]
                                         if path},
               launch_sha256=launch_sha, start_update=start_update,
               branch_source=str(a.branch_from_training_state.resolve()) if a.branch_from_training_state else None,
               branch_source_sha256=hashlib.sha256(a.branch_from_training_state.read_bytes()).hexdigest()
                   if a.branch_from_training_state else None)
        completed = start_update
        budget_stopped = False
        for update, batch in enumerate(collector):
            metrics = scalarize(policy.train_op(batch.to_tensordict()))
            completed = start_update + update + 1
            row = {"update": completed, "frames": completed * frames_per_batch, **metrics}
            if a.train_skill:
                row['train_skill_fraction'] = float(batch['train_skill_mask'].float().mean())
                if a.train_skill == 'hit':
                    row['hit_artificial_reset_fraction'] = (
                        base.hit_artificial_reset_count / max(base.hit_reset_count, 1))
            history.append(row); print(json.dumps(row), flush=True)
            if experiment and a.group == 'C':
                base.set_curriculum_update(completed + 1)
            if (experiment or upgrade) and (completed % a.save_every == 0 or completed == a.updates):
                checkpoint, state_path = save_checkpoint(completed)
                record(status='running', updates_completed=completed, checkpoint=str(checkpoint),
                       training_state=str(state_path), cumulative_gpu_seconds=prior_gpu_seconds + time.monotonic() - started)
            if experiment and prior_gpu_seconds + time.monotonic() - started >= a.max_gpu_seconds:
                budget_stopped = True
                break
            if (a.max_frames is not None and completed < a.updates
                    and completed * frames_per_batch >= a.max_frames):
                budget_stopped = True
                break
            if completed >= a.updates: break
        collector.shutdown()
        final_policy_state = policy.state_dict(); final_state = state_snapshot(final_policy_state)
        final_digest = state_digest(final_state)
        if experiment or upgrade:
            checkpoint, state_path = save_checkpoint(completed)
        else:
            checkpoint = a.output.with_suffix(".pt"); torch.save(final_policy_state, checkpoint)
        launch_actor_checkpoint = None
        launch_actor_export_exact = None
        if a.train_skill == 'intercept':
            source_state = torch.load(a.actor_warmstart, map_location='cpu')
            source_actor = source_state['actor_params']
            trained_actor = dict(policy.actor_params.items(True, True))
            source_actor_items = dict(source_actor.items(True, True))
            assert source_actor_items.keys() == trained_actor.keys()
            with torch.no_grad():
                for key, target_value in source_actor_items.items():
                    trained_value = trained_actor[key]
                    assert target_value.shape == trained_value.shape, (key, target_value.shape, trained_value.shape)
                    target_value.copy_(trained_value.detach().cpu())
            launch_actor_checkpoint = a.output.with_suffix('.launch.pt')
            temp_launch_actor = launch_actor_checkpoint.with_suffix('.tmp')
            torch.save(source_state, temp_launch_actor)
            temp_launch_actor.replace(launch_actor_checkpoint)
            reloaded_actor = torch.load(launch_actor_checkpoint, map_location='cpu')
            launch_actor_export_exact = state_equal(
                state_snapshot(policy.actor_params), state_snapshot(reloaded_actor['actor_params']))
            assert launch_actor_export_exact
            launch_actor_meta = {
                'framework': 'AeroWall',
                'policy_role': ('AeroWallBoundedInterceptV1'
                                if a.actor_action_distribution == 'tanh'
                                else 'AeroWallLateralInterceptV1'),
                'candidate_name': candidate_name,
                'observation_version': a.skill_observation_version,
                'reward_design': a.reward_design,
                'actor_learning_rate': cfg.algo.actor.lr,
                'actor_action_distribution': a.actor_action_distribution,
                'actor_distribution_implementation': (
                    'HCSP.TanhNormalWithEntropy(tanh_loc=True)'
                    if a.actor_action_distribution == 'tanh'
                    else 'HCSP default (IndependentNormal)'
                ),
                'launch_distribution': cfg.task.wall_launch_distribution,
                'launch_distribution_config_sha256': launch_distribution_sha,
                'seed': a.seed,
                'updates_completed': completed,
                'source_actor_warmstart': str(a.actor_warmstart.resolve()),
                'source_actor_warmstart_sha256': hashlib.sha256(a.actor_warmstart.read_bytes()).hexdigest(),
                'training_policy_checkpoint': str(checkpoint),
                'training_policy_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'actor_export_exact': launch_actor_export_exact,
            }
            a.output.with_suffix('.launch.meta.json').write_text(json.dumps(launch_actor_meta, indent=2) + '\n')
        if a.train_skill:
            from aerowall_skill_policies import AeroWallSkillChainPolicy
            reloaded = AeroWallSkillChainPolicy(
                cfg.algo, env.agent_spec['drone'], base.device,
                train_skill=a.train_skill, launch_checkpoint=a.launch_checkpoint,
                launch_observation_version=a.launch_observation_version,
                skill_observation_version=a.skill_observation_version,
                recovery_checkpoint=a.recovery_checkpoint,
                hit_checkpoint=a.hit_checkpoint,
                recovery_observation_version=a.recovery_observation_version,
                hit_observation_version=a.hit_observation_version,
            )
        elif a.launch_checkpoint:
            from aerowall_skill_policies import AeroWallLaunchRecoveryPolicy
            reloaded = AeroWallLaunchRecoveryPolicy(
                cfg.algo, env.agent_spec['drone'], base.device, a.launch_checkpoint,
                recovery_checkpoint=a.recovery_checkpoint,
            )
        else:
            reloaded = MAPPOPolicy(cfg.algo, agent_spec=env.agent_spec["drone"], device=base.device)
        reloaded.load_state_dict(torch.load(checkpoint, map_location=base.device))
        reload_state = state_snapshot(reloaded.state_dict()); reload_digest = state_digest(reload_state)
        weights_changed = not state_equal(initial_state, final_state)
        checkpoint_reload_exact = state_equal(final_state, reload_state)
        assert weights_changed and checkpoint_reload_exact and final_digest == reload_digest
        launch_frozen_exact = state_equal(launch_initial, state_snapshot(policy.launch.state_dict())) if launch_initial is not None else None
        assert launch_frozen_exact is not False
        frozen_sources_exact = {name: state_equal(before, state_snapshot(policy.frozen[name].state_dict()))
                                for name, before in frozen_initial.items()}
        assert all(frozen_sources_exact.values())
        record(status="budget_stopped" if budget_stopped else "passed", frames=completed * frames_per_batch,
               framework='AeroWall', policy_class=policy.__class__.__name__,
               updates_completed=completed, training_history=history,
               final_policy_sha256=final_digest, checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               weights_changed=weights_changed, checkpoint_reload_exact=checkpoint_reload_exact,
               launch_frozen_exact=launch_frozen_exact, frozen_sources_exact=frozen_sources_exact,
               hit_artificial_resets=base.hit_artificial_reset_count if a.train_skill == 'hit' else None,
               hit_total_resets=base.hit_reset_count if a.train_skill == 'hit' else None,
               hit_artificial_reset_fraction=(base.hit_artificial_reset_count / max(base.hit_reset_count, 1)
                                              if a.train_skill == 'hit' else None),
               launch_actor_checkpoint=str(launch_actor_checkpoint) if launch_actor_checkpoint else None,
               launch_actor_checkpoint_sha256=(hashlib.sha256(launch_actor_checkpoint.read_bytes()).hexdigest()
                                               if launch_actor_checkpoint else None),
               launch_actor_export_exact=launch_actor_export_exact,
               launch_candidate_name=launch_candidate_name,
               candidate_name=candidate_name,
               launch_actor_learning_rate=cfg.algo.actor.lr if launch_actor_checkpoint else None,
               actor_action_distribution=a.actor_action_distribution,
               actor_distribution_implementation=(
                   'HCSP.TanhNormalWithEntropy(tanh_loc=True)'
                   if a.actor_action_distribution == 'tanh'
                   else 'HCSP default (IndependentNormal)'
               ),
               launch_distribution=cfg.task.wall_launch_distribution,
               launch_distribution_config_sha256=launch_distribution_sha,
               launch_distribution_ranges={'ball_position_y_m': list(launch_y_bounds),
                                           'ball_velocity_y_mps': list(launch_vy_bounds)},
               training_state=str(state_path) if experiment or upgrade else None,
               cumulative_gpu_seconds=prior_gpu_seconds + time.monotonic() - started)
    except Exception as exc:
        record(status="failed", error=repr(exc), traceback=traceback.format_exc()); raise
    finally:
        if app is not None: app.close()


if __name__ == "__main__": main()
