"""Evaluate one uninterrupted first episode per HCSP wall-task environment."""
import argparse
import hashlib
import inspect
import json
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HCSP = ROOT / "third_party/HCSP"
AEROWALL_ROTOR_RESPONSE_BANK_SHA256 = "cc9d93b5f3ecb2b367e1f3c209b080963ef535b38be63f2d004b38b4c35af0ec"
AEROWALL_ROTOR_RESPONSE_CASE_IDS = (
    "handoff-heldout-0008-env-0008", "handoff-heldout-0012-env-0012",
    "handoff-heldout-0018-env-0018", "handoff-heldout-0033-env-0033",
    "handoff-heldout-0051-env-0051", "handoff-heldout-0059-env-0059",
    "handoff-heldout-0073-env-0073", "handoff-heldout-0082-env-0082",
    "handoff-heldout-0099-env-0099", "handoff-heldout-0107-env-0107",
    "handoff-heldout-0112-env-0112", "handoff-heldout-0113-env-0113",
)
sys.path.insert(0, str(ROOT))


def audit_handoff_reset(base, cases, torch, atol=1e-4):
    """Compare every serialized handoff field with the live Isaac reset state."""
    root, ball_position, ball_velocity = base._kinematics()
    drone_velocity = base.drone.get_velocities(clone=True)[:, 0]
    _, ball_orientation = base.ball.get_world_poses(clone=True)
    actual = {
        'drone_position': root[:, 0, :3],
        'drone_orientation': root[:, 0, 3:7],
        'drone_velocity': drone_velocity,
        'motor_throttle': base.drone.throttle[:, 0],
        'prev_action': base.prev_action[:, 0],
        'action_before': base.action_before[:, 0],
        'ball_position': ball_position[:, 0],
        'ball_orientation': ball_orientation[:, 0],
        'ball_velocity': ball_velocity[:, 0, :3],
        'ball_angular_velocity': ball_velocity[:, 0, 3:6],
        'prev_ball_velocity': base.prev_ball_vel,
        'wall_target': base.wall_target,
        'phase': base.phase,
        'caps': base.caps,
        'walls': base.walls,
        'rallies': base.rallies,
        'streak': base.streak,
        'max_streak': base.max_streak,
        'skill_id': base.skill_id,
        'skill_held_steps': base.skill_held_steps,
    }
    errors = {}
    for name, value in actual.items():
        expected = torch.as_tensor([case[name] for case in cases],
                                   dtype=value.dtype, device=value.device)
        if expected.shape != value.shape:
            raise ValueError(
                f'handoff reset field {name} shape mismatch: '
                f'expected {tuple(expected.shape)}, got {tuple(value.shape)}'
            )
        if not value.is_floating_point():
            errors[name] = 0.0 if torch.equal(value, expected) else float('inf')
        else:
            errors[name] = float((value - expected).abs().max().item())
    failures = {name: error for name, error in errors.items() if error > atol}
    if failures:
        raise ValueError(f'handoff reset state mismatch (atol={atol}): {failures}')
    return {
        'passed': True,
        'case_count': len(cases),
        'field_count': len(actual),
        'absolute_tolerance': atol,
        'max_abs_error_by_field': errors,
    }


def _source_record(label, source):
    if not source:
        return {'path': None, 'sha256': None}
    path = Path(source).resolve()
    return {
        'path': str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
    }


def _module_source_record(label, module):
    try:
        source = inspect.getsourcefile(type(module))
    except (TypeError, OSError):
        source = None
    return _source_record(label, source)


def _reset_state_record(base, torch, np, random_module):
    """Capture post-reset physics/FSM state and random-generator fingerprints."""
    root, ball_position, ball_velocity = base._kinematics()
    fields = {
        'drone_root_state': root[:, 0],
        'drone_velocity': base.drone.get_velocities(clone=True)[:, 0],
        'motor_throttle': base.drone.throttle[:, 0],
        'ball_position': ball_position[:, 0],
        'ball_velocity': ball_velocity[:, 0],
        'phase': base.phase,
        'skill_id': base.skill_id,
        'caps': base.caps,
        'walls': base.walls,
        'rallies': base.rallies,
        'streak': base.streak,
        'max_streak': base.max_streak,
        'prev_action': base.prev_action[:, 0],
        'action_before': base.action_before[:, 0],
        'prev_ball_velocity': base.prev_ball_vel,
        'wall_target': base.wall_target,
    }
    state_hasher = hashlib.sha256()
    values = {}
    for name, value in fields.items():
        array = value.detach().cpu().contiguous().numpy()
        values[name] = array.tolist()
        state_hasher.update(name.encode('utf-8'))
        state_hasher.update(str(array.dtype).encode('ascii'))
        state_hasher.update(pickle.dumps(array.shape, protocol=4))
        state_hasher.update(array.tobytes())

    rng_hashes = {
        'torch_cpu': hashlib.sha256(torch.random.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        'numpy': hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
        'python': hashlib.sha256(pickle.dumps(random_module.getstate(), protocol=4)).hexdigest(),
    }
    if torch.cuda.is_available():
        rng_hashes['torch_cuda'] = [
            hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
            for state in torch.cuda.get_rng_state_all()
        ]
    return {
        'capture_point': 'after_env_reset_before_first_policy_action',
        'state_sha256': state_hasher.hexdigest(),
        'fields_by_env': values,
        'rng_state_sha256': rng_hashes,
    }


def _selected_actor_role(policy, skill_id, caps):
    if caps == 0:
        return 'launch'
    if skill_id == 1:
        return 'trainable' if getattr(policy, 'train_skill', None) == 'hit' else 'hit'
    if getattr(policy, 'train_skill', None) == 'recover' and skill_id == 2:
        return 'trainable'
    return 'recovery'


def _actor_input_record(policy, td, actions, *, first_env_only=False):
    views = getattr(policy, 'last_actor_observation_views', None)
    if not views:
        return None
    skills = td['info', 'skill_id'].squeeze(-1).long()
    caps = td['stats', 'caps'].squeeze(-1).long()
    limit = 1 if first_env_only else int(skills.shape[0])
    skill_values = skills[:limit].detach().cpu().tolist()
    cap_values = caps[:limit].detach().cpu().tolist()
    action_values = actions[:limit, 0].detach().cpu().tolist()
    view_values = {
        name: value[:limit, 0].detach().cpu().tolist()
        for name, value in views.items()
    }
    active_roles = [
        _selected_actor_role(policy, int(skill_values[index]), int(cap_values[index]))
        for index in range(limit)
    ]
    return {
        'env_indices': list(range(limit)),
        'skill_id': skill_values,
        'caps': cap_values,
        'active_actor_role': active_roles,
        'action': action_values,
        'actor_observation_views_by_role': view_values,
        'active_actor_observation': [
            view_values[role][index] if role in view_values else None
            for index, role in enumerate(active_roles)
        ],
    }


def _event_actor_observations(policy, td):
    """Return each environment's active actor observation for this action."""
    views = getattr(policy, 'last_actor_observation_views', None)
    if not views:
        return {}
    skills = td['info', 'skill_id'].squeeze(-1).long().detach().cpu().tolist()
    caps = td['stats', 'caps'].squeeze(-1).long().detach().cpu().tolist()
    result = {}
    for env_index, (skill_id, cap_count) in enumerate(zip(skills, caps)):
        role = _selected_actor_role(policy, int(skill_id), int(cap_count))
        observation = views.get(role)
        if observation is None:
            continue
        result[env_index] = {
            'actor_observation_role': role,
            'actor_observation_version': getattr(policy, 'observation_versions', {}).get(role),
            'actor_observation': observation[env_index, 0].detach().cpu().tolist(),
        }
    return result


def _reproducibility_sources(base, policy):
    records = {
        'evaluator': _source_record('evaluator', __file__),
        'environment': _module_source_record('environment', base),
        'reward_logic': _source_record('reward_logic', ROOT / 'scripts/aerowall_wall_reward_logic.py'),
        'observation_logic': _source_record('observation_logic', ROOT / 'aerowall/wall_rl/observations.py'),
        'skill_fsm': _source_record('skill_fsm', ROOT / 'aerowall/wall_rl/skill_fsm.py'),
        'policy_specs': _source_record('policy_specs', ROOT / 'aerowall/wall_rl/policy_specs.py'),
        'skill_policy': _source_record('skill_policy', ROOT / 'scripts/aerowall_skill_policies.py'),
        'policy': _module_source_record('policy', policy),
    }
    actor = getattr(policy, 'actor', None)
    if actor is not None:
        records['trainable_actor'] = _module_source_record('trainable_actor', actor)
    for role, frozen_policy in getattr(policy, 'frozen', {}).items():
        records[f'frozen_{role}_policy'] = _module_source_record(f'frozen_{role}_policy', frozen_policy)
        frozen_actor = getattr(frozen_policy, 'actor', None)
        if frozen_actor is not None:
            records[f'frozen_{role}_actor'] = _module_source_record(f'frozen_{role}_actor', frozen_actor)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage", choices=["A0", "A1", "A2", "INTERCEPT", "WALL", "RETURN", "RALLY", "A", "B", "C"])
    parser.add_argument("--actor-warmstart", type=Path)
    parser.add_argument('--launch-checkpoint', type=Path)
    parser.add_argument('--recovery-checkpoint', type=Path)
    parser.add_argument('--hit-checkpoint', type=Path)
    parser.add_argument('--launch-observation-version', choices=[
        'legacy', 'relative_v2', 'relative_v3', 'aerowall_goal_v1',
        'aerowall_intercept_target_retention_v1',
    ])
    parser.add_argument('--allow-observation-version-ablation', action='store_true',
                        help='Explicitly allow a role observation view that differs from its training report')
    parser.add_argument('--launch-action-distribution', choices=['default', 'tanh'], default='default',
                        help='Override only the frozen Launch actor distribution in a skill-chain run')
    parser.add_argument('--hit-action-distribution', choices=['default', 'tanh'], default='default',
                        help='Override only the frozen Hit actor distribution in a skill-chain run')
    parser.add_argument('--recovery-action-distribution', choices=['default', 'tanh'], default='default',
                        help='Override only the frozen Recover actor distribution in a skill-chain run')
    parser.add_argument('--recovery-observation-version', choices=['legacy', 'relative_v2', 'relative_v3', 'aerowall_goal_v1'])
    parser.add_argument('--hit-observation-version', choices=['legacy', 'relative_v2', 'relative_v3', 'aerowall_goal_v1'])
    parser.add_argument('--train-skill', choices=['hit', 'recover'])
    parser.add_argument('--skill-chain', action='store_true',
                        help='Evaluate a frozen three-skill route using explicit FSM and observation views')
    parser.add_argument('--observation-version', choices=['legacy', 'relative_v2', 'relative_v3', 'aerowall_goal_v1'])
    parser.add_argument('--skill-observation-version', choices=['legacy', 'relative_v2', 'relative_v3', 'aerowall_goal_v1'],
                        help='Observation view used by the trainable Hit/Recover actor in a chained run')
    parser.add_argument('--reward-design', choices=['legacy', 'causal_v1', 'causal_v2', 'aerowall_causal_v3', 'aerowall_causal_v4', 'aerowall_causal_v5', 'aerowall_causal_v6'])
    parser.add_argument('--hit-window', type=float)
    parser.add_argument('--hit-exit-window', type=float)
    parser.add_argument('--min-dwell-steps', type=int)
    parser.add_argument('--initial-case-bank', type=Path)
    parser.add_argument('--handoff-output', type=Path,
                        help='Write an immutable AeroWall pre-actor handoff bank from the source rollout')
    parser.add_argument('--handoff-role', choices=['launch', 'hit', 'recover'], default='hit',
                        help='Actor role whose state is captured; Launch captures phase-0 precontact states')
    parser.add_argument('--handoff-case-ids',
                        help='Comma-separated natural case IDs to capture in a Launch precontact bank')
    parser.add_argument('--handoff-time-target', type=float, default=0.4,
                        help='Target time-to-contact in seconds for Launch precontact capture')
    parser.add_argument('--handoff-reset-audit-only', action='store_true',
                        help='After Isaac reset, compare every live field with a validated handoff bank and exit')
    parser.add_argument('--record-trajectory', action='store_true')
    parser.add_argument('--aerowall-rotor-override-channel', type=int, choices=range(4),
                        help='AeroWall-only preregistered rotor-response screen: override one direct rotor-command channel')
    parser.add_argument('--aerowall-rotor-override-delta', type=float, choices=(-0.10, 0.10), default=0.10,
                        help='Preregistered signed actor-action offset; valid only with --aerowall-rotor-override-channel')
    parser.add_argument('--aerowall-rotor-override-steps', type=int, choices=(10,), default=10,
                        help='Frozen intervention horizon for AeroWallRotorCommandResponseCounterfactualV1')
    parser.add_argument('--record-reproducibility', action='store_true',
                        help='Record post-reset state/RNG fingerprints and exact skill actor inputs at diagnostic steps')
    parser.add_argument('--aerowall-postcap-ball-velocity-restore-bank', type=Path,
                        help='AeroWall-only no-training state-intervention targets for a registered post-cap ball-velocity screen')
    parser.add_argument('--candidate-name',
                        help='AeroWall candidate identity for reports; HCSP library/framework names remain unchanged')
    parser.add_argument('--candidate-training-report', type=Path,
                        help='Training report whose standalone actor checkpoint is being evaluated')
    parser.add_argument('--hit-candidate-name',
                        help='AeroWall identity for a non-default Hit actor checkpoint')
    parser.add_argument('--hit-training-report', type=Path,
                        help='Training report whose full policy checkpoint is used as the Hit actor')
    parser.add_argument('--action-mapping-ablation-name',
                        help='AeroWall identity for an inference-only action distribution ablation')
    parser.add_argument('--action-mapping-ablation-role', action='append',
                        choices=['launch', 'hit', 'recovery'],
                        help='Role changed by --action-mapping-ablation-name; repeat for multiple roles')
    parser.add_argument('--record-rgb', action='store_true', help='Save original-model camera frames, one environment only')
    parser.add_argument('--experiment-config', type=Path)
    parser.add_argument('--case-bank', type=Path)
    parser.add_argument('--protocol', choices=['natural', 'recovery'], default='natural')
    parser.add_argument('--zero-force-control', action='store_true',
                        help='Diagnostic: call the zero-force API without selecting any disturbance')
    parser.add_argument('--upgrade-config', type=Path,
                        help='Load and hash-check configs/wall_skill_upgrade_v3.json; explicit CLI values override it')
    args = parser.parse_args()
    if args.aerowall_rotor_override_channel is not None:
        if (not args.skill_chain or args.train_skill is not None or args.initial_case_bank is None
                or args.handoff_output is not None or args.handoff_reset_audit_only
                or args.protocol != 'natural'):
            parser.error('AeroWall rotor override requires frozen --skill-chain natural continuation from the registered initial handoff bank')
    if args.aerowall_postcap_ball_velocity_restore_bank is not None:
        if (not args.skill_chain or args.train_skill is not None or args.initial_case_bank is None
                or args.handoff_output is not None or args.handoff_reset_audit_only
                or args.protocol != 'natural' or args.stage != 'RALLY'
                or args.launch_action_distribution != 'tanh'
                or args.hit_action_distribution != 'default'
                or args.recovery_action_distribution != 'default'
                or args.aerowall_rotor_override_channel is not None
                or args.candidate_name or args.candidate_training_report
                or args.hit_candidate_name or args.hit_training_report
                or not args.record_trajectory or not args.record_reproducibility):
            parser.error('AeroWall post-cap velocity restore requires a no-training natural RALLY Launch-Tanh skill-chain audit with trajectory and reproducibility recording')
    candidate_training_report = None
    candidate_training_report_sha256 = None
    candidate_actor_checkpoint_sha256 = None
    if args.candidate_training_report:
        args.candidate_training_report = args.candidate_training_report.resolve()
        if not args.candidate_training_report.is_file():
            parser.error('--candidate-training-report does not exist')
        candidate_training_report = json.loads(args.candidate_training_report.read_text())
        report_candidate_name = candidate_training_report.get('launch_candidate_name')
        if not report_candidate_name:
            parser.error('--candidate-training-report has no launch_candidate_name')
        if args.candidate_name and args.candidate_name != report_candidate_name:
            parser.error('--candidate-name does not match launch_candidate_name in training report')
        args.candidate_name = report_candidate_name
        if not args.launch_checkpoint:
            parser.error('--candidate-training-report requires --launch-checkpoint')
        launch_checkpoint_path = args.launch_checkpoint.resolve()
        if not launch_checkpoint_path.is_file():
            parser.error('--launch-checkpoint does not exist')
        candidate_actor_checkpoint_sha256 = hashlib.sha256(launch_checkpoint_path.read_bytes()).hexdigest()
        expected_actor_sha256 = candidate_training_report.get('launch_actor_checkpoint_sha256')
        if expected_actor_sha256 != candidate_actor_checkpoint_sha256:
            parser.error('launch actor checkpoint SHA256 does not match candidate training report')
        candidate_training_report_sha256 = hashlib.sha256(
            args.candidate_training_report.read_bytes()
        ).hexdigest()
    elif args.candidate_name and args.launch_checkpoint:
        launch_checkpoint_path = args.launch_checkpoint.resolve()
        if not launch_checkpoint_path.is_file():
            parser.error('--launch-checkpoint does not exist')
        candidate_actor_checkpoint_sha256 = hashlib.sha256(launch_checkpoint_path.read_bytes()).hexdigest()
    hit_training_report = None
    hit_training_report_sha256 = None
    hit_checkpoint_sha256 = None
    if args.hit_candidate_name and not args.hit_checkpoint:
        parser.error('--hit-candidate-name requires --hit-checkpoint')
    if args.hit_training_report:
        args.hit_training_report = args.hit_training_report.resolve()
        if not args.hit_training_report.is_file() or not args.hit_checkpoint:
            parser.error('--hit-training-report requires an existing report and --hit-checkpoint')
        hit_training_report = json.loads(args.hit_training_report.read_text())
        if hit_training_report.get('train_skill') != 'hit':
            parser.error('--hit-training-report must identify a train_skill=hit run')
        hit_checkpoint_path = args.hit_checkpoint.resolve()
        if not hit_checkpoint_path.is_file():
            parser.error('--hit-checkpoint does not exist')
        hit_checkpoint_sha256 = hashlib.sha256(hit_checkpoint_path.read_bytes()).hexdigest()
        if hit_training_report.get('checkpoint_sha256') != hit_checkpoint_sha256:
            parser.error('Hit checkpoint SHA256 does not match --hit-training-report')
        hit_training_report_sha256 = hashlib.sha256(args.hit_training_report.read_bytes()).hexdigest()
    elif args.hit_checkpoint:
        hit_checkpoint_path = args.hit_checkpoint.resolve()
        if hit_checkpoint_path.is_file():
            hit_checkpoint_sha256 = hashlib.sha256(hit_checkpoint_path.read_bytes()).hexdigest()
    training_observation_versions_by_role = {
        'launch': ((candidate_training_report or {}).get('skill_observation_version')
                   or (candidate_training_report or {}).get('observation_version')),
        'hit': ((hit_training_report or {}).get('skill_observation_version')
                or (hit_training_report or {}).get('observation_version')),
        'recovery': None,
    }
    for role, argument_name in (
        ('launch', 'launch_observation_version'),
        ('hit', 'hit_observation_version'),
    ):
        trained_version = training_observation_versions_by_role[role]
        if not trained_version:
            continue
        evaluation_version = getattr(args, argument_name)
        if evaluation_version is None:
            setattr(args, argument_name, trained_version)
        elif (evaluation_version != trained_version
              and not args.allow_observation_version_ablation):
            parser.error(
                f'--{argument_name.replace("_", "-")}={evaluation_version} differs from '
                f'{role} training observation {trained_version}; pass '
                '--allow-observation-version-ablation only for an intentional observation ablation'
            )
    if args.action_mapping_ablation_name:
        if not args.skill_chain or args.train_skill:
            parser.error('--action-mapping-ablation-name requires a frozen --skill-chain evaluation')
        if not args.action_mapping_ablation_role:
            parser.error('--action-mapping-ablation-name requires at least one --action-mapping-ablation-role')
        action_distributions = {
            'launch': args.launch_action_distribution,
            'hit': args.hit_action_distribution,
            'recovery': args.recovery_action_distribution,
        }
        invalid_roles = [role for role in args.action_mapping_ablation_role
                         if action_distributions[role] != 'tanh']
        if invalid_roles:
            parser.error('every --action-mapping-ablation-role must use --<role>-action-distribution tanh')
    upgrade = None
    upgrade_sha = None
    if args.upgrade_config:
        from aerowall.wall_rl.upgrade_config import load_upgrade_config, configured_checkpoint, configured_bank
        upgrade = load_upgrade_config(args.upgrade_config, ROOT)
        upgrade_sha = hashlib.sha256(args.upgrade_config.read_bytes()).hexdigest()

        def configured(cli_value, value, fallback):
            return cli_value if cli_value is not None else value if value is not None else fallback

        args.num_envs = configured(args.num_envs, upgrade['evaluation'].get('num_envs'), 128)
        args.seed = configured(args.seed, upgrade['evaluation'].get('seed'), 123)
        args.stage = configured(args.stage, 'RALLY', 'A0')
        args.observation_version = configured(args.observation_version, upgrade['baseline']['observation_version'], 'legacy')
        args.reward_design = configured(args.reward_design, upgrade.get('reward_version'), 'legacy')
        args.hit_window = configured(args.hit_window, upgrade['fsm']['hit_enter_seconds'], 0.18)
        args.hit_exit_window = configured(args.hit_exit_window, upgrade['fsm']['hit_exit_seconds'], args.hit_window)
        args.min_dwell_steps = configured(args.min_dwell_steps, upgrade['fsm']['min_dwell_steps'], 2)
        args.launch_observation_version = configured(args.launch_observation_version,
                                                     upgrade['skills']['launch']['observation_version'], 'legacy')
        args.recovery_observation_version = configured(args.recovery_observation_version,
                                                       upgrade['skills']['recover']['observation_version'], 'legacy')
        args.hit_observation_version = configured(args.hit_observation_version,
                                                  upgrade['skills']['hit']['observation_version'], 'legacy')
        args.skill_observation_version = configured(args.skill_observation_version,
                                                    upgrade['training']['skill_observation_version'], 'relative_v2')
        if args.launch_checkpoint is None:
            args.launch_checkpoint = configured_checkpoint(upgrade, 'launch', ROOT)
        elif args.launch_checkpoint.resolve() != configured_checkpoint(upgrade, 'launch', ROOT):
            parser.error('--launch-checkpoint conflicts with the checkpoint pinned by --upgrade-config')
        if args.recovery_checkpoint is None:
            args.recovery_checkpoint = configured_checkpoint(upgrade, 'c350_recovery', ROOT)
        elif args.recovery_checkpoint.resolve() != configured_checkpoint(upgrade, 'c350_recovery', ROOT):
            parser.error('--recovery-checkpoint conflicts with the checkpoint pinned by --upgrade-config')
        if args.initial_case_bank is None:
            args.initial_case_bank = configured_bank(upgrade, 'fixed_bank', ROOT)
        if args.num_envs != len(json.loads(args.initial_case_bank.read_text())['cases']):
            parser.error('--num-envs must match the configured initial-case bank size')
    else:
        args.num_envs = args.num_envs if args.num_envs is not None else 128
        args.seed = args.seed if args.seed is not None else 123
        args.stage = args.stage or 'A0'
        args.observation_version = args.observation_version or 'legacy'
        args.reward_design = args.reward_design or 'legacy'
        args.hit_window = args.hit_window if args.hit_window is not None else 0.18
        args.hit_exit_window = args.hit_exit_window if args.hit_exit_window is not None else args.hit_window
        args.min_dwell_steps = args.min_dwell_steps if args.min_dwell_steps is not None else 2
        args.launch_observation_version = args.launch_observation_version or 'legacy'
        args.recovery_observation_version = args.recovery_observation_version or 'legacy'
        args.hit_observation_version = args.hit_observation_version or 'legacy'
        args.skill_observation_version = args.skill_observation_version or 'relative_v2'
    experiment = json.loads(args.experiment_config.read_text()) if args.experiment_config else None
    cases = json.loads(args.case_bank.read_text())['cases'] if args.case_bank else None
    initial_bank_payload = json.loads(args.initial_case_bank.read_text()) if args.initial_case_bank else None
    initial_cases = initial_bank_payload['cases'] if initial_bank_payload else None
    ball_velocity_restore_payload = None
    ball_velocity_restore_by_env = {}
    if args.aerowall_postcap_ball_velocity_restore_bank is not None:
        args.aerowall_postcap_ball_velocity_restore_bank = args.aerowall_postcap_ball_velocity_restore_bank.resolve()
        if not args.aerowall_postcap_ball_velocity_restore_bank.is_file():
            parser.error('AeroWall post-cap velocity restore bank does not exist')
        ball_velocity_restore_payload = json.loads(args.aerowall_postcap_ball_velocity_restore_bank.read_text())
        restore_bank_sha = hashlib.sha256(args.aerowall_postcap_ball_velocity_restore_bank.read_bytes()).hexdigest()
        source_bank_sha = hashlib.sha256(args.initial_case_bank.read_bytes()).hexdigest()
        if (ball_velocity_restore_payload.get('experiment_name') != 'AeroWallLaunchTanhPostContactBallVelocityRestoreV1'
                or ball_velocity_restore_payload.get('source_initial_case_bank_sha256') != source_bank_sha
                or source_bank_sha != 'b6ebf13209ba42b1ad77eb452248b6f743c0f0fbcc32805d250fbc162b3804da'
                or ball_velocity_restore_payload.get('seed') != args.seed
                or ball_velocity_restore_payload.get('num_envs') != args.num_envs
                or args.num_envs != 128
                or not initial_cases or len(initial_cases) != args.num_envs):
            parser.error('AeroWall post-cap restore bank source seed, case bank, or environment count does not match')
        case_ids_by_env = {index: case.get('case_id') for index, case in enumerate(initial_cases)}
        expected_ids = tuple(ball_velocity_restore_payload.get('selected_cases', ()))
        restore_cases = ball_velocity_restore_payload.get('cases', [])
        if len(restore_cases) != 13 or tuple(case.get('case_id') for case in restore_cases) != expected_ids:
            parser.error('AeroWall post-cap restore bank has invalid or reordered selected cases')
        for entry in restore_cases:
            env_index = int(entry.get('env', -1))
            if (env_index < 0 or env_index >= len(initial_cases)
                    or case_ids_by_env.get(env_index) != entry.get('case_id')
                    or env_index in ball_velocity_restore_by_env):
                parser.error('AeroWall post-cap restore bank case-to-environment mapping is invalid')
            target_velocity = entry.get('control_post_step_ball_linear_velocity_mps')
            if not isinstance(target_velocity, list) or len(target_velocity) != 3:
                parser.error('AeroWall post-cap restore target must contain exactly three linear velocity values')
            ball_velocity_restore_by_env[env_index] = entry
        if set(ball_velocity_restore_by_env) != {int(item['env']) for item in restore_cases}:
            parser.error('AeroWall post-cap restore bank has duplicate or missing environments')
        ball_velocity_restore_payload['_runtime_bank_sha256'] = restore_bank_sha
    if ball_velocity_restore_payload is not None:
        checkpoint_paths = {
            'launch': args.launch_checkpoint,
            'hit': args.hit_checkpoint,
            'recovery': args.recovery_checkpoint,
        }
        if any(path is None or not path.resolve().is_file() for path in checkpoint_paths.values()):
            parser.error('AeroWall post-cap restore requires all three frozen skill checkpoints')
        checkpoint_hashes = {
            role: hashlib.sha256(path.resolve().read_bytes()).hexdigest()
            for role, path in checkpoint_paths.items()
        }
        if checkpoint_hashes != ball_velocity_restore_payload.get('source_checkpoint_sha256'):
            parser.error('AeroWall post-cap restore checkpoints differ from its paired control targets')
        expected_settings = (
            args.launch_observation_version == 'relative_v3'
            and args.hit_observation_version == 'relative_v3'
            and args.recovery_observation_version == 'relative_v3'
            and args.reward_design == 'legacy'
            and args.hit_window == 0.18
            and args.hit_exit_window == 0.18
            and args.min_dwell_steps == 2
        )
        if not expected_settings:
            parser.error('AeroWall post-cap restore requires the frozen relative_v3/legacy/window/dwell settings')
    if args.aerowall_rotor_override_channel is not None:
        bank_sha = hashlib.sha256(args.initial_case_bank.read_bytes()).hexdigest()
        case_ids = tuple(case.get('case_id') for case in (initial_cases or []))
        if bank_sha != AEROWALL_ROTOR_RESPONSE_BANK_SHA256:
            parser.error('AeroWall rotor override requires the frozen registered 12-case bank SHA256')
        if (len(case_ids) != 12 or case_ids != AEROWALL_ROTOR_RESPONSE_CASE_IDS
                or initial_bank_payload.get('experiment_name') != 'AeroWallLaunchLateralMomentumCounterfactualV1'
                or initial_bank_payload.get('condition') != 'control'):
            parser.error('AeroWall rotor override requires the exact registered Launch control handoff cases')
    handoff_evaluation = bool(initial_cases) and all(
        'handoff_state_version' in case for case in initial_cases
    )
    if initial_cases and any('handoff_state_version' in case for case in initial_cases) and not handoff_evaluation:
        parser.error('initial case bank cannot mix handoff states and natural initial states')
    if args.aerowall_rotor_override_channel is not None:
        if not handoff_evaluation or len(initial_cases) != 12 or args.num_envs != 12:
            parser.error('AeroWall rotor override requires exactly 12 validated Launch handoff environments')
    handoff_role = None
    if handoff_evaluation:
        from aerowall.wall_rl.curriculum import handoff_actor_role, validate_handoff_case
        try:
            handoff_roles = {handoff_actor_role(case) for case in initial_cases}
            if len(handoff_roles) != 1:
                raise ValueError('initial handoff bank cannot mix actor roles')
            handoff_role = handoff_roles.pop()
            for case in initial_cases:
                validate_handoff_case(case)
        except (TypeError, ValueError) as exc:
            parser.error(f'invalid handoff bank: {exc}')
        expected_bank_kind = f'aerowall_{handoff_role}_handoff_v1'
        if initial_bank_payload.get('bank_kind') != expected_bank_kind:
            parser.error(f'{handoff_role.title()} handoff bank must declare bank_kind={expected_bank_kind}')
        if initial_bank_payload.get('handoff_actor_role', handoff_role) != handoff_role:
            parser.error('handoff bank actor role does not match its cases')
        if handoff_role == 'recover' and initial_bank_payload.get('handoff_actor_role') != 'recover':
            parser.error('Recover handoff bank must explicitly declare handoff_actor_role=recover')
        source_audit = initial_bank_payload.get('source_contact_audit', {})
        if (source_audit.get('callback_errors') or
                source_audit.get('legal_cap_events', 0) != source_audit.get('corroborated_cap_events', -1) or
                source_audit.get('wall_events', 0) != source_audit.get('corroborated_wall_events', -1)):
            parser.error(f'{handoff_role.title()} handoff bank source contact audit is incomplete')
    if args.handoff_reset_audit_only and not handoff_evaluation:
        parser.error('--handoff-reset-audit-only requires a validated handoff bank')
    if args.handoff_case_ids and args.handoff_role != 'launch':
        parser.error('--handoff-case-ids is only supported for Launch precontact capture')
    if args.handoff_role == 'launch' and args.handoff_output and not initial_cases:
        parser.error('Launch precontact capture requires --initial-case-bank with natural case IDs')
    if args.handoff_role == 'launch' and args.handoff_output and not (0.38 <= args.handoff_time_target <= 0.42):
        parser.error('Launch precontact capture target must be within 0.38–0.42 seconds')
    if args.handoff_role == 'recover' and not args.handoff_output:
        parser.error('--handoff-role recover requires --handoff-output')
    if args.handoff_output and (args.stage != 'RALLY' or args.protocol != 'natural'):
        parser.error('--handoff-output requires the uninterrupted natural RALLY protocol')
    if initial_cases and len(initial_cases) != args.num_envs:
        parser.error('--initial-case-bank case count must equal --num-envs')
    if args.train_skill and (args.observation_version not in ('relative_v2', 'relative_v3', 'aerowall_goal_v1') or not args.launch_checkpoint):
        parser.error('--train-skill requires a relative observation version and --launch-checkpoint')
    if args.skill_chain and (not args.launch_checkpoint or not args.hit_checkpoint or not args.recovery_checkpoint):
        parser.error('--skill-chain requires launch, hit, and recovery checkpoints')
    if args.launch_action_distribution != 'default' and not (args.skill_chain or args.train_skill):
        parser.error('--launch-action-distribution requires --skill-chain or --train-skill')
    if args.hit_action_distribution != 'default' and not (args.skill_chain or args.train_skill):
        parser.error('--hit-action-distribution requires --skill-chain or --train-skill')
    if args.recovery_action_distribution != 'default' and not (args.skill_chain or args.train_skill):
        parser.error('--recovery-action-distribution requires --skill-chain or --train-skill')
    if args.protocol == 'recovery':
        parser.error('--protocol recovery requires --experiment-config and --case-bank') if not (experiment and cases) else None
        assert len(cases) == args.num_envs
    elif cases:
        parser.error('--case-bank is valid only with --protocol recovery')
    if args.zero_force_control and args.protocol != 'natural':
        parser.error('--zero-force-control requires natural protocol')
    if args.record_rgb and args.num_envs != 1:
        parser.error('--record-rgb requires --num-envs 1')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.handoff_output:
        args.handoff_output = args.handoff_output.resolve()
        if args.handoff_output == args.output:
            parser.error('--handoff-output must differ from --output')
        args.handoff_output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "initializing", "pid": os.getpid(), "seed": args.seed,
              "stage": args.stage, "num_envs": args.num_envs,
              "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
              "protocol": args.protocol,
              "zero_force_control": args.zero_force_control,
              "case_bank_path": str(args.case_bank.resolve()) if args.case_bank else None,
              "experiment_sha256": hashlib.sha256(args.experiment_config.read_bytes()).hexdigest() if args.experiment_config else None,
              "case_bank_sha256": hashlib.sha256(args.case_bank.read_bytes()).hexdigest() if args.case_bank else None}
    report['upgrade_config'] = str(args.upgrade_config.resolve()) if args.upgrade_config else None
    report['upgrade_config_sha256'] = upgrade_sha
    report['initial_case_bank_sha256'] = (hashlib.sha256(args.initial_case_bank.read_bytes()).hexdigest()
                                          if args.initial_case_bank else None)
    report['initial_case_bank_kind'] = (initial_bank_payload.get('bank_kind', 'natural_initial_state')
                                        if initial_bank_payload else None)
    report['handoff_actor_role'] = (
        handoff_role if handoff_evaluation else args.handoff_role if args.handoff_output else None
    )

    def record(**values):
        report.update(values)
        temp = args.output.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.output)
        print(json.dumps(values), flush=True)

    app = None
    record()
    try:
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        sys.path.insert(0, str(HCSP))
        # Prefer the canonical AeroWall modules over stale root-level copies
        # left by earlier experiment checkouts.
        sys.path.insert(0, str(ROOT / "scripts"))
        from hcsp import init_simulation_app
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        with initialize_config_dir(version_base=None, config_dir=str(HCSP / "cfg")):
            cfg = compose(config_name="train", overrides=["task=Serve", "headless=true", "wandb.mode=disabled",
                                                           f"task.env.num_envs={args.num_envs}", f"seed={args.seed}"])
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)
        cfg.sim.dt = 0.0025
        cfg.sim.substeps = 8
        cfg.env.max_episode_length = cfg.task.env.max_episode_length = experiment['horizon_steps'] if experiment else 600
        cfg.task.drone_model = "IrisTest"
        cfg.task.action_transform = None
        cfg.task.wall_curriculum_stage = args.stage
        cfg.task.wall_observation_version = args.observation_version
        cfg.task.wall_reward_design = args.reward_design
        cfg.task.wall_hit_window = args.hit_window
        cfg.task.wall_hit_enter_seconds = args.hit_window
        cfg.task.wall_hit_exit_seconds = args.hit_exit_window
        cfg.task.wall_min_dwell_steps = args.min_dwell_steps
        if upgrade:
            trajectory_cfg = upgrade['trajectory']
            cfg.task.wall_restitution = trajectory_cfg['restitution_x']
            cfg.task.wall_y_bounds = trajectory_cfg['wall_y_bounds_m']
            cfg.task.wall_z_bounds = trajectory_cfg['wall_z_bounds_m']
            cfg.task.wall_contact_height = trajectory_cfg['contact_height_m']
        cfg.task.wall_case_mode = 'bank' if initial_cases else 'fixed'
        cfg.task.wall_draw_ball_trajectory = args.record_rgb
        cfg.task.wall_recenter_reward = False
        cfg.task.wall_reward_variant = experiment.get('wall_reward_variant', 'constant') if experiment else 'constant'
        cfg.task.wall_perturb_mode = ('cases' if args.protocol == 'recovery'
                                      else 'curriculum' if args.zero_force_control else 'none')
        cfg.task.wall_record_events = True
        cfg.task.wall_end_on_ball_drop = bool(experiment.get("end_on_ball_drop", False)) if experiment else False
        if args.record_rgb:
            cfg.viewer.resolution = [1280, 720]
            # Give the high end of the ball arc a clear margin above the wall
            # and video frame. Keep the original viewing direction and translate
            # the camera upward to preserve the HCSP scene composition.
            cfg.viewer.eye = [4.5, -6.5, 5.2]
            cfg.viewer.lookat = [1.2, 0.5, 3.7]
        cfg.algo.actor.lr = 1e-4
        cfg.algo.critic.lr = 5e-4
        cfg.algo.train_every = 64
        cfg.algo.num_minibatches = 16
        cfg.algo.ppo_epochs = 4
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        sys.argv = [sys.argv[0], "--portable", "--portable-root", str(ROOT / ".cache/kit")]
        app = init_simulation_app(cfg)
        from hcsp_offline_assets import configure_local_asset_root
        configure_local_asset_root()
        from hcsp.learning import MAPPOPolicy
        from aerowall_policy_encoder import configure_policy_encoder
        configure_policy_encoder()
        from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
        from aerowall_wall_rally_env import AeroWallSingleWallRallyEnv
        from check_upstream_contacts import contact_reporting_before_initialization
        with contact_reporting_before_initialization():
            base = AeroWallSingleWallRallyEnv(cfg, headless=True)
        if cases:
            base.set_perturbation_cases(cases)
        if initial_cases:
            base.set_initial_cases(initial_cases)
        import carb
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
        from pxr import PhysicsSchemaTools
        carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
        physical_contacts = []
        callback_errors = []
        callback_step = [-1]
        def contact_callback(headers, data):
            try:
                for header in headers:
                    paths = [str(PhysicsSchemaTools.intToSdfPath(getattr(header, k))) for k in ('actor0', 'actor1')]
                    if any(path.endswith('/ball') for path in paths):
                        physical_contacts.append({'step': callback_step[0], 'actors': paths,
                                                  'substep': base.current_substep,
                                                  'contact_count': int(header.num_contact_data)})
            except Exception as exc:
                callback_errors.append(repr(exc))
        subscription = get_physx_simulation_interface().subscribe_contact_report_events(contact_callback)
        env = TransformedEnv(base, Compose(InitTracker())).eval()
        env.set_seed(args.seed)
        if args.train_skill or args.skill_chain:
            from aerowall_skill_policies import AeroWallSkillChainPolicy
            policy = AeroWallSkillChainPolicy(
                cfg.algo, env.agent_spec['drone'], base.device,
              train_skill=args.train_skill, launch_checkpoint=args.launch_checkpoint,
              launch_observation_version=args.launch_observation_version,
              skill_observation_version=args.skill_observation_version,
              recovery_checkpoint=args.recovery_checkpoint,
              hit_checkpoint=args.hit_checkpoint,
              launch_action_distribution=args.launch_action_distribution,
              hit_action_distribution=args.hit_action_distribution,
              recovery_action_distribution=args.recovery_action_distribution,
              recovery_observation_version=args.recovery_observation_version,
              hit_observation_version=args.hit_observation_version,
            )
        elif args.launch_checkpoint:
            from aerowall_skill_policies import AeroWallLaunchRecoveryPolicy
            policy = AeroWallLaunchRecoveryPolicy(
                cfg.algo, env.agent_spec['drone'], base.device, args.launch_checkpoint,
                recovery_checkpoint=args.recovery_checkpoint,
            )
        else:
            policy = MAPPOPolicy(cfg.algo, agent_spec=env.agent_spec["drone"], device=base.device)
        checkpoint_sha = None
        warmstart = None
        if args.checkpoint:
            checkpoint = args.checkpoint.resolve()
            policy.load_state_dict(torch.load(checkpoint, map_location=base.device))
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        elif args.actor_warmstart:
            from train_aerowall_wall_rl import warmstart_actor
            checkpoint = args.actor_warmstart.resolve()
            warmstart = warmstart_actor(policy, checkpoint)
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        policy.eval()
        td = env.reset()
        reproducibility_record = None
        initial_policy_input_trace = None
        if args.record_reproducibility:
            input_files = {
                name: _source_record(name, path)
                for name, path in (
                    ('checkpoint', args.checkpoint),
                    ('actor_warmstart', args.actor_warmstart),
                    ('launch_checkpoint', args.launch_checkpoint),
                    ('hit_checkpoint', args.hit_checkpoint),
                    ('recovery_checkpoint', args.recovery_checkpoint),
                    ('candidate_training_report', args.candidate_training_report),
                    ('hit_training_report', args.hit_training_report),
                    ('upgrade_config', args.upgrade_config),
                    ('experiment_config', args.experiment_config),
                    ('case_bank', args.case_bank),
                    ('initial_case_bank', args.initial_case_bank),
                ) if path is not None
            }
            reproducibility_record = {
                'source_files': _reproducibility_sources(base, policy),
                'input_files': input_files,
                'post_reset_state': _reset_state_record(base, torch, np, random),
                'runtime': {
                    'torch_version': torch.__version__,
                    'cuda_version': torch.version.cuda,
                    'cuda_device_name': torch.cuda.get_device_name(base.device)
                    if torch.cuda.is_available() else None,
                    'hcsp_commit': subprocess.check_output(
                        ['git', '-C', str(HCSP), 'rev-parse', 'HEAD'], text=True
                    ).strip(),
                },
            }
        if args.handoff_reset_audit_only:
            reset_audit = audit_handoff_reset(base, initial_cases, torch)
            record(status='passed', handoff_evaluation=True,
                   handoff_actor_role=handoff_role,
                   handoff_reset_audit=reset_audit,
                   contact_audit={
                       'inherited_source_prefix_events': sum(
                           len(case.get('source_event_prefix', [])) for case in initial_cases
                       ),
                       'inherited_source_prefix_physx_audited': True,
                   })
            return
        rgb_frames = 0
        rgb_directory = args.output.with_suffix('.frames')
        def save_rgb():
            nonlocal rgb_frames
            from PIL import Image
            base.sim.render()
            pixels = np.asarray(base.render(mode='rgb_array')).copy()
            assert pixels.shape == (720, 1280, 3) and pixels.std() > 1., 'Invalid RGB readback'
            Image.fromarray(pixels).save(rgb_directory / f'{rgb_frames:05d}.png')
            rgb_frames += 1
        if args.record_rgb:
            rgb_directory.mkdir(exist_ok=False)
            before_render = base.ball.get_world_poses()[0].clone()
            for _ in range(5):
                base.sim.render()
            save_rgb()
            assert torch.equal(before_render, base.ball.get_world_poses()[0]), 'Rendering advanced physics'
        finished = torch.zeros(args.num_envs, dtype=torch.bool, device=base.device)
        outcomes = [None] * args.num_envs
        first_episode_events = []
        event_cursor = 0
        closest_distance = torch.full((args.num_envs,), float("inf"), device=base.device)
        closest_radial = torch.full_like(closest_distance, float("nan"))
        closest_axial = torch.full_like(closest_distance, float("nan"))
        closest_ball_z = torch.full_like(closest_distance, float("nan"))
        closest_drone_z = torch.full_like(closest_distance, float("nan"))
        closest_up_z = torch.full_like(closest_distance, float("nan"))
        horizontal_history = []
        diagnostic_trace = []
        trajectory = []
        action_intervention_trace = []
        first_legal_cap_by_env = {}
        pending_postcap_actor_input = {}
        post_contact_actor_input_trace = []
        ball_velocity_restore_trace = []
        ball_velocity_restore_applied_envs = set()
        action_change_sum = 0.0
        action_square_sum = 0.0
        action_sample_count = 0
        previous_actions = torch.zeros(args.num_envs, 4, device=base.device)
        skill_step_counts = [0, 0, 0]
        handoff_candidates = []
        handoff_candidate_by_env = {}
        handoff_recorded = torch.zeros(args.num_envs, dtype=torch.bool, device=base.device)
        started = time.monotonic()
        with torch.no_grad():
            for step in range(base.max_episode_length):
                callback_step[0] = step
                active = ~finished.clone()
                if args.handoff_output:
                    eligible = active & ~handoff_recorded
                    launch_contact_time = None
                    if args.handoff_role == 'launch':
                        eligible &= (base.skill_id == 0) & (base.phase == 0)
                        eligible &= (base.caps == 0) & (base.walls == 0)
                        if args.handoff_case_ids:
                            requested_ids = set(args.handoff_case_ids.split(','))
                            eligible &= torch.as_tensor(
                                [case.get('case_id') in requested_ids for case in initial_cases],
                                dtype=torch.bool, device=base.device,
                            )
                        _, ball_probe, velocity_probe = base._kinematics()
                        vertical_gap = ball_probe[:, 0, 2] - base.wall_contact_height
                        vertical_discriminant = velocity_probe[:, 0, 2].square() + 2.0 * 9.81 * vertical_gap
                        valid_vertical_crossing = vertical_discriminant >= 0.0
                        launch_contact_time = (velocity_probe[:, 0, 2]
                                               + vertical_discriminant.clamp_min(0.0).sqrt()) / 9.81
                        valid_vertical_crossing &= launch_contact_time > 0.0
                        eligible &= valid_vertical_crossing
                        eligible &= (launch_contact_time - args.handoff_time_target).abs() <= 0.0200001
                    else:
                        handoff_skill_id = 1 if args.handoff_role == 'hit' else 2
                        eligible &= (base.skill_id == handoff_skill_id) & (base.phase == 2)
                        eligible &= (base.caps > 0) & (base.walls > 0)
                    indices = torch.nonzero(eligible, as_tuple=False).flatten().cpu().tolist()
                    if indices:
                        root_before, ball_pos_before, ball_vel_before = base._kinematics()
                        drone_velocity_before = base.drone.get_velocities(clone=True)[:, 0]
                        _, ball_orientation_before = base.ball.get_world_poses(clone=True)
                        for index in indices:
                            source_case = initial_cases[index] if initial_cases else None
                            source_case_id = (source_case.get('case_id') if source_case
                                              else f'fixed-{index:04d}')
                            candidate = {
                                'case_id': f'handoff-{source_case_id}-env-{index:04d}',
                                'handoff_state_version': (3 if args.handoff_role == 'launch' else
                                                          1 if args.handoff_role == 'hit' else 2),
                                'handoff_actor_role': args.handoff_role,
                                'source_case_id': source_case_id,
                                'source_env': index,
                                'source_policy_step': step,
                                'source_progress_step': int(base.progress_buf[index].item()),
                                'drone_position': root_before[index, 0, :3].detach().cpu().tolist(),
                                'drone_orientation': root_before[index, 0, 3:7].detach().cpu().tolist(),
                                'drone_velocity': drone_velocity_before[index].detach().cpu().tolist(),
                                'motor_throttle': base.drone.throttle[index, 0].detach().cpu().tolist(),
                                'prev_action': base.prev_action[index, 0].detach().cpu().tolist(),
                                'action_before': base.action_before[index, 0].detach().cpu().tolist(),
                                'ball_position': ball_pos_before[index, 0].detach().cpu().tolist(),
                                'ball_orientation': ball_orientation_before[index, 0].detach().cpu().tolist(),
                                'ball_velocity': ball_vel_before[index, 0, :3].detach().cpu().tolist(),
                                'ball_angular_velocity': ball_vel_before[index, 0, 3:6].detach().cpu().tolist(),
                                'prev_ball_velocity': base.prev_ball_vel[index].detach().cpu().tolist(),
                                'wall_target': base.wall_target[index].detach().cpu().tolist(),
                                'phase': int(base.phase[index].item()),
                                'caps': int(base.caps[index].item()),
                                'walls': int(base.walls[index].item()),
                                'rallies': int(base.rallies[index].item()),
                                'streak': int(base.streak[index].item()),
                                'max_streak': int(base.max_streak[index].item()),
                                'skill_id': int(base.skill_id[index].item()),
                                'skill_held_steps': int(base.skill_held_steps[index].item()),
                                'source_event_prefix': [],
                            }
                            if args.handoff_role == 'launch':
                                candidate['source_time_to_contact_s'] = float(launch_contact_time[index].item())
                            handoff_candidates.append(candidate)
                            handoff_candidate_by_env[index] = candidate
                        handoff_recorded[indices] = True
                observation_before = td['agents', 'observation'][0, 0].detach().cpu().tolist()
                if hasattr(policy, 'record_reproducibility'):
                    # When explicitly requested, retain the active actor view
                    # for every step so contact callbacks can be joined to the
                    # exact observation that produced their action.
                    policy.record_reproducibility = bool(args.record_reproducibility)
                td = policy(td, deterministic=True)
                event_actor_observations = (
                    _event_actor_observations(policy, td)
                    if args.record_reproducibility else {}
                )
                policy_actions = td['agents', 'action'][:, 0].clone()
                action_before = policy_actions[0].detach().cpu().tolist()
                if args.record_reproducibility:
                    for env_index, cap_state in list(pending_postcap_actor_input.items()):
                        if step != cap_state['policy_step'] + 1:
                            continue
                        actor_input = event_actor_observations.get(env_index)
                        record_row = {
                            'case_id': initial_cases[env_index]['case_id'] if initial_cases else 'fixed',
                            'env': env_index,
                            'first_legal_cap_policy_step': cap_state['policy_step'],
                            'first_legal_cap_substep': cap_state['substep'],
                            'policy_step': step,
                            'state_intervention_applied': env_index in ball_velocity_restore_applied_envs,
                            'active_actor_observation': actor_input,
                            'policy_action': policy_actions[env_index].detach().cpu().tolist(),
                        }
                        post_contact_actor_input_trace.append(record_row)
                        del pending_postcap_actor_input[env_index]
                actions = policy_actions.clone()
                if (args.aerowall_rotor_override_channel is not None
                        and step < args.aerowall_rotor_override_steps):
                    active_skills = td['info', 'skill_id'].reshape(-1).long()
                    active_phases = base.phase
                    override_rows = active
                    if bool((override_rows & ((active_skills != 0) | (active_phases != 0))).any()):
                        raise RuntimeError('Registered rotor response screen left Launch phase before its 10-step intervention horizon')
                    env_indices = torch.nonzero(override_rows, as_tuple=False).flatten().cpu().tolist()
                    for index in env_indices:
                        channel = args.aerowall_rotor_override_channel
                        before = float(policy_actions[index, channel].item())
                        unbounded = before + args.aerowall_rotor_override_delta
                        after_limit = max(-1.0, min(1.0, unbounded))
                        action_intervention_trace.append({
                            'env': index,
                            'case_id': initial_cases[index]['case_id'],
                            'policy_step': step,
                            'rotor_index': channel,
                            'delta': args.aerowall_rotor_override_delta,
                            'policy_action_before_override': before,
                            'effective_action_before_rotor_limit': unbounded,
                            'rotor_command_after_limit': after_limit,
                            'effective_limited_command_delta': after_limit - max(-1.0, min(1.0, before)),
                        })
                    actions[override_rows, args.aerowall_rotor_override_channel] += args.aerowall_rotor_override_delta
                    action_tensor = td['agents', 'action'].clone()
                    action_tensor[:, 0] = actions
                    td['agents', 'action'] = action_tensor
                for index, candidate in list(handoff_candidate_by_env.items()):
                    if candidate['source_policy_step'] == step:
                        candidate['source_policy_action'] = policy_actions[index].detach().cpu().tolist()
                        del handoff_candidate_by_env[index]
                if args.record_reproducibility and step == 0:
                    initial_policy_input_trace = _actor_input_record(policy, td, policy_actions)
                    if initial_policy_input_trace is not None:
                        initial_policy_input_trace['policy_step'] = step
                        initial_policy_input_trace['observation_versions_by_role'] = getattr(
                            policy, 'observation_versions', None
                        )
                action_change_sum += float((actions[active] - previous_actions[active]).norm(dim=-1).sum())
                action_square_sum += float(actions[active].square().sum())
                action_sample_count += int(active.sum())
                previous_actions[active] = actions[active]
                for skill in range(3):
                    skill_step_counts[skill] += int((active & (td['info', 'skill_id'].flatten().long() == skill)).sum())
                nxt = env.step(td)["next"]
                step_events = base.events[event_cursor:]
                for event in step_events:
                    env_index = int(event['env'])
                    if not (bool(event.get('body')) and bool(event.get('legal_cap'))):
                        continue
                    if env_index in first_legal_cap_by_env:
                        continue
                    cap_state = {
                        'policy_step': step,
                        'substep': int(event.get('substep', -1)),
                    }
                    first_legal_cap_by_env[env_index] = cap_state
                    pending_postcap_actor_input[env_index] = cap_state
                    target_entry = ball_velocity_restore_by_env.get(env_index)
                    if target_entry is None:
                        continue
                    if bool(nxt['done'].flatten()[env_index].item()):
                        raise RuntimeError(f'Cannot apply post-cap velocity restore after a terminal transition: {target_entry["case_id"]}')
                    ball_velocity = base.ball.get_velocities()
                    before_velocity = ball_velocity[env_index, 0, :3].detach().clone()
                    target_tensor = torch.as_tensor(
                        target_entry['control_post_step_ball_linear_velocity_mps'],
                        dtype=ball_velocity.dtype, device=base.device,
                    )
                    full_velocity = ball_velocity[env_index, 0].detach().clone().unsqueeze(0)
                    full_velocity[0, :3] = target_tensor
                    env_ids = torch.tensor([env_index], dtype=torch.long, device=base.device)
                    base.ball.set_velocities(full_velocity, env_ids)
                    after_velocity = base.ball.get_velocities()[env_index, 0, :3].detach().clone()
                    max_abs_error = float((after_velocity - target_tensor).abs().max().item())
                    if max_abs_error > 1e-5:
                        raise RuntimeError(f'Post-cap ball velocity restore readback exceeded tolerance for {target_entry["case_id"]}: {max_abs_error}')
                    # Keep the collision detector's velocity history consistent with the injected physical state.
                    base.prev_ball_vel[env_index] = after_velocity
                    # The environment already produced this transition's reward/done;
                    # refresh the next state and all explicit observation views only.
                    nxt.update(base._compute_state_and_obs())
                    ball_velocity_restore_trace.append({
                        'case_id': target_entry['case_id'],
                        'env': env_index,
                        'policy_step': step,
                        'contact_substep': int(event.get('substep', -1)),
                        'contact_radial_error_m': float(event['radial_error']),
                        'pre_restore_ball_linear_velocity_mps': before_velocity.cpu().tolist(),
                        'target_control_post_step_ball_linear_velocity_mps': target_tensor.cpu().tolist(),
                        'post_restore_ball_linear_velocity_mps': after_velocity.cpu().tolist(),
                        'readback_max_abs_error_mps': max_abs_error,
                        'tolerance_mps': 1e-5,
                        'ball_position_preserved': True,
                        'ball_orientation_and_angular_velocity_preserved': True,
                        'drone_and_fsm_state_preserved': True,
                        'prev_ball_velocity_history_synchronized': True,
                        'next_state_observation_refreshed': True,
                        'restore_bank_sha256': ball_velocity_restore_payload['_runtime_bank_sha256'],
                    })
                    ball_velocity_restore_applied_envs.add(env_index)
                if args.record_rgb and (step + 1) % 2 == 0:
                    save_rgb()
                root, ball_position, _ = base._kinematics()
                if args.record_trajectory:
                    trajectory.append({'drone_state': root[:, 0].detach().cpu().numpy().copy(),
                                       'ball_position': ball_position[:, 0].detach().cpu().numpy().copy(),
                                       'ball_velocity': base.ball.get_velocities()[:, 0].detach().cpu().numpy().copy(),
                                       'skill_id': td['info', 'skill_id'][:, 0].detach().cpu().numpy().copy(),
                                       'active': active.detach().cpu().numpy().copy(),
                                       'action': policy_actions.detach().cpu().numpy().copy(),
                                       'executed_action': actions.detach().cpu().numpy().copy(),
                                       'actuator_command': actions.clamp(-1.0, 1.0).detach().cpu().numpy().copy(),
                                       'motor_throttle': base.drone.throttle[:, 0].detach().cpu().numpy().copy(),
                                       'rotor_thrust_local_z_n': base.drone.thrusts[:, 0, :, 2].detach().cpu().numpy().copy()})
                if bool(active[0]) and (step < 10 or step % 10 == 0):
                    trace_row = {'step': step, 'observation_before': observation_before,
                                 'action': action_before,
                                 'executed_action_before_rotor_limit': actions[0].detach().cpu().tolist(),
                                 'root_after': root[0, 0].detach().cpu().tolist()}
                    if args.record_reproducibility:
                        trace_row['actor_input_record'] = _actor_input_record(
                            policy, td, actions, first_env_only=True,
                        )
                        trace_row['observation_versions_by_role'] = getattr(
                            policy, 'observation_versions', None
                        )
                    diagnostic_trace.append(trace_row)
                horizontal_history.append((root[:, 0, :2] - torch.tensor([1.5, 0.0], device=base.device)).norm(dim=-1).clone())
                if len(horizontal_history) > 100:
                    horizontal_history.pop(0)
                relative = ball_position[:, 0] - root[:, 0, :3]
                up = root[:, 0, 16:19]
                axial = (relative * up).sum(-1)
                radial = (relative - axial[:, None] * up).norm(dim=-1)
                separation = relative.norm(dim=-1)
                nearer = active & (separation < closest_distance)
                closest_distance = torch.where(nearer, separation, closest_distance)
                closest_radial = torch.where(nearer, radial, closest_radial)
                closest_axial = torch.where(nearer, axial, closest_axial)
                closest_ball_z = torch.where(nearer, ball_position[:, 0, 2], closest_ball_z)
                closest_drone_z = torch.where(nearer, root[:, 0, 2], closest_drone_z)
                closest_up_z = torch.where(nearer, up[:, 2], closest_up_z)
                for event in base.events[event_cursor:]:
                    if bool(active[event["env"]]):
                        event_record = {"policy_step": step, **event}
                        event_record.update(event_actor_observations.get(event['env'], {}))
                        first_episode_events.append(event_record)
                event_cursor = len(base.events)
                done = nxt["done"].flatten()
                for index in torch.nonzero(active & done).flatten().cpu().tolist():
                    stats = nxt["stats"][index]
                    outcomes[index] = {
                        "env": index,
                        "case_id": initial_cases[index]['case_id'] if initial_cases else 'fixed',
                        "policy_steps": step + 1,
                        "return": float(stats["return"].item()),
                        "rallies": int(stats["rallies"].item()),
                        "max_streak": int(stats["max_streak"].item()),
                        "caps": int(stats["caps"].item()),
                        "walls": int(stats["walls"].item()),
                        "failure": int(stats["failure"].item()),
                        "failure_reason_bits": int(stats["failure_reason_bits"].item()),
                        "timed_out": bool(nxt['truncated'][index].item()),
                        "perturbation_wanted": bool(base.perturb_wanted[index].item()),
                        "perturbation_applied": bool(base.perturb_applied[index].item()),
                        "perturbation_missed": bool(base.perturb_missed[index].item()),
                        "perturbation_start_at": int(base.perturb_start_at[index].item()),
                        "perturbation_end_at": int(base.perturb_end_at[index].item()),
                        "closest_distance": float(closest_distance[index].item()),
                        "closest_radial": float(closest_radial[index].item()),
                        "closest_axial": float(closest_axial[index].item()),
                        "closest_ball_z": float(closest_ball_z[index].item()),
                        "closest_drone_z": float(closest_drone_z[index].item()),
                        "closest_up_z": float(closest_up_z[index].item()),
                        "horizontal_error_last_100": float(torch.stack(horizontal_history)[..., index].mean().item()),
                    }
                    finished[index] = True
                if bool(finished.all()):
                    break
                if bool(done.any()):
                    nxt["_reset"] = nxt["done"]
                    td = env.reset(nxt)
                else:
                    td = nxt
        assert all(item is not None for item in outcomes)
        if ball_velocity_restore_payload is not None:
            expected_restore_envs = set(ball_velocity_restore_by_env)
            missing_injections = expected_restore_envs - ball_velocity_restore_applied_envs
            missing_followups = expected_restore_envs - {
                row['env'] for row in post_contact_actor_input_trace
            }
            bad_readbacks = [
                row for row in ball_velocity_restore_trace
                if row['readback_max_abs_error_mps'] > row['tolerance_mps']
            ]
            wrong_roles = [
                row for row in post_contact_actor_input_trace
                if row['env'] in expected_restore_envs
                and (not row['active_actor_observation']
                     or row['active_actor_observation']['actor_observation_role'] != 'recovery')
            ]
            if missing_injections or missing_followups or bad_readbacks or wrong_roles:
                raise RuntimeError(
                    'AeroWall post-cap restore audit failed: '
                    f'missing_injections={sorted(missing_injections)}, '
                    f'missing_followups={sorted(missing_followups)}, '
                    f'bad_readbacks={len(bad_readbacks)}, wrong_roles={len(wrong_roles)}'
                )
        rallies = [item["rallies"] for item in outcomes]
        caps = [item["caps"] for item in outcomes]
        walls = [item["walls"] for item in outcomes]
        returns = [item["return"] for item in outcomes]
        distances = [item["closest_distance"] for item in outcomes]
        radials = [item["closest_radial"] for item in outcomes]
        horizontal_errors = [item["horizontal_error_last_100"] for item in outcomes]
        event_path = args.output.with_suffix(".events.jsonl")
        event_path.write_text("".join(json.dumps(item) + "\n" for item in first_episode_events))
        summary = {
            "episodes": len(outcomes),
            "mean_return": float(np.mean(returns)),
            "mean_caps": float(np.mean(caps)),
            "valid_first_contact_rate": float(np.mean(np.asarray(caps) >= 1)),
            "mean_walls": float(np.mean(walls)),
            "mean_rallies": float(np.mean(rallies)),
            "legal_second_hit_rate": float(np.mean(np.asarray(caps) >= 2)),
            "safety_failure_rate": float(np.mean([row['failure'] in (2, 4, 5) for row in outcomes])),
            "failure_counts": {name: sum(row['failure'] == code for row in outcomes)
                               for code, name in ((0, 'none'), (1, 'ball_ground'), (2, 'drone_ground'),
                                                  (3, 'out_of_bounds'), (4, 'illegal_contact'), (5, 'drone_wall'))},
            "mean_action_change": action_change_sum / max(action_sample_count, 1),
            "mean_action_square_proxy": action_square_sum / max(action_sample_count, 1),
            "skill_step_counts": skill_step_counts,
            "one_rally_rate": float(np.mean(np.asarray(rallies) >= 1)),
            "three_rally_rate": float(np.mean(np.asarray(rallies) >= 3)),
            "five_rally_rate": float(np.mean(np.asarray(rallies) >= 5)),
            "max_rallies": int(max(rallies)),
            "mean_closest_distance": float(np.mean(distances)),
            "min_closest_distance": float(np.min(distances)),
            "mean_closest_radial": float(np.mean(radials)),
            "mean_horizontal_error_last_100": float(np.mean(horizontal_errors)),
            "fraction_horizontal_error_below_0p1": float(np.mean(np.asarray(horizontal_errors) < 0.1)),
        }
        if handoff_evaluation:
            initial_caps = np.asarray([int(case['caps']) for case in initial_cases])
            initial_rallies = np.asarray([int(case['rallies']) for case in initial_cases])
            added_caps = np.asarray(caps) - initial_caps
            added_rallies = np.asarray(rallies) - initial_rallies
            summary['valid_first_contact_rate'] = None
            summary['legal_second_hit_rate'] = float(np.mean(added_caps >= 1))
            summary['handoff_conditional'] = {
                'metric_scope': f'continuation from an audited pre-{handoff_role.title()} state',
                'handoff_actor_role': handoff_role,
                'first_legal_hit_rate': float(np.mean(added_caps >= 1)),
                'mean_rallies_after_handoff': float(np.mean(added_rallies)),
                'one_rally_after_handoff_rate': float(np.mean(added_rallies >= 1)),
                'three_rallies_after_handoff_rate': float(np.mean(added_rallies >= 3)),
                'five_rallies_after_handoff_rate': float(np.mean(added_rallies >= 5)),
                'source_prefix_physx_audited': True,
            }
        valid_wall_events = [event for event in first_episode_events if event['expected_wall']]
        if valid_wall_events:
            targeted = [event for event in valid_wall_events
                        if abs(event['ball_position'][1] - float(base.wall_target[event['env'], 0])) <= 1.0
                        and abs(event['ball_position'][2] - float(base.wall_target[event['env'], 1])) <= 1.5]
            summary['wall_target_hit_rate'] = len(targeted) / len(valid_wall_events)
            summary['mean_wall_height_m'] = float(np.mean([event['ball_position'][2] for event in valid_wall_events]))
        else:
            summary['wall_target_hit_rate'] = None
            summary['mean_wall_height_m'] = None
        if args.stage == "A0":
            gate = {"metric": "horizontal_error_last_100", "threshold_m": 0.1,
                    "required_fraction": 0.9,
                    "passed": summary["mean_horizontal_error_last_100"] < 0.1
                              and summary["fraction_horizontal_error_below_0p1"] >= 0.9
                              and all(row['policy_steps'] == base.max_episode_length and row['failure'] == 0 for row in outcomes)}
        elif args.stage in ("A", "A1", "A2"):
            gate = {"metric": "legal_cap_rate", "threshold": 0.8,
                    "value": float(np.mean(np.asarray(caps) >= 1)),
                    "passed": float(np.mean(np.asarray(caps) >= 1)) >= 0.8}
        else:
            gate = None
        corroborated = []
        cap_audited = {}
        for event in first_episode_events:
            if event['legal_cap']:
                expected_ball = f"/World/envs/env_{event['env']}/ball"
                confirmed = any(row['step'] == event['policy_step'] and row['substep'] == event['substep']
                                and row['contact_count'] > 0 and expected_ball in row['actors']
                                and any('/IrisTest_0/' in path for path in row['actors'])
                                for row in physical_contacts)
                corroborated.append(confirmed)
                cap_audited[(event['env'], event['policy_step'], event['substep'])] = confirmed
        contact_audit = {'legal_events': len(corroborated), 'corroborated_events': sum(corroborated),
                         'callback_errors': callback_errors, 'physical_reports': len(physical_contacts)}
        wall_corroborated = []
        wall_audited = {}
        for event in first_episode_events:
            if event['wall']:
                expected_ball = f"/World/envs/env_{event['env']}/ball"
                confirmed = any(row['step'] == event['policy_step'] and row['substep'] == event['substep']
                                and row['contact_count'] > 0 and expected_ball in row['actors']
                                and any(path.endswith('/single_wall') for path in row['actors'])
                                for row in physical_contacts)
                wall_corroborated.append(confirmed)
                wall_audited[(event['env'], event['policy_step'], event['substep'])] = confirmed
        contact_audit.update(wall_events=len(wall_corroborated), corroborated_wall_events=sum(wall_corroborated))
        handoff_capture = None
        if handoff_evaluation:
            contact_audit['inherited_source_prefix_events'] = sum(
                len(case.get('source_event_prefix', [])) for case in initial_cases
            )
            contact_audit['inherited_source_prefix_physx_audited'] = True
        if args.handoff_output:
            accepted_handoffs = []
            rejected_handoffs = 0
            for candidate in handoff_candidates:
                prefix = []
                invalid_history = False
                env_index = candidate['source_env']
                cutoff = candidate['source_policy_step']
                for event in first_episode_events:
                    if event['env'] != env_index or event['policy_step'] >= cutoff:
                        continue
                    marker = (env_index, event['policy_step'], event['substep'])
                    if event['body'] and not event.get('expected_cap', False):
                        invalid_history = True
                        break
                    if event['wall'] and not event['expected_wall']:
                        invalid_history = True
                        break
                    if event.get('expected_cap', False):
                        if not cap_audited.get(marker, False):
                            invalid_history = True
                            break
                        prefix.append({'kind': 'cap', 'y': event['ball_position'][1],
                                       'step': event['policy_step'], 'substep': event['substep']})
                    elif event['expected_wall']:
                        if not wall_audited.get(marker, False):
                            invalid_history = True
                            break
                        prefix.append({'kind': 'wall', 'y': event['ball_position'][1],
                                       'step': event['policy_step'], 'substep': event['substep']})
                if args.handoff_role == 'launch':
                    if invalid_history or prefix:
                        rejected_handoffs += 1
                        continue
                elif invalid_history or len(prefix) < 2 or prefix[-1]['kind'] != 'wall':
                    rejected_handoffs += 1
                    continue
                candidate['source_event_prefix'] = prefix
                from aerowall.wall_rl.curriculum import validate_handoff_case
                validate_handoff_case(candidate)
                accepted_handoffs.append(candidate)
            checkpoint_hashes = {}
            for name, path in (('launch', args.launch_checkpoint),
                               ('recovery', args.recovery_checkpoint),
                               ('hit', args.hit_checkpoint),
                               ('standalone', args.checkpoint)):
                if path:
                    checkpoint_hashes[name] = hashlib.sha256(path.resolve().read_bytes()).hexdigest()
            payload = {
                'handoff_bank_version': 1,
                'bank_kind': f'aerowall_{args.handoff_role}_handoff_v1',
                'handoff_actor_role': args.handoff_role,
                'source_report': str(args.output),
                'source_event_log': str(args.output.with_suffix('.events.jsonl')),
                'source_contact_report': str(args.output.with_suffix('.contacts.json')),
                'source_initial_case_bank_sha256': report['initial_case_bank_sha256'],
                'source_checkpoint_sha256': checkpoint_hashes,
                'source_contact_audit': {
                    'legal_cap_events': len(corroborated),
                    'corroborated_cap_events': sum(corroborated),
                    'wall_events': len(wall_corroborated),
                    'corroborated_wall_events': sum(wall_corroborated),
                    'callback_errors': callback_errors,
                },
                'candidate_count': len(handoff_candidates),
                'rejected_count': rejected_handoffs,
                'cases': accepted_handoffs,
            }
            encoded = json.dumps(payload, indent=2, sort_keys=True) + '\n'
            if args.handoff_output.exists() and args.handoff_output.read_text() != encoded:
                raise ValueError(f'Refusing to overwrite a different frozen {args.handoff_role.title()} handoff bank')
            if not args.handoff_output.exists():
                temp = args.handoff_output.with_suffix(args.handoff_output.suffix + '.tmp')
                temp.write_text(encoded)
                temp.replace(args.handoff_output)
            handoff_capture = {
                'path': str(args.handoff_output),
                'sha256': hashlib.sha256(args.handoff_output.read_bytes()).hexdigest(),
                'handoff_actor_role': args.handoff_role,
                'capture_time_target_s': args.handoff_time_target if args.handoff_role == 'launch' else None,
                'candidate_count': len(handoff_candidates),
                'accepted_count': len(accepted_handoffs),
                'rejected_count': rejected_handoffs,
            }
        from aerowall_wall_reward_logic import score_rally_events
        audited_events = [
            ([dict(event) for event in initial_cases[index].get('source_event_prefix', [])]
             if handoff_evaluation else [])
            for index in range(args.num_envs)
        ]
        for event in first_episode_events:
            marker = (event['env'], event['policy_step'], event['substep'])
            kind = ('cap' if event.get('expected_cap', event['legal_cap']) and cap_audited.get(marker) else
                    'wall' if event['wall'] and wall_audited.get(marker) else None)
            if kind:
                audited_events[event['env']].append({
                    'kind': kind, 'y': event['ball_position'][1],
                    'step': event['policy_step'], 'substep': event['substep'],
                })
        scored_events = [score_rally_events(events) for events in audited_events]
        summary['mean_centered_prefix_rallies'] = float(np.mean(
            [row['centered_prefix_rallies'] for row in scored_events]))
        summary['rally_histogram'] = {str(value): rallies.count(value) for value in sorted(set(rallies))}
        summary['all_outcomes_contact_audited'] = all(
            scored['caps'] == outcome['caps'] and scored['walls'] == outcome['walls']
            and scored['rallies'] == outcome['rallies']
            for scored, outcome in zip(scored_events, outcomes)
        )
        if experiment:
            from aerowall_wall_reward_logic import score_recovery
            for index, outcome in enumerate(outcomes):
                scored = scored_events[index]
                outcome.update(scored)
                outcome['max_abs_wall_y'] = max((abs(y) for y in scored['wall_y_sequence']), default=None)
                failure_names = {0: 'timeout_or_unfinished', 1: 'ball_ground', 2: 'drone_ground',
                                 3: 'out_of_bounds', 4: 'illegal_contact', 5: 'drone_wall'}
                outcome['failure_reason'] = failure_names.get(outcome['failure'], 'unknown')
                outcome['contact_audit_passed'] = (
                    scored['caps'] == outcome['caps'] and scored['walls'] == outcome['walls']
                    and scored['rallies'] == outcome['rallies']
                )
                if args.protocol == 'recovery':
                    end_at = outcome['perturbation_end_at']
                    impulse_end = divmod(end_at, base.substeps) if end_at >= 0 else None
                    outcome.update(score_recovery(audited_events[index], impulse_end,
                                                  outcome['failure'] != 0, outcome['timed_out']))
                    outcome['recovery_success'] &= outcome['perturbation_applied'] and outcome['contact_audit_passed']
                    outcome['case_id'] = cases[index]['case_id']
                    outcome['direction'] = cases[index]['direction']
                    outcome['difficulty'] = cases[index]['difficulty']
                    # Only an off-center wall *before* the first centered
                    # recovery wall is evidence of a rescue. A later drift
                    # after success is a separate failure, not a rescue.
                    candidate = outcome['recovery_wall_ordinal']
                    before_center = scored['wall_y_sequence'][2: candidate - 1 if candidate else 5]
                    outcome['off_center_wall_before_recovery'] = any(abs(y) > 1.0 for y in before_center)
            summary['safe10_count'] = sum(row['safe10'] and row['contact_audit_passed'] for row in outcomes)
            summary['safe10_rate'] = summary['safe10_count'] / len(outcomes)
            summary['safe15_count'] = sum(row['safe15'] and row['contact_audit_passed'] for row in outcomes)
            summary['safe15_rate'] = summary['safe15_count'] / len(outcomes)
            summary['mean_centered_prefix_rallies'] = float(np.mean([row['centered_prefix_rallies'] for row in outcomes]))
            if args.protocol == 'recovery':
                summary['recovery_success_count'] = sum(row['recovery_success'] for row in outcomes)
                summary['recovery_success_rate'] = summary['recovery_success_count'] / len(outcomes)
                summary['recovery_by_direction'] = {
                    direction: {'count': sum(row['direction'] == direction for row in outcomes),
                                'success': sum(row['recovery_success'] for row in outcomes if row['direction'] == direction)}
                    for direction in ('left', 'right')
                }
                summary['perturbation_missed_count'] = sum(row['perturbation_missed'] for row in outcomes)
                displaced = [row for row in outcomes if row['off_center_wall_before_recovery']]
                summary['off_center_recovery_subset'] = {
                    'count': len(displaced), 'recovery_success': sum(row['recovery_success'] for row in displaced)
                }
        if args.stage in ('A', 'A1', 'A2'):
            gate['passed'] = gate['passed'] and bool(corroborated) and all(corroborated) and not callback_errors
        elif args.stage == 'WALL':
            rate = float(np.mean(np.asarray(walls) >= 1))
            gate = {'metric': 'wall_rate', 'value': rate, 'threshold': .8,
                    'passed': rate >= .8 and bool(wall_corroborated) and all(wall_corroborated)
                              and all(corroborated) and not callback_errors}
        elif args.stage == 'RETURN':
            rate = float(np.mean(np.asarray(rallies) >= 1))
            gate = {'metric': 'one_return_rate', 'value': rate, 'threshold': .8,
                    'passed': rate >= .8 and bool(wall_corroborated) and all(wall_corroborated)
                              and bool(corroborated) and all(corroborated) and not callback_errors}
        elif args.stage == 'RALLY':
            audit_ok = bool(wall_corroborated) and all(wall_corroborated) and bool(corroborated) and all(corroborated) and not callback_errors
            gate = {'metric': 'consecutive_rallies', 'minimum_target': 3, 'main_target': 5,
                    'three_rate': summary['three_rally_rate'], 'five_rate': summary['five_rally_rate'],
                    'three_passed': summary['three_rally_rate'] >= .8 and audit_ok,
                    'five_passed': summary['five_rally_rate'] >= .8 and audit_ok,
                    'passed': summary['five_rally_rate'] >= .8 and audit_ok}
        if handoff_evaluation:
            gate = {
                'metric': 'natural_full_episode_retention',
                'applicable': False,
                'passed': False,
                'reason': f'conditional {handoff_role.title()} handoff evaluation; use handoff_conditional metrics and do not treat as natural full-episode acceptance',
            }
        if experiment:
            audit_ok = (bool(corroborated) and all(corroborated) and bool(wall_corroborated)
                        and all(wall_corroborated) and not callback_errors
                        and all(row['contact_audit_passed'] for row in outcomes))
            if args.protocol == 'natural':
                required = (experiment['acceptance']['natural_main_required_of_128'] if args.num_envs == 128
                            else experiment['acceptance']['natural_repeat_required_of_20'] if args.num_envs == 20
                            else int(np.ceil(0.8 * args.num_envs)))
                gate = {'metric': 'safe10', 'required': required, 'count': summary['safe10_count'],
                        'passed': summary['safe10_count'] >= required and audit_ok}
            else:
                left = summary['recovery_by_direction']['left']
                right = summary['recovery_by_direction']['right']
                formal = args.num_envs == 100 and left['count'] == right['count'] == 50
                overall_required = (experiment['acceptance']['recovery_required_of_100'] if formal
                                    else int(np.ceil(0.7 * args.num_envs)))
                side_required = (experiment['acceptance']['recovery_each_direction_required_of_50'] if formal
                                 else int(np.ceil(0.6 * min(left['count'], right['count']))))
                gate = {'metric': 'recovery_success', 'required': overall_required,
                        'each_direction_required': side_required, 'count': summary['recovery_success_count'],
                        'passed': summary['recovery_success_count'] >= overall_required
                                  and left['success'] >= side_required and right['success'] >= side_required
                                  and summary['perturbation_missed_count'] == 0 and audit_ok}
        args.output.with_suffix('.contacts.json').write_text(json.dumps(physical_contacts, indent=2))
        if experiment and args.protocol == 'recovery':
            args.output.with_suffix('.forces.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in base.perturb_events))
        trajectory_path = None
        actuator_trace_summary = None
        if trajectory:
            trajectory_path = args.output.with_suffix('.trajectory.npz')
            trajectory_arrays = {key: np.stack([frame[key] for frame in trajectory]) for key in trajectory[0]}
            np.savez_compressed(trajectory_path, **trajectory_arrays, policy_dt=0.02, physics_dt=0.0025)
            active_rows = trajectory_arrays['active'].astype(bool)
            policy_action = trajectory_arrays['action'][active_rows]
            raw_action = trajectory_arrays['executed_action'][active_rows]
            limited_command = trajectory_arrays['actuator_command'][active_rows]
            motor_throttle = trajectory_arrays['motor_throttle'][active_rows]
            rotor_thrust = trajectory_arrays['rotor_thrust_local_z_n'][active_rows]
            sampled_skill_id = trajectory_arrays['skill_id'][active_rows].astype(np.int64)
            per_skill_action_trace = {}
            for skill_id, skill_name in ((0, 'launch'), (1, 'hit'), (2, 'recovery')):
                skill_rows = sampled_skill_id == skill_id
                if not skill_rows.any():
                    continue
                skill_raw = raw_action[skill_rows]
                skill_limited = limited_command[skill_rows]
                per_skill_action_trace[skill_name] = {
                    'samples': int(skill_rows.sum()),
                    'policy_action_abs_max_before_override': float(np.abs(policy_action[skill_rows]).max()),
                    'raw_action_abs_max': float(np.abs(skill_raw).max()),
                    'fraction_action_values_limited': float(
                        np.mean(np.abs(skill_raw - skill_limited) > 1e-7)
                    ),
                }
            actuator_trace_summary = {
                'schema': ('AeroWallActuatorTraceV3'
                           if args.aerowall_rotor_override_channel is not None
                           else 'AeroWallActuatorTraceV2'),
                'policy_action_abs_max_before_override': float(np.abs(policy_action).max()),
                'raw_action_abs_max': float(np.abs(raw_action).max()),
                'fraction_action_values_limited': float(np.mean(np.abs(raw_action - limited_command) > 1e-7)),
                'per_skill_action_trace': per_skill_action_trace,
                'limited_command_abs_max': float(np.abs(limited_command).max()),
                'motor_throttle_min': float(motor_throttle.min()),
                'motor_throttle_mean': float(motor_throttle.mean()),
                'motor_throttle_max': float(motor_throttle.max()),
                'rotor_thrust_local_z_n_mean_per_rotor': rotor_thrust.mean(axis=0).tolist(),
                'total_rotor_thrust_n_mean': float(rotor_thrust.sum(axis=-1).mean()),
                'total_rotor_thrust_n_min': float(rotor_thrust.sum(axis=-1).min()),
                'total_rotor_thrust_n_max': float(rotor_thrust.sum(axis=-1).max()),
                'sampled_active_control_steps': int(active_rows.sum()),
            }
        record(status="passed", hcsp_commit=subprocess.check_output(["git", "-C", str(HCSP), "rev-parse", "HEAD"], text=True).strip(),
               framework='AeroWall', policy_class=policy.__class__.__name__,
               checkpoint_sha256=checkpoint_sha, env_sha256=hashlib.sha256((ROOT / 'scripts/aerowall_wall_rally_env.py').read_bytes()).hexdigest(),
               environment_class='AeroWallSingleWallRallyEnv',
               reward_logic_sha256=hashlib.sha256((ROOT / 'scripts/aerowall_wall_reward_logic.py').read_bytes()).hexdigest(),
               physics_dt=float(cfg.sim.dt), policy_dt=float(cfg.sim.dt) * int(cfg.sim.substeps),
               summary=summary, outcomes=outcomes, first_episode_events=str(event_path),
               first_episode_event_count=len(first_episode_events), actor_warmstart=warmstart,
               gate=gate, contact_audit=contact_audit, diagnostic_trace=diagnostic_trace,
               reproducibility=reproducibility_record,
               initial_policy_input_trace=initial_policy_input_trace,
               post_contact_actor_input_trace=post_contact_actor_input_trace,
               state_intervention=({
                   'name': 'AeroWallLaunchTanhPostContactBallVelocityRestoreV1',
                   'target_bank_path': str(args.aerowall_postcap_ball_velocity_restore_bank),
                   'target_bank_sha256': ball_velocity_restore_payload['_runtime_bank_sha256'],
                   'selected_case_count': len(ball_velocity_restore_by_env),
                   'applied_case_count': len(ball_velocity_restore_trace),
                   'changed_physical_component': 'ball linear velocity xyz only',
                   'contact_history_sync': 'base.prev_ball_vel set to the injected post-step linear velocity to prevent a synthetic collision delta',
                   'next_state_refresh': 'base._compute_state_and_obs() updated the next TensorDict observation/state; current transition reward/done remained from the completed natural step',
                   'readback_tolerance_mps': 1e-5,
                   'per_case_trace': ball_velocity_restore_trace,
               } if ball_velocity_restore_payload is not None else None),
               launch_checkpoint=str(args.launch_checkpoint) if args.launch_checkpoint else None,
               candidate_name=args.candidate_name,
               candidate_training_report=(str(args.candidate_training_report)
                                          if args.candidate_training_report else None),
               candidate_training_report_sha256=candidate_training_report_sha256,
               candidate_actor_checkpoint_sha256=candidate_actor_checkpoint_sha256,
               hit_candidate_name=args.hit_candidate_name,
               hit_training_report=(str(args.hit_training_report) if args.hit_training_report else None),
               hit_training_report_sha256=hit_training_report_sha256,
               hit_checkpoint_sha256=hit_checkpoint_sha256,
               recovery_checkpoint_sha256=(hashlib.sha256(args.recovery_checkpoint.resolve().read_bytes()).hexdigest()
                                           if args.recovery_checkpoint and args.recovery_checkpoint.resolve().is_file()
                                           else None),
               recovery_checkpoint=str(args.recovery_checkpoint) if args.recovery_checkpoint else None,
               hit_checkpoint=str(args.hit_checkpoint) if args.hit_checkpoint else None,
               policy_route='AeroWallSkillChainPolicy' if args.skill_chain or args.train_skill else
                            'AeroWallLaunchRecoveryPolicy' if args.launch_checkpoint else 'HCSP_MAPPOPolicy',
               train_skill=args.train_skill, observation_version=args.observation_version,
               skill_observation_version=args.skill_observation_version,
               skill_observation_versions_by_role=getattr(policy, 'observation_versions', None),
               training_observation_versions_by_role=training_observation_versions_by_role,
               evaluation_observation_versions_by_role={
                   'launch': args.launch_observation_version,
                   'hit': args.hit_observation_version,
                   'recovery': args.recovery_observation_version,
               },
               observation_version_matches_training_by_role={
                   role: (None if trained_version is None else
                          trained_version == {
                              'launch': args.launch_observation_version,
                              'hit': args.hit_observation_version,
                              'recovery': args.recovery_observation_version,
                          }[role])
                   for role, trained_version in training_observation_versions_by_role.items()
               },
               allow_observation_version_ablation=args.allow_observation_version_ablation,
               launch_action_distribution=args.launch_action_distribution,
               hit_action_distribution=args.hit_action_distribution,
               recovery_action_distribution=args.recovery_action_distribution,
               training_action_distributions_by_role={
                   'launch': (candidate_training_report or {}).get('actor_action_distribution'),
                   'hit': (hit_training_report or {}).get('actor_action_distribution'),
                   'recovery': None,
               },
               evaluation_action_distributions_by_role={
                   'launch': args.launch_action_distribution,
                   'hit': args.hit_action_distribution,
                   'recovery': args.recovery_action_distribution,
               },
               action_mapping_ablation_name=args.action_mapping_ablation_name,
               action_mapping_ablation_roles=args.action_mapping_ablation_role or [],
               action_intervention=({
                   'name': 'AeroWallRotorCommandResponseCounterfactualV1',
                   'channel': args.aerowall_rotor_override_channel,
                   'delta': args.aerowall_rotor_override_delta,
                   'policy_steps': args.aerowall_rotor_override_steps,
                   'bank_sha256': AEROWALL_ROTOR_RESPONSE_BANK_SHA256,
                   'applied_env_step_count': len(action_intervention_trace),
                   'applied_step_indices': sorted({row['policy_step'] for row in action_intervention_trace}),
                   'per_step_effective_command_trace': action_intervention_trace,
               } if args.aerowall_rotor_override_channel is not None else None),
               action_mapping_overrides={
                   name: 'HCSP.TanhNormalWithEntropy(tanh_loc=True)'
                   for name, distribution in (
                       ('launch', args.launch_action_distribution),
                       ('hit', args.hit_action_distribution),
                       ('recovery', args.recovery_action_distribution),
                   ) if distribution == 'tanh'
               },
               action_mapping_experiment=(args.action_mapping_ablation_name or (
                   ('AeroWall' + 'And'.join(
                       label for _, label, distribution in (
                           ('launch', 'Launch', args.launch_action_distribution),
                           ('hit', 'Hit', args.hit_action_distribution),
                           ('recovery', 'Recover', args.recovery_action_distribution),
                       ) if distribution == 'tanh'
                   ) + 'TanhMappingAblationV1'
                    if any(distribution == 'tanh' for distribution in (
                        args.launch_action_distribution, args.hit_action_distribution,
                        args.recovery_action_distribution
                    )) else None)
                   if not args.candidate_name and not args.train_skill else None
               )),
               actor_distribution_by_skill={
                   'launch': ('HCSP.TanhNormalWithEntropy(tanh_loc=True)'
                              if args.launch_action_distribution == 'tanh'
                              else 'HCSP default (IndependentNormal)'),
                   'recovery': ('HCSP.TanhNormalWithEntropy(tanh_loc=True)'
                                if args.recovery_action_distribution == 'tanh'
                                else 'HCSP default (IndependentNormal)'),
                   'hit': ('HCSP.TanhNormalWithEntropy(tanh_loc=True)'
                           if args.hit_action_distribution == 'tanh'
                           else 'HCSP default (IndependentNormal)'),
               },
               reward_design=args.reward_design, hit_window=args.hit_window,
               hit_exit_window=args.hit_exit_window, min_dwell_steps=args.min_dwell_steps,
               handoff_capture=handoff_capture,
               handoff_evaluation=handoff_evaluation,
               actuator_trace_schema=(('AeroWallActuatorTraceV3'
                                        if args.aerowall_rotor_override_channel is not None
                                        else 'AeroWallActuatorTraceV2')
                                       if trajectory_path else None),
               actuator_trace_summary=actuator_trace_summary,
               trajectory=str(trajectory_path) if trajectory_path else None,
               rgb={'directory': str(rgb_directory), 'frames': rgb_frames, 'fps': 25,
                    'initial_frame_time': 0., 'representation': 'HCSP simulator RGB'} if args.record_rgb else None,
               elapsed_seconds=time.monotonic() - started)
    except Exception as exc:
        record(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
