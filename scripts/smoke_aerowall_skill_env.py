"""Run the bounded T4 environment-contract smoke in the pinned Isaac runtime."""

import argparse
import gc
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HCSP = ROOT / "third_party/HCSP"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


def snapshot_state(state):
    import torch

    return {key: value.detach().cpu().clone()
            for key, value in state.items()
            if isinstance(value, torch.Tensor)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=6201)
    parser.add_argument("--observation-version", choices=["relative_v3", "aerowall_goal_v1"])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "initializing", "framework": "AeroWall",
              "environment_class": "AeroWallSingleWallRallyEnv",
              "num_envs": args.num_envs, "seed": args.seed}

    def record(**values):
        report.update(values)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(values), flush=True)

    app = None
    base = None
    env = None
    policy_a = None
    policy_b = None
    record()
    try:
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        sys.path.insert(0, str(HCSP))
        sys.path.insert(0, str(ROOT / "scripts"))
        from hcsp import init_simulation_app
        from aerowall.wall_rl.upgrade_config import load_upgrade_config, configured_checkpoint

        config = load_upgrade_config(ROOT / "configs/wall_skill_upgrade_v3.json", ROOT)
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        with initialize_config_dir(version_base=None, config_dir=str(HCSP / "cfg")):
            cfg = compose(config_name="train", overrides=[
                "task=Serve", "headless=true", "wandb.mode=disabled",
                f"task.env.num_envs={args.num_envs}", f"seed={args.seed}",
            ])
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)
        cfg.sim.dt = 0.0025
        cfg.sim.substeps = 8
        cfg.env.max_episode_length = cfg.task.env.max_episode_length = 64
        cfg.task.drone_model = "IrisTest"
        cfg.task.action_transform = None
        cfg.task.wall_curriculum_stage = "RALLY"
        observation_version = args.observation_version or config["training"]["skill_observation_version"]
        cfg.task.wall_observation_version = observation_version
        cfg.task.wall_reward_design = config["training"]["reward_version"]
        cfg.task.wall_case_mode = "train"
        cfg.task.wall_train_skill = "hit"
        cfg.task.wall_hit_artificial_ratio = config["reset"]["artificial_ratio"]
        cfg.task.wall_hit_fixed_incoming = bool(config["training"].get("fixed_incoming", False))
        cfg.task.wall_goal_y_bounds = config["training"].get("wall_goal_y_bounds_m", [-0.3, 0.3])
        cfg.task.wall_goal_z_bounds = config["training"].get("wall_goal_z_bounds_m", [3.8, 4.2])
        cfg.task.wall_hit_enter_seconds = config["fsm"]["hit_enter_seconds"]
        cfg.task.wall_hit_exit_seconds = config["fsm"]["hit_exit_seconds"]
        cfg.task.wall_min_dwell_steps = config["fsm"]["min_dwell_steps"]
        cfg.task.wall_restitution = config["trajectory"]["restitution_x"]
        cfg.task.wall_y_bounds = config["trajectory"]["wall_y_bounds_m"]
        cfg.task.wall_z_bounds = config["trajectory"]["wall_z_bounds_m"]
        cfg.task.wall_contact_height = config["trajectory"]["contact_height_m"]
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        sys.argv = [sys.argv[0], "--portable", "--portable-root", str(ROOT / ".cache/kit")]
        app = init_simulation_app(cfg)
        from hcsp_offline_assets import configure_local_asset_root
        configure_local_asset_root()
        from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
        from aerowall_wall_rally_env import AeroWallSingleWallRallyEnv
        from aerowall_policy_encoder import configure_policy_encoder
        configure_policy_encoder()
        from hcsp.learning import MAPPOPolicy

        base = AeroWallSingleWallRallyEnv(cfg, headless=True)
        env = TransformedEnv(base, Compose(InitTracker())).train()
        env.set_seed(args.seed)
        initial = env.reset()
        assert base.num_envs == args.num_envs
        assert initial["agents", f"{observation_version}_observation"].shape == (
            args.num_envs, 1, 48 if observation_version == "aerowall_goal_v1" else 46,
        )
        assert torch.isfinite(initial["agents", "observation"]).all()
        assert torch.isfinite(initial["stats", "failure_reason_bits"]).all()

        skill_before = base.skill_id.clone()
        held_before = base.skill_held_steps.clone()
        for _ in range(3):
            base._compute_state_and_obs()
        observation_is_pure = (torch.equal(skill_before, base.skill_id)
                               and torch.equal(held_before, base.skill_held_steps))

        if args.num_envs < 2:
            raise ValueError("selective-reset smoke needs at least two environments")
        base.skill_id[1] = 2
        base.skill_held_steps[1] = 7
        base.caps[1] = 4
        base.episode_id[1] = 19
        before_other = (base.skill_id[1].clone(), base.skill_held_steps[1].clone(),
                        base.caps[1].clone(), base.episode_id[1].clone())
        base._reset_idx(torch.tensor([0], device=base.device))
        selective_reset_isolated = all(torch.equal(before, after) for before, after in zip(
            before_other, (base.skill_id[1], base.skill_held_steps[1], base.caps[1], base.episode_id[1])))

        source_checkpoint = configured_checkpoint(config, "c350_recovery", ROOT)
        from aerowall.wall_rl.policy_specs import agent_spec_for_observation_dim
        checkpoint_spec = agent_spec_for_observation_dim(env.agent_spec["drone"], 46)
        policy_a = MAPPOPolicy(cfg.algo, agent_spec=checkpoint_spec, device=base.device)
        policy_a.load_state_dict(torch.load(source_checkpoint, map_location=base.device))
        checkpoint_state = snapshot_state(policy_a.state_dict())
        temp_checkpoint = args.output.with_suffix(".reload.pt")
        torch.save(policy_a.state_dict(), temp_checkpoint)
        policy_b = MAPPOPolicy(cfg.algo, agent_spec=checkpoint_spec, device=base.device)
        policy_b.load_state_dict(torch.load(temp_checkpoint, map_location=base.device))
        checkpoint_reload_exact = all(torch.equal(value, snapshot_state(policy_b.state_dict())[key])
                                      for key, value in checkpoint_state.items())
        temp_checkpoint.unlink(missing_ok=True)

        counters = {"control_step_calls": 0, "apply_action_calls": 0}
        pre_step = base._pre_sim_step
        apply_action = base.drone.apply_action

        def count_pre_step(td):
            counters["control_step_calls"] += 1
            return pre_step(td)

        def count_apply_action(action):
            counters["apply_action_calls"] += 1
            return apply_action(action)

        base._pre_sim_step = count_pre_step
        base.drone.apply_action = count_apply_action
        td = env.reset()
        td["agents", "action"] = torch.zeros(args.num_envs, 1, 4, device=base.device)
        with torch.no_grad():
            result = env.step(td)["next"]
        reward_finite = bool(torch.isfinite(result["agents", "reward"]).all())
        component_keys = [key for key in result["stats"].keys() if str(key).startswith("reward_")]
        components_finite = all(torch.isfinite(result["stats", key]).all() for key in component_keys)
        motor_cadence_correct = (counters["control_step_calls"] == 1
                                 and counters["apply_action_calls"] == int(cfg.sim.substeps))
        checks = {
            "observation_read_is_pure": observation_is_pure,
            "selective_reset_isolated": selective_reset_isolated,
            "checkpoint_reload_exact": checkpoint_reload_exact,
            "reward_finite": reward_finite,
            "reward_components_finite": components_finite,
            "motor_control_cadence_1_plus_7": motor_cadence_correct,
        }
        record(status="passed" if all(checks.values()) else "failed",
               hcsp_commit=__import__("subprocess").check_output(
                   ["git", "-C", str(HCSP), "rev-parse", "HEAD"], text=True).strip(),
               upgrade_config_sha256=hashlib.sha256(
                   (ROOT / "configs/wall_skill_upgrade_v3.json").read_bytes()).hexdigest(),
               source_checkpoint_sha256=hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
               environment_sha256=hashlib.sha256(
                   (ROOT / "scripts/aerowall_wall_rally_env.py").read_bytes()).hexdigest(),
               counters=counters, reward_component_keys=component_keys,
               events_observed=len(base.events), checks=checks)
        if not all(checks.values()):
            raise RuntimeError(checks)
    except Exception as exc:
        record(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if app is not None:
            policy_b = None
            policy_a = None
            if "checkpoint_spec" in locals():
                checkpoint_spec = None
            gc.collect()
            if env is not None:
                env.close()
                env = None
            elif base is not None:
                base.close()
            del env, base
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()
            app.close()


if __name__ == "__main__":
    main()
