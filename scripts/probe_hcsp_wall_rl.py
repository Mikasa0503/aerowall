"""Run initialization-only contact fixtures against HCSPSingleWallRL."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HCSP = ROOT / "third_party/HCSP"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True, type=Path); a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "initializing", "pid": os.getpid()}
    def record(**values):
        report.update(values); a.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(values), flush=True)
    app = None; record()
    try:
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        sys.path.insert(0, str(HCSP))
        from hcsp import init_simulation_app
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        with initialize_config_dir(version_base=None, config_dir=str(HCSP / "cfg")):
            cfg = compose(config_name="train", overrides=["task=Serve", "headless=true", "wandb.mode=disabled", "task.env.num_envs=5", "seed=20260921"])
        OmegaConf.resolve(cfg); OmegaConf.set_struct(cfg, False)
        cfg.sim.dt = 0.0025; cfg.sim.substeps = 8; cfg.env.max_episode_length = cfg.task.env.max_episode_length = 100
        cfg.task.drone_model = "IrisTest"; cfg.task.action_transform = None
        sys.argv = [sys.argv[0], "--portable", "--portable-root", str(ROOT / ".cache/kit")]
        app = init_simulation_app(cfg)
        from hcsp_offline_assets import configure_local_asset_root
        configure_local_asset_root()
        from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
        from hcsp_wall_rl_env import HCSPSingleWallRL
        base = HCSPSingleWallRL(cfg, headless=True)
        env = TransformedEnv(base, Compose(InitTracker())).eval(); env.set_seed(20260921)
        with torch.no_grad():
            td = env.reset(); base.set_probe_states()
            action = torch.full((5, 1, 4), 0.24, device=base.device)
            outcomes = [None] * 5
            for step in range(100):
                td[("agents", "action")] = action
                nxt = env.step(td)["next"]
                for i in torch.nonzero(nxt["done"].flatten()).flatten().cpu().tolist():
                    if outcomes[i] is None:
                        outcomes[i] = {"step": step + 1, "failure": int(base.failure[i]), "phase": int(base.phase[i]), "caps": int(base.caps[i]), "walls": int(base.walls[i])}
                td = nxt
            for i in range(5):
                if outcomes[i] is None: outcomes[i] = {"step": 100, "failure": int(base.failure[i]), "phase": int(base.phase[i]), "caps": int(base.caps[i]), "walls": int(base.walls[i])}
        checks = {
            "center_is_legal_cap": any(e["env"] == 0 and e["legal_cap"] for e in base.events),
            "edge_is_legal_cap": any(e["env"] == 1 and e["legal_cap"] for e in base.events),
            "outside_racket_is_not_legal": not any(e["env"] == 2 and e["legal_cap"] for e in base.events),
            "wall_reverses_ball": any(e["env"] == 3 and e["wall"] for e in base.events),
            "ground_terminates": outcomes[4]["failure"] == 1,
        }
        record(status="passed" if all(checks.values()) else "failed", checks=checks, outcomes=outcomes, events=base.events,
               hcsp_commit=subprocess.check_output(["git", "-C", str(HCSP), "rev-parse", "HEAD"], text=True).strip(),
               env_sha256=hashlib.sha256((ROOT / "scripts/hcsp_wall_rl_env.py").read_bytes()).hexdigest(), physics_dt=cfg.sim.dt, control_dt=cfg.sim.dt * cfg.sim.substeps)
        if not all(checks.values()): raise RuntimeError(checks)
    except Exception as exc:
        record(status="failed", error=repr(exc), traceback=traceback.format_exc()); raise
    finally:
        if app is not None: app.close()


if __name__ == "__main__": main()
