# SingleJuggle development training

The first sustained development run uses the pinned upstream SingleJuggle shell configuration, including CTBR, dt=0.02, restitution randomization and the original PPO implementation. Only parallelism (512), logging (local), seed and per-run interaction budget are changed. The verified project PID reset adapter is applied. This is pretraining/development evidence; its transitions must remain in the budget ledger for any later method that inherits its weights.

Command:

```sh
./scripts/run_probe.sh scripts/train_single_juggle.py runs/singlejuggle-dev-001.json --updates 100 --num-envs 512 --seed 0 >runs/singlejuggle-dev-001.log 2>&1
```

100 updates × 512 environments × 64 steps = 3,276,800 environment transitions. Initial diagnostic PPO runs are separate deployment experiments and are not loaded into this policy. Save an actual learned checkpoint after the first update, every 50 updates, and the last update. Each checkpoint includes policy, both optimizers, value normalization (within policy state), cumulative frame/update counts, Python/NumPy/Torch/CUDA RNG, source hashes and the resolved learning-configuration hash. JSONL metrics retain every update and completed-episode upstream statistics. Upstream num_true_hits remains a cooldown-based diagnostic proxy, not the plan's contact-event success score.

Resume with --resume PATH starts a new output/run directory and adds the checkpoint's prior frames to the new frames. It checks that the resolved learning configuration is unchanged apart from execution budget/log intervals. It restores learning and RNG state but initializes fresh simulator episodes; it does not claim exact trajectory continuation. Resume requires a real runtime verification before being described as tested.

Next evaluation must replay an unchanged checkpoint on 100 fixed development initial states, count legal ball/bat contact entries rather than upstream cooldown hits, and export real state/contact/action trajectories for the four-phase action analysis. Do not declare 80% five-juggle success from a training reward or averaged upstream hit count. WallRally reward design and attitude constraints remain downstream of that analysis.
