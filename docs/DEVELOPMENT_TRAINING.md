# SingleJuggle development training

The first sustained development run uses the pinned upstream SingleJuggle shell configuration, including CTBR, dt=0.02, restitution randomization and the original PPO implementation. Only parallelism (512), logging (local), seed and per-run interaction budget are changed. The verified project PID reset adapter is applied. This is pretraining/development evidence; its transitions must remain in the budget ledger for any later method that inherits its weights.

Command:

```sh
./scripts/run_probe.sh scripts/train_single_juggle.py runs/singlejuggle-dev-001.json --updates 100 --num-envs 512 --seed 0 >runs/singlejuggle-dev-001.log 2>&1
```

100 updates × 512 environments × 64 steps = 3,276,800 environment transitions. Initial diagnostic PPO runs are separate deployment experiments and are not loaded into this policy. Save an actual learned checkpoint after the first update, every 50 updates, and the last update. Each checkpoint includes policy, both optimizers, value normalization (within policy state), cumulative frame/update counts, Python/NumPy/Torch/CUDA RNG, source hashes and the resolved learning-configuration hash. JSONL metrics retain every update and completed-episode upstream statistics. Upstream num_true_hits remains a cooldown-based diagnostic proxy, not the plan's contact-event success score.

Resume with --resume PATH starts a new output/run directory and adds the checkpoint's prior frames to the new frames. It checks that the resolved learning configuration is unchanged apart from execution budget/log intervals. It restores learning and RNG state but initializes fresh simulator episodes; it does not claim exact trajectory continuation. Resume requires a real runtime verification before being described as tested.

Next evaluation must replay an unchanged checkpoint on 100 fixed development initial states, count legal ball/bat contact entries rather than upstream cooldown hits, and export real state/contact/action trajectories for the four-phase action analysis. Do not declare 80% five-juggle success from a training reward or averaged upstream hit count. WallRally reward design and attitude constraints remain downstream of that analysis.

## First completed batch

singlejuggle-dev-001 completed 100 updates / 3,276,800 transitions in 295.432 training-loop seconds (11,091.6 transitions/second). The last checkpoint is checkpoints/singlejuggle-dev-001/frames-000003276800.pt. The final batch's mean upstream hit proxy was 0.579, not evidence of five-juggle competence. Fixed first-episode evaluation and policy improvement remain required.

The raw air.usd bat collider is a Z cylinder with radius 0.05 m and height 0.11 m, identity local transform. The visual mesh uses a different shape/offset. Evaluation reads collider dimensions from the composed runtime stage rather than assuming the visible mesh or upstream reward proximity radius describes collision geometry.

Evaluator command (use a new report name on retries):

```sh
./scripts/run_probe.sh scripts/evaluate_single_juggle.py runs/singlejuggle-eval-002.json --checkpoint checkpoints/singlejuggle-dev-001/frames-000003276800.pt --config runs/singlejuggle-dev-001.yaml --scenarios artifacts/singlejuggle-development-scenarios-100.json >runs/singlejuggle-eval-002.log 2>&1
```

The first run writes a seeded 100-scenario initial-state roster; subsequent runs require an exact roster match. Evaluation counts only the first episode of each scenario. Physics contact FOUND/LOST events maintain pair state to deduplicate entries; the top-cap classifier remains provisional until physical side/bottom/rotor fixtures are checked. It exports all scenarios' actual states/actions as NPZ and contact evidence as JSONL, including ball velocities before/after impact and bat contact-point velocity with angular contribution. These are development results, not the final 200-scenario wall-task test suite.

Resume verification: singlejuggle-dev-002 resumes frames-000003276800.pt and successfully trains from total update 100 to 110 with cumulative frames 3,604,480. Both optimizer states and configuration compatibility are loaded by the runner. The run adds 900 updates (29,491,200 transitions), targeting 32,768,000 cumulative transitions; it is ongoing, not completed. First resumed-update checkpoint frames-000003309568.pt is saved.

## Aligned body-enabled curriculum (development only)

`scripts/train_aligned_juggle.py` and `aerowall/envs/aligned_juggle.py` retain original observations/CTBR but replace velocity-proxy rewards with lifecycle-qualified physical cap rewards and illegal-contact termination. Dense terms scale with dt; discrete cap/illegal rewards do not. No global tilt constraint or in-flight state rewriting is used. A wall contact terminates this juggling-only curriculum; this is not a WallRally score.

smoke-01: 16 environments, 2 updates, 2,048 frames, 47 cap credits/21 illegal/38 resets, passed. smoke-128-01: 128 environments, 10 updates, 81,920 frames, 2,356 cap credits/393 illegal/586 resets, passed. resume-01: restores checkpoint, optimizer and RNG with fresh physics episodes, 2 further updates/16,384 frames, passed. Total frame accounting includes the original 32,768,000-frame policy; final resume count is 32,866,304. The separate 16-env smoke branch is not part of that checkpoint lineage.

Legacy collector terminal stats show zeros for illegal reasons despite direct contacts; they are explicitly unvalidated. New direct terminal records are captured in the environment before reset. In resume-01 batches, collector done counts equal direct episode records (75 and 60); illegal reasons are 71 and 55, totaling exactly 126 direct illegal contacts. Boundary counts are 4 and 5. Reports retain both fields for diagnosis. No sustained-success claim follows from training statistics.

All 17 existing archive/event contract tests pass. Lazy/eager fixed evaluation arrays and per-scenario outcomes agree exactly (docs/aligned-readback-comparison.json). Aligned policy development baseline remains 46/100 five-cap successes until an independent fixed-roster evaluation proves otherwise.
