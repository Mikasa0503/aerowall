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

## First longer aligned training interruption

aligned-juggle-dev-001 terminates with a strict ledger error after 76 completed updates (622,592 additional frames): PERSIST without FOUND for environment 113 ball/rotor_1. Last durable checkpoint is update 75 at 33,480,704 lineage frames; update 76 and the unfinished rollout are not in that checkpoint. Preserve this failed run when accounting total development compute. There is no claim that all 100 requested updates finished.

The error does not establish its cause. Router diagnostics now include physics progress, reads since each environment reset and active pairs. No ledger acceptance rule was relaxed. aligned-juggle-dev-002 resumes the durable checkpoint with these diagnostics and new physics episodes; a separate frozen-roster evaluation tests the durable checkpoint. Episode-reset manifold persistence remains a hypothesis to verify, not an implemented exception.

## Verified reset boundary cause and scoring alignment

Diagnostic dev-002 fails with env=51, reads_since_reset=0, progress=1, edge=PERSIST, impulse=0, empty active ledger. The physical-contact trained evaluator independently encounters the same first-step condition in env=69. These observations establish that PhysX may retain a manifold across episode reset. The ledger now accepts an unknown PERSIST only on the first fetched step of a new episode epoch. It opens an uncredited interval; zero impulse still produces no impact. Positive current contact is classified normally, illegal priority remains, and missing FOUND later in an episode still raises. Raw event edges remain PERSIST with an explicit episode_reset_reentry marker. A cumulative diagnostic counter is recorded in training. The new unit regression covers zero impulse, delayed positive impulse, single credit, ordinary strict failure and illegal first-step contact; all 18 contract tests pass. dev-003 tests this correction from the durable dev-002 checkpoint.

Legacy-boundary evaluation of the 33,480,704-frame checkpoint scores 47/100, with 47 body failures, 40 upstream terminations, 2 non-cap failures and 11 time limits. Because original frequency-based wrong_hit termination is incompatible with the new physical-contact curriculum, it must not serve as the final curriculum evaluator. `--contact-task` now evaluates both policies under identical actual-contact curriculum boundaries. Original checkpoint baseline under those rules is 44/100 (67 body failures, 3 boundary failures, 30 time limits). Keep legacy 46/100 and contact-task 44/100 as separate baselines; do not combine results across episode definitions. Trained contact-task evaluation is rerun after the evidenced reset correction; the failed first attempt is retained.

Trained contact-task eval-trained-02 completes after the reset correction: 48/100 reach five legal cap credits, with 66 body, 9 non-cap, 8 boundary terminations and 17 time limits. Baseline contact-task is 44/100 and 30 time limits. The four-scenario increase is not a robust recovery improvement; sustained 10-second survival is lower. The 80% development gate remains unmet. At reporting, dev-003 remains active; no claim that its 100-update budget has completed.

## Completed corrected long run

aligned-juggle-dev-003 completes all 100 updates (819,200 transitions), final checkpoint frames-000034308096.pt. It handles observed first-post-reset PERSIST events using the validated epoch rule, rather than disabling strict lifecycle errors. Fixed contact-task eval-trained-03 reaches five legal caps in 59/100: 45 body, 12 non-cap, 7 boundary and 36 time-limit outcomes. Same-rule original baseline is 44/100 with 30 time limits. This is development improvement, not the 80% gate, statistical multi-seed evidence, or a WallRally result. Do not conflate this policy's lineage budget with all consumed development branches and failed rollouts.

Current training uses 20 ms physics. Fast tilted fixtures now reject that step size for wall-task physics. Before scaling wall training, separate physics substeps from the original 50 Hz policy/controller timing and independently re-evaluate; do not silently change action frequency or carry over old gates.

## Physics substeps with original control timing

AlignedJuggle now advances contacts after every physical step while the policy and PID remain at 0.02 s. Motor commands are held between controller calls; rotor response and aerodynamic forces update at physics dt using current body state. Policy-rate finite-difference diagnostics use 0.02 s, and dense reward scales by policy time. Legal contacts are accumulated chronologically: an illegal event suppresses cap credit in that physical step and later substeps, while a prior legal physical event is retained. First physical failure reason is latched. Raw events carry physical substep/time and optional contact-time kinematics. No trajectory state is rewritten outside reset.

The single-substep regression (eval-substep1-02) matches every old saved trajectory array and all 100 outcomes exactly: 59/100 five-cap success, 36 time limits. The added NPZ physics_dt key is metadata. Clock audit: 500 policy/controller calls and 500 physics steps for 10 s.

With the same 34,308,096-frame policy and fixed roster, eval-substep8-01 uses 2.5 ms physics, 500 policy/controller calls and 4,000 physics steps. It reaches five legal caps in 86/100: 26 body, 18 non-cap, 17 boundary failures and 39 time limits. This meets the five-cap development count on this roster, not sustained wall-rally success or formal multi-seed evidence. Independent event audit checks all 8,286 raw events, 2,172 legal credits, event timestamps and terminal per-scenario counts.

16-env training smoke and checkpoint-resume tests each complete two PPO updates. Resume records 128 policy/controller calls, 1,024 physics steps, 2,048 environment transitions, 73 cap credits and 6 illegal contacts. Changing physics dt is an explicit transfer, not silent resume: learning_config_hash includes dt/substeps. Physics steps do not inflate the RL environment-transition budget.

The first evaluation attempt exposed a bat COM array shape mismatch in diagnostic kinematics and was preserved as failed; the reshape/batched quaternion correction is verified by successful evaluation. A 1.25 ms closed-loop reference and original-checkpoint 2.5 ms evaluation are in progress before selecting the curriculum checkpoint.

The trained-policy 1.25 ms reference completes at 86/100, identical aggregate five-cap success to 2.5 ms, with 39 time limits. It uses exactly 500 policy/controller calls and 8,000 physics steps. Per-scenario failures need not agree: 31 body, 11 non-cap, 19 boundary. Both substep event/timing audits pass.

The original 32,768,000-frame checkpoint on the same 2.5 ms body-enabled scene scores 96/100 with 95 time limits, four body failures and one boundary failure. Therefore the original checkpoint is the preferred development curriculum initializer; coarse-step fine-tuning is not selected merely because it used more compute. Its 1.25 ms reference is running before closing this comparison. This is selection on a development set, not formal multi-seed evidence.

Original-checkpoint reference completes: at 1.25 ms it scores 95/100 five-cap successes, 94 time limits, two boundary, two body and two non-cap outcomes. The corresponding 2.5 ms results are 96/100 and 95 time limits. Clock/event audit also passes for the reference. The original checkpoint and 2.5 ms physics / 20 ms policy-PID interval are selected for the next development single-wall-return curriculum, retaining 1.25 ms as the reference. This satisfies the fixed-development juggling target under the tested scene; it does not close wall-return, broad collision accuracy, formal comparisons or stress-test requirements.
