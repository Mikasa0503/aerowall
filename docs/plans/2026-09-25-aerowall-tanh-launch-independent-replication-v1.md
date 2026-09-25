# AeroWall Launch Tanh Independent Replication V1

**Pre-registration date:** 2026-09-25

**Status:** Frozen no-training development replication; not a P2 acceptance test or final frozen test.

**Purpose:** Part 5 action mapping and Part 13 independent evidence.

## Question

Does the prior Launch-only TanhNormal mapping signal replicate on a new 128-case development bank when all process RNGs are seeded, with the same frozen checkpoints and settings?

## Frozen inputs

- Case builder: `scripts/build_aerowall_wall_cases.py --kind heldout --seed 260925 --count 128`.
- Development bank: `configs/wall_cases/devrep-aerowall-tanh-launch-v1-seed-260925.json`, SHA256 `b6ebf13209ba42b1ad77eb452248b6f743c0f0fbcc32805d250fbc162b3804da`. This is not a final test bank.
- Evaluation seed: 260925; 128 environments; natural RALLY; deterministic actor evaluation.
- Launch checkpoint: V6 `aerowall-lateral-intercept-v1-causal-v6-s6101.launch.pt`, SHA256 `165fd2a0eabf43082df62a1f5d458cc4f528f51d02cffa389f32e8bd6671dd8b`.
- Hit and Recover checkpoint: `checkpoints/c-u350.pt`, SHA256 `4e31d16bbe2f3cd82ee6d25a519ed3650b17b91f91a1d40391f9310f0407a659`.
- Observation: `relative_v3`; reward: `legacy`; hit window 0.18 s; exit window 0.18 s; minimum dwell 2 steps.
- HCSP distribution: `IndependentNormal` for control roles and HCSP `TanhNormalWithEntropy(tanh_loc=True)` only for the Launch treatment. HCSP implementation is reused without modification. Experiment identity: `AeroWallLaunchTanhIndependentReplicationV1`.
- Evaluator source SHA256 after adding `random.seed(args.seed)`: `ba24ca57befddf42b768121020c01e2f11a856e27d989f3e070e3ebd062a8502`.

## Paired arms

Run control first with default Launch distribution. Run treatment with only Launch action distribution set to Tanh. Keep Hit and Recover distributions, all checkpoints, FSM, environment, initial case bank, reward, observations, episode horizon, and every other evaluator parameter unchanged. Record reproducibility fingerprints and full trajectories, outcomes, action limiting, events, and PhysX contacts for both arms.

## Endpoints and decision rule

Report paired legal first contacts, legal second hits, 3-rally and 5-rally counts, rally histogram and mean, each safety failure code, ball boundary exits, total and per-role actuator clipping, all contact-audit totals, and source/checkpoint/environment hashes. Verify matching post-reset state and Python, NumPy, Torch, and CUDA RNG fingerprints across arms.

A task-level replication signal requires both legal second-hit and 3-rally counts to exceed control, with safety failures no higher than control. Any safety regression rejects the mapping for candidate use. A positive result remains development evidence only; it cannot pass P2, promote a policy, or authorize P3-P6 without the plan-level multi-seed and final-test requirements.

## Guardrails

- No training, checkpoint changes, policy promotion, formal C350 route change, or final test use.
- Do not tune settings, remove cases, or change thresholds after opening results.
- Preserve every adverse event and callback error. If RNG or initial-state equivalence fails, mark the replication inconclusive and do not replace cases.
