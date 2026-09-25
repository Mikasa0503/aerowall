# AeroWall V6 Same-Bank Launch Tanh Replication V1

**Pre-registration date:** 2026-09-25

**Status:** Frozen paired no-training development evaluation.

**Purpose:** Re-evaluate the V6 Launch Tanh mapping on the exact original V6 heldout development bank with current RNG and reset-state reproducibility recording. This is not a final-test or P2 acceptance run.

## Question

Does changing only the frozen V6 Launch action distribution from HCSP IndependentNormal to HCSP TanhNormalWithEntropy(tanh_loc=True) reduce actuator clipping while improving task-level continuation without increasing safety failures?

## Frozen inputs

- Case bank: configs/wall_cases/heldout-128-v4.json, SHA256 666709adb50f350161a51112695909fa0cf304d829685be41b35ae2f8cfe8f80. This remains a development bank, not the final frozen test set.
- Evaluation: 128 environments, seed 9524, natural RALLY, deterministic actor evaluation.
- Launch checkpoint: artifacts/wall-skill-upgrade-v3/launch-diagnostics/aerowall-lateral-intercept-v1-causal-v6-s6101.launch.pt, SHA256 165fd2a0eabf43082df62a1f5d458cc4f528f51d02cffa389f32e8bd6671dd8b.
- Hit and Recover checkpoints: checkpoints/c-u350.pt, SHA256 4e31d16bbe2f3cd82ee6d25a519ed3650b17b91f91a1d40391f9310f0407a659.
- Environment evaluator SHA256 at freeze: scripts/evaluate_aerowall_wall_rl.py = 1f4c5ca912a51d30d10adece9a23069a7e013db413c3f3a8b72b2148edc746c5; environment SHA256 = 2d7355fe2db5b7498e80e0ea1b63825420465c0d16ba27786d0d048793b6655c; policy SHA256 = cf23c6caa9989755ff41b2bf6e163a3a51b1296c87c061f8c905f584e74eba92; reward logic SHA256 = 0ec90fda95baec92cba313b0c37511a52ef820bfbd620b9aa1ba710f9312f1d7.
- Observation and reward: role-aligned observation versions bound to the same checkpoint reports, all three frozen roles use relative_v3; legacy evaluation reward. FSM, hit and exit windows, dwell, episode horizon, simulator, and all other settings remain fixed.
- Control: default action distribution for Launch, Hit, and Recover.
- Treatment identity: AeroWallLaunchTanhSameBankReplicationV1; only Launch uses HCSP TanhNormalWithEntropy(tanh_loc=True). Hit and Recover remain default. HCSP/upstream implementation is unchanged.
- Both arms record complete trajectories, outcomes, actuator traces, PhysX contact audits, reset state, RNG fingerprints, and source/checkpoint hashes.

## Paired endpoints and decision rule

Record legal first contacts, legal second hits, 3-rally and 5-rally counts, rally histogram, each safety failure class, out-of-bounds outcomes, aggregate and per-role action clipping, contact-audit totals, and all artifact/source/checkpoint hashes. Confirm identical post-reset state, RNG state, case IDs, environment/reward code, observations, and non-Launch settings.

A Launch Tanh replication signal requires both legal second-hit count and 3-rally count to exceed control, with safety failures no higher than control. Clipping reduction or first-contact improvement alone is a diagnostic signal and does not meet this task-level rule. Any mismatch in paired reset/RNG/source checks makes the run inconclusive. No case may be removed or replaced after results are seen.

## Guardrails

- No training, checkpoint changes, promotion, formal C350 route change, or use as a final test.
- This replication checks robustness of the original same-bank result; the separate new-bank replication remains relevant and will be reported alongside it.
- Do not change thresholds or endpoints after opening results.

## Artifacts

Reserved prefix: artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-v6-samebank-launch-tanh-replication-v1.

## Execution record (2026-09-25)

- Both valid arms completed with status passed on the frozen bank/checkpoints/source. Paired reset-state, reset fields, Python/NumPy/Torch CPU/Torch CUDA RNG fingerprints, initial observations, environment/reward source hashes, role observations, and case IDs all matched.
- Control: 110/128 legal first contacts, 5/128 legal second hits, 0/128 three-rallies, safety failures 39/128, aggregate raw action clipping 29.96%.
- Launch Tanh: 119/128 legal first contacts, 5/128 legal second hits, 0/128 three-rallies, safety failures 33/128, aggregate raw action clipping 16.19%. Illegal contact changed 38→31; drone-wall changed 1→2; out-of-bounds stayed 19; ball-ground changed 70→76.
- PhysX: control 117/117 legal and 108/108 wall events corroborated; Tanh 127/127 legal and 106/106 wall events corroborated; no callback errors.
- The preregistered task-level rule failed because legal second hits and three-rallies did not exceed control. Do not launch an additional training run from this replication or promote a candidate. The previously trained AeroWallBoundedInterceptV1-CausalV6 remains an existing, separately named candidate; this replication does not justify retraining it.
- Valid reports: control SHA256 06be00b171ab29e2cd22290c690cd83210a345cbbdb1d2801ece115407d15c3a; Tanh SHA256 25801acc329dfcf2ac0726c77e1aa7946ace3d4d8079529338b39d991a2c73c6. Machine analysis SHA256 f7733dbeb1cce650057d2284f7886b6176d6e4952759be06428506401de7f71f.
- Two earlier control-only attempts had mismatched role observations and are excluded. Their reports and companion artifacts are preserved in artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/invalid-comparisons/.
