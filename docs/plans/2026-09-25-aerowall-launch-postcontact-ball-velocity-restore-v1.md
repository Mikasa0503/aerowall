# AeroWall Launch Post-contact Ball Velocity Restore V1

**Protocol freeze:** 2026-09-25, before the no-intervention reproduction reruns. A code-lock amendment was recorded after those reruns and before the intervention arm.

**Scope:** Post-hoc, no-training mechanism screen on a selected development subset. This is not a natural-policy comparison, P2 acceptance test, or promotion route.

## Why this screen

A seeded 128-case Launch-Tanh replication showed more legal first contacts but fewer legal second hits and more safety failures. In the 13 selected cases where both arms had first contact but only Tanh later had an illegal-contact failure, the median paired post-contact ball-velocity difference was 0.631 m/s. This follow-up asks whether restoring only the post-cap ball's linear velocity is enough to reduce those later illegal contacts.

## Frozen source and cases

- Source development bank: configs/wall_cases/devrep-aerowall-tanh-launch-v1-seed-260925.json, SHA256 b6ebf13209ba42b1ad77eb452248b6f743c0f0fbcc32805d250fbc162b3804da.
- Selected IDs, frozen from the completed phase report: heldout-0007, heldout-0013, heldout-0015, heldout-0033, heldout-0035, heldout-0037, heldout-0054, heldout-0061, heldout-0075, heldout-0090, heldout-0098, heldout-0101, heldout-0103.
- Source phase-report SHA256: 9907c17713cc7221deecb170dd839a3780d7ed31169548d046c7209d4696bd76.
- Checkpoints, observations, reward, FSM, windows, dwell, seed, case bank, and other actor distributions stay at the independent-replication settings. Launch-Tanh remains the only policy mapping change.

## Intervention and logging

Run three 128-environment natural RALLY arms: default Launch control, Launch-Tanh natural control, and Launch-Tanh with the registered restore applied only to the 13 selected cases. For each selected environment, detect its first PhysX-audited legal Launch cap. After that policy step completes and before the next actor action, replace only the ball's three linear velocity components with the same case's post-step value from the fresh default-Launch control trajectory. Preserve ball position/orientation/angular velocity, the drone state, FSM, counters, and all checkpoints. Synchronize the environment's prev_ball_vel contact-detector history to the injected velocity, refresh the next observation/state TensorDict, and verify the live velocity readback within 1e-5 m/s. Do not inject on later contacts.

Record the exact contact step/substep, pre/post intervention velocity, immediate next active actor observation/action, full trajectory, reset/RNG fingerprints, all outcomes, and PhysX contact sidecars. The default and Tanh no-injection reruns must reproduce the prior saved trajectories exactly; otherwise stop and mark the screen inconclusive.

## Endpoints and decision

Primary endpoint: the count of selected cases ending in illegal contact (failure code 4), and whether a later nonlegal body contact occurs after the first cap. Report legal second hits, rallies, every failure code, ball boundary exits, contact audits, and case-level deltas as secondary endpoints.

A local ball-velocity mechanism signal is supported only if at least 7/13 selected cases avoid the later illegal contact without adding ball-ground, out-of-bounds, drone-ground, or drone-wall failures and without losing a legal second hit. Otherwise the single-velocity hypothesis is rejected or inconclusive. This post-hoc subset cannot pass P2 or support a population claim under either result. Keep C350 unchanged; do not train or promote. If any source reset/RNG or contact audit fails, stop the interpretation and retain all outputs as inconclusive evidence.

## Reproduction and code lock

The first protocol draft SHA256 was `2ffcbb351cb6da25a8a610333d8560602366d67dc3d850f8406e563d6fd90813`. Its first fresh-control attempt stopped before Isaac initialization because the new restore-settings guard referenced its payload before the payload was loaded; it produced no simulation report. The guard was moved after restore-bank loading. This opt-in-only ordering fix changed the evaluator source hash from `fa5acdcc1c070a0bb199d37dcb6d8cdf845f0025c49bc98c15462fa0800874d7` to `1f4c5ca912a51d30d10adece9a23069a7e013db413c3f3a8b72b2148edc746c5`.

Before the intervention arm, fresh default-Launch and Launch-Tanh runs both passed and reproduced their previous trajectory files byte-for-byte: default Launch control SHA256 `51a233a5d5b788942e82bf50fc88480e5b02034b4678567eb51b1c6dec33cbd1`; Launch-Tanh SHA256 `fb4ed6426f0d4a6d71cbefc5ded8c3e63399405f61efc43e0e88f333f5fd336f`. Each run also passed full contact corroboration (95/95 and 102/102 events, respectively). The final opt-in evaluator is locked at SHA256 `1f4c5ca912a51d30d10adece9a23069a7e013db413c3f3a8b72b2148edc746c5`; the paired target-bank builder is locked at SHA256 `3f7cfb95ddcdd5381b7a5d351692d203a5ac9a922815b26b52dd6caa26daabc1`. The target bank records the final amended protocol SHA256 before the intervention run.
