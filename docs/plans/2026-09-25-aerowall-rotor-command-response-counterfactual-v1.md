# AeroWall Rotor Command Response Counterfactual V1

**Pre-registration date:** 2026-09-25

**Status:** Frozen no-training diagnostic; not a P2 acceptance test and cannot authorize training or promotion.

**Scope:** Part 5 action-interface audit and Part 13 local-controllability evidence.

## Rationale and falsifiable question

The earlier AeroWallLaunchLateralMomentumCounterfactualV1 changed drone world-y velocity directly in 12 selected pre-contact failure states. It increased legal first contacts from 3/12 to 6/12 and reduced median paired radial error by 0.0279 m, but two intervention cases later crossed the ball's world-y boundary. That state edit did not show whether the existing policy can produce a similar velocity correction through its actions.

The active AeroWall wall-rally evaluator wraps the environment only with InitTracker; the four actor outputs go directly to drone.apply_action. HCSP RotorGroup.forward clips each motor command to [-1, 1], maps it to sqrt(clamp((u+1)/2, 0, 1)), and applies motor lag. No PIDRateController_flightmare is attached to this route. The channels are rotor indices 0–3; this protocol does not assign them body axes.

**Question:** From the same saved Launch states, can a small, bounded change to a direct rotor command measurably move drone world-y velocity toward that case's pre-matched legal reference?

## Frozen inputs and policy

- Use the 12 failure handoffs from aerowall-launch-lateral-momentum-counterfactual-v1-control-s9524.json, SHA256 cc9d93b5f3ecb2b367e1f3c209b080963ef535b38be63f2d004b38b4c35af0ec. Do not replace or resample any case.
- Keep the 12 matched legal references fixed as listed in the existing AeroWallLaunchLateralMomentumCounterfactualV1 protocol.
- Seed 9524, deterministic actor, natural RALLY continuation, same AeroWall environment/FSM/reward/observation/PhysX settings, and frozen checkpoints:
  - Launch AeroWallBoundedInterceptV1-CausalV6, SHA256 bb256fd0a10a9e7653791636f60969323a0ad6f606fe5f4d72884c9a4edf6f50
  - Hit AeroWallBoundedGoalHitV1-Tanh-S6201, SHA256 d970228e9c2fdc662fc6fa9b0aead0672628ffd0da947dd04a93685a31968dd7
  - Recover C350, SHA256 4e31d16bbe2f3cd82ee6d25a519ed3650b17b91f91a1d40391f9310f0407a659
- Keep Launch/Hit TanhNormal and Recover default distributions, relative_v3 environment observations, relative_v2 trainable-skill observation, aerowall_causal_v6 reward, and the existing hit/FSM timing unchanged.

## Paired action arms

For each of the same 12 saved states, run one control and eight treatment continuations:

| Arm | Intervention during policy steps 0–9 |
| --- | --- |
| Control | No override |
| Treatments | Add +0.10 or -0.10 to exactly one actor output channel, rotor index 0, 1, 2, or 3 |

Apply the signed offset to that channel on every one of the first ten policy decisions, before env.step; let the deterministic actor produce all other actions and resume the unmodified actor after step 9. The environment's existing rotor limit then clips the command. Record both the actor output before override and the command after override/clipping, so a boundary-limited perturbation is visible rather than excluded.

Run each arm from the same source bank and seed in the same evaluator configuration. Continue naturally until contact or task termination. Do not alter state, action distribution, checkpoint, reward, FSM, environment bounds, or any other action channel. Total scope is 108 short selected-case continuations; this is a mechanism screen, not a population-rate estimate.

## Outcomes and decision rules

**Primary measure:** For each case and each arm, record drone world-y velocity at reset and after policy step 9. Compare the treated-minus-control change with the direction and magnitude needed to reach that case's matched legal-reference world-y velocity. Report the complete per-case, per-rotor, per-sign response matrix; do not select a best arm and present it as a policy result.

A shared direct-command direction is considered a local signal only if the same rotor index and sign moves world-y velocity toward its reference in at least 9/12 cases and the median absolute change is at least 0.02 m/s. Otherwise, reject a single shared rotor-command correction rule. A negative result does not prove state-conditioned control is impossible.

**Task and safety endpoints:** Continue every episode to its normal terminal event and record first-contact legality/radial error, wall contact, rallies, safety failure codes, callback errors, and ball boundary termination. Explicitly flag ball world-y boundary crossings at abs(y) > 3 m, including cases that first accumulate rallies. Any treatment arm that introduces a ball boundary failure or new drone-ground, drone-wall, or illegal-contact failure relative to its paired control is rejected for candidate use.

Reset state fields, first action reproduction, effective command deltas, source/checkpoint/environment/reward hashes, and all PhysX contact audits must be recorded. A missing contact audit invalidates that case-arm for contact claims; do not replace it.

## Guardrails

- No training, checkpoint change, policy promotion, formal C350 change, or P3–P6 start.
- Even a positive response matrix does not pass P2. It only shows local action sensitivity in these 12 selected development states.
- Report all eight treatment arms, including negative responses and any command clipping. Do not change the perturbation size, horizon, case set, or threshold after observing results.
- The Tanh mapping ablation remains a separate completed experiment; this diagnostic does not retrain its candidate.
