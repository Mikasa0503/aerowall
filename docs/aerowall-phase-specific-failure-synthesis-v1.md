# AeroWall Phase-Specific Failure Mode Synthesis V1

**Date:** 2026-09-25

**Scope:** Read-only synthesis of seed 9524, the 128-case v4 development bank, and the selected 33-case contact replay. This is not final acceptance evidence. No simulator run, training, checkpoint change, or promotion was performed for this synthesis.

## Findings by rally phase

| Phase | Direct evidence | What it supports |
|---|---|---|
| 0 — Launch before first legal cap | Full-bank control and Recover-Tanh both have 116 legal and 12 illegal first contacts, with identical case labels. All 12 illegal contacts exceed the 0.20 m radial limit; 9 are already outside it before the final action, and 3 cross during that action interval. All 12 are produced by the Launch actor. | The Recover mapping is not the cause of these first-contact differences. The direct symptom is a Launch radial miss. In 12 matched illegal/legal approach pairs, 7 have standardized incoming-state distance ≤1 and 11 ≤1.5; at 0.40 s before the final action, illegal cases have median local-y −0.144 m and drone world-vy +0.136 m/s relative to their legal matches. This is an association in one development bank, not a causal action rule. |
| 1 — Ball travelling to the wall | The failure-enriched 33-case replay contains one Tanh-only phase-1 illegal contact, `heldout-0083`, produced by Recover. | Record the case for follow-up; one selected event does not establish a phase-1 failure rate or a general mechanism. |
| 2 — Return after the wall | In the selected replay, control has 11 illegal return contacts and Recover-Tanh has 14. The same 11 occur in both; Tanh adds `heldout-0037`, `0053`, and `0124`. Actual actors are Hit 10 / Recover 1 in control and Hit 6 / Recover 8 in Tanh; six shared cases switch from Hit to Recover. The paired radial-error change is heterogeneous: five improve and six worsen. Median pre-final-action radial error is 0.309 m vs 0.318 m. | Recover-Tanh does not provide a consistent correction. These are failure-enriched examples; they do not estimate population rates or isolate a single action mechanism. |

## Cross-experiment checks

- The Recover-Tanh mapping removes Recover command clipping (29.958% to 0%; full-chain clipping 19.006% to 0%) but reduces legal second hits from 14 to 11, three-rally cases from 5 to 3, five-rally cases from 4 to 2, and raises safety failures from 28 to 36/128. It is not a candidate for training or promotion.
- `AeroWallLaunchInterceptTargetRetentionV1` changes Launch actions in 14/128 cases, but first-contact labels and rally counts do not change; safety failures rise from 28 to 29/128. At all 14 first action-divergence points, six saved pre-action state/motor fields are bitwise identical across the paired trajectories. This confirms an immediate action response on a common recorded state, while the closed-loop result remains negative.
- In `heldout-0060`, both runs make legal phase-0 contact, with radial error 0.0341 m in the ablation and 0.0635 m in control. Later, the ablation crosses the environment world-x < 0.5 m task threshold; neither PhysX sidecar records a drone/single-wall contact. The threshold is not evidence of physical wall impact.
- Correction: the active AeroWall evaluator sends its four actor outputs directly to the rotor commands; HCSP RotorGroup clips each command to [-1,1], applies the square-root throttle map, then models motor lag. It does not attach `PIDRateController_flightmare`. The earlier body-rate decoding is superseded. Recomputed matched illegal-minus-legal rotor-command medians vary by lead and channel (lead 20: `[+0.099,+0.154,+0.093,-0.017]`; lead 10: `[-0.057,-0.017,-0.056,+0.000]`; lead 0: `[+0.089,+0.004,+0.021,-0.001]`). These are descriptive development-bank associations; they neither support one fixed rotor-channel offset nor establish action controllability.

## Decision and next gate

The strongest repeatable failure symptom is the phase-0 Launch radial miss; phase-2 return misses are a separate, actor-role-heterogeneous problem. The tested observation and action-mapping changes did not improve the P2 ability/safety gate. Keep formal C350, set P2=false, and keep P3–P6 gated. Do not train or promote from these diagnostics.

Before another policy change, require one named, phase-specific hypothesis and a pre-registered paired evaluation with explicit safety and rally rejection criteria. Any next diagnostic remains inference-only and must preserve checkpoint, reward, FSM, and formal routing. If no single mechanism can be isolated from the existing evidence, stop this candidate line instead of adding a fixed action offset.

## Follow-up: Launch lateral-momentum counterfactual

A separate preregistered no-training screen captured the same 12 phase-0 failed cases and 12 matched legal references at a 0.40±0.02 s predicted contact lead. It reset each failed state twice and changed only drone world-y linear velocity to the matched legal snapshot value. Both continuation arms reproduced all 12 states with zero reset error across 20 fields; the control first action matched the source action within 2.4e-7. Every first body contact in both arms matched a positive PhysX `base_link`—ball contact at the same env/step/substep; callback errors were empty.

Intervention produced 6/12 legal first contacts versus 3/12 in control, a net paired gain of 3/12. Median paired first-contact radial error improved by 0.027883 m, meeting the preregistered 0.02 m threshold. Eleven of twelve velocity substitutions lowered world-y velocity; one raised it by 0.03164 m/s, so the association is not a uniform directional rule. Two cases later ended because the ball crossed `abs(world_y) > 3 m` (heldout-0033 at trajectory index 68; heldout-0107 at index 316, after four rallies). They were ball boundary terminations, not drone ground/wall contacts.

This supports only a mechanism screen on the selected v4 development pairs. The intervention edits physical state directly and does not show the current actor can realize that momentum change through its actions. It does not pass P2 or authorize training. Formal C350 and the P3–P6 gate remain unchanged. Full results and hashes are in `artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-launch-lateral-momentum-counterfactual-v1-analysis-s9524.json`.

## Reproducibility

The machine-readable synthesis is [the JSON report](../artifacts/wall-skill-upgrade-v3/launch-diagnostics/reproducibility/aerowall-phase-specific-failure-mode-synthesis-v1-s9524.json). It verifies 10 source artifacts by SHA256 and rechecks the pre-action state equality at each of the 14 first Launch action divergences. The recomputation script is [analyze_aerowall_phase_specific_failure_modes.py](../scripts/analyze_aerowall_phase_specific_failure_modes.py).
