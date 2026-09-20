# WallRally development integration

The task runs native physics at 2.5 ms with held CTBR commands and PID updates at 20 ms. It retains physical reaction forces and terminates on illegal contact, bounds or numerical failure. It does not terminate at the first valid rally. This is an implementation gate, not a learned wall-return result.

The actor receives the original 24-feature juggling prefix plus angular velocity (3), wall geometry (6), current target (3), prior action (4) and observable phase (3). The critic adds four clock features. Actual restitution and future targets are absent. Initial actor transfer uses separate prefix/suffix normalization and zero weights for added inputs. On 64 sampled prefixes and suffix scales 1 and 10, maximum action difference was 4.7684e-7; the critic is initialized separately for the new reward.

Development reward: 100 per actual legal cap–wall–cap rally, 20 additional for a target-success rally, -20 on failure, up to 3 for ballistic launch shaping only at an actual legal cap. Dense interception and action smoothness terms are scaled by the 20 ms policy interval. The ballistic estimate never constitutes a wall hit or actor input. Targets and reward weights are provisional. No global tilt restriction or flip termination is used.

## Verified evidence

- `runs/wall-task-contract-01.json` and `.exit.json`: 16 physical wall hits, current-target publication, zero false rallies, timeout/termination separation, selected-environment reset and observation shapes passed. These are explicit initialization/clock fixtures, not policy trajectories. The hidden-parameter check tests observation dependency, not randomized material behavior.
- `runs/wall-rally-smoke-04.json`: 2 PPO updates, 2048 transitions, saved checkpoint; 6 actual wall hits and zero rallies. No first-step reset failures.
- `runs/wall-rally-resume-01.json` and `.exit.json`: restored policy, optimizer and learning RNG; another 2048 transitions, total update counter 4 and cumulative counted frames 32,772,096. Simulator episodes restart; bitwise physical continuation is not claimed.
- `docs/reset-guard-regression.json`: original policy remains 96/100 for at least five legal cap contacts. Outcomes and all 13 saved trajectory arrays exactly match the prior 2.5 ms run.
- All 18 existing archive and rally contract unit tests pass.

## Reset-contact correction

PhysX can report a pre-reset CPU PERSIST point at ground height after the body was reset to 1.1 m. First-read-only filtering rejects points outside the current source collider's conservative geometric radius plus 5 cm and one-step travel. It preserves lifecycle edges and stores discarded-point diagnostics. This is not a cooldown. Geometry is computed from actual Cube size, Cylinder radius/height or Mesh points because authored extents underestimated the base collider.

`runs/reset-contact-epoch-04.json` accepts all 16 true first-step ground contacts, accepts zero ground contacts after teleporting clear, and accepts all 16 renewed contacts. This fixture did not generate stale points; the two actual stale-point rejections are recorded in `wall-rally-smoke-04`. Earlier failed probes are retained (missing bounds, unsupported bounding-box API, then underestimated authored extents).

## Remaining work

The single-wall-return development target (70%) has not been achieved. `wall-rally-dev-001` begins a bounded 128-environment, 25-update run (204800 additional transitions), initialized from the selected 32,768,000-frame juggling policy. Inspect its terminal report before claiming completion. Full recovery analysis, frozen wall evaluation, randomization, GRU/estimator methods, equal-budget three-seed comparisons, ablations and demonstrations remain outstanding.
