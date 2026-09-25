# Wall skill v2: dynamic-ball experiment protocol

This is an experimental successor to the accepted fixed-ball C350 policy. The
legacy observation and reward remain the defaults. No result from v2 is a
validated improvement until the paired evaluation below is complete.

## Frozen distributions

`configs/wall_cases/fixed-128-v1.json` reproduces the old initial state.
`train-128-v1.json` samples initial ball lateral positions within ±0.4 m and
velocities within ±0.3 m/s. `heldout-128-v1.json` uses disjoint lateral bands
of 0.45–0.75 m and 0.35–0.6 m/s. All are generated deterministically by
`scripts/build_aerowall_wall_cases.py`; the feasibility filter checks only
scene boundaries and conservative straight-line reachability. It is not a
successful-contact oracle. The simulator's `--case-mode train` samples the
training ranges on every reset. Checkpoint and case-bank SHA-256 values must
be captured in the evaluation reports.

The frozen fixed, train and held-out bank hashes are respectively
`e65ab1ef8147a4f19e7c02ddd10546bc1be0d62ac090c697de3859f4d7616735`,
`b29e5f3f483fbb38cfbbd20bfe1b3d1a8d9bacfbe0e287d9880fd46f043c9172`,
and `963a887a4c48abac3d7e25771e38856426345b8ac73f852248fcdadef6643947`.

## Controlled comparisons

Run the frozen C350 chain first on fixed, train and held-out banks. Then run
three paired training seeds from that same actor, holding train frames,
upstream source checkpoints, PPO configuration, and evaluation banks fixed.
Compare one change at a time: (1) fixed versus dynamic reset; (2) legacy
versus `relative_v2` observation; (3) without versus with frozen-prefix
`AeroWallSkillChainPolicy`; (4) legacy versus `causal_v1` reward. Retain the final
checkpoint at the predeclared update, regardless of intermediate scores.
Each comparison uses three independent training seeds and all 128 held-out
cases per seed; parallel environments are cases, not independent seeds.

The primary metrics are physical-contact-audited legal second-hit rate and
at-least-three-rally rate on the held-out bank. Both must improve over the
frozen C350. The fixed-bank mean centered rally prefix may drop by at most
one rally. Drone-ground, drone-wall and illegal-contact terminations must not
increase. Every reported legal ball and wall event must match a PhysX contact
report. Report per-seed paired values, failure reasons and the full rally
distribution. A missed condition rejects the candidate; do not choose a
different checkpoint based on its video.

## Implementation and verification sequence

1. Run the simulator-free unit suite and verify the bank hashes and disjoint
   ranges. A 16-environment Isaac Sim smoke must verify reset states, 46-wide
   observations, finite actions/rewards, motor-step cadence, legal contact
   events, per-environment policy masks and checkpoint reload.
2. Run frozen C350 on the three banks using `--initial-case-bank`. Do not use
   the new reward for this baseline. Check that fixed-bank behavior matches
   the prior accepted evaluator before interpreting dynamic results.
3. Train a one-contact `INTERCEPT` skill with `relative_v2` and dynamic
   resets. This diagnostic isolates the ability to reach a varied incoming
   ball; it is not a successful wall-volley result. Use its final checkpoint
   as the frozen prefix for `hit`, then train `recover` with a frozen learned
   hit checkpoint. Run small pilot jobs before the paired budget.
4. Evaluate the complete chain on every bank and perform the reward ablation.
   Only after passing the gates consider broader randomization, a learned
   event-driven high-level controller or joint fine-tuning.

Isaac Sim, HCSP and checkpoints live outside Git. The Python-only tests do not
prove simulator behavior or policy learning.

## First diagnostic run (not an accepted policy)

On the 128-case held-out bank, the frozen launch policy made 0/128 legal first
contacts with 5/128 illegal-contact failures. A 409,600-frame, one-seed `INTERCEPT`
run from the same launch actor reached 100/128 legal first contacts, all
confirmed by PhysX, but had 28/128 illegal-contact failures. An earlier hover-dense
diagnostic reached 67/128 contacts and 29/128 illegal-contact failures at the same
budget. These are one-seed diagnostics, not the three-seed paired result.
Neither trained skill passes the safety gate; retain the old policy for the
accepted fixed-throw demo. The event-dominated `INTERCEPT` checkpoint may be
used as an experimental source for the `hit` feasibility probe, with this
failure recorded rather than hidden.

The first 409,600-frame `hit` probe used that frozen `INTERCEPT` source and
the original frozen C350 recovery actor. Against its untrained same-source
counterpart on the same 128 cases, real wall contacts changed from 35 to 42,
while legal second contacts remained zero in both runs. Safety failures rose
from 56 to 61 and target-region accuracy among wall contacts fell from 40.0%
to 28.6%. The small wall-contact gain does not justify promotion. All 42
reported wall contacts and 105 legal ball contacts in the trained run matched
PhysX contact reports.

The trained `hit` rollout's 42 wall contacts occurred at mean height 2.15 m
(range 0.33–3.95 m), below the commanded 3.5–4.5 m wall-target heights on
average. This explains why a stronger wall-hit count alone is a poor proxy for
recoverable returns. These numbers come from the first-episode event JSONL,
not a rendered video.

An equally budgeted, frozen-launch/frozen-hit `recover` probe from the same
relative-observation actor still achieved 0/128 legal second hits on held-out
cases. It produced 36 audited wall contacts and 67/128 safety failures. On the
fixed 128-case bank, the accepted C350 chain retained 8.0 mean centered
rallies with zero safety failures, whereas the full experimental chain had
zero walls, zero rallies and 128/128 illegal-contact terminations. These are
decisive rejections under the predeclared held-out, fixed-preservation and
safety gates. Keep the legacy policy and reward as the defaults. The three
training-seed ablation, higher-level RL and joint skill adaptation are not
triggered by these results.

The follow-up below implemented that isolated Hit curriculum and reward
screen. Those probes also failed the promotion conditions, so no three-seed
expansion or higher-level/joint training was started. Keep the accepted policy
and reward as defaults; the held-out banks remain frozen.

## Follow-up Hit curriculum

`make_hit_start` constructs a descending ball whose predicted contact is
0.35–0.45 s away and lies inside the 0.2 m racket disk. The first version used
0.14–0.175 s and was replaced when the Hit window was widened. Hit training
defaults to 70% of resets from this feasible near-contact distribution and
30% from real upstream policy execution; change to 20% artificial only after
the early stage has passed its gate. The actual reset fraction is recorded in
every training report. `causal_v2` is a separate reward arm: it gives a smooth
score for predicted wall-target height/lateral position and conservative
return reachability, then scores the measured wall impact location. `causal_v1`
and legacy rewards keep their previous definitions.

The second held-out bank is `heldout-128-v2.json`, generated with seed 9224
before the new Hit evaluation (SHA-256
`8fa2cacb6e9a030f5f52266b8d99aa3d2deb866f5099091ff67980bf74eac36f`). The
first comparison holds the event-trained Intercept checkpoint, C350 recovery
checkpoint, 70% artificial reset ratio, 409,600 frames, and this bank fixed;
the only changed factor is reward design across `legacy`, `causal_v1`, and
`causal_v2`. Seed 6101 screens the three arms. Expand promising arms to seeds
6102 and 6103 only when a screen shows held-out progress without destroying
fixed-bank behavior.

On the new bank, the untrained same-source chain produced 99 legal first
contacts, 35 wall contacts, one legal second contact, a 2.53 m mean wall height
and 55 safety failures. The `causal_v1` mixed-start pilot produced 105 first
contacts and 39 wall contacts, but zero second contacts, a 2.24 m mean wall
height, 30.8% target accuracy and 57 safety failures. All reported contacts
were PhysX-corroborated. This shows that the near-contact mixture improves
first contact slightly but does not teach a useful outbound shot by itself;
the `causal_v2` pilot produced 106 first contacts, 34 wall contacts, two
legal second contacts, a 2.71 m mean wall height, 23.5% target accuracy and 66
safety failures. The same candidate had zero walls and 128 illegal-contact
terminations on the fixed bank. The same-seed legacy-reward control produced
97 first contacts, 48 wall contacts, zero legal second contacts and 57 safety
failures. All reported events were PhysX-corroborated. The reward change did
not pass; the larger wall count in the legacy arm also failed to produce a
legal return.

The source launch, accepted C350, experimental intercept, hit and recover
checkpoint SHA-256 values are respectively `1886527f2bb3f60f28d59c94882bee60eef5ee7af180d36aaeaff729b28aa40e`,
`4e31d16bbe2f3cd82ee6d25a519ed3650b17b91f91a1d40391f9310f0407a659`,
`289b319b425ef3f425e54a5a0286e7e56acf849d3d5ea6ade8a1a3d8fa09ba22`,
`1109fce0b28c5bc293cc4e145f626ac887846b816260328e0b835f47d99a8292`,
and `7110811cd15f236b896ceb9c3cf20bf876e3756ea7732fd82f705574a27de3da`.
The detailed JSON reports and PhysX event JSONL are in the ignored local
`artifacts/wall-skill-v2/` directory and the isolated 138 workspace's `runs/`
directory; model weights remain outside Git.

## Hit window follow-up

The v1 and v2 reward screens did not meet the promotion gate, so no three-seed
expansion was run. Their comparison showed that the 0.18 s Hit window allowed
only about 18% of chain actions to train the Hit actor. The next screen extends
the Hit window and synthetic feasible contact time together to 0.42 s (about
21 control steps), while holding the 70% reset mixture and `causal_v2`
constant. It uses a fresh held-out bank, `heldout-128-v3.json` (seed 9324,
SHA-256 `ecec4fffdbff6439a1eda5ad904d76ecccd5b0826270149de9c70f4b6f709650`).
Before training, the same source chain with the wider window scored 103 legal
first contacts, 50 wall contacts, 4 legal second contacts, 42.9% wall-target
accuracy and 62 safety failures. The reported events all passed PhysX audit.

The 409,600-frame widened-window candidate reached 95/128 first contacts,
17 walls, one legal second hit, 35.3% target accuracy and 93 safety failures
when evaluated with the 0.42 s window. Re-evaluating the same weights at
0.18 s produced 105 first contacts, 22 walls, two legal second hits, 22.7%
target accuracy and 72 safety failures. The trained model made fewer wall and
second-hit events, had lower target accuracy, and incurred more safety
failures than its matching untrained source chain at both windows; at 0.18 s
it had two more first contacts. All reported contacts were
PhysX-corroborated. This rejects the widened-window candidate, and no
three-seed expansion was run.

## C350 Hit-actor compatibility screen

To separate a poor learned Hit initialization from the Intercept-to-Hit
handoff, the accepted C350 actor was loaded into the trainable Hit slot and
given its original `legacy` observation while the frozen Intercept policy
continued to receive `relative_v2`. The frozen recovery actor was also C350.
This checks integration and transfer behavior; it is not a trained candidate.
The new v4 held-out bank was frozen before evaluation (seed 9424, SHA-256
`666709adb50f350161a51112695909fa0cf304d829685be41b35ae2f8cfe8f80`).

On v4, the original C350 policy alone made zero legal first contacts and had
zero safety terminations. The chained C350 Hit/Recover policy made 102/128
legal first contacts, 29 audited wall contacts and 4/128 legal second hits;
only 1/128 episodes reached at least three rallies. Its safety-failure rate
was 65/128 (6 drone-ground, 30 illegal-contact and 29 drone-wall failures).
The target-region rate was 41.4%, and the mean wall height was 2.76 m. All
109 legal ball events and all 29 wall events were corroborated by PhysX. This
is a real first-contact improvement over direct C350 on diverse starts, but it
does not meet the rally or safety gates.

On the frozen fixed bank, the same chain reached exactly four rallies in every
episode, then ended with 113 drone-wall and 15 illegal-contact failures. Its
mean centered prefix was 4.0 rallies, versus 8.0 for accepted C350, a four
rally regression against the one-rally allowance. All 640 legal contacts and
all 512 wall events were PhysX-corroborated. Reject this chain and retain
C350 as the default. The experiments do not isolate whether the regression
comes from the learned Intercept's arrival state or another handoff detail;
do not start Stage III, add randomization, or train a high-level policy until
that failure is explained and a safer chain passes the frozen gates.

The 16-case smoke report, full v4 held-out report, fixed-bank report, contact
audits and event JSONL are in the ignored local `artifacts/wall-skill-v2/`
directory and the corresponding 138 `runs/` directory. The smoke report
confirmed policy loading and the legacy observation switch; the 128-case
reports are the evaluation evidence. No checkpoint from these screens is
promoted, and the C350 checkpoint SHA remains
`4e31d16bbe2f3cd82ee6d25a519ed3650b17b91f91a1d40391f9310f0407a659`.
