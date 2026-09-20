# Runtime gate evidence

These are deployment checks, not juggling or wall-rally success results.

| Requirement | Evidence | Scope and remaining work |
| --- | --- | --- |
| Native A100 GPU physics | runs/gate1-native-03.json | 200 steps, 4 rebounds, CUDA tensors, GPU dynamics enabled; clean-process log has no Error |
| 16 environments, 10,000 policy steps | runs/singlejuggle-10000-reset-safe-01.json | Passed with the project PID-reset adapter; 3,881 episode resets and finite tensors |
| Selective state reset | same report, reset_checks | Environments 0/3/7 reset; other states unchanged; controller output matches a fresh controller exactly |
| Actual ball/bat contact | runs/singlejuggle-contacts-03.json | All 16 environments report ball and bat actor/collider paths; relative vertical velocity reverses |
| Cross-environment collision filtering | same report, pair_frames | Deliberately intersecting original balls from environments 0/1 pass through, with no pair contact or horizontal velocity change; not exhaustive over all 120 pairs |
| Short PPO update/save/reload | runs/singlejuggle-ppo-02.json | Passed: 3 updates, 3,072 environment transitions, actor parameters changed; checkpoint reload action error 0; 128 evaluation steps with finite observations/rewards |
| Rendering | runs/singlejuggle-render-02.json and docs/render-review.json | Native 960x720 RGB readback and visual inspection passed; frame changes track real motion and render-only updates preserve physics; no task demo video yet |

The contact fixtures initialize states explicitly and then let PhysX evolve them. They are not policy trajectories and cannot be counted toward task performance. The controlled ball/bat check uses the original collision assets. It does not replace the later 1,000-impact, half-timestep convergence, illegal-contact, reaction-force and scoring-deduplication validation.

Isaac Sim 2023.1.0 requires the contact-report API to be attached before its initial physics scene parsing in this setup. Adding it after initialization produced actual rebounds but no contact reports. The diagnostic temporarily wraps initial SimulationContext.reset to attach instrumentation, restores the method immediately, and reads get_contact_report after each physics fetch. Physics parameters and collision groups remain unchanged.

Original upstream reset behavior failed a fresh-controller comparison by 0.01747644 in normalized rotor-command units. The project transform clears selected rows of PID integral/filter history at the reset callback. It leaves done semantics and other environments unchanged. Both failing and passing reports are retained.

SimulationApp fast shutdown can return process code zero after a caught Python error. scripts/run_probe.sh therefore requires both a successful process result and a passed JSON verdict. Inspect check coverage rather than treating an exit code as proof of all gates.

The first PPO attempt failed because diagnostic warmup stepping retained autograd state. The retry performs environment stepping/reset under no_grad and explicitly asserts collected tensors do not require gradients; PPO optimization itself keeps autograd enabled. The original PPO implementation is unchanged.

The first learned checkpoint is checkpoints/singlejuggle-ppo-02.pt. It includes policy and optimizer states, update/frame counts and Torch RNG states. Its source hashes and parsed configuration are linked from the report. It is a deployment artifact, not evidence of sustained juggling performance.

Three missing ground textures were copied from the pinned HCSP reference into JuggleRL's expected relative asset directory. docs/asset-restoration.json records commit, hashes and copied files; scripts/prepare_assets.py reproduces the copy and refuses unexpected content. No existing upstream source, USD, physics or control file changed. The default wide camera makes the drone small; a presentation camera is still needed for the final demo.

128-environment deployment benchmark: runs/singlejuggle-ppo-128-01.json passed 3 PPO updates (24,576 transitions), with collection rates 5,009 / 4,396 / 4,184 transitions per second and update times 0.656 / 0.583 / 0.583 seconds. These three batches establish initial throughput only, not sustained task learning. A 512-environment comparison is running before choosing the development training scale.
