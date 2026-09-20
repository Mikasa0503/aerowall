# Runtime gate evidence

These are deployment checks, not juggling or wall-rally success results.

| Requirement | Evidence | Scope and remaining work |
| --- | --- | --- |
| Native A100 GPU physics | runs/gate1-native-03.json | 200 steps, 4 rebounds, CUDA tensors, GPU dynamics enabled; clean-process log has no Error |
| 16 environments, 10,000 policy steps | runs/singlejuggle-10000-reset-safe-01.json | Passed with the project PID-reset adapter; 3,881 episode resets and finite tensors |
| Selective state reset | same report, reset_checks | Environments 0/3/7 reset; other states unchanged; controller output matches a fresh controller exactly |
| Actual ball/bat contact | runs/singlejuggle-contacts-03.json | All 16 environments report ball and bat actor/collider paths; relative vertical velocity reverses |
| Cross-environment collision filtering | runs/singlejuggle-contacts-04.json | All 120 original ball-ball environment pairs deliberately intersect and pass through, with no pair contact or horizontal velocity change; other body-type pairs are not exhaustively tested |
| Short PPO update/save/reload | runs/singlejuggle-ppo-02.json | Passed: 3 updates, 3,072 environment transitions, actor parameters changed; checkpoint reload action error 0; 128 evaluation steps with finite observations/rewards |
| Rendering | runs/singlejuggle-render-02.json and docs/render-review.json | Native 960x720 RGB readback and visual inspection passed; frame changes track real motion and render-only updates preserve physics; no task demo video yet |

The contact fixtures initialize states explicitly and then let PhysX evolve them. They are not policy trajectories and cannot be counted toward task performance. The controlled ball/bat check uses the original collision assets. It does not replace the later 1,000-impact, half-timestep convergence, illegal-contact, reaction-force and scoring-deduplication validation.

Isaac Sim 2023.1.0 requires the contact-report API to be attached before its initial physics scene parsing in this setup. Adding it after initialization produced actual rebounds but no contact reports. The diagnostic temporarily wraps initial SimulationContext.reset to attach instrumentation, restores the method immediately, and reads get_contact_report after each physics fetch. Physics parameters and collision groups remain unchanged.

Original upstream reset behavior failed a fresh-controller comparison by 0.01747644 in normalized rotor-command units. The project transform clears selected rows of PID integral/filter history at the reset callback. It leaves done semantics and other environments unchanged. Both failing and passing reports are retained.

SimulationApp fast shutdown can return process code zero after a caught Python error. scripts/run_probe.sh therefore requires both a successful process result and a passed JSON verdict. Inspect check coverage rather than treating an exit code as proof of all gates.

The first PPO attempt failed because diagnostic warmup stepping retained autograd state. The retry performs environment stepping/reset under no_grad and explicitly asserts collected tensors do not require gradients; PPO optimization itself keeps autograd enabled. The original PPO implementation is unchanged.

The first learned checkpoint is checkpoints/singlejuggle-ppo-02.pt. It includes policy and optimizer states, update/frame counts and Torch RNG states. Its source hashes and parsed configuration are linked from the report. It is a deployment artifact, not evidence of sustained juggling performance.

Three missing ground textures were copied from the pinned HCSP reference into JuggleRL's expected relative asset directory. docs/asset-restoration.json records commit, hashes and copied files; scripts/prepare_assets.py reproduces the copy and refuses unexpected content. No existing upstream source, USD, physics or control file changed. The default wide camera makes the drone small; a presentation camera is still needed for the final demo.

128-environment deployment benchmark: runs/singlejuggle-ppo-128-01.json passed 3 PPO updates (24,576 transitions), with collection rates 5,009 / 4,396 / 4,184 transitions per second and update times 0.656 / 0.583 / 0.583 seconds. These three batches establish initial throughput only, not sustained task learning. The 512-environment comparison also passed (runs/singlejuggle-ppo-512-01.json): 98,304 transitions across 3 updates, 8.492 seconds in the training loop, combined throughput 11,576 transitions/second. Checkpoint reload error is zero. 512 environments is the provisional development scale; sustained-learning throughput remains unmeasured.

Ball-wall calibration: runs/ball-wall-calibration-02.json and its two dt child reports passed 1,000 identical seeded cases at each of 0.02 and 0.01 seconds (2,000 total physical impacts). All impacts have exactly one CONTACT_FOUND, a contact point, and no wrong pairs. The maximum paired normal rebound velocity difference is 0.001508694 (0.151%); measured mechanical energy never grows across the impact. Initial normal speeds span 1–8 m/s with tangential ratios in [-0.75, 0.75]. This fixture does not cover ball/bat reaction forces, illegal contacts or learned performance.

The wall material range [0.65, 0.95] yields effective ball-wall restitution about [0.72555, 0.87477], with the ball material fixed at 0.8. The dt=0.01 empirical fit is effective_e = 0.5000000 * wall_material_e + 0.4000004. This is consistent with averaging material restitutions, and is not evidence that effective e spans [0.65, 0.95]. docs/ball-wall-calibration-analysis.json records the fit and errors. The task distribution remains to be frozen after feasibility analysis.

The initial calibration attempt (runs/ball-wall-calibration-01.exit.json) crashed with a native segmentation fault in PhysX tensor set_transforms near the time-step transition; its status is not a pass. The second attempt isolates each time step in a fresh process, verifies child exit codes and reports, saves every batch, and joins records by the same case IDs. This avoids live time-step switching; it does not establish the exact internal cause of the initial crash.
