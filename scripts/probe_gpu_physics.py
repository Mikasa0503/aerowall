"""Gate 1 only: same Kit experience as upstream, one real GPU-physics bouncing ball.

Run through scripts/python.sh after the runtime archive has been inspected.
This is not the 16-environment/10,000-policy-step or PPO gate.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.steps < 100:
        parser.error("At least 100 physics steps are needed to observe gravity and rebound")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "gate": "minimal_gpu_physics", "status": "starting", "pid": os.getpid(),
              "requested_steps": args.steps, "physics_dt": 0.02,
              "training_gate_passed": False, "rendering_tested": False}

    def record(**values):
        report.update(values)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps(values), flush=True)

    app = None
    record(status="importing_simulation_app")
    try:
        from omni.isaac.kit import SimulationApp

        experience = str(Path(os.environ["EXP_PATH"]) / "omni.isaac.sim.python.kit")
        record(status="initializing_kit", experience=experience)
        app = SimulationApp({"headless": True, "anti_aliasing": 0}, experience=experience)
        record(status="kit_initialized")

        # Physics modules are deliberately imported after Kit initialization.
        import numpy as np
        import torch
        import omni.usd
        from pxr import PhysxSchema
        from omni.isaac.core.simulation_context import SimulationContext
        from omni.isaac.core.materials import PhysicsMaterial
        from omni.isaac.core.objects import DynamicSphere, FixedCuboid
        from omni.isaac.core.prims import RigidPrimView

        assert torch.cuda.is_available(), "CUDA not available"
        sim = SimulationContext(stage_units_in_meters=1.0, physics_dt=0.02, rendering_dt=0.02,
                                backend="torch", device="cuda:0",
                                sim_params={"use_gpu": True, "use_gpu_pipeline": True,
                                            "gravity": [0, 0, -9.81]})
        material = PhysicsMaterial("/World/material", restitution=0.8,
                                   static_friction=0.0, dynamic_friction=0.0)
        FixedCuboid("/World/floor", translation=np.array([0., 0., -0.1]),
                     scale=np.array([10., 10., 0.2]), size=1.0, physics_material=material)
        DynamicSphere("/World/ball", translation=np.array([0., 0., 1.0]),
                      radius=0.04, mass=0.0472, physics_material=material)
        balls = RigidPrimView("/World/ball", name="probe_ball")
        sim.reset()
        balls.initialize()
        scene = omni.usd.get_context().get_stage().GetPrimAtPath(sim.get_physics_context().prim_path)
        gpu_dynamics = PhysxSchema.PhysxSceneAPI(scene).GetEnableGPUDynamicsAttr().Get()
        assert gpu_dynamics is True, f"GPU dynamics not enabled: {gpu_dynamics}"
        record(status="stepping", gpu=torch.cuda.get_device_name(0),
               torch_version=torch.__version__, gpu_dynamics_enabled=gpu_dynamics)
        started = time.monotonic()
        samples = []
        previous_vz = 0.0
        rebounds = 0
        for step in range(args.steps):
            sim.step(render=False)
            positions, rotations = balls.get_world_poses()
            velocities = balls.get_velocities()
            assert positions.device.type == "cuda", "Positions not on CUDA pipeline"
            assert velocities.device.type == "cuda", "Velocities not on CUDA pipeline"
            assert torch.isfinite(positions).all() and torch.isfinite(rotations).all() and torch.isfinite(velocities).all()
            z, vz = float(positions[0, 2].item()), float(velocities[0, 2].item())
            assert -0.1 < z < 2.0, f"Ball escaped floor/bounds: {z}"
            if previous_vz < -0.1 and vz > 0.1:
                rebounds += 1
            previous_vz = vz
            samples.append({"step": step + 1, "t": (step + 1) * 0.02, "z": z, "vz": vz})
        assert samples[9]["z"] < 0.95, "No observed free fall"
        assert rebounds > 0, "No observed collision rebound"
        record(status="passed", completed_steps=args.steps, elapsed_seconds=time.monotonic() - started,
               observed_rebounds=rebounds, samples=samples,
               scope="one ball/floor; upstream multirotor, reset isolation, PPO and 1000-collision calibration remain untested")
        return 0
    except Exception as error:
        record(status="failed", error=repr(error), traceback=traceback.format_exc())
        return 1
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
