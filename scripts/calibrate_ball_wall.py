"""Paired 1,000-case wall-impact calibration using upstream sphere and solver settings.

Fixture initialization is explicit. No state is overwritten during each impact.
This does not validate policy performance or ball/bat reaction forces.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cases', type=int, default=1000)
    parser.add_argument('--dt', type=float, choices=[.02, .01], help='Internal single-process time step')
    args = parser.parse_args()
    if args.cases < 1000:
        parser.error('At least 1000 paired cases are required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'initializing', 'pid': os.getpid(), 'scope': 'controlled ball-wall impacts only',
              'case_count': args.cases, 'time_steps': [args.dt] if args.dt else [0.02, 0.01],
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save(**values):
        report.update(values)
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(args.output)
        print(json.dumps(values), flush=True)

    app = None
    save()
    if args.dt is None:
        children = []
        records = []
        for dt in [.02, .01]:
            child = args.output.with_name(args.output.stem + f'-dt{dt:.2f}.json')
            if child.exists():
                raise FileExistsError(child)
            save(status='calibrating', current_dt=dt, execution='one fresh process per time step')
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output', str(child),
                                     '--cases', str(args.cases), '--dt', str(dt)])
            data = json.loads(child.read_text()) if child.exists() else {'status': 'missing'}
            children.append({'report': str(child), 'exit_code': result.returncode, 'status': data['status']})
            if result.returncode != 0 or data['status'] != 'passed':
                save(status='failed', children=children, error='Single-time-step calibration failed')
                return 1
            records.extend(json.loads(child.with_suffix('.impacts.json').read_text()))
        by_key = {(row['case_id'], row['dt']): row for row in records}
        deltas = [abs(by_key[i, .02]['outgoing_vx'] - by_key[i, .01]['outgoing_vx']) /
                  max(abs(by_key[i, .01]['outgoing_vx']), 1e-8) for i in range(args.cases)]
        raw = args.output.with_suffix('.impacts.json')
        raw.write_text(json.dumps(records, indent=2) + '\n')
        passed = len(records) == 2 * args.cases and len(by_key) == 2 * args.cases and max(deltas) < .05
        save(status='passed' if passed else 'failed', children=children, impact_count=len(records),
             unique_case_dt_count=len(by_key), max_relative_velocity_difference=max(deltas),
             half_dt_velocity_difference_below_five_percent=max(deltas) < .05,
             raw_impacts=str(raw), raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
        return 0 if passed else 1
    try:
        from omni.isaac.kit import SimulationApp
        sys.argv = [sys.argv[0], '--portable', '--portable-root', str(ROOT / '.cache/kit')]
        app = SimulationApp({'headless': True, 'anti_aliasing': 0},
                            experience=str(Path(os.environ['EXP_PATH']) / 'omni.isaac.sim.python.kit'))
        import numpy as np
        import torch
        import omni.usd
        from omegaconf import OmegaConf
        from omni.isaac.core import SimulationContext
        from omni.isaac.core.materials import PhysicsMaterial
        from omni.isaac.core.objects import DynamicSphere, FixedCuboid
        from omni.isaac.core.prims import RigidPrimView
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysxSchema, PhysicsSchemaTools

        sim_file = ROOT / 'third_party/JuggleRL_train/cfg/base/sim_base.yaml'
        sim_params = OmegaConf.to_container(OmegaConf.load(sim_file).sim, resolve=True)
        sim_params['dt'] = args.dt
        sim = SimulationContext(stage_units_in_meters=1., physics_dt=args.dt, rendering_dt=args.dt,
                                backend='torch', device='cuda:0', sim_params=sim_params)
        stage = omni.usd.get_context().get_stage()
        count, mass, radius, gravity = 16, .0472, .04, 9.81
        wall_materials = []
        ball_material = PhysicsMaterial('/World/ball_material', restitution=.8)
        for index in range(count):
            path = f'/World/cases/case_{index}'
            material = PhysicsMaterial(f'/World/wall_material_{index}', restitution=.8)
            wall_materials.append(material)
            FixedCuboid(path + '/wall', translation=np.array([0., index * 8., 2.]),
                         scale=np.array([.2, 4., 4.]), size=1., physics_material=material)
            ball = DynamicSphere(path + '/ball', translation=np.array([-.45, index * 8., 1.5]),
                                 radius=radius, mass=mass, physics_material=ball_material)
            PhysxSchema.PhysxContactReportAPI.Apply(ball.prim).CreateThresholdAttr().Set(0.)
        balls = RigidPrimView('/World/cases/case_*/ball', name='calibration_balls')
        sim.reset()
        balls.initialize()
        assert PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath(sim.get_physics_context().prim_path)).GetEnableGPUDynamicsAttr().Get()
        # View order can be lexical (0,1,10,...), so derive fixture coordinates from paths.
        ids = [int(path.split('/case_')[1].split('/')[0]) for path in balls.prim_paths]
        id_to_row = {index: row for row, index in enumerate(ids)}
        rng = np.random.default_rng(20260921)
        specs = [{'id': i, 'normal_speed': float(rng.uniform(1., 8.)),
                  'tangent_ratio': float(rng.uniform(-.75, .75)),
                  'wall_restitution': float(rng.uniform(.65, .95))} for i in range(args.cases)]
        args.output.with_suffix('.cases.json').write_text(json.dumps(specs, indent=2) + '\n')
        save(status='calibrating', sim_config=sim_params,
             sim_config_sha256=hashlib.sha256(sim_file.read_bytes()).hexdigest(),
             sphere={'mass': mass, 'radius': radius, 'restitution': .8,
                     'static_friction': ball_material.get_static_friction(),
                     'dynamic_friction': ball_material.get_dynamic_friction()},
             fixture={'wall_dimensions': [.2, 4., 4.], 'start_x': -.45, 'start_z': 1.5},
             gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__, view_order=ids)
        interface = get_physx_simulation_interface()
        records = []

        def energy(pos, vel):
            linear = .5 * mass * (vel[:3] ** 2).sum()
            angular = .5 * (.4 * mass * radius ** 2) * (vel[3:] ** 2).sum()
            return float((linear + angular + mass * gravity * pos[2]).item())

        for dt in [args.dt]:
            for offset in range(0, args.cases, count):
                active = min(count, args.cases - offset)
                batch = [specs[min(offset + row, args.cases - 1)] for row in range(count)]
                positions = torch.tensor([[-.45, index * 8., 1.5] for index in ids], device='cuda:0')
                rotations = torch.zeros(count, 4, device='cuda:0'); rotations[:, 0] = 1
                velocities = torch.zeros(count, 6, device='cuda:0')
                for row, spec in enumerate(batch):
                    velocities[row, 0] = spec['normal_speed']
                    velocities[row, 1] = spec['normal_speed'] * spec['tangent_ratio']
                    wall_materials[ids[row]].set_restitution(spec['wall_restitution'])
                balls.set_world_poses(positions, rotations)
                balls.set_velocities(velocities)
                previous_pos, previous_vel = positions.clone(), velocities.clone()
                found = [0] * count
                hits = [None] * count
                wrong_pairs = []
                contact_points = [None] * count
                for step in range(math.ceil(.5 / dt)):
                    sim.step(render=False)
                    pos, vel = balls.get_world_poses()[0].clone(), balls.get_velocities().clone()
                    assert pos.device.type == 'cuda' and torch.isfinite(pos).all() and torch.isfinite(vel).all()
                    headers, data = interface.get_contact_report()
                    for header in headers:
                        if header.type != ContactEventType.CONTACT_FOUND:
                            continue
                        actors = [str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                                  str(PhysicsSchemaTools.intToSdfPath(header.actor1))]
                        if not any(path.endswith('/ball') for path in actors):
                            continue
                        ball_path = next(path for path in actors if path.endswith('/ball'))
                        index = int(ball_path.split('/case_')[1].split('/')[0])
                        row = id_to_row[index]
                        if set(actors) != {f'/World/cases/case_{index}/ball', f'/World/cases/case_{index}/wall'}:
                            wrong_pairs.append(actors)
                        found[row] += 1
                        if header.num_contact_data:
                            point = data[header.contact_data_offset]
                            contact_points[row] = {'position': list(point.position), 'normal': list(point.normal)}
                    for row in range(active):
                        if hits[row] is None and previous_vel[row, 0] > 0 and vel[row, 0] < 0:
                            incoming = float(previous_vel[row, 0].item())
                            outgoing = float(vel[row, 0].item())
                            before = energy(previous_pos[row], previous_vel[row])
                            after = energy(pos[row], vel[row])
                            kinetic = float((.5 * mass * (previous_vel[row, :3] ** 2).sum()).item())
                            hits[row] = {'case_id': batch[row]['id'], 'dt': dt, 'step': step + 1,
                                         'input': batch[row], 'incoming_vx': incoming, 'outgoing_vx': outgoing,
                                         'effective_restitution': -outgoing / incoming,
                                         'before_position': previous_pos[row].cpu().tolist(),
                                         'after_position': pos[row].cpu().tolist(),
                                         'before_velocity': previous_vel[row].cpu().tolist(),
                                         'after_velocity': vel[row].cpu().tolist(),
                                         'mechanical_energy_before': before, 'mechanical_energy_after': after,
                                         'energy_growth_fraction': max(0., after - before) / max(kinetic, 1e-8)}
                    previous_pos, previous_vel = pos, vel
                for row in range(active):
                    result = hits[row] or {'case_id': batch[row]['id'], 'dt': dt, 'missed': True}
                    result.update(contact_found_count=found[row], contact=contact_points[row], wrong_pairs=wrong_pairs)
                    records.append(result)
                args.output.with_suffix('.impacts.json').write_text(json.dumps(records, indent=2) + '\n')
                if (offset // count) % 8 == 0:
                    save(status='calibrating', current_dt=dt, completed_cases_this_dt=offset + active)
        raw_path = args.output.with_suffix('.impacts.json')
        raw_path.write_text(json.dumps(records, indent=2) + '\n')
        valid = [row for row in records if not row.get('missed')]
        by_key = {(row['case_id'], row['dt']): row for row in valid}
        deltas = [abs(by_key[i, .02]['outgoing_vx'] - by_key[i, .01]['outgoing_vx']) /
                  max(abs(by_key[i, .01]['outgoing_vx']), 1e-8)
                  for i in range(args.cases) if (i, .02) in by_key and (i, .01) in by_key]
        checks = {'no_missed_collisions': len(valid) == args.cases,
                  'one_contact_entry_each': all(row['contact_found_count'] == 1 for row in records),
                  'contact_points_present': all(row['contact'] is not None for row in records),
                  'no_wrong_pairs': all(not row['wrong_pairs'] for row in records),
                  'normal_energy_bounded': all(row['effective_restitution'] <= 1.01 for row in valid),
                  'mechanical_energy_growth_below_one_percent': all(row['energy_growth_fraction'] < .01 for row in valid)}
        save(status='passed' if all(checks.values()) else 'failed', checks=checks,
             raw_impacts=str(raw_path), raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
             impact_count=len(records), max_relative_velocity_difference=max(deltas, default=None),
             max_energy_growth_fraction=max((row['energy_growth_fraction'] for row in valid), default=None),
             effective_restitution_range=[min((row['effective_restitution'] for row in valid), default=None),
                                          max((row['effective_restitution'] for row in valid), default=None)])
        return 0 if report['status'] == 'passed' else 1
    except Exception as error:
        save(status='failed', error=repr(error), traceback=traceback.format_exc())
        return 1
    finally:
        if app is not None:
            app.close()


if __name__ == '__main__':
    raise SystemExit(main())
