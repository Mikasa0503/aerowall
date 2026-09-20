"""Measure equal-and-opposite ball/articulation impact momentum in free fall.

Uses the actual upstream rigid bodies and GPU masses, including all drone links.
Only fixture initialization writes state. No controller forces are applied during
measurement; gravity is explicitly removed from each one-step momentum change.
"""
import re
import torch


def check_reaction(env, base, record):
    import omni.usd
    from pxr import UsdPhysics, PhysicsSchemaTools
    from omni.isaac.core.prims import RigidPrimView
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType, SETTING_DISABLE_CONTACT_PROCESSING
    import carb.settings

    stage = omni.usd.get_context().get_stage()
    paths = [str(p.GetPath()) for p in stage.Traverse()
             if '/Air_0/' in str(p.GetPath()) and str(p.GetPath()).startswith('/World/envs/')
             and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    # This 2023 runtime accepts one path expression, not a list of paths.
    suffixes = sorted({p.split('/Air_0/', 1)[1] for p in paths})
    views = []
    for i, suffix in enumerate(suffixes):
        view = RigidPrimView('/World/envs/env_*/Air_0/' + suffix,
                             name=f'reaction_link_{i}', reset_xform_properties=False)
        view.initialize()
        views.append(view)
    body_paths = [p for view in views for p in view.prim_paths]
    assert set(body_paths) == set(paths), 'Rigid-body inventory coverage mismatch'
    bats = RigidPrimView('/World/envs/env_*/Air_0/bat', name='reaction_bats', reset_xform_properties=False)
    bats.initialize()
    env_ids = torch.tensor([int(re.search(r'/env_(\d+)/', p).group(1)) for p in body_paths], device=base.device)
    masses = torch.cat([v.get_masses().flatten() for v in views]).to(base.device)
    ball_mass = base.ball.get_masses().reshape(base.num_envs, 1).to(base.device)
    total_mass = torch.zeros(base.num_envs, device=base.device).index_add_(0, env_ids, masses)
    assert torch.isfinite(masses).all() and (masses > 0).all()
    dt = base.sim.get_physics_dt()
    gravity_step = torch.tensor([0., 0., -9.81 * dt], device=base.device)
    settings = carb.settings.get_settings()
    previous = settings.get_as_bool(SETTING_DISABLE_CONTACT_PROCESSING)
    settings.set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
    interface = get_physx_simulation_interface()

    def drone_momentum():
        values = masses[:, None] * torch.cat([v.get_velocities() for v in views])[:, :3]
        return torch.zeros(base.num_envs, 3, device=base.device).index_add_(0, env_ids, values)

    try:
        control = []
        for trial in ['control', 'impact']:
            env.reset()
            masses = torch.cat([v.get_masses().flatten() for v in views]).to(base.device)
            ball_mass = base.ball.get_masses().reshape(base.num_envs, 1).to(base.device)
            total_mass = torch.zeros(base.num_envs, device=base.device).index_add_(0, env_ids, masses)
            pos = base.envs_positions[:, None, :].clone()
            pos[..., 2] += 3.
            quat = torch.zeros(base.num_envs, 1, 4, device=base.device); quat[..., 0] = 1.
            base.drone.set_world_poses(pos, quat)
            base.drone.set_velocities(torch.zeros(base.num_envs, 1, 6, device=base.device))
            base.sim.step(render=False)
            ball_pos = bats.get_world_poses()[0].clone().reshape(base.num_envs, 1, 3)
            ball_pos[..., 2] += .20
            if trial == "control":
                ball_pos[..., 0] += 5.
            ball_vel = torch.zeros(base.num_envs, 1, 6, device=base.device); ball_vel[..., 2] = -2.
            base.ball.set_world_poses(ball_pos, quat)
            base.ball.set_velocities(ball_vel)
            impacts, frames = [], []
            credited=set()
            for step in range(16):
                drone_before = drone_momentum()
                ball_before = ball_mass * base.ball.get_velocities()[:, 0, :3]
                base.sim.step(render=False)
                drone_after = drone_momentum()
                ball_after = ball_mass * base.ball.get_velocities()[:, 0, :3]
                drone_impulse = drone_after - drone_before - total_mass[:, None] * gravity_step
                ball_impulse = ball_after - ball_before - ball_mass * gravity_step
                assert torch.isfinite(drone_impulse).all() and torch.isfinite(ball_impulse).all()
                headers, data = interface.get_contact_report()
                if trial == 'control':
                    control.append({'before': drone_before.clone(), 'after': drone_after.clone(),
                                    'background_impulse': drone_impulse.clone()})
                    continue
                for h in headers:
                    if h.type == ContactEventType.CONTACT_LOST:
                        continue
                    # Contact margins can report FOUND before actual impact.
                    impulse_sum=sum(sum(float(v)*float(v) for v in data[h.contact_data_offset+j].impulse)**.5 for j in range(h.num_contact_data))
                    if impulse_sum<=1e-6:continue
                    actors = [str(PhysicsSchemaTools.intToSdfPath(h.actor0)), str(PhysicsSchemaTools.intToSdfPath(h.actor1))]
                    if not any(p.endswith('/ball') for p in actors) or not any(p.endswith('/bat') for p in actors):
                        continue
                    index = int(re.search(r'/env_(\d+)/', next(p for p in actors if p.endswith('/ball'))).group(1))
                    if index in credited:continue
                    credited.add(index)
                    ball_j = ball_impulse[index]
                    raw_drone_j = drone_impulse[index]
                    background = control[step]['background_impulse'][index]
                    drone_j = raw_drone_j - background
                    pre_error = float(torch.linalg.norm(drone_before[index] - control[step]['before'][index]).item())
                    residual = float((torch.linalg.norm(ball_j + drone_j) / torch.linalg.norm(ball_j).clamp_min(1e-8)).item())
                    impacts.append({'env_id': index, 'step': step, 'actors': actors,
                                    'ball_impulse': ball_j.cpu().tolist(), 'drone_impulse': drone_j.cpu().tolist(),
                                    'relative_momentum_residual': residual,
                                    'raw_drone_impulse': raw_drone_j.cpu().tolist(),
                                    'control_background_impulse': background.cpu().tolist(),
                                    'precontact_drone_momentum_control_error': pre_error,
                                    'passed': bool(ball_j[2] > .01 and drone_j[2] < -.01 and residual < .01 and pre_error < 1e-5)})
                frames.append({'step': step, 'drone_momentum': drone_after.cpu().tolist(),
                               'ball_momentum': ball_after.cpu().tolist()})
        result = {'passed': len(impacts) == base.num_envs and len({v['env_id'] for v in impacts}) == base.num_envs
                  and all(v['passed'] for v in impacts), 'fixture_only': True,
                  'scope': 'paired free-fall control and centered ball-bat impact per environment; linear reaction momentum',
                  'correction': 'subtract measured same-step control drone momentum residual; precontact equality required',
                  'dt': dt, 'body_paths': body_paths, 'body_masses': masses.cpu().tolist(),
                  'drone_total_mass': total_mass.cpu().tolist(), 'ball_mass': ball_mass.cpu().tolist(),
                  'impacts': impacts, 'frames': frames}
        record(reaction_checks=result)
        return result
    finally:
        settings.set_bool(SETTING_DISABLE_CONTACT_PROCESSING, previous)
