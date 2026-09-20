"""Controlled diagnostics on the actual upstream ball, bat and collision groups.

State placement is a test fixture, never a policy success trajectory. Bodies move
only through PhysX between fixture initialization and measurements.
"""
import re
import itertools
import torch
from contextlib import contextmanager


@contextmanager
def contact_reporting_before_initialization(enable_body_collisions=False):
    """Attach contact instrumentation before PhysX parses the upstream stage."""
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics
    from omni.isaac.core.simulation_context import SimulationContext
    original = SimulationContext.reset

    def reset_with_reports(context, *args, **kwargs):
        for prim in omni.usd.get_context().get_stage().Traverse():
            path = str(prim.GetPath())
            if enable_body_collisions and path.startswith('/World/envs/') and '/Air_0/' in path and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(True)
            if str(prim.GetPath()).startswith('/World/envs/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
        return original(context, *args, **kwargs)

    SimulationContext.reset = reset_with_reports
    try:
        yield
    finally:
        SimulationContext.reset = original


def check_contacts(env, base, record):
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
    import carb.settings
    from pxr import PhysxSchema, PhysicsSchemaTools
    from omni_drones.views import RigidPrimView

    stage = omni.usd.get_context().get_stage()
    settings = carb.settings.get_settings()
    previous_disabled = settings.get_as_bool(SETTING_DISABLE_CONTACT_PROCESSING)
    settings.set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
    for index in range(base.num_envs):
        prim = stage.GetPrimAtPath(f'/World/envs/env_{index}/ball')
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
    events, callback_errors = [], []
    phase, step_index = 'setup', 0

    def callback(headers, data):
        try:
            for header in headers:
                paths = {key: str(PhysicsSchemaTools.intToSdfPath(getattr(header, key)))
                         for key in ('actor0', 'actor1', 'collider0', 'collider1')}
                event = {'phase': phase, 'step': step_index, 'type': str(header.type), **paths,
                         'contact_count': int(header.num_contact_data)}
                if header.num_contact_data:
                    point = data[header.contact_data_offset]
                    event['position'] = list(point.position)
                    event['normal'] = list(point.normal)
                    event['impulse'] = str(point.impulse)
                events.append(event)
        except Exception as error:
            callback_errors.append(repr(error))

    interface = get_physx_simulation_interface()

    def physics_step():
        base.sim.step(render=False)
        headers, data = interface.get_contact_report()
        callback(headers, data)
    bats = RigidPrimView('/World/envs/env_*/Air_0/bat', reset_xform_properties=False, shape=(-1, 1))
    bats.initialize()
    report = {'passed': False, 'fixture_only': True, 'callback_errors': callback_errors,
              'contact_processing_was_disabled': previous_disabled,
              'report_method': 'get_contact_report_after_fetch_results'}
    try:
        env.reset()
        drone_pos = base.envs_positions[:, None, :].clone()
        drone_pos[..., 2] += 2.0
        quat = torch.zeros(base.num_envs, 1, 4, device=base.device)
        quat[..., 0] = 1
        base.drone.set_world_poses(drone_pos, quat)
        base.drone.set_velocities(torch.zeros(base.num_envs, 1, 6, device=base.device))
        physics_step()
        bat_pos = bats.get_world_poses()[0].clone()
        ball_pos = bat_pos.clone()
        ball_pos[..., 2] += 0.20
        ball_vel = torch.zeros(base.num_envs, 1, 6, device=base.device)
        ball_vel[..., 2] = -2.0
        base.ball.set_world_poses(ball_pos, quat)
        base.ball.set_velocities(ball_vel)
        phase = 'ball_bat'
        frames = []
        for step_index in range(20):
            physics_step()
            pos = base.ball.get_world_poses()[0].clone()
            vel = base.ball.get_velocities().clone()
            assert torch.isfinite(pos).all() and torch.isfinite(vel).all()
            frames.append({'step': step_index, 'ball_pos': pos.cpu().tolist(),
                           'ball_vel': vel.cpu().tolist(),
                           'bat_vel': bats.get_velocities().cpu().tolist()})
        bat_envs = set()
        cross_events = []
        for event in events:
            if event['phase'] != phase:
                continue
            ids = [int(match) for key in ('actor0', 'actor1')
                   for match in re.findall(r'/env_(\d+)/', event[key])]
            if len(ids) == 2 and ids[0] != ids[1]:
                cross_events.append(event)
            if '/bat' in event['actor0'] + event['actor1'] and event['contact_count']:
                bat_envs.update(ids)
        relative_vz = [[f['ball_vel'][e][0][2] - f['bat_vel'][e][0][2] for f in frames]
                       for e in range(base.num_envs)]
        reversal = [values[0] < -0.1 and max(values) > 0.1 for values in relative_vz]
        report.update(ball_bat_contact_envs=sorted(bat_envs), ball_bat_frames=frames,
                      relative_vertical_velocity_reversal=reversal,
                      unexpected_cross_environment_contacts=cross_events)
        record(contact_checks=report)

        # Exhaust all original ball pairs across environments. Each pair is
        # initialized to intersect; no state is overwritten during its flight.
        pair_results = []
        for first, second in itertools.combinations(range(base.num_envs), 2):
            parked = base.envs_positions[:, None, :].clone()
            parked[..., 0] += 100.
            parked[..., 2] = 3.
            base.ball.set_world_poses(parked, quat)
            base.ball.set_velocities(torch.zeros(base.num_envs, 1, 6, device=base.device))
            center = base.envs_positions[0] + torch.tensor([2., 0., 3.], device=base.device)
            pair_pos = torch.stack([center + torch.tensor([-.15, 0., 0.], device=base.device),
                                    center + torch.tensor([.15, 0., 0.], device=base.device)])[:, None, :]
            pair_vel = torch.zeros(2, 1, 6, device=base.device)
            pair_vel[0, 0, 0], pair_vel[1, 0, 0] = 2., -2.
            indices = torch.tensor([first, second], device=base.device)
            base.ball.set_world_poses(pair_pos, quat[:2], indices)
            base.ball.set_velocities(pair_vel, indices)
            phase = f'cross_environment_pair_{first}_{second}'
            pair_frames = []
            for step_index in range(10):
                physics_step()
                pos = base.ball.get_world_poses()[0][indices].clone()
                vel = base.ball.get_velocities()[indices].clone()
                pair_frames.append({'step': step_index, 'pos': pos.cpu().tolist(),
                                    'vel': vel.cpu().tolist(),
                                    'center_distance': float(torch.linalg.norm(pos[0] - pos[1]).item())})
            paths = {f'/World/envs/env_{first}/ball', f'/World/envs/env_{second}/ball'}
            pair_contacts = [e for e in events if e['phase'] == phase and
                             {e['actor0'], e['actor1']} == paths]
            crossed = pair_frames[-1]['pos'][0][0][0] > pair_frames[-1]['pos'][1][0][0]
            overlap = min(f['center_distance'] for f in pair_frames) < 2 * base.ball_radius
            vx_error = max(abs(f['vel'][0][0][0] - 2.) + abs(f['vel'][1][0][0] + 2.) for f in pair_frames)
            pair_results.append({'env_ids': [first, second], 'frames': pair_frames,
                                 'contact_events': pair_contacts, 'crossed': crossed,
                                 'overlap_opportunity': overlap, 'vx_error': vx_error,
                                 'passed': crossed and overlap and vx_error < 1e-5 and not pair_contacts})
        report.update(cross_environment_pair_checks=pair_results,
                      tested_pair_count=len(pair_results), contact_events=events,
                      isolation_scope='all original ball-ball environment pairs; other body types not exhaustively tested')
        report['passed'] = (len(bat_envs) == base.num_envs and all(reversal) and not cross_events
                            and not callback_errors and len(pair_results) == base.num_envs * (base.num_envs - 1) // 2
                            and all(row['passed'] for row in pair_results))
        record(contact_checks=report)
        return report
    finally:
        settings.set_bool(SETTING_DISABLE_CONTACT_PROCESSING, previous_disabled)
