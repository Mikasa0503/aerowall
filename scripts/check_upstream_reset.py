"""Runtime checks on actual upstream state, independent of reward hit counters."""
import torch


def check_selective_reset(env, base, controller, td):
    mask = torch.zeros(base.num_envs, 1, dtype=torch.bool, device=base.device)
    mask[[0, 3, 7]] = True
    selected = mask[:, 0]
    untouched = ~selected

    def snapshot():
        ball_pos, ball_rot = base.ball.get_world_poses()
        drone_pos, drone_rot = base.drone.get_world_poses()
        return {name: value.clone() for name, value in {
            'ball_pos': ball_pos, 'ball_rot': ball_rot, 'ball_vel': base.ball.get_velocities(),
            'drone_pos': drone_pos, 'drone_rot': drone_rot, 'drone_vel': base.drone.get_velocities(),
            'progress': base.progress_buf, 'restitution': base.current_restitution,
            'hit_time': base.last_hit_t, 'hit_step': base.last_hit_step,
        }.items()}

    before = snapshot()
    controller_before = {'integ': controller.integ.clone(),
                         'last_body_rate': controller.last_body_rate.clone()}
    request = td.clone()
    request.set('_reset', mask)
    reset_td = env.reset(request)
    after = snapshot()
    history_unchanged = {name: bool(torch.equal(value[untouched], getattr(controller, name)[untouched]))
                         for name, value in controller_before.items()}
    unchanged = {name: bool(torch.equal(before[name][untouched], value[untouched]))
                 for name, value in after.items()}
    cleared = {'progress': bool((base.progress_buf[selected] == 0).all()),
               'hit_time': bool((base.last_hit_t[selected] == -999).all()),
               'hit_step': bool((base.last_hit_step[selected] == -100).all()),
               'near_ball': bool((~base.racket_near_ball[selected]).all()),
               'ball_height_max': bool((base.ball_height_max[selected] == 0).all())}
    state = reset_td['info', 'drone_state'][..., :13].clone()
    rates = torch.zeros(base.num_envs, 1, 3, device=base.device)
    thrust = torch.full((base.num_envs, 1, 1), 9.81, device=base.device)
    # The upstream transform obtains reset_pid from done, not InitTracker.is_init.
    actual_mask = reset_td['done'].expand(-1, state.shape[1])
    from omni_drones.controllers import PID_controller_flightmare
    fresh = PID_controller_flightmare(base.cfg.sim.dt, base.drone.params, base.device).to(base.device)
    expected = fresh(state, rates, thrust, torch.ones_like(actual_mask))
    actual = controller(state, rates, thrust, actual_mask)
    error = (actual[selected] - expected[selected]).abs().max().item()
    return {'selected_envs': [0, 3, 7], 'unselected_state_unchanged': unchanged,
            'selected_state_cleared': cleared,
            'physics_reset_passed': all(unchanged.values()) and all(cleared.values()),
            'post_reset_done': reset_td['done'][selected].tolist(),
            'post_reset_is_init': reset_td['is_init'][selected].tolist(),
            'unselected_controller_history_unchanged': history_unchanged,
            'controller_fresh_reset_max_error': error,
            'controller_reset_passed': error < 1e-6 and all(history_unchanged.values())}


def scene_inventory():
    import omni.usd
    from pxr import UsdGeom, UsdPhysics
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
                             useExtentsHint=False, ignoreVisibility=True)
    rows = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith('/World/envs/env_0/'):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.MassAPI):
            bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
            rows.append({'path': path, 'type': prim.GetTypeName(), 'usd_authored_mass': mass,
                         'collision': prim.HasAPI(UsdPhysics.CollisionAPI),
                         'usd_world_min': None if bound.IsEmpty() else list(bound.GetMin()),
                         'usd_world_max': None if bound.IsEmpty() else list(bound.GetMax())})
    return {'scope': 'USD authored geometry only; GPU tensor poses and masses can differ from USD',
            'prims': rows}
