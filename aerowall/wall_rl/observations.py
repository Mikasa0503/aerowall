"""Versioned wall-policy observations with explicit coordinate semantics."""


def _quat_rotate_inverse(quaternion, vector):
    """Rotate vectors by the inverse of scalar-first unit quaternions."""
    import torch

    scalar = quaternion[..., :1]
    axis = quaternion[..., 1:]
    cross = torch.cross(axis, vector, dim=-1)
    projection = (axis * vector).sum(dim=-1, keepdim=True)
    return vector * (2.0 * scalar.square() - 1.0) - 2.0 * scalar * cross + 2.0 * axis * projection


def _build_relative_observation(goto_prefix, root, ball_pos, ball_vel,
                                intercept, flight_time, feasible, phase,
                                wall_target, previous_action, *, legacy_velocity):
    import torch
    import torch.nn.functional as functional

    drone_pos = root[..., :3]
    drone_body_vel = root[..., 7:10]
    quat = root[..., 3:7]
    rel_pos = _quat_rotate_inverse(quat, ball_pos - drone_pos)
    if legacy_velocity:
        # Preserve the original relative_v2 feature semantics for reproducing
        # checkpoints trained before the coordinate-frame bug was corrected.
        rel_vel = _quat_rotate_inverse(quat, ball_vel) - drone_body_vel
    else:
        # root[..., 7:10] is world-axis velocity in HCSP's state contract.
        # Subtract in one coordinate frame, then rotate the difference.
        rel_vel = _quat_rotate_inverse(quat, ball_vel - drone_body_vel)
    rel_intercept = _quat_rotate_inverse(quat, intercept - drone_pos)
    phase_onehot = functional.one_hot(phase, 3).float()[:, None]
    target = wall_target[:, None]
    extras = torch.cat((rel_pos, rel_vel, rel_intercept, phase_onehot,
                        target, flight_time[:, None, None],
                        feasible.float()[:, None, None], previous_action), -1)
    assert goto_prefix.shape[-1] == 26 and extras.shape[-1] == 20
    return torch.cat((goto_prefix, extras), -1)


def build_relative_observation(goto_prefix, root, ball_pos, ball_vel,
                               intercept, flight_time, feasible, phase,
                               wall_target, previous_action):
    """Build relative_v3: 46 features with body-frame relative velocity."""
    return _build_relative_observation(
        goto_prefix, root, ball_pos, ball_vel, intercept, flight_time,
        feasible, phase, wall_target, previous_action, legacy_velocity=False,
    )


def build_relative_observation_v2(goto_prefix, root, ball_pos, ball_vel,
                                  intercept, flight_time, feasible, phase,
                                  wall_target, previous_action):
    """Reproduce historical relative_v2, including its mixed-frame velocity."""
    return _build_relative_observation(
        goto_prefix, root, ball_pos, ball_vel, intercept, flight_time,
        feasible, phase, wall_target, previous_action, legacy_velocity=True,
    )


def build_aerowall_intercept_target_retention_v1_observation(
        relative_v3, drone_position, intercept, geometry_valid, phase):
    """Retain a valid AeroWall intercept target for phase-0 Launch guidance.

    This evaluation-only view changes the first three goto-delta channels when
    a bounded plane crossing exists but the conservative drone-reachability
    test may fail. The remaining relative_v3 features and all non-phase-0
    goto targets are copied unchanged.
    """
    import torch

    if relative_v3.shape[-1] != 46:
        raise ValueError("AeroWall target-retention view requires 46-D relative_v3 observations")
    target_delta = intercept - torch.tensor(
        (0.0, 0.0, 0.18), dtype=intercept.dtype, device=intercept.device,
    ) - drone_position
    use_intercept = geometry_valid & (phase == 0)
    view = relative_v3.clone()
    view[..., :3] = torch.where(
        use_intercept[:, None, None], target_delta[:, None, :], view[..., :3],
    )
    return view


def build_aerowall_goal_observation(legacy_observation, wall_target,
                                   y_bounds=(-0.3, 0.3),
                                   z_bounds=(3.8, 4.2)):
    """Append a normalized wall-goal command while preserving legacy features.

    The original 46 values keep their exact order and meaning. The two added
    channels encode lateral and vertical wall-goal offsets relative to the
    configured target range, so a zero-initialized policy extension reproduces
    the source actor before it learns to use the command.
    """
    import torch

    if legacy_observation.shape[-1] != 46 or wall_target.shape[-1] != 2:
        raise ValueError("AeroWall goal observation requires 46 legacy features and a 2D wall target")
    y_mid, z_mid = (float(y_bounds[0]) + float(y_bounds[1])) / 2.0, (float(z_bounds[0]) + float(z_bounds[1])) / 2.0
    y_half, z_half = (float(y_bounds[1]) - float(y_bounds[0])) / 2.0, (float(z_bounds[1]) - float(z_bounds[0])) / 2.0
    if y_half <= 0.0 or z_half <= 0.0:
        raise ValueError("wall-goal target bounds must be increasing")
    goal = torch.stack(((wall_target[..., 0] - y_mid) / y_half,
                        (wall_target[..., 1] - z_mid) / z_half), dim=-1)
    return torch.cat((legacy_observation, goal.unsqueeze(-2)), dim=-1)
