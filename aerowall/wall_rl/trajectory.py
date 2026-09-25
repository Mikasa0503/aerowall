"""Ballistic intercept and outbound feasibility, without simulator writes."""

from __future__ import annotations

import math

DEFAULT_WALL_Y_BOUNDS = (-2.8, 2.8)
DEFAULT_WALL_Z_BOUNDS = (0.3, 7.7)
DEFAULT_INTERCEPT_X_BOUNDS = (0.5, 6.0)


def predict_intercept(ball_position, ball_velocity, plane_z=2.18, wall_x=0.2,
                      restitution=0.8, wall_y_bounds=DEFAULT_WALL_Y_BOUNDS,
                      wall_z_bounds=DEFAULT_WALL_Z_BOUNDS):
    """Predict a descending crossing and reflect only from the finite wall."""
    x, y, z = ball_position
    vx, vy, vz = ball_velocity
    discriminant = vz * vz + 19.62 * (z - plane_z)
    if discriminant < 0:
        return None
    t = (vz + math.sqrt(discriminant)) / 9.81
    if not 0 < t <= 2.0:
        return None
    ix = x + vx * t
    wall_t = None
    if vx < -0.05:
        candidate_wall_t = (wall_x - x) / vx
        if 0 < candidate_wall_t < t:
            wall_y = y + vy * candidate_wall_t
            wall_z = z + vz * candidate_wall_t - 4.905 * candidate_wall_t ** 2
            finite_wall_hit = (wall_y_bounds[0] <= wall_y <= wall_y_bounds[1]
                               and wall_z_bounds[0] <= wall_z <= wall_z_bounds[1])
            if finite_wall_hit:
                wall_t = candidate_wall_t
                ix = wall_x - restitution * vx * (t - candidate_wall_t)
    return {"position": (ix, y + vy * t, plane_z), "time": t,
            "wall_time": wall_t}


def predict_intercept_batch(ball_pos, ball_vel, drone_pos, plane_z=2.18,
                            wall_x=0.2, restitution=0.8,
                            wall_y_bounds=DEFAULT_WALL_Y_BOUNDS,
                            wall_z_bounds=DEFAULT_WALL_Z_BOUNDS):
    """Torch-vectorized version; imports Torch only inside the simulator path."""
    import torch

    vx, vy, vz = ball_vel.unbind(-1)
    x, y, z = ball_pos.unbind(-1)
    disc = vz.square() + 19.62 * (z - plane_z)
    t = (vz + disc.clamp_min(0).sqrt()) / 9.81
    valid_t = (disc >= 0) & (t > 0) & (t <= 2.0)
    wall_t = (wall_x - x) / torch.where(vx < -0.05, vx, torch.full_like(vx, -0.05))
    candidate_y = y + vy * wall_t
    candidate_z = z + vz * wall_t - 4.905 * wall_t.square()
    finite_wall_hit = (candidate_y >= wall_y_bounds[0]) & (candidate_y <= wall_y_bounds[1])
    finite_wall_hit &= (candidate_z >= wall_z_bounds[0]) & (candidate_z <= wall_z_bounds[1])
    bounce = (vx < -0.05) & (wall_t > 0) & (wall_t < t) & finite_wall_hit
    ix = torch.where(bounce, wall_x - restitution * vx * (t - wall_t), x + vx * t)
    target = torch.stack((ix, y + vy * t, torch.full_like(ix, plane_z)), -1)
    distance = (target - drone_pos).norm(dim=-1)
    target_bounds_valid = (target[:, 0] >= DEFAULT_INTERCEPT_X_BOUNDS[0])
    target_bounds_valid &= target[:, 0] <= DEFAULT_INTERCEPT_X_BOUNDS[1]
    target_bounds_valid &= target[:, 1].abs() <= DEFAULT_WALL_Y_BOUNDS[1]
    feasible = valid_t & target_bounds_valid
    feasible &= distance <= 0.35 + 2.0 * t
    return target, t.clamp(0, 2), feasible


def intercept_target_geometry_valid_batch(ball_position, ball_velocity, intercept,
                                          plane_z=2.18):
    """Validate plane-crossing time and bounds without applying drone reachability."""
    import torch

    z = ball_position[:, 2]
    vz = ball_velocity[:, 2]
    discriminant = vz.square() + 19.62 * (z - plane_z)
    crossing_time = (vz + discriminant.clamp_min(0).sqrt()) / 9.81
    crossing_valid = (discriminant >= 0) & (crossing_time > 0) & (crossing_time <= 2.0)
    bounds_valid = (intercept[:, 0] >= DEFAULT_INTERCEPT_X_BOUNDS[0])
    bounds_valid &= intercept[:, 0] <= DEFAULT_INTERCEPT_X_BOUNDS[1]
    bounds_valid &= intercept[:, 1].abs() <= DEFAULT_WALL_Y_BOUNDS[1]
    return crossing_valid & bounds_valid


def outbound_quality(ball_pos, ball_vel, wall_target, drone_pos, wall_x=0.2,
                     wall_y_bounds=DEFAULT_WALL_Y_BOUNDS,
                     wall_z_bounds=DEFAULT_WALL_Z_BOUNDS, restitution=0.8):
    """Score a legal outbound ball by target hit and reachable return, in [0,1]."""
    import torch

    x, y, z = ball_pos.unbind(-1)
    vx, vy, vz = ball_vel.unbind(-1)
    wall_t = (wall_x - x) / torch.where(vx < -0.05, vx, torch.full_like(vx, -0.05))
    wall_y = y + vy * wall_t
    wall_z = z + vz * wall_t - 4.905 * wall_t.square()
    valid = (vx < -0.05) & (wall_t > 0) & (wall_t < 2.0)
    valid &= (wall_y >= wall_y_bounds[0]) & (wall_y <= wall_y_bounds[1])
    valid &= (wall_z >= wall_z_bounds[0]) & (wall_z <= wall_z_bounds[1])
    target_error = ((wall_y - wall_target[:, 0]) / 1.0).square()
    target_error += ((wall_z - wall_target[:, 1]) / 1.5).square()
    # Reflect the horizontal velocity at the wall and predict the next
    # descending racket-height crossing from the physical wall impact state.
    wall_pos = torch.stack((torch.full_like(x, wall_x), wall_y, wall_z), -1)
    wall_vel = torch.stack((-restitution * vx, vy, vz - 9.81 * wall_t), -1)
    return_point, return_t, reachable = predict_intercept_batch(
        wall_pos, wall_vel, drone_pos, wall_x=wall_x,
        wall_y_bounds=wall_y_bounds, wall_z_bounds=wall_z_bounds,
        restitution=restitution,
    )
    return torch.where(valid & reachable & (return_t > 0.05),
                       torch.exp(-target_error), torch.zeros_like(x))


def target_quality(ball_position, wall_target, *, lateral_scale=0.7,
                   vertical_scale=1.0):
    """Smooth [0,1] score for a measured or predicted wall impact location."""
    import torch

    error = (ball_position[..., 1] - wall_target[..., 0]) / lateral_scale
    error = error.square() + ((ball_position[..., 2] - wall_target[..., 1]) / vertical_scale).square()
    return torch.exp(-0.5 * error)


def outbound_quality_v2(ball_pos, ball_vel, wall_target, drone_pos, wall_x=0.2,
                        wall_y_bounds=DEFAULT_WALL_Y_BOUNDS,
                        wall_z_bounds=DEFAULT_WALL_Z_BOUNDS, restitution=0.8):
    """Dense but bounded target/reachability score for the causal_v2 ablation.

    A valid path to the finite wall receives a smooth wall-target score. The
    return term decays with the distance beyond the drone's conservative
    reachable radius, so merely hitting the wall low or far off-axis gives a
    weak learning signal instead of a binary zero.
    """
    import torch

    x, y, z = ball_pos.unbind(-1)
    vx, vy, vz = ball_vel.unbind(-1)
    wall_t = (wall_x - x) / torch.where(vx < -0.05, vx, torch.full_like(vx, -0.05))
    wall_y = y + vy * wall_t
    wall_z = z + vz * wall_t - 4.905 * wall_t.square()
    valid = (vx < -0.05) & (wall_t > 0) & (wall_t < 2.0)
    valid &= (wall_y >= wall_y_bounds[0]) & (wall_y <= wall_y_bounds[1])
    valid &= (wall_z >= wall_z_bounds[0]) & (wall_z <= wall_z_bounds[1])
    wall_position = torch.stack((torch.full_like(x, wall_x), wall_y, wall_z), -1)
    target_score = target_quality(wall_position, wall_target)

    wall_velocity = torch.stack((-restitution * vx, vy, vz - 9.81 * wall_t), -1)
    return_point, return_t, return_feasible = predict_intercept_batch(
        wall_position, wall_velocity, drone_pos, wall_x=wall_x,
        wall_y_bounds=wall_y_bounds, wall_z_bounds=wall_z_bounds,
        restitution=restitution,
    )
    distance = (return_point - drone_pos).norm(dim=-1)
    reachable_radius = 0.35 + 2.0 * return_t
    reach_score = torch.exp(-torch.relu(distance - reachable_radius) / 0.5)
    reach_score = torch.where(return_feasible, torch.ones_like(reach_score), reach_score)
    score = target_score * (0.25 + 0.75 * reach_score)
    return torch.where(valid & (return_t > 0.05), score, torch.zeros_like(score))
