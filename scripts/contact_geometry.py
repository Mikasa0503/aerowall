"""Geometric cap diagnostic for the upstream identity-local Z cylinder.

Inputs use current PhysX body poses and world contact data. A paired side/bottom
fixture is required before using this diagnostic in a scored task rule.
"""
import torch


def classify_cylinder_cap(point, normal, body_position, body_quaternion, radius):
    from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse
    local = quat_rotate_inverse(body_quaternion.unsqueeze(0), (point-body_position).unsqueeze(0))[0]
    axis = quat_rotate(body_quaternion.unsqueeze(0), point.new_tensor([[0.,0.,1.]]))[0]
    alignment = torch.abs(torch.dot(normal, axis))
    top = bool(local[2] > 0 and torch.linalg.norm(local[:2]) <= radius+.01 and alignment >= .8)
    return top, local, axis, float(alignment.item())
