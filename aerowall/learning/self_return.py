"""Outgoing velocity for one physical wall reflection and a chosen receiving point.

This gravity-only reference shapes real cap events; it never modifies ball state.
"""
import torch


def self_return_velocity(position, wall_front, ball_radius, return_position,
                         flight_time=1., restitution=.8):
    assert flight_time>0 and restitution>0
    target=torch.as_tensor(return_position,dtype=position.dtype,device=position.device)
    plane=torch.as_tensor(wall_front,dtype=position.dtype,device=position.device)-ball_radius
    outward_distance=plane-position[...,0]
    return_distance=plane-target[...,0]
    vx=(outward_distance+return_distance/restitution)/flight_time
    vy=(target[...,1]-position[...,1])/flight_time
    vz=(target[...,2]-position[...,2])/flight_time+4.905*flight_time
    velocity=torch.stack([vx,vy,vz],-1)
    wall_time=outward_distance/vx.clamp_min(1e-8)
    valid=(outward_distance>.05)&(return_distance>.05)&(wall_time>0)&(wall_time<flight_time)&torch.isfinite(velocity).all(-1)
    return velocity,valid
