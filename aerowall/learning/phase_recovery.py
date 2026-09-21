"""HCSP-inspired, phase-limited home recovery shaping for development.

This is a fixed-home recovery diagnostic, not a ball interception controller.
All inputs are current observable quantities; no state or action is modified.
HCSP reference: Receive_hover.py at 009961b8f5702dd0c1c943cef0e01e09dfcd138d.
"""
import torch


def phase_recovery_score(position, up_z, angular_velocity, ball_vx, phase, done, target):
    distance=(position-target).norm(dim=-1)
    pose=1/(1+(1.2*distance).square())
    upright=((up_z.clamp(-1,1)+1)/2).square()
    # Penalize rotation about every axis after launch; HCSP uses yaw spin.
    rotation=1/(1+angular_velocity.square().sum(-1))
    enabled=(((phase==1)&(ball_vx>.5))|(phase==2))&~done.bool()
    score=pose*(1+upright+rotation)*enabled.float()
    return score,enabled
