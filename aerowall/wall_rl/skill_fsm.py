"""Event-driven skill selection. Switches take effect on the next control step."""

from __future__ import annotations

from enum import IntEnum


class Skill(IntEnum):
    INTERCEPT = 0
    HIT = 1
    RECOVER = 2


def select_skill(previous: Skill, *, caps: int, phase: int,
                 contact_time: float, feasible: bool, held_steps: int,
                 hit_window: float = 0.18, hit_enter_seconds: float | None = None,
                 hit_exit_seconds: float | None = None,
                 min_dwell_steps: int = 2,
                 artificial_hit: bool = False) -> Skill:
    """Choose the next executor; contact events override the dwell timer."""
    enter = hit_window if hit_enter_seconds is None else hit_enter_seconds
    exit_ = enter if hit_exit_seconds is None else hit_exit_seconds
    if caps > 0 and phase == 1:
        return Skill.RECOVER

    if (previous == Skill.HIT and artificial_hit and caps == 0 and phase == 0
            and contact_time > 0):
        return Skill.HIT
    if (previous == Skill.HIT and caps == 0 and phase in (0, 2)
            and feasible and 0 < contact_time <= exit_):
        return Skill.HIT

    if held_steps < min_dwell_steps:
        return previous
    if feasible and 0 < contact_time <= enter and phase in (0, 2):
        return Skill.HIT
    return Skill.INTERCEPT if caps == 0 else Skill.RECOVER


def select_skill_batch(previous, caps, phase, contact_time, feasible, held_steps,
                       hit_window=0.18, hit_enter_seconds=None,
                       hit_exit_seconds=None, min_dwell_steps=2,
                       artificial_hit=None):
    import torch

    enter = hit_window if hit_enter_seconds is None else hit_enter_seconds
    exit_ = enter if hit_exit_seconds is None else hit_exit_seconds
    result = torch.where(caps > 0, torch.full_like(previous, int(Skill.RECOVER)),
                         torch.full_like(previous, int(Skill.INTERCEPT)))
    hit = feasible & (contact_time > 0) & (contact_time <= enter)
    hit &= (phase == 0) | (phase == 2)
    result = torch.where(hit, torch.full_like(previous, int(Skill.HIT)), result)
    retain_hit = (previous == int(Skill.HIT)) & (caps == 0) & feasible
    retain_hit &= (contact_time > 0) & (contact_time <= exit_)
    retain_hit &= (phase == 0) | (phase == 2)
    if artificial_hit is not None:
        owner = (previous == int(Skill.HIT)) & artificial_hit & (caps == 0)
        owner &= (phase == 0) & (contact_time > 0)
        retain_hit |= owner
    result = torch.where(retain_hit, torch.full_like(previous, int(Skill.HIT)), result)

    # Legal contact is a physical handoff and cannot be delayed by dwell.
    contact_handoff = (caps > 0) & (phase == 1)
    result = torch.where(contact_handoff, torch.full_like(previous, int(Skill.RECOVER)), result)
    dwell_ready = held_steps >= min_dwell_steps
    return torch.where(dwell_ready | contact_handoff | retain_hit, result, previous)
