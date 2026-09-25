"""Versioned reward functions; the accepted legacy reward stays the default."""

from __future__ import annotations


def intercept_potential(drone_pos, intercept, feasible):
    distance = (drone_pos - intercept).norm(dim=-1)
    return -distance.div(1.5).clamp(0, 1) * feasible.float()


def compute_skill_reward(*, potential_before, potential_after, approach_mask,
                         legal_contact, next_contact, legal_wall, outbound_quality,
                         illegal, crash, out, action, previous_action,
                         wall_quality=None, gamma=0.995):
    """One control-step reward from disjoint, one-shot physical events."""
    shaping = 0.5 * (gamma * potential_after - potential_before) * approach_mask.float()
    smoothness = 0.002 * (action - previous_action).square().sum(dim=-1)
    wall_event = 10.0 * legal_wall.float()
    if wall_quality is not None:
        wall_event = legal_wall.float() * (2.0 + 8.0 * wall_quality)
    return (shaping + 2.0 * legal_contact.float()
            + 2.0 * outbound_quality * legal_contact.float()
            + wall_event + 30.0 * next_contact.float()
            - 10.0 * illegal.float() - 20.0 * crash.float()
            - 10.0 * out.float() - smoothness)


def compute_aerowall_causal_v3_reward_terms(*, potential_before, potential_after,
                                            potential_before_valid=None,
                                            potential_after_valid=None,
                                            approach_mask, terminal=None,
                                            legal_contact, next_contact, legal_wall,
                                            outbound_quality, illegal, crash, out,
                                            action, previous_action, wall_quality=None,
                                            gamma=0.995):
    """Return AeroWall's causal-v3 reward components with safe boundaries.

    A transiently invalid prediction suppresses shaping for that transition
    instead of turning an invalid potential (zero) into an artificial gain.
    Terminal transitions receive event/failure rewards but no shaping delta.
    """
    import torch

    valid_before = (torch.ones_like(potential_before, dtype=torch.bool)
                    if potential_before_valid is None else potential_before_valid.bool())
    valid_after = valid_before if potential_after_valid is None else potential_after_valid.bool()
    if terminal is None:
        terminal_mask = torch.zeros_like(valid_before)
    else:
        terminal_mask = terminal.bool()
    shaping_mask = approach_mask & valid_before & valid_after & ~terminal_mask
    shaping = 0.5 * (gamma * potential_after - potential_before) * shaping_mask.float()
    contact = 2.0 * legal_contact.float()
    outbound = 2.0 * outbound_quality * legal_contact.float()
    if wall_quality is None:
        wall = 10.0 * legal_wall.float()
    else:
        wall = legal_wall.float() * (2.0 + 8.0 * wall_quality)
    rally = 30.0 * next_contact.float()
    illegal_penalty = -10.0 * illegal.float()
    crash_penalty = -20.0 * crash.float()
    out_penalty = -10.0 * out.float()
    action_smoothness = -0.002 * (action - previous_action).square().sum(dim=-1)
    terms = {
        "approach_shaping": shaping,
        "legal_contact": contact,
        "outbound_quality": outbound,
        "wall_quality": wall,
        "next_contact": rally,
        "illegal_penalty": illegal_penalty,
        "crash_penalty": crash_penalty,
        "out_penalty": out_penalty,
        "action_smoothness": action_smoothness,
    }
    terms["total"] = sum(terms.values())
    return terms


def compute_aerowall_causal_v5_reward_terms(*, potential_before, potential_after,
                                            potential_before_valid=None,
                                            potential_after_valid=None,
                                            approach_mask, terminal=None,
                                            legal_contact, next_contact, legal_wall,
                                            outbound_quality, illegal, phase0_illegal,
                                            phase2_illegal, crash, out, action,
                                            previous_action, wall_quality=None,
                                            gamma=0.995):
    """AeroWall causal-v5: add a targeted late-return safety cost.

    Reuse causal-v4 in full, then add -30 for phase-2 illegal contact events.
    Phase 0 and phase 2 therefore carry a total -40 illegal-contact cost;
    phase 1 remains at -10.
    """
    terms = compute_aerowall_causal_v4_reward_terms(
        potential_before=potential_before,
        potential_after=potential_after,
        potential_before_valid=potential_before_valid,
        potential_after_valid=potential_after_valid,
        approach_mask=approach_mask,
        terminal=terminal,
        legal_contact=legal_contact,
        next_contact=next_contact,
        legal_wall=legal_wall,
        outbound_quality=outbound_quality,
        illegal=illegal,
        phase0_illegal=phase0_illegal,
        crash=crash,
        out=out,
        action=action,
        previous_action=previous_action,
        wall_quality=wall_quality,
        gamma=gamma,
    )
    terms.pop("total")
    terms["phase2_return_safety_penalty"] = -30.0 * phase2_illegal.float()
    terms["total"] = sum(terms.values())
    return terms


def compute_aerowall_causal_v6_reward_terms(*, potential_before, potential_after,
                                            potential_before_valid=None,
                                            potential_after_valid=None,
                                            approach_mask, terminal=None,
                                            legal_contact, next_contact, legal_wall,
                                            outbound_quality, illegal, phase0_illegal,
                                            phase2_illegal, crash, out, action,
                                            previous_action, wall_quality=None,
                                            gamma=0.995):
    """AeroWall causal-v6: further target first-intercept safety.

    Reuse causal-v5 (phase 0 and phase 2 total -40), then add -40 only to
    phase-0 illegal contact events, making phase 0 -80 and phase 2 -40.
    """
    terms = compute_aerowall_causal_v5_reward_terms(
        potential_before=potential_before,
        potential_after=potential_after,
        potential_before_valid=potential_before_valid,
        potential_after_valid=potential_after_valid,
        approach_mask=approach_mask,
        terminal=terminal,
        legal_contact=legal_contact,
        next_contact=next_contact,
        legal_wall=legal_wall,
        outbound_quality=outbound_quality,
        illegal=illegal,
        phase0_illegal=phase0_illegal,
        phase2_illegal=phase2_illegal,
        crash=crash,
        out=out,
        action=action,
        previous_action=previous_action,
        wall_quality=wall_quality,
        gamma=gamma,
    )
    terms.pop("total")
    terms["phase0_intercept_safety_penalty_v6"] = -40.0 * phase0_illegal.float()
    terms["total"] = sum(terms.values())
    return terms


def compute_aerowall_causal_v4_reward_terms(*, potential_before, potential_after,
                                            potential_before_valid=None,
                                            potential_after_valid=None,
                                            approach_mask, terminal=None,
                                            legal_contact, next_contact, legal_wall,
                                            outbound_quality, illegal, phase0_illegal,
                                            crash, out, action, previous_action,
                                            wall_quality=None, gamma=0.995):
    """AeroWall causal-v4: increase only first-intercept illegal-contact cost.

    All causal-v3 terms are reused unchanged. The extra -30 applies only to
    illegal contacts observed before the first expected legal cap, making the
    phase-0 penalty -40 while leaving other phases at -10.
    """
    terms = compute_aerowall_causal_v3_reward_terms(
        potential_before=potential_before,
        potential_after=potential_after,
        potential_before_valid=potential_before_valid,
        potential_after_valid=potential_after_valid,
        approach_mask=approach_mask,
        terminal=terminal,
        legal_contact=legal_contact,
        next_contact=next_contact,
        legal_wall=legal_wall,
        outbound_quality=outbound_quality,
        illegal=illegal,
        crash=crash,
        out=out,
        action=action,
        previous_action=previous_action,
        wall_quality=wall_quality,
        gamma=gamma,
    )
    terms.pop("total")
    phase0_penalty = -30.0 * phase0_illegal.float()
    terms["phase0_intercept_safety_penalty"] = phase0_penalty
    terms["total"] = sum(terms.values())
    return terms
