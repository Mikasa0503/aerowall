"""Pure scoring and ballistic helpers for the single-drone wall experiment."""

from __future__ import annotations

import math


def center_band_wall_reward(y):
    """Bounded reward for the measured lateral position of a legal wall hit.

    Accepts a scalar for offline audit or a Torch tensor in the simulator.
    The quadratic center scores every visited wall position. It reaches zero
    at the 1 m acceptance boundary and -10 at 1.5 m from the wall center.
    """
    radius = abs(y)
    if hasattr(radius, "clamp"):
        inner = radius.clamp(0.0, 1.0)
        outer = (radius - 1.0).clamp(0.0, 0.5)
    else:
        inner = max(0.0, min(1.0, radius))
        outer = max(0.0, min(0.5, radius - 1.0))
    return 10.0 - 10.0 * inner * inner - 20.0 * outer


def recenter_prediction(y: float, z: float, vx: float, vy: float, vz: float,
                        wall_x: float = 0.2, x: float = 1.5):
    """Predict lateral landing coordinates after a legal outbound cap.

    Returns None for geometry where the desired return cannot be inferred.
    The runtime environment uses the same equations in vectorized Torch form.
    """
    discriminant = vz * vz + 19.62 * (z - 2.18)
    if discriminant < 0 or vx >= -0.05:
        return None
    duration = (vz + math.sqrt(discriminant)) / 9.81
    wall_time = (wall_x - x) / vx
    if not (0 < wall_time < duration <= 2.0):
        return None
    desired_vy = max(-1.2, min(1.2, -y / duration))
    return {
        "time_to_return": duration,
        "time_to_wall": wall_time,
        "desired_vy": desired_vy,
        "return_y": y + vy * duration,
        "wall_y": y + vy * wall_time,
    }


def score_rally_events(events, center_band: float = 1.0):
    """Score a first episode from ordered, audited cap/wall event records.

    Each record has ``kind``, ``y``, ``step``, and ``substep``. Duplicate
    reports in one physical contact window must be removed by the caller.
    """
    phase = "cap"
    caps = walls = rallies = centered_prefix = 0
    wall_y = []
    for event in events:
        kind = event["kind"]
        if kind == "cap" and phase == "cap":
            caps += 1
            if caps == 1:
                phase = "wall"
            else:
                rallies += 1
                phase = "wall"
        elif kind == "wall" and phase == "wall":
            walls += 1
            wall_y.append(float(event["y"]))
            if centered_prefix == walls - 1 and abs(event["y"]) <= center_band:
                centered_prefix += 1
            phase = "cap"
    return {
        "caps": caps,
        "walls": walls,
        "rallies": rallies,
        "centered_prefix_rallies": min(centered_prefix, rallies),
        "wall_y_sequence": wall_y,
        "safe10": rallies >= 10 and centered_prefix >= 10 and caps >= 11,
        "safe15": rallies >= 15 and centered_prefix >= 15 and caps >= 16,
    }


def score_recovery(events, impulse_end, failed: bool, timed_out: bool = False,
                   center_band: float = 1.0):
    """Apply the predeclared 3-wall return + 2-rally recovery contract.

    Events are audited physical events in chronological order. A success
    reached before a later failure remains a success.
    """
    phase = "cap"
    walls = rallies = 0
    candidate_wall = None
    achieved = False
    for event in events:
        if event["kind"] == "cap" and phase == "cap":
            if walls:
                rallies += 1
            phase = "wall"
            if candidate_wall is not None and rallies >= candidate_wall + 2:
                achieved = True
        elif event["kind"] == "wall" and phase == "wall":
            walls += 1
            phase = "cap"
            marker = (int(event["step"]), int(event["substep"]))
            if (candidate_wall is None and impulse_end is not None
                    and marker > impulse_end and 3 <= walls <= 5
                    and abs(event["y"]) <= center_band):
                candidate_wall = walls
        if event.get("failure_before"):
            break
    return {
        "recovery_wall_ordinal": candidate_wall,
        "post_recovery_rallies": max(0, rallies - (candidate_wall or rallies)),
        "recovery_success": achieved,
        "failed": failed,
        "timed_out": timed_out,
    }
