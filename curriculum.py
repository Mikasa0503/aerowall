"""Deterministic, disjoint initial-state banks for wall-volley experiments."""

from __future__ import annotations

import random

from .trajectory import predict_intercept


CASE_VERSION = 1
HANDOFF_CASE_VERSION = 1
RECOVER_HANDOFF_CASE_VERSION = 2


def handoff_actor_role(case: dict) -> str:
    """Return the actor role named by a serialized handoff state schema."""
    version = case.get("handoff_state_version")
    if version == HANDOFF_CASE_VERSION:
        role = "hit"
    elif version == RECOVER_HANDOFF_CASE_VERSION:
        role = "recover"
    else:
        raise ValueError("unsupported handoff state version")
    declared_role = case.get("handoff_actor_role")
    if declared_role is not None and declared_role != role:
        raise ValueError(f"handoff state version {version} must identify the {role} actor")
    if version == RECOVER_HANDOFF_CASE_VERSION and declared_role != "recover":
        raise ValueError("Recover handoff state must explicitly identify the recover actor")
    return role


def sample_hit_starts(drone_positions, plane_z=2.18, generator=None):
    """Vectorized Hit resets using the same legal window as ``make_hit_start``."""
    import torch

    if drone_positions.ndim != 2 or drone_positions.shape[-1] != 3:
        raise ValueError("drone_positions must have shape (batch, 3)")
    count = drone_positions.shape[0]
    rand = lambda: torch.rand(count, device=drone_positions.device, generator=generator)
    contact_time = 0.35 + 0.10 * rand()
    vx = -0.4 + 0.8 * rand()
    vy = -0.4 + 0.8 * rand()
    vz = -1.1 + 0.6 * rand()
    contact_x = drone_positions[:, 0] - 0.08 + 0.16 * rand()
    contact_y = drone_positions[:, 1] - 0.12 + 0.24 * rand()
    ball_position = torch.stack((
        contact_x - vx * contact_time,
        contact_y - vy * contact_time,
        plane_z - vz * contact_time + 4.905 * contact_time.square(),
    ), dim=-1)
    ball_velocity = torch.stack((vx, vy, vz), dim=-1)
    contact_position = torch.stack((contact_x, contact_y,
                                    torch.full_like(contact_time, plane_z)), dim=-1)
    return {"ball_position": ball_position, "ball_velocity": ball_velocity,
            "contact_position": contact_position, "time_to_contact": contact_time}


def make_hit_start(rng: random.Random) -> dict:
    """Sample a near-racket falling state with a provably reachable crossing."""
    drone = [rng.uniform(1.35, 1.65), rng.uniform(-0.15, 0.15), 2.0]
    time_to_contact = rng.uniform(0.35, 0.45)
    vx, vy = rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)
    vz = rng.uniform(-1.1, -0.5)
    # Keep the predicted contact inside the legal racket disk while varying
    # the incoming position and velocity seen by the parameterized Hit skill.
    contact_x = drone[0] + rng.uniform(-0.08, 0.08)
    contact_y = drone[1] + rng.uniform(-0.12, 0.12)
    ball = [contact_x - vx * time_to_contact,
            contact_y - vy * time_to_contact,
            2.18 - vz * time_to_contact + 4.905 * time_to_contact ** 2]
    velocity = [vx, vy, vz]
    prediction = predict_intercept(ball, velocity, plane_z=2.18)
    if prediction is None or abs(prediction["time"] - time_to_contact) > 1e-5:
        raise AssertionError("constructed Hit start has an invalid ballistic crossing")
    px, py, pz = prediction["position"]
    radial = ((px - drone[0]) ** 2 + (py - drone[1]) ** 2) ** 0.5
    if radial > 0.2 or not 0.0 <= pz - drone[2] <= 0.2:
        raise AssertionError("constructed Hit start is outside the legal racket region")
    return {"drone_position": drone, "ball_position": ball,
            "ball_velocity": velocity, "time_to_contact": time_to_contact,
            "contact_position": [px, py, pz]}


def case_is_feasible(case: dict) -> bool:
    """Reject states outside the scene or unreachable before the first drop."""
    x, y, z = case["ball_position"]
    vx, vy, vz = case["ball_velocity"]
    dx, dy, dz = case["drone_position"]
    if not (1.0 <= x <= 2.0 and abs(y) <= 0.9 and 3.8 <= z <= 5.3):
        return False
    if not (0.7 <= dx <= 2.3 and abs(dy) <= 1.1 and 1.5 <= dz <= 2.5):
        return False
    if not (abs(vx) <= 0.4 and abs(vy) <= 0.75 and -0.5 <= vz <= 0.2):
        return False
    # A conservative straight-line test. It is a sampling filter, not a
    # promise that a particular learned policy can reach the contact point.
    discriminant = vz * vz + 2 * 9.81 * (z - 2.18)
    if discriminant <= 0:
        return False
    t = (vz + discriminant ** 0.5) / 9.81
    intercept = (x + vx * t, y + vy * t, 2.0)
    distance = sum((a - b) ** 2 for a, b in zip(intercept, (dx, dy, dz))) ** 0.5
    return 0.2 <= t <= 1.2 and distance <= 0.35 + 2.0 * t


def validate_handoff_case(case: dict) -> None:
    """Validate a state-complete, PhysX-audited Hit or Recover handoff."""
    import math

    role = handoff_actor_role(case)
    vectors = {
        "drone_position": 3,
        "drone_orientation": 4,
        "drone_velocity": 6,
        "motor_throttle": 4,
        "prev_action": 4,
        "action_before": 4,
        "ball_position": 3,
        "ball_orientation": 4,
        "ball_velocity": 3,
        "ball_angular_velocity": 3,
        "prev_ball_velocity": 3,
        "wall_target": 2,
    }
    for name, width in vectors.items():
        value = case.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != width:
            raise ValueError(f"handoff field {name} must contain {width} numbers")
        if not all(math.isfinite(float(item)) for item in value):
            raise ValueError(f"handoff field {name} must be finite")
    quaternion = [float(value) for value in case["drone_orientation"]]
    qnorm = math.sqrt(sum(value * value for value in quaternion))
    if abs(qnorm - 1.0) > 1e-3:
        raise ValueError("drone_orientation must be a normalized wxyz quaternion")
    ball_quaternion = [float(value) for value in case["ball_orientation"]]
    ball_qnorm = math.sqrt(sum(value * value for value in ball_quaternion))
    if abs(ball_qnorm - 1.0) > 1e-3:
        raise ValueError("ball_orientation must be a normalized wxyz quaternion")
    if any(not 0.0 <= float(value) <= 1.0 for value in case["motor_throttle"]):
        raise ValueError("motor_throttle must be within [0, 1]")
    if int(case.get("phase", -1)) != 2 or int(case.get("caps", 0)) < 1 or int(case.get("walls", 0)) < 1:
        raise ValueError(f"{role.title()} handoff must begin after an audited cap and wall event")
    expected_skill_id = 1 if role == "hit" else 2
    if int(case.get("skill_id", -1)) != expected_skill_id:
        raise ValueError(f"{role.title()} handoff skill_id must identify the {role} actor")
    for name in ("rallies", "streak", "max_streak", "skill_held_steps"):
        if int(case.get(name, -1)) < 0:
            raise ValueError(f"handoff field {name} must be nonnegative")
    prefix = case.get("source_event_prefix")
    if not isinstance(prefix, list) or len(prefix) < 2:
        raise ValueError("handoff state needs its audited cap/wall event prefix")
    expected = "cap"
    for event in prefix:
        if event.get("kind") != expected or not all(key in event for key in ("y", "step", "substep")):
            raise ValueError("handoff source event prefix must alternate cap and wall events")
        if not math.isfinite(float(event["y"])):
            raise ValueError("handoff source event y must be finite")
        expected = "wall" if expected == "cap" else "cap"
    if prefix[-1]["kind"] != "wall":
        raise ValueError(f"{role.title()} handoff event prefix must end at a wall return")


def make_cases(kind: str, seed: int, count: int) -> list[dict]:
    """Produce frozen train, held-out, or fixed cases with separate ranges."""
    if kind not in {"fixed", "train", "heldout"} or count < 1:
        raise ValueError("kind must be fixed/train/heldout and count positive")
    if kind == "fixed":
        return [{"case_id": f"fixed-{i:04d}", "ball_position": [1.5, 0.0, 4.8],
                 "ball_velocity": [0.0, 0.0, 0.0], "drone_position": [1.5, 0.0, 2.0],
                 "wall_target": [0.0, 4.0]} for i in range(count)]
    rng = random.Random(seed)
    result = []
    while len(result) < count:
        if kind == "train":
            y = rng.uniform(-0.4, 0.4)
            vy = rng.uniform(-0.3, 0.3)
            x = rng.uniform(1.35, 1.65)
            z = rng.uniform(4.3, 5.0)
        else:
            side = rng.choice((-1.0, 1.0))
            y = side * rng.uniform(0.45, 0.75)
            vy = side * rng.uniform(0.35, 0.6)
            x = rng.uniform(1.25, 1.75)
            z = rng.uniform(4.1, 5.15)
        case = {
            "case_id": f"{kind}-{len(result):04d}",
            "ball_position": [x, y, z],
            "ball_velocity": [rng.uniform(-0.15, 0.15), vy, rng.uniform(-0.25, 0.0)],
            "drone_position": [rng.uniform(1.35, 1.65), rng.uniform(-0.15, 0.15), 2.0],
            "wall_target": [rng.uniform(-0.6, 0.6), rng.uniform(3.5, 4.5)],
        }
        if case_is_feasible(case):
            result.append(case)
    return result
