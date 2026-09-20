"""Contact-lifecycle qualification and per-environment WallRally state.

No rewards, simulator state writes, cooldowns, or hidden material parameters.
One advance call represents one physical observation step, not an arbitrary
unordered collection of contacts spanning several steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math


class Kind(str, Enum):
    CAP = 'cap'
    WALL = 'wall'
    BALL_GROUND = 'ball_ground'
    BALL_BODY = 'ball_body'
    NON_CAP = 'non_cap'
    DRONE_WALL = 'drone_wall'
    DRONE_GROUND = 'drone_ground'


ILLEGAL = {Kind.BALL_GROUND, Kind.BALL_BODY, Kind.NON_CAP, Kind.DRONE_WALL, Kind.DRONE_GROUND}


@dataclass(frozen=True)
class Impact:
    kind: Kind
    pair: tuple
    impulse: float
    point: tuple | None = None
    target_eligible: bool = True


@dataclass
class ContactLedger:
    """Caller must supply every lifecycle edge, including zero-impulse entries."""
    threshold: float = 1e-6
    active: dict = field(default_factory=dict)

    def observe(self, pair, edge, kind, impulse=0., point=None, target_eligible=True, *, episode_start=False):
        pair = tuple(sorted(pair))
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError('A contact must identify two different collider paths')
        kind = Kind(kind)
        if edge not in ('found','persist','lost'):
            raise ValueError(f'Unknown contact edge {edge}')
        if not math.isfinite(impulse) or impulse < 0:
            raise ValueError('Impulse must be a finite nonnegative magnitude')
        if edge == 'lost':
            self.active.pop(pair, None)
            return None
        if edge == 'found':
            # Duplicate FOUND cannot renew an interval without a preceding LOST.
            self.active.setdefault(pair, False)
        elif pair not in self.active:
            if not episode_start:
                raise RuntimeError(f'PERSIST without observed FOUND: {pair}')
            # Teleport/reset starts a new scoring epoch, while PhysX may retain
            # the old manifold and report PERSIST on its first fetched step.
            # This opens an uncredited interval, never an artificial impulse.
            self.active[pair] = False
        if impulse <= self.threshold:
            return None
        if point is not None and (len(point) != 3 or not all(math.isfinite(v) for v in point)):
            raise ValueError('Contact position must contain three finite coordinates')
        # The manifold can slide from a cap onto a side during the same interval.
        if kind in ILLEGAL:
            return Impact(kind, pair, impulse, point, target_eligible)
        if self.active[pair]:
            return None
        self.active[pair] = True
        return Impact(kind, pair, impulse, point, target_eligible)

    def reset(self):
        self.active.clear()


@dataclass
class RallyState:
    phase: str = 'wait_bat'
    rallies: int = 0
    streak: int = 0
    max_streak: int = 0
    joint_rallies: int = 0
    joint_streak: int = 0
    max_joint_streak: int = 0
    wall_hits: int = 0
    target_index: int = 0
    pending_target_hit: bool = False
    pending_target_error: float | None = None
    steps: int = 0
    terminated: bool = False
    truncated: bool = False
    reason: str | None = None

    def break_chain(self):
        self.phase = 'wait_bat'
        self.streak = self.joint_streak = 0
        self.pending_target_hit = False
        self.pending_target_error = None

    def advance(self, impacts, target, target_radius=.35, failure=None, time_limit=False):
        """target is the current wall-plane world point, before target publication.

        Euclidean contact-point error avoids treating side/back-face hits as
        in-plane target successes. The sensing adapter also sets target_eligible
        only for the intended front face. No material parameter is consumed.
        """
        if self.terminated or self.truncated:
            return {'ignored_after_done': True, 'rally_completed': False, 'publish_next_target': 0}
        self.steps += 1
        result = {'rally_completed': False, 'joint_completed': False, 'publish_next_target': 0,
                  'ambiguous_order': False, 'wall_target_errors': []}
        if failure is not None and failure not in ('out_of_bounds','numerical_failure'):
            raise ValueError(f'Unknown task failure {failure}')
        illegal = sorted({e.kind.value for e in impacts if e.kind in ILLEGAL})
        if failure or illegal:
            self.terminated = True
            self.reason = failure or illegal[0]
            self.break_chain()
            result['termination_reasons'] = ([failure] if failure else []) + illegal
            return result
        caps = [e for e in impacts if e.kind == Kind.CAP]
        walls = [e for e in impacts if e.kind == Kind.WALL]
        if walls:
            if len(target) != 3 or not all(math.isfinite(v) for v in target) or target_radius <= 0 or not math.isfinite(target_radius):
                raise ValueError('A wall event requires a finite target and positive radius')
            for e in walls:
                if e.point is None or len(e.point) != 3 or not all(math.isfinite(v) for v in e.point):
                    raise ValueError('A wall event requires a measured finite contact point')
                result['wall_target_errors'].append(math.dist(e.point, target))
            self.wall_hits += len(walls)
            self.target_index += len(walls)
            result['publish_next_target'] = len(walls)
        if len(caps) > 1 or len(walls) > 1 or (caps and walls):
            self.break_chain()
            result['ambiguous_order'] = True
        elif caps:
            if self.phase == 'to_bat':
                self.rallies += 1
                self.streak += 1
                self.max_streak = max(self.max_streak, self.streak)
                result['rally_completed'] = True
                result['completed_target_error'] = self.pending_target_error
                result['joint_completed'] = self.pending_target_hit
                if self.pending_target_hit:
                    self.joint_rallies += 1
                    self.joint_streak += 1
                    self.max_joint_streak = max(self.max_joint_streak, self.joint_streak)
                else:
                    self.joint_streak = 0
            else:
                self.break_chain()
            self.phase = 'to_wall'
            self.pending_target_hit = False
            self.pending_target_error = None
        elif walls:
            if self.phase == 'to_wall':
                self.phase = 'to_bat'
                self.pending_target_error = result['wall_target_errors'][0]
                self.pending_target_hit = walls[0].target_eligible and self.pending_target_error <= target_radius
            else:
                self.break_chain()
        if time_limit:
            self.truncated = True
            self.reason = 'time_limit'
        return result


class RallyBatch:
    def __init__(self, count):
        if count < 1:
            raise ValueError('count must be positive')
        self.states = [RallyState() for _ in range(count)]
        self.ledgers = [ContactLedger() for _ in range(count)]

    def reset(self, indices):
        indices = list(indices)
        if any(not 0 <= i < len(self.states) for i in indices):
            raise IndexError('Reset index outside environment batch')
        for i in indices:
            self.states[i] = RallyState()
            self.ledgers[i].reset()
