"""Compatibility import for historical AeroWall/HCSP scripts.

New code should import :class:`AeroWallSingleWallRallyEnv` from
``aerowall_wall_rally_env``. The implementation is AeroWall-owned and is not
an upstream HCSP environment.
"""

from aerowall_wall_rally_env import AeroWallSingleWallRallyEnv

HCSPSingleWallRL = AeroWallSingleWallRallyEnv

__all__ = ["AeroWallSingleWallRallyEnv", "HCSPSingleWallRL"]
