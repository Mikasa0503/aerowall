"""Compatibility imports for existing AeroWall experiment scripts.

New code should import the AeroWall-owned policies from
``aerowall_skill_policies``. Their MAPPO implementation and simulator base
still come from the pinned HCSP release.
"""

from aerowall_skill_policies import (
    AeroWallLaunchRecoveryPolicy,
    AeroWallSkillChainPolicy,
)

ChainedRecoveryPolicy = AeroWallLaunchRecoveryPolicy
SkillChainPolicy = AeroWallSkillChainPolicy

__all__ = [
    "AeroWallLaunchRecoveryPolicy",
    "AeroWallSkillChainPolicy",
    "ChainedRecoveryPolicy",
    "SkillChainPolicy",
]
