"""Small project-owned corrections around the pinned upstream runtime."""
import torch
from omni_drones.utils.torchrl.transforms import PIDRateController_flightmare


class ResetSafePIDRateController(PIDRateController_flightmare):
    """Reset PID history at the actual environment reset boundary."""

    def _reset(self, tensordict, tensordict_reset):
        result = super()._reset(tensordict, tensordict_reset)
        if self.controller.init_flag:
            return result
        mask = tensordict.get('_reset', None) if tensordict is not None else None
        if mask is None:
            mask = torch.ones(self.controller.integ.shape[0], dtype=torch.bool,
                              device=self.controller.integ.device)
        else:
            mask = mask.reshape(mask.shape[0], -1).any(dim=-1)
            rows = self.controller.integ.shape[0]
            assert rows % mask.numel() == 0
            mask = mask.repeat_interleave(rows // mask.numel())
        with torch.no_grad():
            self.controller.integ[mask] = 0
            self.controller.last_body_rate[mask] = 0
        return result
