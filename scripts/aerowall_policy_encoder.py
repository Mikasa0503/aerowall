"""AeroWall adapter preserving HCSP's pretrained normalization with new inputs."""
import torch
from torch import nn
from torch.nn import functional as F


class SplitLayerNorm(nn.LayerNorm):
    def forward(self, value):
        return torch.cat([
            F.layer_norm(value[..., :26], (26,), self.weight[:26], self.bias[:26], self.eps),
            F.layer_norm(value[..., 26:], (value.shape[-1] - 26,), self.weight[26:], self.bias[26:], self.eps),
        ], dim=-1)


class SplitGoalLayerNorm(nn.LayerNorm):
    """Keep the 26+20 legacy groups intact and scale appended goal channels."""

    def forward(self, value):
        legacy = torch.cat([
            F.layer_norm(value[..., :26], (26,), self.weight[:26], self.bias[:26], self.eps),
            F.layer_norm(value[..., 26:46], (20,), self.weight[26:46], self.bias[26:46], self.eps),
        ], dim=-1)
        goal = value[..., 46:] * self.weight[46:] + self.bias[46:]
        return torch.cat((legacy, goal), dim=-1)


def configure_policy_encoder():
    import hcsp.learning.mappo as mappo
    if getattr(mappo.make_encoder, '_wall_split', False):
        return
    original = mappo.make_encoder
    def make_encoder(cfg, spec):
        encoder = original(cfg, spec)
        if spec.shape[-1] == 46:
            encoder[0] = SplitLayerNorm(46)
        elif spec.shape[-1] == 48:
            encoder[0] = SplitGoalLayerNorm(48)
        return encoder
    make_encoder._wall_split = True
    mappo.make_encoder = make_encoder
