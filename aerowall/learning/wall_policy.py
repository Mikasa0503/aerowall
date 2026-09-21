"""PPO extension preserving the pretrained 24-feature actor at initialization."""
import torch
from torch import nn
from torch.nn import functional as F
from omni_drones.learning import MAPPOPolicy


class SplitInputLayerNorm(nn.LayerNorm):
    def forward(self, value):
        prefix=F.layer_norm(value[...,:24],(24,),self.weight[:24],self.bias[:24],self.eps)
        suffix=F.layer_norm(value[...,24:],(value.shape[-1]-24,),self.weight[24:],self.bias[24:],self.eps)
        return torch.cat([prefix,suffix],-1)


class WallMAPPOPolicy(MAPPOPolicy):
    def make_actor(self):
        super().make_actor()
        # Preserve functional parameter paths, optimizer references and shapes.
        norm=self.actor.module.encoder[0]
        assert isinstance(norm,nn.LayerNorm) and norm.normalized_shape==(43,)
        norm.__class__=SplitInputLayerNorm

    def initialize_wall_actor(self, payload):
        """Copy a compatible wall actor; critic/value normalizer remain fresh."""
        source=payload['policy']['actor_params']
        source_keys=set(source.keys(True,True))
        assert source_keys==set(self.actor_params.keys(True,True))
        with torch.no_grad():
            for key,value in self.actor_params.items(True,True):
                assert value.shape==source[key].shape,(key,value.shape,source[key].shape)
                value.copy_(source[key])

    def audit_wall_actor(self, payload, count=64):
        from tensordict import TensorDict
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            torch.manual_seed(20260921)
            observation=torch.randn(count,1,43,device=self.device)
            td=TensorDict({self.obs_name:observation},batch_size=[count,1],device=self.device)
            expected=self.actor(td.clone(),payload['policy']['actor_params'],deterministic=True)[self.act_name]
            actual=self.actor(td.clone(),self.actor_params,deterministic=True)[self.act_name]
            assert torch.equal(expected,actual)
            return {'passed':True,'observations':count,'max_action_difference':float((actual-expected).abs().max()),
                    'scope':'Compatible 43-feature actor copied exactly; fresh critic, value normalizer and optimizers'}

    def initialize_juggle_actor(self, payload):
        old=payload['policy']['actor_params']
        norm_keys={('module','encoder','0','weight'),('module','encoder','0','bias')}
        input_key=('module','encoder','1','layers','0','weight')
        with torch.no_grad():
            for key,value in self.actor_params.items(True,True):
                source=old[key]
                if key in norm_keys:
                    assert source.shape==(24,) and value.shape==(43,)
                    value[:24].copy_(source)
                elif key==input_key:
                    assert source.shape[-1]==24 and value.shape[-1]==43
                    value.zero_();value[:,:24].copy_(source)
                else:
                    assert value.shape==source.shape,(key,value.shape,source.shape)
                    value.copy_(source)
        # Critic and value normalizer intentionally start fresh for rally rewards.

    def audit_juggle_actor(self, payload, count=64):
        from tensordict import TensorDict
        from tensordict.nn import TensorDictModule, make_functional
        from torchrl.data import UnboundedContinuousTensorSpec
        from omni_drones.learning.mappo import make_ppo_actor
        assert self.cfg.share_actor, 'Transfer audit currently covers shared FF actor'
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            torch.manual_seed(20260921)
            original=TensorDictModule(make_ppo_actor(self.cfg.actor,UnboundedContinuousTensorSpec((1,24),device=self.device),self.agent_spec.action_spec),in_keys=self.actor_in_keys,out_keys=self.actor_out_keys).to(self.device)
            make_functional(original)
            prefix=torch.randn(count,1,24,device=self.device)
            old_td=TensorDict({self.obs_name:prefix},batch_size=[count,1],device=self.device)
            expected=original(old_td,payload['policy']['actor_params'],deterministic=True)[self.act_name]
            errors=[]
            for scale in (1.,10.):
                extra=torch.randn(count,1,19,device=self.device)*scale
                td=TensorDict({self.obs_name:torch.cat([prefix,extra],-1)},batch_size=[count,1],device=self.device)
                actual=self.actor(td,self.actor_params,deterministic=True)[self.act_name]
                errors.append(float((expected-actual).abs().max()))
                assert torch.allclose(expected,actual,atol=1e-5,rtol=1e-5)
            return {'passed':True,'observations':count,'extra_feature_scales':[1.,10.],
                    'max_action_differences':errors,'scope':'Initial actor output equivalence; critic is new and behavior changes during training'}
