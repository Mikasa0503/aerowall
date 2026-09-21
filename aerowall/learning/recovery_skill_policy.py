"""Frozen launch actor plus trainable post-launch recovery skill.

Observable phase/velocity selects the skill before the action. PPO actor updates
use only actions actually sampled from the recovery actor. The critic estimates
the complete switched policy and is trained on all transitions.
"""
import torch
from aerowall.learning.wall_policy import WallMAPPOPolicy


class RecoverySkillPolicy(WallMAPPOPolicy):
    mask_key='recovery_actor_active'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.cfg.share_actor and self.agent_spec.n==1
        assert not hasattr(self,'minibatch_seq_len'), 'This skill adapter is FF only'
        self.train_in_keys.append(self.mask_key)
        self.frozen_launch_params=None

    def initialize_launch_actor(self,payload):
        source=payload['policy']['actor_params']
        assert source['module','encoder','0','weight'].shape[-1]==43
        self.frozen_launch_params=source.clone().detach().to(self.device)
        assert not any(v.requires_grad for v in self.frozen_launch_params.values(True,True))

    @staticmethod
    def recovery_mask(observation):
        phase=observation[...,40:43].argmax(-1)
        return ((phase==1)&(observation[...,21]>.5))|(phase==2)

    def __call__(self,tensordict,deterministic=False):
        assert self.frozen_launch_params is not None
        mask=self.recovery_mask(tensordict[self.obs_name])
        result=super().__call__(tensordict,deterministic)
        actor_input=result.select(*self.actor_in_keys).clone()
        actor_input.batch_size=[*result.batch_size,self.agent_spec.n]
        with torch.no_grad():
            launch=self.actor(actor_input,self.frozen_launch_params,deterministic=deterministic)
        for key in self.actor_out_keys:
            selected=mask
            while selected.ndim<result[key].ndim:selected=selected.unsqueeze(-1)
            result[key]=torch.where(selected,result[key],launch[key])
        result[self.mask_key]=mask
        return result

    def update_actor(self,batch):
        assert len(batch.batch_size)==1
        selected=batch[self.mask_key].reshape(-1)
        if not selected.any():
            # Skipping Adam entirely also prevents momentum updates on empty data.
            return {'policy_loss':0.,'actor_grad_norm':0.,'entropy':0.,'ESS':0.,
                    'recovery_actor_fraction':0.,'actor_update_skipped':1.}
        info=super().update_actor(batch[selected])
        info.update(recovery_actor_fraction=float(selected.float().mean()),actor_update_skipped=0.)
        return info

    @torch.no_grad()
    def audit_rollout(self,data):
        flat=data.reshape(-1);mask=flat[self.mask_key].reshape(-1)
        assert torch.equal(mask,self.recovery_mask(flat[self.obs_name]).reshape(-1))
        errors={}
        for name,selected,params in [('recovery',mask,self.actor_params),('launch',~mask,self.frozen_launch_params)]:
            if not selected.any():errors[name]=None;continue
            rows=flat[selected]
            actor_input=rows.select(*self.actor_in_keys).clone()
            actor_input.batch_size=[*rows.batch_size,self.agent_spec.n]
            replay=self.actor(actor_input,params,eval_action=True)[self.act_logps_name]
            error=float((replay-rows[self.act_logps_name]).abs().max())
            assert error<2e-4,(name,error)
            errors[name]=error
        return {'passed':True,'recovery_transitions':int(mask.sum()),'launch_transitions':int((~mask).sum()),
                'logp_replay_max_errors':errors}

    def state_dict(self):
        result=super().state_dict()
        assert self.frozen_launch_params is not None
        result['frozen_launch_params']=self.frozen_launch_params
        return result

    def load_state_dict(self,state_dict):
        state=dict(state_dict)
        self.frozen_launch_params=state.pop('frozen_launch_params').clone().detach().to(self.device)
        return super().load_state_dict(state)
