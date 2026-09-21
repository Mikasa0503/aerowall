"""Single executing actor with a frozen reference used only in PPO training.

This is a differentiable loss regularizer, not HCSP's environment reward penalty.
KL direction is current || reference; physical task rewards stay unchanged.
"""
import torch
from aerowall.learning.wall_policy import WallMAPPOPolicy
from aerowall.learning.skill_distillation import gaussian_kl


class ReferenceWallPolicy(WallMAPPOPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.cfg.share_actor and self.agent_spec.n==1
        assert not hasattr(self,'minibatch_seq_len')
        assert self.cfg.actor.output_dist_params
        self.reference_params=None
        self.reference_kl_coef=float(self.cfg.reference_kl_coef)
        assert self.reference_kl_coef>0

    def initialize_reference(self):
        self.reference_params=self.actor_params.clone().detach()
        assert not any(v.requires_grad for v in self.reference_params.values(True,True))

    def update_actor(self,batch):
        assert self.reference_params is not None and len(batch.batch_size)==1
        actor_input=batch.select(*self.actor_in_keys).clone()
        actor_input.batch_size=[*batch.batch_size,1]
        output=self.actor(actor_input.clone(),self.actor_params,eval_action=True)
        with torch.no_grad():
            reference=self.actor(actor_input.clone(),self.reference_params,eval_action=True)
        loc=output['debug','action_loc'];scale=output['debug','action_scale']
        kl=gaussian_kl(loc,scale,reference['debug','action_loc'],reference['debug','action_scale'])
        advantages=batch['advantages'];old=batch[self.act_logps_name]
        new=output[self.act_logps_name];entropy=output[f'{self.agent_spec.name}.action_entropy']
        assert advantages.shape==old.shape==new.shape==entropy.shape
        ratio=(new-old).exp()
        surrogate=torch.minimum(ratio*advantages,ratio.clamp(1-self.clip_param,1+self.clip_param)*advantages)
        policy_loss=-(surrogate*self.act_dim).mean()
        loss=policy_loss-self.cfg.entropy_coef*entropy.mean()+self.reference_kl_coef*kl.mean()
        assert torch.isfinite(loss)
        self.actor_opt.zero_grad();loss.backward()
        parameters=list(self.actor_params.values(True,True))
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in parameters)
        norm=torch.nn.utils.clip_grad_norm_(self.actor_opt.param_groups[0]['params'],self.cfg.max_grad_norm)
        self.actor_opt.step()
        ess=(2*ratio.logsumexp(0)-(2*ratio).logsumexp(0)).exp().mean()/ratio.shape[0]
        return {'policy_loss':float(policy_loss.detach()),'actor_grad_norm':float(norm),
                'entropy':float(entropy.mean().detach()),'ESS':float(ess.detach()),
                'reference_kl':float(kl.mean().detach()),'reference_kl_max':float(kl.max().detach()),
                'reference_penalty':float(self.reference_kl_coef*kl.mean().detach())}

    def state_dict(self):
        state=super().state_dict()
        assert self.reference_params is not None
        state['reference_actor_params']=self.reference_params
        return state

    def load_state_dict(self,state):
        state=dict(state)
        self.reference_params=state.pop('reference_actor_params').clone().detach().to(self.device)
        return super().load_state_dict(state)
