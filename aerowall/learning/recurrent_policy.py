"""Single-agent GRU adapter around pinned upstream MAPPO, development only.

Reuses upstream encoder, distributions, critic, GAE and checkpoint format.
Actor PPO objective follows the upstream MIT implementation by Botian Xu;
upstream copyright/license remains in third_party/JuggleRL_train.
"""
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictParams, make_functional
from omni_drones.learning.mappo import MAPPOPolicy, make_ppo_actor
from aerowall.learning.wall_policy import SplitInputLayerNorm
from aerowall.learning.recurrent_protocol import adapt_recurrent_actor


class RecurrentWallPolicy(MAPPOPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # In this pinned TensorDict version selecting both next and a child
        # can prune the other next fields. Keep the complete transition.
        self.train_in_keys=[k for k in self.train_in_keys if not (isinstance(k,tuple) and k[0]=='next')]

    def make_actor(self):
        assert self.cfg.share_actor and self.agent_spec.n==1
        cfg=self.cfg.actor
        assert cfg.output_dist_params and cfg.rnn.cls.lower()=='gru'
        self.hidden_key=f'{self.agent_spec.name}.actor_rnn_state'
        self.minibatch_seq_len=int(cfg.rnn.train_seq_len)
        assert 1<self.minibatch_seq_len<=self.cfg.train_every
        self.actor_in_keys=[self.obs_name,self.act_name,self.hidden_key,'is_init']
        self.actor_out_keys=[self.act_name,self.act_logps_name,f'{self.agent_spec.name}.action_entropy',
                             ('next',self.hidden_key),('debug','action_loc'),('debug','action_scale')]
        module=make_ppo_actor(cfg,self.agent_spec.observation_spec,self.agent_spec.action_spec)
        assert module.rnn.cell.input_size==module.rnn.cell.hidden_size
        assert module.encoder[0].normalized_shape==(43,)
        module.encoder[0].__class__=SplitInputLayerNorm
        adapt_recurrent_actor(module)
        self.actor=TensorDictModule(module,in_keys=self.actor_in_keys,out_keys=self.actor_out_keys).to(self.device)
        self.actor_params=TensorDictParams(make_functional(self.actor))
        self.actor_opt=torch.optim.Adam(self.actor_params.parameters(),lr=cfg.lr)

    def sequence_actor_output(self,batch):
        assert len(batch.batch_size)==2
        # Shared parameters have no agent batch dimension. Remove the single
        # agent axis instead of vmapping over a nonexistent parameter batch.
        td=TensorDict({self.obs_name:batch[self.obs_name].squeeze(2),
                       self.act_name:batch[self.act_name].squeeze(2),
                       self.hidden_key:batch[self.hidden_key].squeeze(2),
                       'is_init':batch['is_init']},batch_size=batch.batch_size,device=self.device)
        return self.actor(td,self.actor_params,eval_action=True)

    def update_actor(self,batch):
        output=self.sequence_actor_output(batch)
        logp=output[self.act_logps_name]
        old=batch[self.act_logps_name].squeeze(2)
        advantage=batch['advantages'].squeeze(2)
        entropy=output[f'{self.agent_spec.name}.action_entropy']
        assert logp.shape==old.shape==advantage.shape==entropy.shape
        ratio=(logp-old).exp()
        surrogate=torch.minimum(ratio*advantage,ratio.clamp(1-self.clip_param,1+self.clip_param)*advantage)
        loss=-surrogate.mean()*self.act_dim-self.entropy_coef*entropy.mean()
        self.actor_opt.zero_grad();loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(self.actor_params.parameters(),self.cfg.max_grad_norm)
        self.actor_opt.step()
        flat=ratio.detach().flatten();ess=flat.sum().square()/(flat.square().sum()*flat.numel()).clamp_min(1e-12)
        return {'policy_loss':float(-surrogate.mean()*self.act_dim),'entropy':float(entropy.mean()),
                'actor_grad_norm':float(grad),'ESS':float(ess)}
