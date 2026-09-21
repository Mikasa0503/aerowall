"""Verify reference-loss gradients and frozen storage with the real upstream actor."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import UnboundedContinuousTensorSpec as Spec
from aerowall.learning.reference_policy import ReferenceWallPolicy
from aerowall.learning.skill_distillation import gaussian_kl
cfg=OmegaConf.load('runs/skill-distillation-dev-001.yaml').algo
cfg.reference_kl_coef=1.;cfg.actor.output_dist_params=True;cfg.actor.lr=1e-5
spec=SimpleNamespace(name='drone',n=1,observation_spec=Spec((1,43)),action_spec=Spec((1,4)),state_spec=Spec((47,)),reward_spec=Spec((1,1)))
p=ReferenceWallPolicy(cfg,agent_spec=spec,device='cpu')
payload=torch.load('checkpoints/skill-distillation-dev-001/frames-000033857536.pt',map_location='cpu')
p.initialize_wall_actor(payload);p.initialize_reference()
ref=p.reference_params.clone()
obs=torch.load('runs/skill-distillation-dev-001.teacher-data.pt',map_location='cpu')['held_out']['observation'][:128]
batch=TensorDict({p.obs_name:obs},batch_size=[128])
def distribution(params):
 td=TensorDict({p.obs_name:obs},batch_size=[128,1])
 return p.actor(td,params,deterministic=True)
def divergence():
 a=distribution(p.actor_params);b=distribution(p.reference_params)
 return gaussian_kl(a['debug','action_loc'],a['debug','action_scale'],b['debug','action_loc'],b['debug','action_scale']).mean()
with torch.no_grad():
 assert float(divergence())==0.
 torch.manual_seed(41)
 for v in p.actor_params.values(True,True):v.add_(.002*torch.randn_like(v))
 before=float(divergence());out=distribution(p.actor_params)
 for key in p.actor_out_keys:batch[key]=out[key].detach()
 batch['advantages']=torch.zeros_like(batch[p.act_logps_name])
info=p.update_actor(batch)
with torch.no_grad():after=float(divergence())
assert 0<=after<before,(before,after)
assert all(torch.equal(v,p.reference_params[k]) for k,v in ref.items(True,True))
assert not any(v.requires_grad for v in p.reference_params.values(True,True))
result={'status':'passed','initial_kl_zero':True,'zero_advantage_kl_before':before,'zero_advantage_kl_after':after,
 'reference_unchanged':True,'reference_requires_grad':False,'metrics':info,'scope':'Real actor optimizer step with zero advantage reduces deliberately induced reference deviation; physical success not tested'}
Path('runs/reference-policy-contracts.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
