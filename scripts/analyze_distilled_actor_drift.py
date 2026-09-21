"""Compare saved single actors on the same held-out teacher states, without physics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule,make_functional
from torchrl.data import UnboundedContinuousTensorSpec
from omegaconf import OmegaConf
from omni_drones.learning.mappo import make_ppo_actor
from aerowall.learning.wall_policy import SplitInputLayerNorm
from aerowall.learning.skill_distillation import gaussian_kl

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--distillation-report',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,action='append',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
r=json.loads(a.distillation_report.read_text());assert r['status']=='passed'
data_path=Path(r['dataset']);assert sha(data_path)==r['dataset_sha256']
held=torch.load(data_path,map_location='cpu')['held_out']
cfg=OmegaConf.load(a.distillation_report.with_suffix('.yaml')).algo.actor
cfg.output_dist_params=True
actor=TensorDictModule(make_ppo_actor(cfg,UnboundedContinuousTensorSpec((1,43)),UnboundedContinuousTensorSpec((1,4))),
 in_keys=[('agents','observation')],out_keys=[('agents','action'),'logp','entropy',('debug','action_loc'),('debug','action_scale')])
actor.module.encoder[0].__class__=SplitInputLayerNorm
make_functional(actor)
results=[]
with torch.no_grad():
 for path in a.checkpoint:
  payload=torch.load(path,map_location='cpu');assert 'frozen_launch_params' not in payload['policy']
  totals={k:[0.,0.,0] for k in ['launch','recovery']}
  for start in range(0,len(held['observation']),512):
   b={k:v[start:start+512] for k,v in held.items()}
   obs=b['observation'];td=TensorDict({('agents','observation'):obs},batch_size=list(obs.shape[:-1]))
   out=actor(td,payload['policy']['actor_params'],deterministic=True)
   loc=out['debug','action_loc'];scale=out['debug','action_scale']
   kl=gaussian_kl(b['loc'],b['scale'],loc,scale).reshape(-1)
   mse=(loc-b['loc']).square().mean(-1).reshape(-1)
   assert torch.isfinite(kl).all() and torch.isfinite(mse).all()
   mask=b['recovery'].reshape(-1).bool()
   for name,m in [('launch',~mask),('recovery',mask)]:
    totals[name][0]+=float(kl[m].sum());totals[name][1]+=float(mse[m].sum());totals[name][2]+=int(m.sum())
  scores={name:{'rows':n,'mean_kl':k/n,'action_mean_rmse':math.sqrt(e/n)} for name,(k,e,n) in totals.items()}
  baseline=sha(path)==r['checkpoint_sha256']
  if baseline:
   for name in scores:
    for key in ['mean_kl','action_mean_rmse']:
     assert math.isclose(scores[name][key],r['held_out_after'][name][key],rel_tol=2e-4,abs_tol=2e-6),(name,key,scores[name],r['held_out_after'][name])
  results.append({'checkpoint':str(path),'sha256':sha(path),'environment_frames':payload['environment_frames'],
    'baseline_matches_original_assessment':baseline,'held_out':scores})
assert any(x['baseline_matches_original_assessment'] for x in results),'Include original distilled actor as replay validation'
result={'status':'passed','scope':'Same held-out teacher states; distribution drift diagnosis only, not closed-loop success or causal attribution',
 'source_sha256':sha(Path(__file__)),'distillation_report_sha256':sha(a.distillation_report),
 'dataset_sha256':sha(data_path),'results':results}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
