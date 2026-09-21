"""CPU audit of pinned GRU module and actor return protocol, not RL training."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Optional, Union
import torch
from torch import nn
from tensordict import TensorDict

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1]/'third_party/JuggleRL_train/omni_drones/learning'
path=root/'modules/rnn.py'
spec=importlib.util.spec_from_file_location('pinned_gru',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
torch.manual_seed(20260921);torch.set_num_threads(4)
report={'scope':'CPU module and actor protocol audit; not complete recurrent PPO or simulator validation',
        'source_hashes':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [path,root/'mappo.py']}}
try:
    mod.GRU(256,128)(torch.randn(3,256))
except RuntimeError as e:report['documented_128_hidden_with_256_encoder_error']=str(e)
else:raise AssertionError('Expected pinned residual-size incompatibility was not reproduced')
gru=mod.GRU(256,256);x=torch.randn(3,7,256);h=torch.randn(3,256)
reset=torch.zeros(3,7,1,dtype=torch.bool);reset[0,3]=True;reset[2,5]=True
seq,hs=gru(x,h.clone(),reset);steps=[];state=h.clone()
for t in range(7):
    y,state=gru(x[:,t],state,reset[:,t]);steps.append(y)
assert torch.allclose(seq,torch.stack(steps,1),atol=1e-6)
assert torch.allclose(hs[:,-1],state,atol=1e-6)
new,_=gru(x[0:1,3],torch.zeros(1,256),torch.ones(1,1,dtype=torch.bool))
assert torch.allclose(seq[0:1,3],new,atol=1e-6)
report['equal_width_sequence_step_and_selected_reset_passed']=True
# Compile the exact pinned Actor and distribution predicate without importing
# the simulator-dependent package; supply only their declared dependencies.
tree=ast.parse((root/'mappo.py').read_text())
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['Actor','_is_independent_normal']]
ns={'torch':torch,'nn':nn,'Optional':Optional,'Union':Union,'TensorDict':TensorDict}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(root/'mappo.py'),'exec'),ns)
class Dist(nn.Module):
    def forward(self,v):return torch.distributions.Independent(torch.distributions.Normal(v[...,:4],torch.ones_like(v[...,:4])),1)
actor=ns['Actor'](nn.Identity(),Dist(),gru,output_dist_params=True)
result=actor(torch.randn(3,256),deterministic=True)
expected_keys=['action','logp','entropy','actor_rnn_state','debug.action_loc','debug.action_scale']
report['rollout_return_count']=len(result);report['configured_actor_output_keys']=expected_keys
report['rollout_fourth_return_is_none']=result[3] is None
report['rollout_fifth_return_shape']=list(result[4].shape)
assert len(result)==7 and len(expected_keys)==6 and result[3] is None
report['recurrent_ppo_ready']=False
report['blockers']=['Residual addition requires encoder and GRU hidden widths to match',
                    'Recurrent rollout returns seven values for six configured output keys; hidden state is shifted by extra None',
                    'Collector hidden-state carry and sequence-minibatch handling still require end-to-end validation']
report['status']='passed'
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
