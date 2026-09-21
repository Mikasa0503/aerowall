"""Check receiving-reference bounds and fallback without a simulator."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aerowall.learning.return_reference import recovery_target_from_observation
obs=torch.zeros(5,43);obs[:,18:21]=torch.tensor([.7,.4,1.4]);obs[:,21]=-2.
obs[:,27]=.9;obs[:,28]=-1.;obs[:,42]=1.
obs[1,19]=2.;obs[2,21]=2.
obs[3:,40:43]=torch.tensor([0.,1.,0.]);obs[3,21]=3.;obs[4,21]=.2
before=obs.clone();home=torch.tensor([0.,0.,1.]);saved=home.clone()
target,accepted=recovery_target_from_observation(obs,home,[-2.5,-1.8,.08],[1.1,1.8,5.5])
assert accepted.tolist()==[True,False,False,True,False]
assert torch.equal(target[~accepted],home.expand(3,3))
assert torch.all(target[:,2]==1.) and torch.all(target[accepted,0]<.65)
assert torch.allclose(target[accepted,1],torch.full((2,),.4))
assert torch.equal(obs,before) and torch.equal(home,saved)
print(json.dumps({'status':'passed','checks':['incoming intercept reference','future wall reflection reference','out-of-bounds reference falls back','invalid phase/velocity falls back','crossing before wall falls back','reference drone height preserved','inputs unchanged'],'scope':'Reference selection only; no guarantee of reachability or successful interception'}))
