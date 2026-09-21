"""Analytic contracts for observable ballistic return references."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aerowall.learning.return_reference import predict_return_reference
obs=torch.zeros(5,43);obs[:,27]=.9;obs[:,28]=-1;obs[:,18:21]=torch.tensor([0.,.1,1.])
obs[:,21:24]=torch.tensor([2.,.3,2.]);obs[:,41]=1
# Exact descending return to the initial height at t=4/g, with one reflection.
old=obs.clone();target,t,valid,reflected=predict_return_reference(obs,ball_radius=.04,contact_height=1.)
expected_t=4/9.81
# Wall would be reached at .43 s, after this crossing: no reflection yet.
assert valid.all() and not reflected.any()
assert torch.allclose(t,torch.full((5,),expected_t))
assert torch.allclose(target[:,0],torch.full((5,),2*expected_t))
obs[:,21]=3.;target,t,valid,reflected=predict_return_reference(obs,contact_height=1.)
assert reflected.all()
assert torch.allclose(target[:,0],torch.full((5,),.86-.8*3*(expected_t-.86/3)))
assert torch.allclose(target[:,1],torch.full((5,),.1+.3*expected_t))
obs[0,40:43]=torch.tensor([0.,0.,1.]);obs[0,21]=-2.
obs[1,21]=0.;obs[2,20]=-10.;obs[3,28]=0.;obs[4,21]=float('nan')
before=obs.clone();target,t,valid,reflected=predict_return_reference(obs,contact_height=1.)
assert valid.tolist()==[True,False,False,False,False]
assert not reflected.any() and torch.isfinite(target).all() and torch.isfinite(t).all()
assert torch.allclose(obs,before,equal_nan=True)
assert torch.allclose(target[0,0],torch.tensor(-2*expected_t))
print(json.dumps({'status':'passed','checks':['analytic free flight','single future wall reflection','already returning ball','invalid geometry and trajectories masked','inputs unchanged'],'scope':'Predictor contracts, not physical accuracy or learned control'}))
