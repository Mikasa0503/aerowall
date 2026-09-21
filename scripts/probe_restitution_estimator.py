"""Analytic observation-history contracts; no simulator restitution input."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aerowall.learning.restitution_estimator import RestitutionEstimator


def obs(vx,phase,n=2):
    o=torch.zeros(n,43);o[:,28]=-1;o[:,21]=vx;o[:,40+phase]=1
    return o

checks=[]
for delay in (0,1,2):
    e=RestitutionEstimator(2,'cpu',velocity_delay_steps=delay)
    e.update(obs(2.,1))
    before=obs(-1.6 if delay==0 else 2.,2);saved=before.clone()
    result=e.update(before)
    assert torch.equal(before,saved)
    if delay:assert e.samples.sum()==0
    for j in range(1,delay+1):
        result=e.update(obs(-1.6 if j==delay else 2.,2))
        if j<delay:assert e.samples.sum()==0
    assert torch.allclose(result[:,0],torch.full((2,),.8)) and (e.samples==1).all()
    e.update(obs(-1.6,2));assert (e.samples==1).all(), 'persistent phase counted twice'
    e.reset(torch.tensor([True,False]));assert e.samples.tolist()==[0,1]
    assert e.previous_phase.tolist()==[-1,2]
    checks.append('exact delayed reflection, single update, selective reset: '+str(delay))
e=RestitutionEstimator(2,'cpu')
e.update(obs(.1,1));e.update(obs(-.08,2));assert (e.reason==2).all() and not e.samples.any()
e.reset(torch.ones(2,dtype=torch.bool));e.update(obs(2.,1));e.update(obs(1.,2));assert (e.reason==3).all()
e.reset(torch.ones(2,dtype=torch.bool));e.update(obs(2.,1));e.update(obs(-4.,2));assert (e.reason==4).all()
e=RestitutionEstimator(2,'cpu',velocity_delay_steps=2)
e.update(obs(2.,1));e.update(obs(2.,2));e.update(obs(2.,1));assert (e.reason==5).all() and not e.samples.any()
bad=obs(2.,1);bad[:,21]=float('nan');e.update(bad);assert (e.reason==6).all()
e.update(obs(-1.6,2));assert not e.samples.any(), 'invalid history reused'
e=RestitutionEstimator(2,'cpu');e.update(obs(2.,1));e.update(obs(-1.6,2),active=torch.tensor([True,False]));assert e.samples.tolist()==[1,0]
checks+=['low incidence rejected','missing reversal rejected','implausible ratio rejected','interrupted delayed pair rejected','nonfinite history cleared','inactive slot preserved']
print(json.dumps({'status':'passed','checks':checks,'scope':'Analytic observation histories, not physical calibration or learned-policy performance'}))
