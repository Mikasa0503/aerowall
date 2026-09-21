"""Check phase isolation and observable-only, bounded recovery shaping."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aerowall.learning.phase_recovery import phase_recovery_score
p=torch.tensor([[0.,0.,1.]]).repeat(7,1);p[6,0]=2
up=torch.ones(7);up[4]=-1
omega=torch.zeros(7,3);omega[5,0]=4
vx=torch.tensor([2.,2.,2.,0.,2.,2.,2.])
phase=torch.tensor([0,1,2,1,1,1,1]);done=torch.tensor([False,False,True,False,False,False,False])
inputs=[p,up,omega,vx,phase,done];copies=[t.clone() for t in inputs]
s,mask=phase_recovery_score(*inputs,torch.tensor([0.,0.,1.]))
assert s[0]==s[2]==s[3]==0,'Reward before launch, after termination or on vertical juggle'
assert s[1]==3 and 0<s[4]<s[1] and 0<s[5]<s[1] and 0<s[6]<s[1]
assert all(torch.equal(x,y) for x,y in zip(inputs,copies))
assert ((s>=0)&(s<=3)).all()
phase[2]=2;done[2]=False
s2,_=phase_recovery_score(p,up,omega,vx,phase,done,torch.tensor([0.,0.,1.]))
assert s2[2]==3,'No recovery score after wall return'
print(json.dumps({'status':'passed','checks':['no prelaunch or vertical-juggle reward','no terminal reward','return phase enabled','position/up/rotation responses','score bounded 0 to 3','inputs unmodified'],'scope':'Reward contracts only; not learned recovery or successful wall return'}))
