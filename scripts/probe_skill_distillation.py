import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aerowall.learning.skill_distillation import gaussian_kl,balanced_branch_loss
torch.manual_seed(4)
a=torch.randn(17,1,4);b=torch.rand_like(a)+.2;c=torch.randn_like(a).requires_grad_();d=torch.rand_like(a)+.2
actual=gaussian_kl(a,b,c,d)
expected=torch.distributions.kl_divergence(torch.distributions.Independent(torch.distributions.Normal(a,b),1),torch.distributions.Independent(torch.distributions.Normal(c,d),1))
assert torch.allclose(actual,expected,atol=2e-6,rtol=2e-6)
assert torch.equal(gaussian_kl(a,b,a,b),torch.zeros(17,1))
m=torch.zeros(17,1,dtype=torch.bool);m[0]=True
loss=balanced_branch_loss(actual,m);assert torch.allclose(loss,(actual[0].mean()+actual[1:].mean())/2)
loss.backward();assert c.grad is not None and torch.isfinite(c.grad).all()
assert torch.equal(balanced_branch_loss(actual.detach(),torch.zeros_like(m)),actual.detach().mean())
print(json.dumps({'status':'passed','gaussian_kl_matches_torch':True,'identical_distribution_zero':True,'branch_balance_and_empty_branch':True,'finite_student_gradient':True}))
