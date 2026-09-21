"""Policy-rate effective wall restitution from actor observations only.

For a static vertical wall, e = outgoing normal speed / incoming normal speed.
The observable TO_WALL -> TO_BAT transition locates the sampling interval.
Known velocity delay postpones the ratio calculation by that many samples.
No contact callback, impulse, simulator material or unobserved state is read.
Unobserved multiple collisions inside a sampling interval remain an aliasing
limitation: accepted means the observable checks pass, not certified truth.
"""
import torch


class RestitutionEstimator:
    REASONS = ('none', 'accepted', 'low_incidence', 'missing_sign_reversal',
               'ratio_out_of_range', 'interrupted', 'invalid_observation')

    def __init__(self, count, device, velocity_delay_steps=0, prior=.8,
                 minimum_incidence=.5, smoothing=.5):
        assert velocity_delay_steps in (0, 1, 2)
        assert minimum_incidence > 0 and 0 < smoothing <= 1
        self.count=count; self.device=device; self.delay=velocity_delay_steps
        self.prior=float(prior); self.minimum_incidence=float(minimum_incidence)
        self.smoothing=float(smoothing)
        self.estimate=torch.full((count,),self.prior,device=device)
        self.variance=torch.zeros(count,device=device)
        self.samples=torch.zeros(count,dtype=torch.long,device=device)
        self.previous_velocity=torch.zeros(count,3,device=device)
        self.previous_phase=torch.full((count,),-1,dtype=torch.long,device=device)
        self.pending=torch.full((count,),-1,dtype=torch.long,device=device)
        self.reason=torch.zeros(count,dtype=torch.long,device=device)
        self.last_ratio=torch.full((count,),float('nan'),device=device)

    def reset(self, mask):
        mask=mask.flatten().bool()
        self.estimate[mask]=self.prior; self.variance[mask]=0; self.samples[mask]=0
        self.previous_velocity[mask]=0; self.previous_phase[mask]=-1
        self.pending[mask]=-1; self.reason[mask]=0; self.last_ratio[mask]=float('nan')

    @torch.no_grad()
    def update(self, observation, active=None):
        assert observation.shape == (self.count,43)
        if active is None:active=torch.ones(self.count,dtype=torch.bool,device=self.device)
        active=active.flatten().bool()
        self.reason.zero_(); self.last_ratio.fill_(float('nan'))
        normal=observation[:,28:31]; velocity=observation[:,21:24]
        phase_values=observation[:,40:43]; phase=phase_values.argmax(-1)
        valid=torch.isfinite(observation).all(-1)
        valid &= (normal.norm(dim=-1)-1).abs()<1e-5
        # This version assumes a static vertical wall: gravity has no normal component.
        valid &= normal[:,2].abs()<1e-5
        valid &= ((phase_values==0)|(phase_values==1)).all(-1) & (phase_values.sum(-1)==1)
        invalid=active & ~valid
        self.reason[invalid]=6; self.pending[invalid]=-1; self.previous_phase[invalid]=-1
        use=active & valid
        interrupted=use & (self.pending>=0) & (phase!=2)
        self.reason[interrupted]=5; self.pending[interrupted]=-1
        waiting=use & (self.pending>=0)
        self.pending[waiting]-=1
        transition=use & (self.previous_phase==1) & (phase==2)
        self.pending[transition]=self.delay
        mature=use & (self.pending==0)
        incoming=-(self.previous_velocity*normal).sum(-1)
        outgoing=(velocity*normal).sum(-1)
        ratio=outgoing/incoming.clamp_min(1e-8)
        low=mature & (incoming<self.minimum_incidence)
        sign=mature & ~low & (outgoing<=0)
        outside=mature & ~low & ~sign & ((ratio<=0)|(ratio>1.5)|~torch.isfinite(ratio))
        accepted=mature & ~low & ~sign & ~outside
        self.reason[low]=2; self.reason[sign]=3; self.reason[outside]=4; self.reason[accepted]=1
        self.last_ratio[mature]=ratio[mature]
        first=accepted & (self.samples==0)
        more=accepted & ~first
        delta=ratio-self.estimate; alpha=self.smoothing
        self.estimate[more]+=alpha*delta[more]
        self.variance[more]=(1-alpha)*(self.variance[more]+alpha*delta[more].square())
        self.estimate[first]=ratio[first]; self.variance[first]=0
        self.samples[accepted]+=1
        self.pending[mature]=-1
        self.previous_velocity[use]=velocity[use]
        self.previous_phase[use]=phase[use]
        # Count support is a heuristic, explicitly not calibrated probability.
        support=self.samples.float()/(self.samples.float()+1)
        return torch.stack([self.estimate,support,self.variance,accepted.float()],-1).clone()
