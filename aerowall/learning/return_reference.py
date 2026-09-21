"""Ballistic return-height reference from current actor-visible state only.

Assumes a static x-normal wall, gravity only, at most one future wall impact,
and a supplied effective restitution estimate/prior. This is a prediction, not
future simulator truth or proof of drone reachability. No inputs are modified.
"""
import torch


def predict_return_reference(observation, restitution=.8, ball_radius=.04,
                             contact_height=1.123, maximum_horizon=1.5):
    assert observation.shape[-1]>=43
    position=observation[...,18:21];velocity=observation[...,21:24]
    front=observation[...,27];phase=observation[...,40:43].argmax(-1)
    e=torch.as_tensor(restitution,dtype=position.dtype,device=position.device)
    e=torch.broadcast_to(e,position.shape[:-1])
    z=position[...,2];vz=velocity[...,2];vx=velocity[...,0]
    discriminant=vz.square()+2*9.81*(z-contact_height)
    time=(vz+discriminant.clamp_min(0).sqrt())/9.81
    target=position+velocity*time.unsqueeze(-1)
    target=target.clone();target[...,2]=contact_height
    plane=front-ball_radius
    wall_time=(plane-position[...,0])/vx.clamp_min(1e-6)
    reflect=(vx>0)&(wall_time>=0)&(wall_time<time)
    reflected_x=plane-e*vx*(time-wall_time)
    target[...,0]=torch.where(reflect,reflected_x,target[...,0])
    normal=observation[...,28:31]
    geometry=(normal[...,0]+1).abs()<1e-5
    geometry &= normal[...,1:].abs().amax(-1)<1e-5
    valid=(torch.isfinite(observation).all(-1)&torch.isfinite(e)&(e>0)&(e<=1.5)
           &(discriminant>=0)&(time>1e-4)&(time<=maximum_horizon)&geometry
           &(((phase==1)&(vx>.1))|((phase==2)&(vx<-.1))))
    # Invalid rows are explicitly masked and zeroed, rather than exposing NaNs
    # or silently clipping a predicted target into the drone's reachable region.
    target=torch.where(valid.unsqueeze(-1),target,torch.zeros_like(target))
    time=torch.where(valid,time,torch.zeros_like(time))
    return target,time,valid,reflect&valid
