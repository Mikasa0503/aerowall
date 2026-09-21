"""Observable-state ballistic/PD recovery diagnostic, not a learned policy.

Uses a fixed restitution prior, never simulation material labels. Returns CTBR
actions through the existing PID path and never writes physical state.
"""
import torch


class RecoveryController:
    def __init__(self, count, device, wall_front, ball_radius, bat_top=.083):
        self.active=torch.zeros(count,dtype=torch.bool,device=device)
        self.wall_front=float(wall_front);self.ball_plane=self.wall_front-float(ball_radius)
        self.intercept_height=1.+float(bat_top)+float(ball_radius)
        self.activations=0;self.control_steps=0

    def reset(self, mask):
        self.active[mask.flatten()]=False

    def command(self, drone_position, quaternion, drone_velocity, ball_position, ball_velocity, outbound):
        ball_velocity=ball_velocity[:,:3]
        trigger=outbound & (ball_velocity[:,0]>.5)
        self.activations+=int((trigger & ~self.active).sum())
        self.active |= trigger
        self.control_steps+=int(self.active.sum())
        g=9.81;vz=ball_velocity[:,2]
        discriminant=vz.square()+2*g*(ball_position[:,2]-self.intercept_height)
        flight=((vz+discriminant.clamp_min(0).sqrt())/g).clamp(.05,.8)
        target=ball_position+ball_velocity*flight[:,None]
        # Reflect only the modeled horizontal component if it would cross
        # the known wall plane. The fixed 0.8 prior is not a true parameter.
        wall_time=(self.ball_plane-ball_position[:,0])/ball_velocity[:,0].clamp_min(.001)
        reflect=(ball_velocity[:,0]>0)&(wall_time>=0)&(wall_time<flight)
        reflected_x=self.ball_plane-.8*ball_velocity[:,0]*(flight-wall_time)
        target[:,0]=torch.where(reflect,reflected_x,target[:,0])
        target[:,0].clamp_(-2.2,self.wall_front-.25);target[:,1].clamp_(-1.5,1.5);target[:,2]=1.
        acceleration=6.*(target-drone_position)-4.*drone_velocity[:,:3]
        acceleration[:,:2].clamp_(-6.,6.);acceleration[:,2]=(acceleration[:,2]+g).clamp(2.,14.5)
        norm=acceleration.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        desired_up=acceleration/norm
        w,x,y,z=quaternion.unbind(-1)
        up=torch.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],-1)
        world_rate=6.*torch.cross(up,desired_up,dim=-1)
        qv=-quaternion[:,1:]
        body_rate=world_rate+2*torch.cross(qv,torch.cross(qv,world_rate,dim=-1)+w[:,None]*world_rate,dim=-1)
        rate=(body_rate/torch.pi).clamp(-.999,.999)
        thrust=(2*norm.clamp(max=15.)/15.-1).clamp(-.999,.999)
        action=torch.atanh(torch.cat([rate,thrust],-1))
        assert torch.isfinite(action).all()
        return action,self.active.clone()
