"""Explicit single-body adapter for the released full HCSP phase-one controller.

Three own-role observation slots share the physical body. Opponent slots are
mirrored placeholders, not simulated agents. Learned high-level decisions are
arbitrated FirstPass > SecPass > Attack when multiple hitting roles are selected.
"""
import torch
import numpy as np
from tensordict import TensorDict
from hcsp.envs.co_self_play.coselfplay_phase_one import Coselfplay_Phase_one
from hcsp.robots.drone import MultirotorBase
from hcsp.utils.volleyball.common import transfer_root_state_to_the_other_side
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims.rigid_contact_view import RigidContactView

class SingleBodyRoles:
    def __init__(self,real,owner):self.real=real;self.owner=owner;self.n=6
    def __getattr__(self,key):return getattr(self.real,key)
    def get_state(self,*args,**kw):
        state=self.real.get_state(*args,**kw)
        mirror=transfer_root_state_to_the_other_side(state)
        return torch.cat([mirror,mirror,state,state,state,mirror],1)
    @property
    def pos(self):
        p=self.real.pos;mirror=p.clone();mirror[...,:2]*=-1
        return torch.cat([mirror,mirror,p,p,p,mirror],1)
    def set_world_poses(self,positions,orientations,env_ids=None):
        return self.real.set_world_poses(positions[:,2:3],orientations[:,2:3],env_ids)
    def set_velocities(self,velocities,env_ids=None):return self.real.set_velocities(velocities[:,2:3],env_ids)
    def apply_action(self,actions):
        o=self.owner;h=o.info['high_level_action'][:,0]
        role=torch.full((o.num_envs,),2,device=o.device,dtype=torch.long)
        role=torch.where((h[:,2]==1)|(h[:,2]==2),4,role)
        role=torch.where(h[:,1]==1,3,role)
        role=torch.where((h[:,0]==3)|(h[:,0]==4),2,role)
        if o.cfg.get('single_priority','firstpass')=='set':
            role=torch.where(h[:,1]==1,3,role)
        elif o.cfg.get('single_priority','firstpass')=='attack':
            role=torch.where((h[:,2]==1)|(h[:,2]==2),4,role)
        role=torch.where(~o.is_rally,2,role)
        o.executed_role=role
        # Codes: FP goto/pass/hover/serve/serve-hover, Set goto/hit/hover,
        # Attack goto/hit/hover. Record the actually applied branch.
        skill=torch.where(h[:,0]<=2,0,torch.where(h[:,0]<=4,1,2))
        skill=torch.where(role==3,torch.where(h[:,1]==0,5,torch.where(h[:,1]==1,6,7)),skill)
        skill=torch.where(role==4,torch.where(h[:,2]==0,8,torch.where(h[:,2]<=2,9,10)),skill)
        skill=torch.where(~o.is_rally,torch.where(o.serve_step==0,3,4),skill)
        o.executed_skill=skill
        o.executed_action=actions[torch.arange(o.num_envs,device=o.device),role].unsqueeze(1).clone()
        effort=self.real.apply_action(o.executed_action)
        return effort.expand(-1,6)

class HCSPSingleWall(Coselfplay_Phase_one):
    def _design_scene(self):
        model=MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        original=model.spawn
        def spawn_one(robot,*args,**kwargs):
            kwargs['translations']=kwargs['translations'][:1]
            return original(robot,*args,**kwargs)
        model.spawn=spawn_one
        try:paths=super()._design_scene()
        finally:model.spawn=original
        self.physical_drone=self.drone
        self.drone=SingleBodyRoles(self.physical_drone,self)
        material=PhysicsMaterial('/World/Physics_Materials/single_wall_material',restitution=0.8,static_friction=0.,dynamic_friction=0.)
        FixedCuboid('/World/envs/env_0/single_wall',position=np.array([0.,0.,4.]),scale=np.array([.2,8.,8.]),physics_material=material)
        return paths
    def __init__(self,cfg,headless):
        super().__init__(cfg,headless)
        assert self.physical_drone.n==1
        self.executed_role=torch.full((self.num_envs,),2,device=self.device,dtype=torch.long)
        self.executed_action=torch.zeros(self.num_envs,1,4,device=self.device)
        self.executed_skill=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.contact_points=[]
        self.global_step=-1
        self.episode_index=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.wall_views=[]
        for idx in range(self.num_envs):
            view=RigidContactView(f'/World/envs/env_{idx}/ball',
                [f'/World/envs/env_{idx}/{self.physical_drone.name}_0/base_link',f'/World/envs/env_{idx}/single_wall'],
                name=f'single_wall_events_{idx}',prepare_contact_sensors=False,disable_stablization=False,
                apply_rigid_body_api=False,max_contact_count=32)
            view.initialize();assert view.num_shapes==1 and view.num_filters==2
            self.wall_views.append(view)
        self.contact_live=torch.zeros(self.num_envs,2,device=self.device,dtype=torch.bool)
        self.contact_impulse=torch.zeros(self.num_envs,2,device=self.device)
        self.contact_entry=self.contact_live.clone()
        self.phase=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.wall_returns=torch.zeros_like(self.phase)
    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)
        # Native near-side serve distribution, now with only its physical server.
        pos=self.serve_pos_dist.sample(env_ids.shape).unsqueeze(1)
        quat=torch.zeros(len(env_ids),1,4,device=self.device);quat[...,0]=1
        self.physical_drone.set_world_poses(pos+self.envs_positions[env_ids,None,:],quat,env_ids)
        self.physical_drone.set_velocities(torch.zeros(len(env_ids),1,6,device=self.device),env_ids)
        bp=self.ball_pos_dist.sample((*env_ids.shape,1))
        self.ball.set_world_poses(bp+self.envs_positions[env_ids,None,:],quat,env_ids)
        self.ball.set_velocities(torch.zeros(len(env_ids),1,6,device=self.device),env_ids)
        self.serve_turn[env_ids]=False;self.ball_side[env_ids]=False;self.last_hit_side[env_ids]=True
        self.ball_last_vel[env_ids]=0;self.ball_init_vel[env_ids]=0
        if hasattr(self,'phase'):
            self.episode_index[env_ids]+=1
            self.phase[env_ids]=0;self.wall_returns[env_ids]=0;self.contact_live[env_ids]=False
    def _step(self,td):
        self.global_step+=1
        self._pre_sim_step(td)
        self.sim.step(self._should_render(0));self._post_sim_step(td);self.progress_buf+=1
        # Refresh physical state before observations and event bookkeeping.
        self.physical_drone.get_state()
        impulses=torch.stack([v.get_contact_force_matrix(dt=1.)[0].norm(dim=-1) for v in self.wall_views])
        live=impulses>1e-8;entry=live&~self.contact_live
        from hcsp.utils.torch import quat_rotate_inverse
        body_pos,body_quat=self.physical_drone.base_link.get_world_poses()
        for idx,view in enumerate(self.wall_views):
            if not live[idx].any():continue
            magnitudes,points,normals,separations,counts,starts=view.get_contact_force_data(dt=1.)
            assert int(counts.sum())<32
            for kind in (0,1):
                start=int(starts[0,kind]);count=int(counts[0,kind])
                for ci in range(start,start+count):
                    if abs(float(magnitudes[ci]))<=1e-8:continue
                    q=body_quat[idx,0];pos=body_pos[idx,0]
                    point_local=quat_rotate_inverse(q[None],(points[ci]-pos)[None])[0]
                    normal_local=quat_rotate_inverse(q[None],normals[ci][None])[0]
                    self.contact_points.append({'env':idx,'step':int(self.progress_buf[idx])-1,'global_step':self.global_step,'episode':int(self.episode_index[idx]),'kind':kind,
                        'impulse':float(magnitudes[ci]),'position':points[ci].cpu().tolist(),
                        'normal':normals[ci].cpu().tolist(),'body_local_point':point_local.cpu().tolist(),
                        'body_local_normal':normal_local.cpu().tolist(),
                        'ball_body_delta':(self.ball.get_world_poses()[0][idx,0]-pos).cpu().tolist()})
        self.contact_live=live;self.contact_impulse=impulses;self.contact_entry=entry
        body=entry[:,0];wall=entry[:,1];ambiguous=body&wall
        self.wall_returns+=(body&(self.phase==2)&~ambiguous).long()
        self.phase=torch.where(body,1,self.phase)
        self.phase=torch.where(wall&(self.phase==1),2,self.phase)
        self.phase=torch.where(ambiguous,0,self.phase)
        self.racket_hit_ball.zero_();self.drone_hit_ball=torch.zeros_like(self.racket_hit_ball)
        self.drone_hit_ball[torch.arange(self.num_envs,device=self.device),self.executed_role]=body
        # Contact is used only to rearm the wall task, never labelled a legal bat hit.
        self.serve_step+=body.long()
        self.is_rally|=wall
        for name in ('FirstPass','SecPass','Att','Opp_FirstPass','Opp_SecPass','Opp_Att'):
            getattr(self,name+'_already_hit')[wall]=False
        self.info['switch_turn']=(body|wall).unsqueeze(-1)
        self.ball_side.zero_();self.last_hit_side=torch.where(wall,True,self.last_hit_side)
        out=self._compute_state_and_obs(is_step=True)
        bp=self.ball_pos[:,0];dp=self.physical_drone.pos[:,0]
        ground=(bp[:,2]<=self.ball_radius)|(dp[:,2]<.3)
        bounds=(bp[:,0]<-.1)|(bp[:,0]>8)|(bp[:,1].abs()>4)|(bp[:,2]>8)|(dp[:,0]<.5)|(dp[:,0]>8)|(dp[:,1].abs()>4)
        terminated=(ground|bounds).unsqueeze(-1)
        truncated=(self.progress_buf>=self.max_episode_length).unsqueeze(-1)
        self.stats['episode_len']=self.progress_buf[:,None].float()
        self.stats['done']= (terminated|truncated).float()
        self.stats['done_ball_hit_the_ground']=(bp[:,2]<=self.ball_radius)[:,None].float()
        self.stats['done_drone_hit_the_ground']=(dp[:,2]<.3)[:,None].float()
        out.update(self.reward_spec.zero())
        out.update(TensorDict({'done':terminated|truncated,'terminated':terminated,'truncated':truncated},self.batch_size))
        return out
