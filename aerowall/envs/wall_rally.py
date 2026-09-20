"""Development WallRally with substep event scoring and observable target/phase.

Targets are a configured sequence, indexed only by actual wall events. No ball
state writes occur beyond the inherited initial episode reset.
"""
import torch
from tensordict import TensorDict
from torchrl.data import UnboundedContinuousTensorSpec
from aerowall.envs.aligned_juggle import AlignedJuggle
from aerowall.rally_events import ILLEGAL, Kind


class WallRally(AlignedJuggle):
    illegal_contact_kinds = ILLEGAL
    extra_observation_dim = 19  # omega3, wall geometry6, target3, action4, phase3

    def __init__(self, cfg, headless=True):
        self.wall_spec = cfg.wall_task
        self.targets = None
        super().__init__(cfg, headless=headless)
        self.target_sequence = torch.tensor(self.wall_spec.targets_yz, device=self.device)
        assert self.target_sequence.ndim == 2 and self.target_sequence.shape[1] == 2
        self.targets = self.envs_positions.clone()
        self.targets[:,0] += self.wall_front
        self.targets[:,1:] += self.target_sequence[0]
        self.reset_boundary_failures = []
        self.wall_totals = {'rallies':0,'joint_rallies':0,'wall_hits':0,'episodes':0}

    def _set_specs(self):
        super()._set_specs()
        assert not self.cfg.task.time_encoding, 'Legacy transfer prefix is 24 observable features'
        self.observation_spec['agents','observation'] = UnboundedContinuousTensorSpec((self.num_envs,1,43),device=self.device)
        self.observation_spec['agents','state'] = UnboundedContinuousTensorSpec((self.num_envs,47),device=self.device)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # The inherited reset only initialized thrust; clear all four newly
        # observed prior-action channels so episodes cannot leak history.
        self.info['prev_action'][env_ids]=0.
        self.info['policy_action'][env_ids]=0.
        self.prev_actions[env_ids]=0.
        if self.targets is not None:
            self.targets[env_ids] = self.envs_positions[env_ids]
            self.targets[env_ids,0] += self.wall_front
            self.targets[env_ids,1:] += self.target_sequence[0]

    def _compute_state_and_obs(self):
        td = super()._compute_state_and_obs()
        n = self.num_envs
        geometry = torch.tensor([self.wall_front,-1.,0.,0.,float(self.cfg.wall_fixture.dimensions[1]),float(self.cfg.wall_fixture.dimensions[2])],device=self.device).expand(n,1,6)
        target = (self.targets-self.envs_positions)[:,None,:] if self.targets is not None else torch.zeros(n,1,3,device=self.device)
        phase = torch.zeros(n,1,3,device=self.device)
        if self.router is not None:
            for i,state in enumerate(self.router.batch.states):
                phase[i,0,{'wait_bat':0,'to_wall':1,'to_bat':2}[state.phase]]=1.
        omega = self.drone.get_velocities()[...,3:]
        extra = torch.cat([omega,geometry,target,self.info['prev_action'],phase],dim=-1)
        obs = torch.cat([td['agents','observation'],extra],dim=-1)
        time = (self.progress_buf/self.max_episode_length)[:,None].expand(n,4)
        td['agents','observation']=obs
        td['agents','state']=torch.cat([obs[:,0],time],dim=-1)
        return td

    def _begin_policy_step(self):
        self.event_reward = torch.zeros(self.num_envs,1,device=self.device)
        self.wall_cap_counts = torch.zeros_like(self.event_reward)

    def _after_physics_contacts(self, impacts, events, substep):
        bp = self.ball.get_world_poses()[0][:,0]-self.envs_positions
        dp = self.drone.get_world_poses()[0][:,0]-self.envs_positions
        bv = self.ball.get_velocities()[:,0,:3]
        low = torch.tensor(self.wall_spec.bounds_low,device=self.device)
        high = torch.tensor(self.wall_spec.bounds_high,device=self.device)
        outside = ((bp<low)|(bp>high)|(dp<low)|(dp>high)).any(-1)
        finite = torch.isfinite(torch.cat([bp,dp,bv,self.drone.get_velocities()[:,0]],-1)).all(-1)
        outside_cpu=outside.cpu().tolist();finite_cpu=finite.cpu().tolist()
        target_cpu=self.targets.cpu().tolist();progress_cpu=self.progress_buf.cpu().tolist()
        timeout = substep==self.substeps-1 and max(progress_cpu)+1>=self.max_episode_length
        for i,rows in enumerate(impacts):
            state = self.router.batch.states[i]
            was_done = state.terminated or state.truncated
            result = state.advance(rows,target_cpu[i],target_radius=float(self.wall_spec.target_radius),
                failure='numerical_failure' if not finite_cpu[i] else 'out_of_bounds' if outside_cpu[i] else None,
                time_limit=timeout and int(progress_cpu[i])+1>=self.max_episode_length)
            cap_credit=not was_done and not state.terminated and any(v.kind==Kind.CAP for v in rows)
            if cap_credit:self.wall_cap_counts[i]+=1
            for event in events:
                if event['env_id']==i:event['wall_cap_credit']=cap_credit and event['kind']=='cap' and event['credited']
            if was_done:continue
            if state.terminated:
                self.event_reward[i]-=float(self.wall_spec.failure_penalty)
                if progress_cpu[i]==0 and len(self.reset_boundary_failures)<16:
                    self.reset_boundary_failures.append({'env_id':i,'substep':substep,'reason':state.reason,'drone_local_position':dp[i].cpu().tolist(),'ball_local_position':bp[i].cpu().tolist(),'events':[e for e in events if e['env_id']==i]})
            if result['rally_completed']:
                self.event_reward[i]+=float(self.wall_spec.rally_reward)
                self.wall_totals['rallies']+=1
                if result.get('joint_completed'):
                    self.event_reward[i]+=float(self.wall_spec.target_reward)
                    self.wall_totals['joint_rallies']+=1
            if not state.terminated and any(v.kind==Kind.CAP for v in rows):
                # Small launch shaping at a real cap event; never scored as a wall hit.
                flight=(self.wall_front-bp[i,0])/bv[i,0].clamp_min(.1)
                predicted=bp[i,1:]+bv[i,1:]*flight
                predicted=predicted.clone();predicted[1]-=4.905*flight.square()
                target_yz=self.targets[i,1:]-self.envs_positions[i,1:]
                valid=(bv[i,0]>.1)&(flight>0)&(flight<2.)
                self.event_reward[i]+=float(self.wall_spec.launch_reward)*torch.exp(-2.*(predicted-target_yz).square().sum())*valid.float()
            count=result.get('publish_next_target',0)
            if count:
                self.wall_totals['wall_hits']+=count
                self.targets[i,1:]=self.envs_positions[i,1:]+self.target_sequence[state.target_index%len(self.target_sequence)]

    def _compute_reward_and_done(self):
        states = self.router.batch.states
        terminated=torch.tensor([s.terminated for s in states],device=self.device)[:,None]
        truncated=torch.tensor([s.truncated for s in states],device=self.device)[:,None]
        done=terminated|truncated
        receiving=torch.tensor([s.phase!='to_wall' for s in states],device=self.device)[:,None]
        distance=(self.ball_pos-self.drone.pos).norm(dim=-1)
        dense=torch.exp(-2.*distance)*receiving.float()
        smooth=self.action_error_order1.reshape(self.num_envs,-1).mean(-1,keepdim=True)
        reward=self.event_reward+self.policy_dt*(float(self.wall_spec.intercept_weight)*dense-float(self.wall_spec.smoothness_weight)*smooth)
        self.ball_last_2_vel=self.ball_last_vel.clone();self.ball_last_vel=self.ball_linear_vel.clone();self.hited_mark.zero_()
        self.stats['return'].add_(reward);self.stats['episode_len'][:]=self.progress_buf[:,None]
        self.stats['num_true_hits'].add_(self.wall_cap_counts)
        self.stats['truncated'].add_(truncated.float())
        for i in done.squeeze(-1).nonzero().flatten().cpu().tolist():
            state=states[i]
            self.completed_episodes.append({'env_id':i,'return':float(self.stats['return'][i]),'steps':int(self.progress_buf[i]),
                'legal_caps':int(self.stats['num_true_hits'][i]),'rallies':state.rallies,'max_streak':state.max_streak,
                'joint_rallies':state.joint_rallies,'wall_hits':state.wall_hits,'reason':state.reason})
            self.wall_totals['episodes']+=1
        self.contact_totals['legal_cap']+=int(self.wall_cap_counts.sum())
        self.contact_totals['illegal']+=int(self.action_illegal.sum())
        self.contact_totals['gpu_queries']+=self.action_gpu_queries
        self.contact_totals['reset_reentries']=self.router.reset_reentries
        self.contact_totals['stale_reset_points']=self.router.stale_reset_points
        return TensorDict({'stats':self.stats.clone(),'agents':{'reward':reward.unsqueeze(-1)},'done':done,'terminated':terminated,'truncated':truncated},self.num_envs)
