"""Route native GPU points and PhysX lifecycle edges to WallRally events.

All enabled Air collider owners must have one collision shape; this explicit
restriction prevents attributing a body's aggregated point buffer to the wrong
shape. Every positive contact manifold point is checked for cap legality.
"""
import re
from collections import Counter
import torch
from aerowall.gpu_contacts import GPUContactReadback
from aerowall.rally_events import Kind,RallyBatch


class WallContactRouter:
    def __init__(self,base,eager_gpu=True):
        import omni.usd
        from pxr import UsdPhysics,UsdGeom
        from omni.isaac.core.prims import RigidPrimView
        from omni.physx import get_physx_simulation_interface
        self.eager_gpu=eager_gpu
        self.reads_since_reset=[0]*base.num_envs
        self.base=base;self.interface=get_physx_simulation_interface();self.batch=RallyBatch(base.num_envs)
        stage=omni.usd.get_context().get_stage();owners=[]
        for prim in stage.Traverse():
            path=str(prim.GetPath())
            if path.startswith('/World/envs/env_0/Air_0/') and prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                owner=prim
                while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):owner=owner.GetParent()
                if not owner:raise RuntimeError(f'Collider without a rigid-body owner: {path}')
                owners.append(str(owner.GetPath()).split('/Air_0/')[1])
        counts=Counter(owners)
        if not counts or any(v!=1 for v in counts.values()) or 'bat' not in counts:
            raise RuntimeError(f'Unsupported collider-owner layout: {counts}')
        self.links=sorted(counts);self.ground='/World/defaultGroundPlane/GroundPlane/CollisionPlane'
        if not stage.GetPrimAtPath(self.ground):raise RuntimeError('Unexpected ground collider path')
        cylinder=UsdGeom.Cylinder(stage.GetPrimAtPath('/World/envs/env_0/Air_0/bat/collisions'))
        assert cylinder.GetAxisAttr().Get()=='Z'
        import numpy as np
        transform=UsdGeom.Xformable(cylinder).GetLocalTransformation()
        assert np.allclose(np.array(transform)[:3,:3],np.eye(3))
        self.collider_offset=torch.tensor(tuple(transform.ExtractTranslation()),device=base.device)
        self.radius=cylinder.GetRadiusAttr().Get()
        self.bats=RigidPrimView('/World/envs/env_*/Air_0/bat',name='wall_router_bats',reset_xform_properties=False);self.bats.initialize()
        self.bat_order=torch.tensor(sorted(range(base.num_envs),key=lambda j:int(re.search(r'/env_(\d+)/',self.bats.prim_paths[j]).group(1))),device=base.device)
        self.sensors=[];self.routes={}
        for i in range(base.num_envs):
            root=f'/World/envs/env_{i}';ball=root+'/ball';wall=root+'/wall'
            body_paths=[root+'/Air_0/'+name for name in self.links]
            filters=[wall,self.ground]+body_paths
            sensor=GPUContactReadback(ball,filters,f'wall_router_ball_{i}')
            si=len(self.sensors);self.sensors.append(sensor)
            for slot,path in enumerate(filters):
                kind=Kind.WALL if slot==0 else Kind.BALL_GROUND if slot==1 else Kind.CAP if path.endswith('/bat') else Kind.BALL_BODY
                self.routes[tuple(sorted((ball,path)))]=(i,si,slot,kind)
            for j,path in enumerate(body_paths):
                si=len(self.sensors);self.sensors.append(GPUContactReadback(path,[wall,self.ground],f'wall_router_body_{i}_{j}'))
                self.routes[tuple(sorted((path,wall)))]=(i,si,0,Kind.DRONE_WALL)
                self.routes[tuple(sorted((path,self.ground)))]=(i,si,1,Kind.DRONE_GROUND)

    def reset(self,indices):
        indices=list(indices)
        self.batch.reset(indices)
        for i in indices:self.reads_since_reset[i]=0

    def read(self):
        from pxr import PhysicsSchemaTools
        from contact_geometry import classify_cylinder_cap
        from omni_drones.utils.torch import quat_rotate
        data=[s.read() for s in self.sensors] if self.eager_gpu else {}
        def points_for(si,slot):
            if not self.eager_gpu and si not in data:data[si]=self.sensors[si].read()
            return data[si][slot]
        bp,bq=self.bats.get_world_poses();bp,bq=bp[self.bat_order],bq[self.bat_order]
        collision_center=bp+quat_rotate(bq,self.collider_offset.expand(self.base.num_envs,3))
        headers,cpu_data=self.interface.get_contact_report();impacts=[[] for _ in self.batch.states];events=[];observed=set()
        for header in headers:
            pair=tuple(sorted(str(PhysicsSchemaTools.intToSdfPath(getattr(header,k))) for k in ['actor0','actor1']))
            ids={int(m) for path in pair for m in re.findall(r'/env_(\d+)/',path)}
            if len(ids)>1:raise RuntimeError(f'Cross-environment contact: {pair}')
            route=self.routes.get(pair)
            if route is None:
                if any(path.endswith('/ball') for path in pair):raise RuntimeError(f'Unregistered ball contact: {pair}')
                continue # Intra-drone constraints/contacts are not external task pairs.
            i,si,slot,kind=route;edge=str(header.type).rsplit('_',1)[-1].lower();observed.add((si,slot))
            # Current CPU records can be valid for CPU-narrowphase shapes (e.g.
            # the original cylinder), while box contacts need GPU readback.
            # GPU buffers may retain a departed contact after a teleport, so
            # they never create events without a current pair header.
            cpu_samples=[]
            import math
            for offset in range(header.num_contact_data):
                sample=cpu_data[header.contact_data_offset+offset]
                impulse=tuple(sample.impulse);normal=tuple(sample.normal);position=tuple(sample.position)
                magnitude_cpu=math.sqrt(sum(v*v for v in impulse))
                norm=math.sqrt(sum(v*v for v in normal))
                if magnitude_cpu>1e-6 and .99<norm<1.01 and all(math.isfinite(v) for v in position+normal+impulse):
                    cpu_samples.append({'impulse':magnitude_cpu,'impulse_vector':impulse,'point':position,'normal':normal})
            samples=[] if edge=='lost' else cpu_samples if cpu_samples else points_for(si,slot)
            point_source='cpu_current_report' if cpu_samples else 'gpu_with_current_pair_header'
            magnitude=sum(s['impulse'] for s in samples)
            point=tuple(sum(s['point'][j]*s['impulse'] for s in samples)/magnitude for j in range(3)) if magnitude else None
            eligible=True
            if kind==Kind.CAP and samples:
                legal=[classify_cylinder_cap(torch.tensor(s['point'],device=self.base.device),torch.tensor(s['normal'],device=self.base.device),collision_center[i],bq[i],self.radius)[0] for s in samples]
                if not all(legal):kind=Kind.NON_CAP
            if kind==Kind.WALL and samples:
                x=float(self.base.envs_positions[i,0])+self.base.wall_front
                eligible=all(abs(s['point'][0]-x)<.01 and abs(s['normal'][0])>.99 for s in samples)
            try:
                impact=self.batch.ledgers[i].observe(pair,edge,kind,magnitude,point,eligible)
            except RuntimeError as exc:
                raise RuntimeError(f'{exc}; env={i}, reads_since_reset={self.reads_since_reset[i]}, progress={float(self.base.progress_buf[i])}, current_edge={edge}, impulse={magnitude}, active_pairs={self.batch.ledgers[i].active}') from exc
            if impact is not None:impacts[i].append(impact)
            events.append({'env_id':i,'pair':pair,'edge':edge,'kind':kind.value,'points':samples,'credited':impact is not None,'target_eligible':eligible,'point_source':point_source})
        unmatched=[(si,slot) for si,slots in enumerate(data) for slot,samples in enumerate(slots) if samples and (si,slot) not in observed] if self.eager_gpu else []
        self.gpu_queries_this_step=len(data)
        self.unmatched_gpu_scan_performed=self.eager_gpu
        self.unmatched_gpu_slots=unmatched # Diagnostic stale buffers, never score without current lifecycle headers.
        self.reads_since_reset=[v+1 for v in self.reads_since_reset]
        return impacts,events
