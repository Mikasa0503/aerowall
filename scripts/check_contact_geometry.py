"""Physical top/side/bottom and non-bat contact fixtures on the original asset."""
import re
import torch


def check_geometry(env, base, record, pitch_rate=0.):
    import omni.usd
    import carb.settings
    from pxr import UsdGeom, UsdPhysics, PhysicsSchemaTools
    from omni.isaac.core.prims import RigidPrimView
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType, SETTING_DISABLE_CONTACT_PROCESSING
    from omni_drones.utils.torch import quat_rotate
    from contact_geometry import classify_cylinder_cap
    bats = RigidPrimView('/World/envs/env_*/Air_0/bat', name='geometry_bats',reset_xform_properties=False); bats.initialize()
    rotors = RigidPrimView('/World/envs/env_*/Air_0/rotor_0', name='geometry_rotors',reset_xform_properties=False); rotors.initialize()
    def order(view):
        return torch.tensor(sorted(range(base.num_envs),key=lambda i:int(re.search(r'/env_(\d+)/',view.prim_paths[i]).group(1))),device=base.device)
    bat_order, rotor_order = order(bats), order(rotors)
    cylinder = UsdGeom.Cylinder(omni.usd.get_context().get_stage().GetPrimAtPath('/World/envs/env_0/Air_0/bat/collisions'))
    radius = cylinder.GetRadiusAttr().Get()
    collision_flags = {str(p.GetPath()):UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
                       for p in omni.usd.get_context().get_stage().Traverse()
                       if str(p.GetPath()).startswith('/World/envs/env_0/Air_0/') and p.HasAPI(UsdPhysics.CollisionAPI)}
    interface = get_physx_simulation_interface()
    settings = carb.settings.get_settings(); previous = settings.get_as_bool(SETTING_DISABLE_CONTACT_PROCESSING)
    settings.set_bool(SETTING_DISABLE_CONTACT_PROCESSING,False)
    trials = []
    try:
        for phase in ['top','side','bottom','rotor']:
            env.reset()
            pos = base.envs_positions[:,None,:].clone(); pos[...,2] += 3.
            angles = torch.linspace(-.7,.7,base.num_envs,device=base.device)
            quat = torch.zeros(base.num_envs,1,4,device=base.device)
            quat[:,0,0] = torch.cos(angles/2); quat[:,0,2] = torch.sin(angles/2)
            base.drone.set_world_poses(pos,quat)
            initial_velocity = torch.zeros(base.num_envs,1,6,device=base.device)
            initial_velocity[:,0,4] = torch.linspace(-pitch_rate,pitch_rate,base.num_envs,device=base.device)
            base.drone.set_velocities(initial_velocity)
            base.sim.step(render=False)
            bp,bq = bats.get_world_poses(); bp,bq = bp[bat_order].clone(),bq[bat_order].clone()
            offsets = {'top':[0.,0.,.2],'side':[.14 if pitch_rate else .2,0.,.025],'bottom':[0.,0.,-.2],'rotor':[0.,0.,.18]}
            speeds = {'top':[0.,0.,-2.],'side':[-2.,0.,0.],'bottom':[0.,0.,2.],'rotor':[0.,0.,-2.]}
            center = rotors.get_world_poses()[0][rotor_order].clone() if phase == 'rotor' else bp
            offset = torch.tensor(offsets[phase],device=base.device).expand(base.num_envs,3)
            velocity = quat_rotate(bq,torch.tensor(speeds[phase],device=base.device).expand(base.num_envs,3))
            body_velocity = bats.get_velocities()[bat_order]
            velocity += body_velocity[:,:3]
            # Initialize relative incidence in the rotating body's local frame.
            # After this fixture initialization, both bodies evolve only in PhysX.
            velocity += torch.cross(body_velocity[:,3:],center-bp+quat_rotate(bq,offset),dim=-1)
            bv = torch.zeros(base.num_envs,1,6,device=base.device); bv[:,0,:3] = velocity
            base.ball.set_world_poses((center+quat_rotate(bq,offset))[:,None,:],quat)
            base.ball.set_velocities(bv)
            contacts = []
            for step in range(10):
                base.sim.step(render=False)
                bp,bq = bats.get_world_poses(); bp,bq = bp[bat_order],bq[bat_order]
                headers,data = interface.get_contact_report()
                for h in headers:
                    if h.type != ContactEventType.CONTACT_FOUND or not h.num_contact_data: continue
                    actors = [str(PhysicsSchemaTools.intToSdfPath(h.actor0)),str(PhysicsSchemaTools.intToSdfPath(h.actor1))]
                    if not any(p.endswith('/ball') for p in actors): continue
                    index = int(re.search(r'/env_(\d+)/',next(p for p in actors if p.endswith('/ball'))).group(1))
                    point = data[h.contact_data_offset]
                    top,local,axis,alignment = classify_cylinder_cap(torch.tensor(list(point.position),device=base.device),
                        torch.tensor(list(point.normal),device=base.device),bp[index],bq[index],radius)
                    is_bat = any(p.endswith('/bat') for p in actors)
                    contacts.append({'env_id':index,'step':step,'actors':actors,'bat_contact':is_bat,
                                     'classified_top':top if is_bat else False,'local_point':local.cpu().tolist(),'alignment':alignment})
            covered = {e['env_id'] for e in contacts if e['bat_contact']} if phase != 'rotor' else {e['env_id'] for e in contacts if any('/rotor_' in p for p in e['actors'])}
            wrong = [e for e in contacts if e['bat_contact'] and e['classified_top'] != (phase == 'top')]
            result = {'phase':phase,'covered_envs':sorted(covered),'contacts':contacts,'wrong_classifications':wrong,
                      'passed':len(covered)==base.num_envs and not wrong}
            trials.append(result); record(contact_geometry_progress=result)
        result = {'passed':all(t['passed'] for t in trials),'trials':trials,'tilt_range_radians':[-.7,.7],
                  'scope':'controlled top/side/bottom/rotor contact fixtures; coverage limited to specified initial tilts/rates',
                  'pitch_rate_range':[-pitch_rate,pitch_rate],
                  'side_initial_offset':[.14 if pitch_rate else .2,0.,.025],
                  'incidence':'body-local approach plus initial rigid-body point velocity',
                  'angular_motion_coverage':pitch_rate != 0.,'collision_flags':collision_flags}
        record(contact_geometry_checks=result)
        return result
    finally:
        settings.set_bool(SETTING_DISABLE_CONTACT_PROCESSING,previous)
