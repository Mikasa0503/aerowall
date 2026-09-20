"""Original SingleJuggle physics plus a cloned wall, for contact integration.

This scene intentionally retains upstream observation/reward methods only as
fixture infrastructure. It is not registered as a finished WallRally task and
must not be trained as one. Task observations/rewards follow the action gate.
"""
import numpy as np
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.materials import PhysicsMaterial
from omni_drones.envs.pinball.single_juggle import SingleJuggle


class WallContactScene(SingleJuggle):
    def _design_scene(self):
        global_paths=super()._design_scene()
        spec=self.cfg.wall_fixture
        self.bat_overlay={'enabled':False}
        if spec.get('align_cap_to_visual_top',False):
            # Project-stage override only; do not edit the pinned asset file.
            import omni.usd
            from pxr import UsdGeom,Usd,Gf
            stage=omni.usd.get_context().get_stage()
            root=stage.GetPrimAtPath('/World/envs/env_0/Air_0/bat')
            visual=stage.GetPrimAtPath(str(root.GetPath())+'/visuals')
            collider=stage.GetPrimAtPath(str(root.GetPath())+'/collisions')
            cache=UsdGeom.XformCache(Usd.TimeCode.Default())
            relative=cache.ComputeRelativeTransform(visual,root)[0]
            visual_top=relative.Transform(Gf.Vec3d(*UsdGeom.Mesh(visual).GetExtentAttr().Get()[1]))[2]
            half_height=UsdGeom.Cylinder(collider).GetHeightAttr().Get()/2
            offset=float(visual_top-half_height)
            collider.GetAttribute('xformOp:translate').Set(Gf.Vec3d(0.,0.,offset))
            self.bat_overlay={'enabled':True,'collider_z_offset':offset,'visual_top_z':float(visual_top),
                              'physical_top_z':offset+half_height,'note':'Collider translation only; auto COM/inertia must be revalidated'}
        self.wall_front=float(spec.center[0])-float(spec.dimensions[0])/2
        material=PhysicsMaterial('/World/Physics_Materials/wall_fixture',restitution=float(spec.restitution))
        FixedCuboid('/World/envs/env_0/wall',translation=np.array(spec.center),
                    scale=np.array(spec.dimensions),size=1.,physics_material=material)
        return global_paths
