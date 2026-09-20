"""Read original USD collider dimensions and compare central top surfaces."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from pxr import Usd,UsdGeom,Gf
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
s=Usd.Stage.Open(str(a.asset));cache=UsdGeom.XformCache(Usd.TimeCode.Default())
base=s.GetPrimAtPath('/air/base_link/collisions');bat=s.GetPrimAtPath('/air/bat/collisions')
assert base.GetTypeName()=='Cube' and bat.GetTypeName()=='Cylinder'
size=UsdGeom.Cube(base).GetSizeAttr().Get();half=size/2
corners=[cache.GetLocalToWorldTransform(base).Transform(Gf.Vec3d(*v)) for v in itertools.product([-half,half],repeat=3)]
height=UsdGeom.Cylinder(bat).GetHeightAttr().Get();radius=UsdGeom.Cylinder(bat).GetRadiusAttr().Get()
assert UsdGeom.Cylinder(bat).GetAxisAttr().Get()=='Z'
cap=cache.GetLocalToWorldTransform(bat).Transform(Gf.Vec3d(0,0,height/2))
base_top=max(v[2] for v in corners)
r={'asset':str(a.asset),'asset_sha256':hashlib.sha256(a.asset.read_bytes()).hexdigest(),
   'body_collision_box_min':[min(v[j] for v in corners) for j in range(3)],
   'body_collision_box_max':[max(v[j] for v in corners) for j in range(3)],
   'bat_cylinder_radius':radius,'bat_cylinder_height':height,'bat_top_center':list(cap),
   'base_top_above_bat_top_m':base_top-cap[2],
   'body_collision_enabled_in_original_asset':base.GetAttribute('physics:collisionEnabled').Get(),
   'bat_collision_enabled_in_original_asset':bat.GetAttribute('physics:collisionEnabled').Get(),
   'scope':'Authored relative asset geometry. Physics reset pose/solver results are verified separately. Body-enabled central cap access is not established.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
