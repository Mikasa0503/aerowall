"""Conservative collider radius from actual geometry, not authored render extents."""
from itertools import product
import math
from pxr import Usd, UsdGeom, Gf


def collider_owner_radius(prim, owner):
    if prim.IsA(UsdGeom.Mesh):
        points=UsdGeom.Mesh(prim).GetPointsAttr().Get()
        assert points, f'Mesh has no points: {prim.GetPath()}'
    elif prim.IsA(UsdGeom.Cube):
        half=float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
        points=list(product((-half,half),repeat=3))
    elif prim.IsA(UsdGeom.Cylinder):
        shape=UsdGeom.Cylinder(prim);r=float(shape.GetRadiusAttr().Get());h=float(shape.GetHeightAttr().Get())/2
        extents=[r,r,r];extents[{'X':0,'Y':1,'Z':2}[str(shape.GetAxisAttr().Get())]]=h
        points=list(product(*[(-v,v) for v in extents]))
    else:
        raise RuntimeError(f'Unsupported contact source shape: {prim.GetPath()} {prim.GetTypeName()}')
    transform=UsdGeom.XformCache(Usd.TimeCode.Default()).ComputeRelativeTransform(prim,owner)[0]
    radius=max(math.sqrt(sum(float(x)**2 for x in transform.Transform(Gf.Vec3d(*point)))) for point in points)
    assert math.isfinite(radius) and radius>0
    return radius
