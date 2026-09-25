"""Audit the exact HCSP assets and task parameters used by the wall-RL task."""
import argparse
import hashlib
import json
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics
from omegaconf import OmegaConf


def attrs(prim, names):
    return {name: str(prim.GetAttribute(name).Get()) for name in names if prim.HasAttribute(name)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    hcsp = root / "third_party/HCSP"
    asset = hcsp / "hcsp/robots/assets/usd/iris_batVisualOnly.usd"
    iris_cfg = hcsp / "hcsp/robots/assets/usd/iris.yaml"
    serve_cfg = hcsp / "cfg/task/Serve.yaml"
    serve_source = hcsp / "hcsp/envs/low_level_skill/serve.py"
    stage = Usd.Stage.Open(str(asset))
    assert stage is not None
    colliders = []
    bat_prims = []
    for prim in stage.Traverse():
        row = {"path": str(prim.GetPath()), "type": prim.GetTypeName()}
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            row["geometry"] = attrs(prim, ["size", "radius", "height", "extent", "xformOp:scale", "xformOp:translate"])
            colliders.append(row)
        if "bat" in str(prim.GetPath()).lower():
            row["collision_api"] = prim.HasAPI(UsdPhysics.CollisionAPI)
            row["geometry"] = attrs(prim, ["extent", "xformOp:scale", "xformOp:translate"])
            bat_prims.append(row)
    cfg = OmegaConf.load(serve_cfg)
    vehicle = OmegaConf.load(iris_cfg)
    text = serve_source.read_text()
    assertions = {
        "asset_has_no_bat_collision_api": not any(v["collision_api"] for v in bat_prims),
        "base_link_is_collision_body": any(v["path"].endswith("/base_link/collisions") for v in colliders),
        "ball_mass_kg_is_0p005": float(cfg.ball_mass) == 0.005,
        "ball_radius_m_is_0p1": float(cfg.ball_radius) == 0.1,
        "vehicle_mass_kg_is_1p52": float(vehicle.mass) == 1.52,
        "source_racket_radius_m_is_0p2": "self.racket_r = 0.2" in text,
        "source_racket_height_is_two_ball_radii": "2.0 * self.ball_radius" in text,
        "source_material_restitution_is_0p8": "restitution=0.8" in (hcsp / "hcsp/envs/low_level_skill/volley_env.py").read_text(),
    }
    assert all(assertions.values()), assertions
    tracked = [asset, iris_cfg, serve_cfg, serve_source, hcsp / "hcsp/envs/low_level_skill/volley_env.py"]
    report = {
        "status": "passed",
        "scope": "Static audit of exact HCSP Iris/ball/racket model inputs",
        "hcsp_commit": __import__("subprocess").check_output(["git", "-C", str(hcsp), "rev-parse", "HEAD"], text=True).strip(),
        "assertions": assertions,
        "model": {
            "drone": "Iris/IrisTest (same iris_batVisualOnly.usd and iris.yaml)",
            "mass_kg_config": float(vehicle.mass),
            "ball_mass_kg": float(cfg.ball_mass),
            "ball_radius_m": float(cfg.ball_radius),
            "material_restitution_parameter": 0.8,
            "racket_radius_m_geometric_classifier": 0.2,
            "racket_height_m_geometric_classifier": 2 * float(cfg.ball_radius),
            "visible_bat_has_collision_api": False,
            "physical_ball_response_source": "base_link collision; legal-racket label is a geometric classifier around the visible bat",
        },
        "colliders": colliders,
        "bat_prims": bat_prims,
        "sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked},
        "limitations": [
            "This audit proves asset/config structure, not runtime contact behavior.",
            "A separate physics probe must verify center, edge, body and wall contacts before training.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "assertions": assertions}, indent=2))


if __name__ == "__main__":
    main()
