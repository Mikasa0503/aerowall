"""Local namespace for Orbit imports; not a full NVIDIA asset distribution."""
from pathlib import Path


def configure_local_asset_root():
    import carb
    root = Path(__file__).resolve().parents[1]
    assets = root / '.cache/local-asset-namespace'
    for name in ('Isaac', 'NVIDIA'):
        (assets / name).mkdir(parents=True, exist_ok=True)
    # HCSP uses its own local USD files. Orbit imports unused camera/converter
    # modules which nevertheless demand these namespace directories. Any later
    # request for an unavailable asset still fails normally at its actual path.
    carb.settings.get_settings().set('/persistent/isaac/asset_root/default', str(assets))
    return str(assets)
