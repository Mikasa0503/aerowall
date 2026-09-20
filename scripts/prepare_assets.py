"""Restore three missing ground textures from the pinned HCSP reference checkout."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DONOR_COMMIT = '009961b8f5702dd0c1c943cef0e01e09dfcd138d'
NAMES = ['Wireframe_blue.png', 'WireframeBlur_blue.png', 'WireframeBlur_basecolor.png']
EXPECTED = ['940b96ee8c8e53bd1acbc770d4ca0ac98bed9c96439ab386d1b27dbc9790cc4e',
            '44fc0e82007bb0bd94631cd7ec45986a71ed74b9413899fbffac13efe151be4b',
            'deab39088eaf39ed7be951966c5e6bd5e58b90ab64743b0cdc8e5579fd5a99d9']
donor = ROOT / 'third_party/HCSP'
actual = subprocess.check_output(['git', '-C', str(donor), 'rev-parse', 'HEAD'], text=True).strip()
assert actual == DONOR_COMMIT, (actual, DONOR_COMMIT)
source = donor / 'hcsp/envs/assets/Materials/Textures'
destination = ROOT / 'third_party/JuggleRL_train/omni_drones/envs/assets/Materials/Textures'
destination.mkdir(parents=True, exist_ok=True)
rows = []
for name, expected in zip(NAMES, EXPECTED):
    incoming = source / name
    target = destination / name
    digest = hashlib.sha256(incoming.read_bytes()).hexdigest()
    assert digest == expected, f'Donor texture changed: {incoming}'
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, f'Preserve different existing file: {target}'
    else:
        shutil.copyfile(incoming, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    rows.append({'source': str(incoming.relative_to(ROOT)), 'destination': str(target.relative_to(ROOT)),
                 'sha256': digest, 'bytes': target.stat().st_size})
report = {'source_repository': 'https://github.com/thu-uav/HCSP', 'source_commit': DONOR_COMMIT,
          'scope': 'copy missing textures only; no USD, physics, controller or existing source edits',
          'license_provenance': 'Original HCSP checkout and its license retained; Isaac asset license files remain with runtime',
          'files': rows}
(ROOT / 'docs/asset-restoration.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
