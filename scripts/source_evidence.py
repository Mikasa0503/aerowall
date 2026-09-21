"""Preserve source bytes and reject edits during simulator initialization."""
import hashlib
from pathlib import Path


def verify_sources(root, manifest):
    changed=[]
    for name, expected in manifest.items():
        path=Path(root)/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            changed.append(name)
    if changed:
        raise RuntimeError(f'Sources changed during initialization: {changed}')


def snapshot_sources(root, manifest, destination):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    for name, expected in manifest.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Not a project-relative source: {name}')
        data=(Path(root)/relative).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:
            raise RuntimeError(f'Source changed before snapshot: {name}')
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
    return str(destination)
