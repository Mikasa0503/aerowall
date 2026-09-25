"""Load and validate the frozen wall-skill upgrade configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(path: str, config_path: Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    # Plan paths are repository-relative, even if the config is copied to 138.
    return (repo_root / candidate).resolve()


def verify_hashed_file(path: Path, expected_sha256: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch for {path}: {actual} != {expected_sha256}")
    return actual


def load_upgrade_config(path: Path, repo_root: Path, *, verify_banks: bool = True) -> dict:
    """Load schema v3 and verify every frozen initial-state bank by default."""
    config_path = path.resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != 3:
        raise ValueError("wall-skill upgrade config must use schema_version 3")
    if config.get("experiment") != "aerowall_skill_upgrade_v3":
        raise ValueError("unexpected wall-skill experiment identifier")
    for version in ("legacy", "relative_v2", "relative_v3", "aerowall_goal_v1"):
        if version not in config.get("observation_versions", {}):
            raise ValueError(f"missing observation semantic declaration: {version}")
    if "aerowall_causal_v3" not in config.get("reward_versions", []):
        raise ValueError("missing aerowall_causal_v3 reward declaration")
    if verify_banks:
        evaluation = config["evaluation"]
        banks = [config["baseline"]["initial_case_bank"],
                 evaluation["fixed_bank"], evaluation["training_bank"],
                 *evaluation["development_challenge_banks"]]
        seen = set()
        for item in banks:
            identity = (item["path"], item["sha256"])
            if identity in seen:
                continue
            seen.add(identity)
            bank_path = resolve_repo_path(item["path"], config_path, repo_root)
            verify_hashed_file(bank_path, item["sha256"], "initial-state bank")
    return config


def configured_checkpoint(config: dict, role: str, repo_root: Path, *, verify: bool = True) -> Path:
    """Return a checkpoint declared in the config and verify its pinned hash."""
    item = config["checkpoints"][role]
    path = Path(item["path"])
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if verify:
        verify_hashed_file(path, item["sha256"], f"{role} checkpoint")
    return path


def configured_bank(config: dict, name: str, repo_root: Path, *, verify: bool = True) -> Path:
    item = config["evaluation"][name]
    path = resolve_repo_path(item["path"], Path(), repo_root)
    if verify:
        verify_hashed_file(path, item["sha256"], f"{name} bank")
    return path
