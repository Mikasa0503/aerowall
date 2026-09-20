"""Extract the audited archive into a fresh project-local runtime; never execute it."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from audit_runtime_archive import PREFIX, REQUIRED, check_member

ROOT = Path(__file__).resolve().parents[1]


def rebase(member):
    problem = check_member(member)
    if problem:
        raise ValueError(f"{member.name}: {problem}")
    if not member.name.startswith(PREFIX + "/"):
        return None
    member = copy.copy(member)
    member.name = member.name[len(PREFIX) + 1:]
    if member.islnk():
        member.linkname = member.linkname[len(PREFIX) + 1:]
    return member


def main():
    assert hasattr(tarfile, "data_filter"), "Python safety filter is required"
    download = json.loads((ROOT / "runs/runtime-download.json").read_text())
    audit = json.loads((ROOT / "runs/runtime-archive-audit.json").read_text())
    assert download["status"] == "downloaded_not_extracted", download
    assert audit["path_layout_passed"] and audit["gzip_integrity"] == "passed", audit
    source = Path(download["path"]).resolve()
    assert source.parent == ROOT / ".cache/downloads"
    assert source.stat().st_size == download["bytes"]
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == download["sha256"], "Archive changed after download"
    destination = ROOT / "third_party/isaac-sim-2023.1.0-hotfix.1"
    staging = destination.with_name(destination.name + ".staging")
    assert not destination.exists() and not staging.exists(), "Preserve existing installation/staging"
    assert shutil.disk_usage(destination.parent).free > audit["unpacked_bytes"] + 5 * 1024**3
    staging.mkdir(mode=0o700)
    seen = set()
    count = 0
    with tarfile.open(source, "r|gz") as archive:
        for original in archive:
            member = rebase(original)
            if member is None:
                continue
            if not member.isdir() and member.name in seen:
                raise ValueError(f"Duplicate non-directory path: {member.name}")
            seen.add(member.name)
            # data_filter rejects symlink traversal, external links and special files,
            # and removes ownership and unsafe permissions. HOME is never restored.
            archive.extract(member, staging, filter="data")
            count += 1
    assert REQUIRED <= seen
    staging.rename(destination)
    report = {"status": "extracted_not_executed", "source_sha256": digest.hexdigest(),
              "destination": str(destination), "extracted_members": count,
              "authenticity": "author_link_only_no_independent_vendor_checksum"}
    (ROOT / "runs/runtime-extraction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
