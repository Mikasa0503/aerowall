"""Inspect a complete runtime tarball without extracting or executing it."""
import argparse
import gzip
import json
from pathlib import Path, PurePosixPath
import posixpath
import tarfile

PREFIX = "home/chenyinuo/isaac-sim/isaac-sim-2023.1.0-hotfix.1"
REQUIRED = {"setup_conda_env.sh", "setup_python_env.sh", "python.sh", "kit/kit"}


def check_member(member):
    name = member.name.rstrip("/")
    parts = PurePosixPath(name).parts
    if name.startswith("/") or ".." in parts:
        return "absolute or parent-traversing member path"
    if name != PREFIX and not name.startswith(PREFIX + "/"):
        if member.isdir() and PREFIX.startswith(name + "/"):
            return None
        return "outside expected runtime root"
    if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
        return "special file type"
    if member.mode & 0o6000:
        return "setuid or setgid permission"
    if member.issym() or member.islnk():
        target = member.linkname
        if target.startswith("/"):
            return "absolute link target"
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target)
                                     if member.issym() else target)
        if resolved != PREFIX and not resolved.startswith(PREFIX + "/"):
            return "link target escapes runtime root"
    return None


def audit(path):
    # Read through the gzip footer before trusting tar metadata or considering extraction.
    with gzip.open(path, "rb") as stream:
        while stream.read(8 * 1024 * 1024):
            pass
    found = set()
    count = 0
    unpacked_bytes = 0
    problems = []
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            count += 1
            unpacked_bytes += member.size
            problem = check_member(member)
            if problem:
                problems.append({"path": member.name, "reason": problem, "link": member.linkname})
            if member.name.startswith(PREFIX + "/"):
                found.add(member.name[len(PREFIX) + 1:])
    return {"archive": str(path), "gzip_integrity": "passed", "expected_prefix": PREFIX,
            "members": count, "unpacked_bytes": unpacked_bytes, "problems": problems,
            "missing_required": sorted(REQUIRED - found),
            "path_layout_passed": not problems and REQUIRED <= found,
            "authenticity": "not_proven_by_archive_checks", "execution": "not_performed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "problems"}))
    print(f"Unsafe or unexpected entries: {len(report['problems'])}")
    return 0 if report["path_layout_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
