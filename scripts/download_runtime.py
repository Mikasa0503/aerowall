"""Download the author-linked archive without extracting or executing its contents."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SIZE = 10246767862  # HEAD response, 2026-09-20; not an authenticity proof.
CACHE = ROOT / ".cache/downloads"
STATE = ROOT / "runs/runtime-download.json"


def save(**values):
    state = {"time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "pid": os.getpid(), "source": "HCSP and VolleyBots pinned README Google Drive link",
             "file_id": "1Rt4B3U3nGtnvqXrzTAEa6JcH5OaMxqfY", **values}
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(STATE)
    print(json.dumps(state), flush=True)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    destination = CACHE / "isaac-sim-author-2023.1.0-hotfix.1.tar.gz"
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        raise RuntimeError(f"Existing download preserved; inspect it before retrying: {destination}")
    if shutil.disk_usage(CACHE).free < EXPECTED_SIZE + 30 * 1024**3:
        raise RuntimeError("Less than download size plus 30 GiB reserve")
    url = (ROOT / ".cache/drive-download-url.txt").read_text().strip()
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "drive.usercontent.google.com":
        raise RuntimeError("Unexpected public download destination")
    if parse_qs(parsed.query).get("id") != ["1Rt4B3U3nGtnvqXrzTAEa6JcH5OaMxqfY"]:
        raise RuntimeError("Unexpected file ID")
    save(status="downloading", path=str(partial), expected_bytes=EXPECTED_SIZE)
    args = ["curl", "--fail", "--location", "--proto", "=https", "--proto-redir", "=https",
            "--proxy", "http://127.0.0.1:17891", "--connect-timeout", "20", "--retry", "3",
            "--retry-delay", "10", "--speed-limit", "1024", "--speed-time", "90",
            "--continue-at", "-", "--output", str(partial), "--dump-header", str(CACHE / "runtime-headers.txt"), url]
    result = subprocess.run(args)
    if result.returncode:
        save(status="download_failed", curl_exit=result.returncode, bytes=partial.stat().st_size if partial.exists() else 0)
        return result.returncode
    size = partial.stat().st_size
    with partial.open("rb") as stream:
        magic = stream.read(2)
    if size != EXPECTED_SIZE or magic != b"\x1f\x8b":
        save(status="validation_failed", bytes=size, gzip_magic=magic.hex())
        return 2
    save(status="hashing", bytes=size)
    digest = hashlib.sha256()
    with partial.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    partial.replace(destination)
    save(status="downloaded_not_extracted", path=str(destination), bytes=size,
         sha256=digest.hexdigest(), authenticity="author_link_only_no_independent_vendor_checksum")
    return 0


if __name__ == "__main__":
    sys.exit(main())
