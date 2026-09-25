import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_runtime import rebase

PREFIX = "home/example-user/isaac-sim/isaac-sim-2023.1.0-hotfix.1"


class RuntimeExtractionTests(unittest.TestCase):
    def test_safe_prefix_rebase_and_link(self):
        item = tarfile.TarInfo(PREFIX + "/python/bin/python")
        item.type = tarfile.SYMTYPE
        item.linkname = "python3.10"
        result = rebase(item, PREFIX)
        self.assertEqual(result.name, "python/bin/python")
        self.assertEqual(result.linkname, "python3.10")
        self.assertEqual(item.name, PREFIX + "/python/bin/python")

    def test_archive_hardlink_rebase(self):
        item = tarfile.TarInfo(PREFIX + "/b")
        item.type = tarfile.LNKTYPE
        item.linkname = PREFIX + "/a"
        self.assertEqual(rebase(item, PREFIX).linkname, "a")

    @unittest.skipUnless(hasattr(tarfile, "data_filter"), "Requires patched Python tar extraction filters; verified in the isolated Linux environment")
    def test_existing_symlink_cannot_redirect_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "runtime"
            destination.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (destination / "link").symlink_to(outside, target_is_directory=True)
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                item = tarfile.TarInfo(PREFIX + "/link/payload")
                item.size = 1
                archive.addfile(item, io.BytesIO(b"x"))
            buffer.seek(0)
            with tarfile.open(fileobj=buffer) as archive:
                with self.assertRaises(tarfile.OutsideDestinationError):
                    archive.extract(rebase(archive.next(), PREFIX), destination, filter="data")
            self.assertFalse((outside / "payload").exists())


if __name__ == "__main__":
    unittest.main()
