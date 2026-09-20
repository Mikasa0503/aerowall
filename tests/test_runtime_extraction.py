import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_runtime import PREFIX, rebase


class RuntimeExtractionTests(unittest.TestCase):
    def test_safe_prefix_rebase_and_link(self):
        item = tarfile.TarInfo(PREFIX + "/python/bin/python")
        item.type = tarfile.SYMTYPE
        item.linkname = "python3.10"
        result = rebase(item)
        self.assertEqual(result.name, "python/bin/python")
        self.assertEqual(result.linkname, "python3.10")
        self.assertEqual(item.name, PREFIX + "/python/bin/python")

    def test_archive_hardlink_rebase(self):
        item = tarfile.TarInfo(PREFIX + "/b")
        item.type = tarfile.LNKTYPE
        item.linkname = PREFIX + "/a"
        self.assertEqual(rebase(item).linkname, "a")

    @unittest.skipUnless(hasattr(tarfile, "data_filter"), "Requires patched Python tar extraction filters; verified on 138")
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
                    archive.extract(rebase(archive.next()), destination, filter="data")
            self.assertFalse((outside / "payload").exists())


if __name__ == "__main__":
    unittest.main()
