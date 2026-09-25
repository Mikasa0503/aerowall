import importlib.util
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("audit", Path(__file__).resolve().parents[1] / "scripts/audit_runtime_archive.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PREFIX = "home/example-user/isaac-sim/isaac-sim-2023.1.0-hotfix.1"


class ArchivePathTests(unittest.TestCase):
    def member(self, name, kind=tarfile.REGTYPE, link=""):
        value = tarfile.TarInfo(name)
        value.type = kind
        value.linkname = link
        return value

    def test_regular_and_relative_internal_link(self):
        self.assertIsNone(module.check_member(self.member(PREFIX + "/python.sh"), PREFIX))
        self.assertIsNone(module.check_member(self.member(PREFIX + "/lib/libfoo.so", tarfile.SYMTYPE, "libfoo.so.1"), PREFIX))

    def test_parent_traversal_and_absolute_path(self):
        for name in ["/etc/profile", PREFIX + "/../../escape", "unrelated/file"]:
            self.assertIsNotNone(module.check_member(self.member(name), PREFIX))

    def test_absolute_and_escaping_links(self):
        for target in ["/etc/passwd", "../../../../../../outside"]:
            self.assertIsNotNone(module.check_member(self.member(PREFIX + "/lib/a", tarfile.SYMTYPE, target), PREFIX))

    def test_special_files_and_privilege_bits(self):
        self.assertIsNotNone(module.check_member(self.member(PREFIX + "/device", tarfile.CHRTYPE), PREFIX))
        member = self.member(PREFIX + "/executable")
        member.mode = 0o4755
        self.assertIsNotNone(module.check_member(member, PREFIX))

    def test_only_ancestor_directories_allowed(self):
        self.assertIsNone(module.check_member(self.member("home/example-user", tarfile.DIRTYPE), PREFIX))
        self.assertIsNotNone(module.check_member(self.member("home/example-user"), PREFIX))

    def test_archive_runtime_root_is_discovered_without_a_user_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                archive.addfile(self.member(PREFIX + "/python.sh"))
            self.assertEqual(module.discover_prefix(path), PREFIX)

    def test_archive_with_multiple_runtime_roots_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ambiguous.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                archive.addfile(self.member(PREFIX + "/python.sh"))
                other = PREFIX.replace("example-user", "other-user")
                archive.addfile(self.member(other + "/python.sh"))
            with self.assertRaisesRegex(ValueError, "found 2"):
                module.discover_prefix(path)


if __name__ == "__main__":
    unittest.main()
