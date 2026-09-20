import importlib.util
from pathlib import Path
import tarfile
import unittest

spec = importlib.util.spec_from_file_location("audit", Path(__file__).resolve().parents[1] / "scripts/audit_runtime_archive.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ArchivePathTests(unittest.TestCase):
    def member(self, name, kind=tarfile.REGTYPE, link=""):
        value = tarfile.TarInfo(name)
        value.type = kind
        value.linkname = link
        return value

    def test_regular_and_relative_internal_link(self):
        self.assertIsNone(module.check_member(self.member(module.PREFIX + "/python.sh")))
        self.assertIsNone(module.check_member(self.member(module.PREFIX + "/lib/libfoo.so", tarfile.SYMTYPE, "libfoo.so.1")))

    def test_parent_traversal_and_absolute_path(self):
        for name in ["/etc/profile", module.PREFIX + "/../../escape", "unrelated/file"]:
            self.assertIsNotNone(module.check_member(self.member(name)))

    def test_absolute_and_escaping_links(self):
        for target in ["/etc/passwd", "../../../../../../outside"]:
            self.assertIsNotNone(module.check_member(self.member(module.PREFIX + "/lib/a", tarfile.SYMTYPE, target)))

    def test_special_files_and_privilege_bits(self):
        self.assertIsNotNone(module.check_member(self.member(module.PREFIX + "/device", tarfile.CHRTYPE)))
        member = self.member(module.PREFIX + "/executable")
        member.mode = 0o4755
        self.assertIsNotNone(module.check_member(member))

    def test_only_ancestor_directories_allowed(self):
        self.assertIsNone(module.check_member(self.member("home/chenyinuo", tarfile.DIRTYPE)))
        self.assertIsNotNone(module.check_member(self.member("home/chenyinuo")))


if __name__ == "__main__":
    unittest.main()
